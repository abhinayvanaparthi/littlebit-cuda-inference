"""
cuda_ops.py - Loads the custom packed-binary GEMM CUDA extension and exposes:

  packed_matvec_cuda(X, W_packed_int32, orig_cols) -> Y
      Register-level GEMM: Y = X @ W^T without materializing W in VRAM.

  littlebit_forward_cuda(x, layer) -> y
      Full LittleBit forward pass using the custom CUDA kernel for both
      V and U matrix multiplications. Matches the reference in kernel_ref.py.

The extension is JIT-compiled on first import via torch.utils.cpp_extension.load.
Arch target: sm_86 (NVIDIA Ampere, RTX 30xx).
"""

from __future__ import annotations
import os
import sys
import torch
import torch.nn as nn
from typing import Any

# ---------------------------------------------------------------------------
# 1. Build / load the CUDA extension
# ---------------------------------------------------------------------------

# Must be set before importing cpp_extension loader so nvcc picks the right arch
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

_EXT_DIR  = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_EXT_DIR, "build_cuda_unpack")

def _load_extension():
    from torch.utils.cpp_extension import load

    # Windows MSVC flags passed through extra_cflags (for cuda_binding.cpp)
    # nvcc flags go in extra_cuda_cflags (for cuda_unpack.cu)
    _extra_cflags = [
        "/O2",          # MSVC optimisation
        "/EHsc",        # Exception handling (required by pybind11)
        "/std:c++17",   # C++17 for MSVC
    ]

    _extra_cuda_cflags = [
        "-O3",
        "-arch=sm_86",
        # NOTE: --use_fast_math is intentionally omitted: it demotes double
        # accumulations and breaks the 1e-3 accuracy threshold for large K.
        "-std=c++17",
        # Suppress noisy warnings from CUDA headers on MSVC
        "-Xcompiler", "/wd4819",
        "-Xcompiler", "/wd4267",
        "-Xcompiler", "/wd4244",
    ]

    ext = load(
        name="cuda_unpack_ext",
        sources=[
            os.path.join(_EXT_DIR, "cuda_binding.cpp"),
            os.path.join(_EXT_DIR, "cuda_unpack.cu"),
        ],
        extra_cflags=_extra_cflags,
        extra_cuda_cflags=_extra_cuda_cflags,
        build_directory=_BUILD_DIR,
        verbose=True,
    )
    return ext


os.makedirs(_BUILD_DIR, exist_ok=True)

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        print("[cuda_ops] Compiling CUDA extension (first run, may take ~30s)...")
        _ext = _load_extension()
        print("[cuda_ops] CUDA extension loaded successfully.")
    return _ext


# ---------------------------------------------------------------------------
# 2. Public API: packed_matvec_cuda
# ---------------------------------------------------------------------------

def packed_matvec_cuda(
    X: torch.Tensor,
    W_packed: torch.Tensor,
    orig_cols: int,
) -> torch.Tensor:
    """
    Register-level packed binary GEMM: Y = X @ W^T.

    Parameters
    ----------
    X : torch.Tensor
        Shape (B, orig_cols), dtype float32 or float16, on CUDA.
    W_packed : torch.Tensor
        Shape (out_features, words_per_row), dtype torch.uint32 or torch.int32
        (the kernel interprets the bits as uint32 regardless).
    orig_cols : int
        Unpadded inner dimension (columns >= orig_cols are masked out).

    Returns
    -------
    torch.Tensor
        Shape (B, out_features), same dtype and device as X.
    """
    ext = _get_ext()

    # ATen doesn't dispatch uint32 natively; view as int32 (same bit layout)
    if W_packed.dtype == torch.uint32:
        W_int32 = W_packed.view(torch.int32)
    elif W_packed.dtype == torch.int32:
        W_int32 = W_packed
    else:
        raise TypeError(
            f"W_packed must be uint32 or int32, got {W_packed.dtype}"
        )

    if not X.is_contiguous():
        X = X.contiguous()
    if not W_int32.is_contiguous():
        W_int32 = W_int32.contiguous()

    return ext.packed_matvec(X, W_int32, orig_cols)


# ---------------------------------------------------------------------------
# 3. Public API: littlebit_forward_cuda
# ---------------------------------------------------------------------------

def littlebit_forward_cuda(
    x: torch.Tensor,
    layer: Any,
) -> torch.Tensor:
    """
    Full LittleBit forward pass using register-level CUDA kernel.

    Forward equation:
      y_main = ((((x * v2) @ V.T) * (v1 * u2)) @ U.T) * u1
      if residual:
        y += ((((x * v2_R) @ V_R.T) * (v1_R * u2_R)) @ U_R.T) * u1_R
      if bias is not None:
        y += bias

    The '@V.T' and '@U.T' steps are computed by the custom CUDA kernel
    without materializing V or U in VRAM.

    Parameters
    ----------
    x : torch.Tensor
        Input activations, shape (..., in_features), float32 or float16, CUDA.
    layer : PackedLittleBitLayer or dict-like
        Packed layer with buffers: V_packed, U_packed, v2, v1, u2, u1,
        and optionally V_R_packed, U_R_packed, v2_R, v1_R, u2_R, u1_R, bias.

    Returns
    -------
    torch.Tensor
        Output of shape (..., out_features), same dtype as x.
    """
    orig_shape  = x.shape
    orig_dtype  = x.dtype
    calc_dtype  = x.dtype if x.dtype in (torch.float16, torch.float32) else torch.float32

    # Read layer metadata
    in_features  = int(getattr(layer, "in_features",  layer["in_features"]))
    out_features = int(getattr(layer, "out_features", layer["out_features"]))
    split_dim    = int(getattr(layer, "split_dim",    layer["split_dim"]))
    residual     = bool(getattr(layer, "residual",    layer["residual"]))

    hidden_dim = orig_shape[-1]
    if hidden_dim != in_features:
        raise ValueError(
            f"Input dim {hidden_dim} does not match layer in_features {in_features}"
        )

    # Flatten to 2D (B, in_features)
    x_flat = x.view(-1, hidden_dim).to(calc_dtype)
    B = x_flat.shape[0]
    dev = x_flat.device

    def _get(key: str) -> torch.Tensor:
        t = getattr(layer, key, None) if hasattr(layer, key) else layer.get(key, None)
        if t is None:
            raise KeyError(f"Layer missing tensor '{key}'")
        return t.to(device=dev, dtype=calc_dtype)

    def _get_packed(key: str) -> torch.Tensor:
        t = getattr(layer, key, None) if hasattr(layer, key) else layer.get(key, None)
        if t is None:
            raise KeyError(f"Layer missing packed tensor '{key}'")
        return t.to(device=dev)

    # --- Main branch ---
    v2 = _get("v2")
    v1 = _get("v1")
    u2 = _get("u2")
    u1 = _get("u1")
    V_packed = _get_packed("V_packed")
    U_packed = _get_packed("U_packed")

    a = x_flat * v2                                           # (B, in_features)
    h = packed_matvec_cuda(a, V_packed, in_features)         # (B, split_dim)
    g = h * (v1 * u2)                                        # (B, split_dim)
    z = packed_matvec_cuda(g, U_packed, split_dim)           # (B, out_features)
    y = z * u1                                               # (B, out_features)

    # --- Residual branch (optional) ---
    has_residual_weights = (
        hasattr(layer, "V_R_packed") or
        (hasattr(layer, "keys") and "V_R_packed" in layer.keys()) or
        (isinstance(layer, dict) and "V_R_packed" in layer)
    )
    if residual and has_residual_weights:
        v2_R = _get("v2_R")
        v1_R = _get("v1_R")
        u2_R = _get("u2_R")
        u1_R = _get("u1_R")
        V_R_packed = _get_packed("V_R_packed")
        U_R_packed = _get_packed("U_R_packed")

        a_R = x_flat * v2_R
        h_R = packed_matvec_cuda(a_R, V_R_packed, in_features)
        g_R = h_R * (v1_R * u2_R)
        z_R = packed_matvec_cuda(g_R, U_R_packed, split_dim)
        y_R = z_R * u1_R
        y   = y + y_R

    # --- Bias (optional) ---
    bias_tensor = getattr(layer, "bias", None)
    if bias_tensor is None and isinstance(layer, dict):
        bias_tensor = layer.get("bias", None)
    if bias_tensor is not None:
        y = y + bias_tensor.to(device=dev, dtype=calc_dtype)

    # Reshape and cast back to original dtype
    output_shape = (*orig_shape[:-1], out_features)
    return y.to(orig_dtype).view(output_shape)
