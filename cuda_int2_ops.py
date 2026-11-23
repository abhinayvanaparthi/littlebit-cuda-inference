"""
cuda_int2_ops.py
Standalone PyTorch wrapper and JIT loader for 2-bit (INT2) quantization:
- pack_int2_weight: Vectorized INT2 packing in PyTorch (16 weights per uint32, codes 0..3 mapping to {-1.5, -0.5, +0.5, +1.5})
- unpack_int2_weight: Reference PyTorch dequantizer to FP32
- unpack_int2_gemv: High-performance sm_86 CUDA register-unpacking GEMV kernel wrapper
"""

import os
import torch
from typing import Tuple
from torch.utils.cpp_extension import load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(CURRENT_DIR, "build_cuda_int2")
os.makedirs(BUILD_DIR, exist_ok=True)

_CUDA_INT2_EXT = None


def get_cuda_int2_ext():
    """Lazily JIT-compiles and returns the standalone CUDA INT2 extension."""
    global _CUDA_INT2_EXT
    if _CUDA_INT2_EXT is None:
        source_cu = os.path.join(CURRENT_DIR, "cuda_int2_unpack.cu")
        source_cpp = os.path.join(CURRENT_DIR, "cuda_int2_binding.cpp")

        _CUDA_INT2_EXT = load(
            name="cuda_int2_unpack_ext",
            sources=[source_cpp, source_cu],
            build_directory=BUILD_DIR,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "-gencode=arch=compute_86,code=sm_86",
            ],
            extra_cflags=["/O2"],
            verbose=False,
        )
    return _CUDA_INT2_EXT


def pack_int2_weight(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Packs a 2D weight matrix [N, K] into:
      W_packed: [N, K // 16] (int32)
      scales:   [N] (float16)

    Packing specification:
    - 16 weights per uint32 word (2 bits per weight).
    - Symmetric 4-level quantization grid: {-1.5, -0.5, +0.5, +1.5}.
    - Decision boundaries at -1.0, 0.0, +1.0.
    - Codes:
        0 -> -1.5
        1 -> -0.5
        2 -> +0.5
        3 -> +1.5
    - Per-row FP16 scale: s_i = max(|W_i|) / 1.5.
    """
    assert W.dim() == 2, f"Expected 2D weight matrix, got shape {W.shape}"
    N, K = W.shape
    assert K % 16 == 0, f"Column dimension K ({K}) must be a multiple of 16"

    W_f32 = W.to(torch.float32)

    # Per-row scale: s_i = max(|W_i|) / 1.5
    max_vals = W_f32.abs().amax(dim=1, keepdim=True)
    scales = torch.clamp(max_vals / 1.5, min=1e-5)

    # Normalized weights in range [-1.5, +1.5]
    W_norm = W_f32 / scales

    # Quantize to codes 0..3:
    #   W_norm < -1.0 -> code 0
    #   -1.0 <= W_norm < 0.0 -> code 1
    #   0.0 <= W_norm < 1.0 -> code 2
    #   W_norm >= 1.0 -> code 3
    # Equivalent to clamp(floor(W_norm + 2.0), 0, 3)
    codes = torch.clamp((W_norm + 2.0).floor(), 0, 3).to(torch.int32)

    # Pack 16 codes into each int32 word:
    # Weight 0 in bits [0..1], Weight 1 in bits [2..3], ..., Weight 15 in bits [30..31]
    codes_16 = codes.view(N, K // 16, 16)
    shifts = torch.arange(0, 32, 2, dtype=torch.int32, device=W.device)
    shifted = codes_16 << shifts.unsqueeze(0).unsqueeze(0)

    # Disjoint bit summation produces the packed int32 representation
    W_packed = shifted.sum(dim=-1, dtype=torch.int32)

    return W_packed, scales.squeeze(1).to(torch.float16)


def unpack_int2_weight(W_packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Unpacks W_packed [N, K // 16] and scales [N] back to an explicit FP32 matrix [N, K].
    Formula: W_dequant[i, j] = (code[i, j] - 1.5) * scales[i]
    """
    assert W_packed.dim() == 2, f"Expected 2D packed matrix, got {W_packed.shape}"
    N, words = W_packed.shape
    K = words * 16
    device = W_packed.device

    packed_i64 = W_packed.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(0, 32, 2, dtype=torch.int64, device=device)

    # [N, words, 16]
    codes = (packed_i64.unsqueeze(-1) >> shifts) & 0x3
    levels = (codes.to(torch.float32) - 1.5).view(N, K)

    return levels * scales.to(torch.float32).unsqueeze(1)


def unpack_int2_gemv(x: torch.Tensor, W_packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Executes single-token decode GEMV (M == 1): Y = X @ W_packed.T * scales.
    Uses sm_86 warp-per-row register-level bit unpacking with FP32 accumulators.

    Args:
        x: [K] or [1, K] in float16 (CUDA)
        W_packed: [N, K // 16] in int32 (CUDA)
        scales: [N] in float16 (CUDA)

    Returns:
        Y: [N] or [1, N] in float16 (CUDA)
    """
    ext = get_cuda_int2_ext()
    x_fp16 = x if x.dtype == torch.float16 else x.to(torch.float16)
    scales_fp16 = scales if scales.dtype == torch.float16 else scales.to(torch.float16)
    return ext.packed_int2_gemv_cuda(x_fp16, W_packed, scales_fp16)
