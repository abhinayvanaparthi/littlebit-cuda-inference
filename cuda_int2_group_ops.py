"""
cuda_int2_group_ops.py
Standalone PyTorch wrapper and JIT loader for Stage 2: Asymmetric Per-Group INT2 Quantization:
- pack_int2_group_weight: Vectorized group-wise asymmetric INT2 packing with fast MSE clipping grid search
- unpack_int2_group_weight: Reference PyTorch dequantizer to FP32
- unpack_int2_group_gemv: High-performance sm_86 CUDA register-unpacking GEMV kernel wrapper
"""

import os
import torch
from typing import Tuple
from torch.utils.cpp_extension import load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(CURRENT_DIR, "build_cuda_int2_group")
os.makedirs(BUILD_DIR, exist_ok=True)

_CUDA_INT2_GROUP_EXT = None


def get_cuda_int2_group_ext():
    """Lazily JIT-compiles and returns the standalone CUDA INT2 group extension."""
    global _CUDA_INT2_GROUP_EXT
    if _CUDA_INT2_GROUP_EXT is None:
        source_cu = os.path.join(CURRENT_DIR, "cuda_int2_group_unpack.cu")
        source_cpp = os.path.join(CURRENT_DIR, "cuda_int2_group_binding.cpp")

        _CUDA_INT2_GROUP_EXT = load(
            name="cuda_int2_group_unpack_ext",
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
    return _CUDA_INT2_GROUP_EXT


def pack_int2_group_weight(
    W: torch.Tensor,
    group_size: int = 32,
    num_clip_steps: int = 6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Packs a 2D weight matrix [N, K] into:
      W_packed:    [N, K // 16] (int32)
      scales:      [N, K // group_size] (float16)
      zero_points: [N, K // group_size] (float16)

    Asymmetric Per-Group INT2 Specification:
    - Weight reconstruction: w_hat = scale * q + zero_point, where q in {0, 1, 2, 3}.
    - Fast MSE clipping loop: tests clip ratios from 0.75 to 1.00 of group min/max
      to prevent outlier weights from collapsing the resolution of inlier weights.
    - 16 2-bit codes packed into each uint32 word.
    """
    assert W.dim() == 2, f"Expected 2D weight matrix, got shape {W.shape}"
    N, K = W.shape
    assert K % group_size == 0, f"Column dimension K ({K}) must be divisible by group_size ({group_size})"
    assert group_size in (16, 32), f"Supported group sizes are 16 and 32, got {group_size}"

    W_f32 = W.to(torch.float32)
    num_groups = K // group_size
    Wg = W_f32.view(N, num_groups, group_size)

    w_min = Wg.amin(dim=-1, keepdim=True)
    w_max = Wg.amax(dim=-1, keepdim=True)

    # Grid search over clip ratios [0.75 .. 1.00] to minimize MSE
    best_mse = torch.full((N, num_groups, 1), float("inf"), device=W.device, dtype=torch.float32)
    best_scale = torch.zeros((N, num_groups, 1), device=W.device, dtype=torch.float32)
    best_zp = torch.zeros((N, num_groups, 1), device=W.device, dtype=torch.float32)
    best_q = torch.zeros((N, num_groups, group_size), dtype=torch.int32, device=W.device)

    clip_ratios = torch.linspace(0.75, 1.0, num_clip_steps, device=W.device, dtype=torch.float32)

    for alpha in clip_ratios:
        # Symmetric range shrinkage around zero
        c_min = w_min * alpha
        c_max = w_max * alpha
        scale = torch.clamp((c_max - c_min) / 3.0, min=1e-5)
        zp = c_min

        # Quantize to codes {0, 1, 2, 3}
        q = torch.clamp(torch.round((Wg - zp) / scale), 0, 3).to(torch.int32)
        w_hat = scale * q.float() + zp
        mse = (Wg - w_hat).pow(2).mean(dim=-1, keepdim=True)

        mask = mse < best_mse
        best_mse = torch.where(mask, mse, best_mse)
        best_scale = torch.where(mask, scale, best_scale)
        best_zp = torch.where(mask, zp, best_zp)
        best_q = torch.where(mask, q, best_q)

    # Reshape codes to [N, K // 16, 16] for 16-element uint32 packing
    codes_flat = best_q.view(N, K)
    codes_16 = codes_flat.view(N, K // 16, 16)
    shifts = torch.arange(0, 32, 2, dtype=torch.int32, device=W.device)
    shifted = codes_16 << shifts.unsqueeze(0).unsqueeze(0)

    W_packed = shifted.sum(dim=-1, dtype=torch.int32)
    scales_fp16 = best_scale.squeeze(-1).to(torch.float16)
    zero_points_fp16 = best_zp.squeeze(-1).to(torch.float16)

    return W_packed, scales_fp16, zero_points_fp16


def unpack_int2_group_weight(
    W_packed: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor:
    """
    Unpacks W_packed [N, K // 16], scales [N, num_groups], and zero_points [N, num_groups]
    back to an explicit FP32 matrix [N, K].
    Formula: W_dequant[i, j] = scale[i, j // group_size] * q[i, j] + zero_point[i, j // group_size]
    """
    assert W_packed.dim() == 2, f"Expected 2D packed matrix, got {W_packed.shape}"
    N, words = W_packed.shape
    K = words * 16
    device = W_packed.device

    packed_i64 = W_packed.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(0, 32, 2, dtype=torch.int64, device=device)

    # [N, words, 16] -> [N, K]
    codes = ((packed_i64.unsqueeze(-1) >> shifts) & 0x3).view(N, K).to(torch.float32)

    # Expand scales and zero_points from [N, num_groups] to [N, K]
    num_groups = K // group_size
    scales_f32 = scales.to(torch.float32).unsqueeze(-1).repeat(1, 1, group_size).view(N, K)
    zp_f32 = zero_points.to(torch.float32).unsqueeze(-1).repeat(1, 1, group_size).view(N, K)

    return scales_f32 * codes + zp_f32


def unpack_int2_group_gemv(
    x: torch.Tensor,
    W_packed: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor:
    """
    Executes single-token decode GEMV (M == 1) with per-group scales and zero points.
    Uses sm_86 warp-per-row register-level bit unpacking with FP32 accumulators.

    Args:
        x: [K] or [1, K] in float16 (CUDA)
        W_packed: [N, K // 16] in int32 (CUDA)
        scales: [N, K // group_size] in float16 (CUDA)
        zero_points: [N, K // group_size] in float16 (CUDA)
        group_size: 16 or 32

    Returns:
        Y: [N] or [1, N] in float16 (CUDA)
    """
    ext = get_cuda_int2_group_ext()
    x_fp16 = x if x.dtype == torch.float16 else x.to(torch.float16)
    scales_fp16 = scales if scales.dtype == torch.float16 else scales.to(torch.float16)
    zp_fp16 = zero_points if zero_points.dtype == torch.float16 else zero_points.to(torch.float16)
    return ext.packed_int2_group_gemv_cuda(x_fp16, W_packed, scales_fp16, zp_fp16, group_size)
