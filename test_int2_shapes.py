"""
test_int2_shapes.py
Verifies Int2GroupLinear across all Qwen3-8B projection matrix shapes:
1. 1024 x 4096  (k_proj, v_proj) with group_size=16
2. 12288 x 4096 (gate_proj, up_proj) with group_size=32
3. 4096 x 12288 (down_proj) with group_size=32
4. 4096 x 4096  (q_proj, o_proj) with group_size=16

Tests both decode (M=1) and prefill (M=16) across 1D, 2D, and 3D activation shapes.
"""

import sys
import torch
import torch.nn as nn
from cuda_int2_group_ops import Int2GroupLinear, unpack_int2_group_weight

def verify_shape(out_f: int, in_f: int, group_size: int, name: str):
    print(f"\n---> Testing {name}: out_features={out_f}, in_features={in_f}, group_size={group_size}")
    linear = nn.Linear(in_f, out_f, bias=False, device="cuda", dtype=torch.float16)
    nn.init.normal_(linear.weight, mean=0.0, std=0.02)

    int2_layer = Int2GroupLinear.from_linear(linear, group_size=group_size, device="cuda")

    # Reconstruct weight for ground truth check
    W_dequant = unpack_int2_group_weight(int2_layer.W_packed, int2_layer.scales, int2_layer.zero_points, group_size=group_size)

    # 1. Single-token decode test (1D: [K])
    x_1d = torch.randn(in_f, device="cuda", dtype=torch.float16)
    y_cuda_1d = int2_layer(x_1d)
    y_ref_1d = (x_1d.to(torch.float32) @ W_dequant.t()).to(torch.float16)
    err_1d = (y_cuda_1d.to(torch.float32) - y_ref_1d.to(torch.float32)).abs().max().item()
    assert err_1d < 2e-3, f"1D decode failed with error {err_1d}"
    print(f"     [OK] 1D Decode (K): Max Abs Error = {err_1d:.6f}")

    # 2. 2D single-token decode test (2D: [1, K])
    x_2d = torch.randn(1, in_f, device="cuda", dtype=torch.float16)
    y_cuda_2d = int2_layer(x_2d)
    y_ref_2d = (x_2d.to(torch.float32) @ W_dequant.t()).to(torch.float16)
    err_2d = (y_cuda_2d.to(torch.float32) - y_ref_2d.to(torch.float32)).abs().max().item()
    assert err_2d < 2e-3, f"2D decode failed with error {err_2d}"
    print(f"     [OK] 2D Decode (1, K): Max Abs Error = {err_2d:.6f}")

    # 3. 3D multi-token prefill test (3D: [1, 16, K])
    x_3d = torch.randn(1, 16, in_f, device="cuda", dtype=torch.float16)
    y_cuda_3d = int2_layer(x_3d)
    y_ref_3d = (x_3d.to(torch.float32) @ W_dequant.t()).to(torch.float16)
    err_3d = (y_cuda_3d.to(torch.float32) - y_ref_3d.to(torch.float32)).abs().max().item()
    cos_3d = torch.cosine_similarity(y_cuda_3d.float().view(1, -1), y_ref_3d.float().view(1, -1)).item()
    assert err_3d < 6e-3 and cos_3d > 0.9999, f"3D prefill failed with error {err_3d}, cos={cos_3d}"
    print(f"     [OK] 3D Prefill (1, 16, K): Max Abs Error = {err_3d:.6f}, Cos Sim = {cos_3d:.6f}")

def main():
    print("=" * 70)
    print("VERIFYING Int2GroupLinear ACROSS ALL QWEN3-8B MATRIX SHAPES")
    print("=" * 70)

    # 1. 1024 x 4096 (k_proj, v_proj)
    verify_shape(1024, 4096, 16, "k_proj / v_proj (1024x4096, G=16)")

    # 2. 12288 x 4096 (gate_proj, up_proj)
    verify_shape(12288, 4096, 32, "gate_proj / up_proj (12288x4096, G=32)")

    # 3. 4096 x 12288 (down_proj)
    verify_shape(4096, 12288, 32, "down_proj (4096x12288, G=32)")

    # 4. 4096 x 4096 (q_proj, o_proj)
    verify_shape(4096, 4096, 16, "q_proj / o_proj (4096x4096, G=16)")

    print("\n" + "=" * 70)
    print("ALL QWEN3-8B MATRIX SHAPES AND BATCH DIMS VERIFIED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
