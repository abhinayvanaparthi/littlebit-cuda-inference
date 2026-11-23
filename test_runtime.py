"""
test_runtime.py - Verification suite for standalone bit-packed LittleBit runtime on Windows 11.

Instantiates real LittleBitLinear layers from Samsung's repository:
  - Standard dimensions: M=4096, K=4096, R=128 (residual=False & residual=True)
  - Non-multiple-of-32 dimensions: M=1024, K=300, R=100 (residual=False & residual=True)
  - Non-multiple-of-32 latent: M=512, K=350, R=88 (residual=True)

Validates:
  - pack_binary_matrix & bit-exact unpacking
  - export_layer_to_safetensors & load_packed_layer
  - Vectorized and register-simulation GEMV against Samsung's LittleBitLinear.forward()
  - Assert max_abs_error < 1e-3
  - Exact memory footprint before vs after packing + compression ratios
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn

# Ensure LittleBit is on sys.path
sys.path.insert(0, os.path.abspath("./LittleBit"))

# Samsung imports
from quantization.hub import LittleBitModel
from quantization.modules.littlebit import LittleBitLinear
from quantization.functions.binary import STEBinary

# Our runtime imports
from packer import (
    pack_binary_matrix,
    export_layer_to_safetensors,
    load_packed_layer,
    PackedLittleBitLayer,
)
from kernel_ref import (
    unpack_binary_matrix,
    unpack_and_matvec_ref,
    matvec_register_sim,
    littlebit_forward_ref,
)


def compute_memory_stats(
    in_features: int,
    out_features: int,
    split_dim: int,
    residual: bool,
    has_bias: bool,
):
    """
    Computes exact byte counts for:
      1) Dense FP16 baseline
      2) Decomposed LittleBit FP16 (unpacked)
      3) Bit-packed uint32 + FP16 scales
    """
    # 1. Dense FP16 baseline
    dense_fp16_bytes = in_features * out_features * 2
    if has_bias:
        dense_fp16_bytes += out_features * 2

    # Number of branches: 1 (main) or 2 (main + residual)
    num_branches = 2 if residual else 1

    # 2. Decomposed LittleBit FP16 parameters
    # V: (split_dim, in_features), U: (out_features, split_dim)
    # scales: v2 (in_features), v1 (split_dim), u2 (split_dim), u1 (out_features)
    weights_fp16_per_branch = (split_dim * in_features + out_features * split_dim) * 2
    scales_fp16_per_branch = (in_features + 2 * split_dim + out_features) * 2
    unpacked_fp16_bytes = num_branches * (weights_fp16_per_branch + scales_fp16_per_branch)
    if has_bias:
        unpacked_fp16_bytes += out_features * 2

    # 3. Packed uint32 + FP16 scales
    # V_packed: split_dim * ceil(in_features / 32) * 4 bytes
    # U_packed: out_features * ceil(split_dim / 32) * 4 bytes
    v_words = (in_features + 31) // 32
    u_words = (split_dim + 31) // 32
    packed_weights_per_branch = (split_dim * v_words * 4) + (out_features * u_words * 4)
    packed_scales_per_branch = scales_fp16_per_branch
    packed_bytes = num_branches * (packed_weights_per_branch + packed_scales_per_branch)
    if has_bias:
        packed_bytes += out_features * 2

    return {
        "dense_fp16_bytes": dense_fp16_bytes,
        "unpacked_fp16_bytes": unpacked_fp16_bytes,
        "packed_bytes": packed_bytes,
        "packed_weights_bytes": num_branches * packed_weights_per_branch,
        "scales_and_bias_bytes": num_branches * packed_scales_per_branch + (out_features * 2 if has_bias else 0),
        "ratio_vs_dense": dense_fp16_bytes / packed_bytes,
        "ratio_vs_unpacked": unpacked_fp16_bytes / packed_bytes,
    }


def create_and_init_littlebit_linear(
    in_features: int,
    out_features: int,
    split_dim: int,
    residual: bool = False,
    bias: bool = True,
    seed: int = 42,
) -> LittleBitLinear:
    """Helper to instantiate and decompose a real LittleBitLinear layer."""
    torch.manual_seed(seed)
    lin = nn.Linear(in_features, out_features, bias=bias)
    lin.__class__ = LittleBitLinear
    lin.__quant_convert__(
        do_train=True,
        quant_func=STEBinary,
        split_dim=split_dim,
        residual=residual,
    )
    lin.eval()
    return lin


def run_layer_test(
    test_name: str,
    in_features: int,
    out_features: int,
    requested_split_dim: int,
    residual: bool,
    batch_size: int = 4,
    device: str = "cpu",
    test_reg_sim: bool = True,
):
    print("=" * 80)
    print(f"RUNNING TEST: {test_name}")
    print(f"Config: M={out_features}, K={in_features}, req_R={requested_split_dim}, residual={residual}, device={device}")

    # 1. Instantiate Samsung LittleBitLinear
    layer = create_and_init_littlebit_linear(
        in_features=in_features,
        out_features=out_features,
        split_dim=requested_split_dim,
        residual=residual,
        bias=True,
    )
    actual_split_dim = layer.split_dim
    print(f"Samsung finalized split_dim: {actual_split_dim}")

    # Move layer to device
    layer = layer.to(device)

    # 2. Run Samsung forward pass (FP32)
    torch.manual_seed(123)
    x = torch.randn(batch_size, in_features, device=device)
    with torch.no_grad():
        y_samsung = layer(x)

    # 3. Export to safetensors
    with tempfile.TemporaryDirectory() as tmpdir:
        safetensors_path = os.path.join(tmpdir, "packed_layer.safetensors")
        export_info = export_layer_to_safetensors(layer, safetensors_path)
        file_size = os.path.getsize(safetensors_path)
        print(f"Safetensors exported: {os.path.basename(safetensors_path)} ({file_size:,} bytes on disk)")

        # 4. Load packed layer
        packed_layer = load_packed_layer(safetensors_path, device=device)
        print(f"Loaded packed layer: {packed_layer}")

        # 5. Run inference with our vectorized reference unpacker
        with torch.no_grad():
            y_ref = packed_layer(x)

        max_err_vec = (y_ref - y_samsung).abs().max().item()
        print(f"-> Vectorized Reference Max Absolute Error: {max_err_vec:.6e}")
        assert max_err_vec < 1e-3, f"Vectorized max_err {max_err_vec} >= 1e-3!"

        # 6. Run inference with register simulation loop (on first sample for speed)
        if test_reg_sim:
            with torch.no_grad():
                y_sim = littlebit_forward_ref(x[:2], packed_layer, use_register_sim=True)
            max_err_sim = (y_sim - y_samsung[:2]).abs().max().item()
            print(f"-> Register-Sim Oracle Max Absolute Error: {max_err_sim:.6e}")
            assert max_err_sim < 1e-3, f"Register-sim max_err {max_err_sim} >= 1e-3!"

    # 7. Compute & Print Memory Footprint
    mem = compute_memory_stats(
        in_features=in_features,
        out_features=out_features,
        split_dim=actual_split_dim,
        residual=residual,
        has_bias=True,
    )
    print("\n[Memory Footprint Analysis]")
    print(f"  - Dense FP16 Baseline:             {mem['dense_fp16_bytes']:,} bytes ({mem['dense_fp16_bytes'] / 1024:.2f} KB / {mem['dense_fp16_bytes'] / (1024*1024):.3f} MB)")
    print(f"  - Decomposed FP16 (Unpacked):       {mem['unpacked_fp16_bytes']:,} bytes ({mem['unpacked_fp16_bytes'] / 1024:.2f} KB / {mem['unpacked_fp16_bytes'] / (1024*1024):.3f} MB)")
    print(f"  - Bit-Packed (uint32 + FP16 scales): {mem['packed_bytes']:,} bytes ({mem['packed_bytes'] / 1024:.2f} KB / {mem['packed_bytes'] / (1024*1024):.3f} MB)")
    print(f"      * uint32 packed weights:        {mem['packed_weights_bytes']:,} bytes")
    print(f"      * FP16 scale vectors + bias:    {mem['scales_and_bias_bytes']:,} bytes")
    print(f"  - Compression vs Dense FP16:        {mem['ratio_vs_dense']:.2f}x reduction")
    print(f"  - Compression vs Unpacked FP16:     {mem['ratio_vs_unpacked']:.2f}x reduction")
    print(f"STATUS: PASSED (max_abs_error < 1e-3)\n")


def test_bit_exact_packing_roundtrip():
    print("=" * 80)
    print("RUNNING BIT-LEVEL PACK/UNPACK ROUNDTRIP TEST")
    
    # Test arbitrary non-multiple-of-32 matrix
    rows, cols = 17, 305
    torch.manual_seed(999)
    # Generate random {-1.0, +1.0}
    w = torch.randint(0, 2, (rows, cols)).float() * 2.0 - 1.0

    packed, orig_cols = pack_binary_matrix(w, return_orig_dim=True)
    assert packed.dtype == torch.uint32
    assert orig_cols == cols
    expected_words = (cols + 31) // 32
    assert packed.shape == (rows, expected_words)

    # Unpack with kernel_ref
    w_unpacked = unpack_binary_matrix(packed, orig_cols=cols, dtype=torch.float32)
    assert torch.equal(w, w_unpacked), "Bit-exact pack/unpack roundtrip failed!"
    print(f"-> Roundtrip exact match for shape ({rows}, {cols}) -> uint32 shape {packed.shape}: PASSED (0 bit errors)")


def main():
    print("================================================================================")
    print("  Samsung LittleBit Standalone Bit-Packed Inference Runtime Test Suite")
    print("  Host OS: Windows 11 | Torch: " + torch.__version__ + " | CUDA: " + str(torch.cuda.is_available()))
    print("================================================================================\n")

    # 1. Bit-level pack/unpack test
    test_bit_exact_packing_roundtrip()

    # 2. Test standard dimensions: M=4096, K=4096, R=128 (No residual)
    run_layer_test(
        test_name="Standard Dimension - Single Branch (No Residual)",
        in_features=4096,
        out_features=4096,
        requested_split_dim=128,
        residual=False,
        batch_size=4,
        device="cpu",
        test_reg_sim=False,  # 4096x4096 in pure python loop is slow; tested vectorized
    )

    # 3. Test standard dimensions: M=4096, K=4096, R=128 (With residual)
    run_layer_test(
        test_name="Standard Dimension - Multi-Branch (Residual=True)",
        in_features=4096,
        out_features=4096,
        requested_split_dim=128,
        residual=True,
        batch_size=4,
        device="cpu",
        test_reg_sim=False,
    )

    # 4. Test non-multiple-of-32 dimensions: M=1024, K=300, R=100 (No residual)
    # K=300 requires 20 pad bits. requested_split_dim=100 is rounded by Samsung to 96.
    run_layer_test(
        test_name="Non-Multiple-of-32 Dimension - Single Branch (K=300, R=100)",
        in_features=300,
        out_features=1024,
        requested_split_dim=100,
        residual=False,
        batch_size=2,
        device="cpu",
        test_reg_sim=True,
    )

    # 5. Test non-multiple-of-32 dimensions: M=1024, K=300, R=100 (With residual)
    run_layer_test(
        test_name="Non-Multiple-of-32 Dimension - Multi-Branch Residual (K=300, R=100)",
        in_features=300,
        out_features=1024,
        requested_split_dim=100,
        residual=True,
        batch_size=2,
        device="cpu",
        test_reg_sim=True,
    )

    # 6. Test odd dimensions where split_dim is also not a multiple of 32:
    # 88 is a multiple of 8 (accepted by Samsung's _finalize_split_dim) but NOT multiple of 32!
    # K=350 is also not multiple of 32!
    run_layer_test(
        test_name="Odd Multiples of 8 (K=350, R=88, Residual=True)",
        in_features=350,
        out_features=512,
        requested_split_dim=88,
        residual=True,
        batch_size=2,
        device="cpu",
        test_reg_sim=True,
    )

    # 7. If CUDA is available, run on GPU as well
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n[CUDA GPU Detected: {gpu_name}] Running CUDA validation...")
        run_layer_test(
            test_name=f"CUDA Execution ({gpu_name}) - M=1024, K=300, R=100 (Residual=True)",
            in_features=300,
            out_features=1024,
            requested_split_dim=100,
            residual=True,
            batch_size=4,
            device="cuda",
            test_reg_sim=False,
        )

    print("=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("All max_abs_error assertions passed (< 1e-3).")
    print("================================================================================")


if __name__ == "__main__":
    main()
