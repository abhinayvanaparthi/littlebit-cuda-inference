"""
benchmark_int2_stage2.py
Stage 2 Benchmark: Asymmetric Per-Group FP16 Scales and Zero Points for INT2.

Compares:
1. Stage 1 INT2 (Single Row Scale)
2. Stage 2 INT2 (group_size = 32, asymmetric scale + zero_point)
3. Stage 2 INT2 (group_size = 16, asymmetric scale + zero_point)
4. FP16 cuBLAS Reference (Unquantized)

Evaluated on:
- Qwen3-8B Layer 0 self_attn.q_proj (4096 x 4096 reconstructed FP32 ground truth)
- Synthetic 4096 x 4096 Gaussian matrix
- Metrics: Packed Buffer Size (MB), Median / Mean / P95 Latency (us) over 1,000 runs,
  Max Abs Error, Mean Abs Error, Output Cosine Similarity, and Weight Cosine Similarity.
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Dict, Any, Tuple
from safetensors import safe_open
from huggingface_hub import hf_hub_download

# Local modules
from cuda_int2_ops import (
    pack_int2_weight,
    unpack_int2_gemv,
    get_cuda_int2_ext,
)
from cuda_int2_group_ops import (
    pack_int2_group_weight,
    unpack_int2_group_weight,
    unpack_int2_group_gemv,
    get_cuda_int2_group_ext,
)
from run_qwen3_littlebit import (
    PackedLittleBitLinear,
    build_packed_linear,
    _unpack_to_fp32,
)

REPO_ID = "LittleBitLLM/littlebit-qwen3-8b-0.55bpw"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def reconstruct_fp32_littlebit_matrix(layer: PackedLittleBitLinear) -> torch.Tensor:
    in_feat = layer.in_features
    out_feat = layer.out_features
    split_dim = layer.split_dim

    V_f32 = _unpack_to_fp32(layer.V_packed, in_feat).to(DEVICE)
    U_f32 = _unpack_to_fp32(layer.U_packed, split_dim).to(DEVICE)
    u1 = layer.u1.to(device=DEVICE, dtype=torch.float32).view(out_feat, 1)
    u2 = layer.u2.to(device=DEVICE, dtype=torch.float32).squeeze()
    v1 = layer.v1.to(device=DEVICE, dtype=torch.float32).squeeze()
    v2 = layer.v2.to(device=DEVICE, dtype=torch.float32).view(1, in_feat)

    W_main = (U_f32 * u1) @ torch.diag(v1 * u2) @ (V_f32 * v2)

    if layer.residual and hasattr(layer, "V_R_packed"):
        V_R_f32 = _unpack_to_fp32(layer.V_R_packed, in_feat).to(DEVICE)
        U_R_f32 = _unpack_to_fp32(layer.U_R_packed, split_dim).to(DEVICE)
        u1_R = layer.u1_R.to(device=DEVICE, dtype=torch.float32).view(out_feat, 1)
        u2_R = layer.u2_R.to(device=DEVICE, dtype=torch.float32).squeeze()
        v1_R = layer.v1_R.to(device=DEVICE, dtype=torch.float32).squeeze()
        v2_R = layer.v2_R.to(device=DEVICE, dtype=torch.float32).view(1, in_feat)
        W_res = (U_R_f32 * u1_R) @ torch.diag(v1_R * u2_R) @ (V_R_f32 * v2_R)
        return W_main + W_res

    return W_main


def benchmark_gemv_latency(fn, warmup: int = 200, timed: int = 1000) -> Tuple[float, float, float]:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies = []

    for _ in range(timed):
        start_event.record()
        _ = fn()
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event) * 1000.0)

    arr = np.array(latencies)
    return float(np.median(arr)), float(np.mean(arr)), float(np.percentile(arr, 95))


def run_benchmark():
    if DEVICE != "cuda":
        print("ERROR: CUDA is not available. Please run with run_with_msvc.bat.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print("=" * 80)
    print(f"STAGE 2: ASYMMETRIC PER-GROUP INT2 BENCHMARK — NVIDIA {gpu_name} (sm_{cc[0]}{cc[1]})")
    print("=" * 80)

    # 1. Compile and load CUDA extensions
    print("\n[1/5] Compiling and loading CUDA extensions...")
    _ = get_cuda_int2_ext()
    _ = get_cuda_int2_group_ext()
    print("      Both Stage 1 and Stage 2 CUDA extensions loaded successfully.")

    # 2. Load Real Model weights & generate Synthetic matrix
    print(f"\n[2/5] Loading Qwen3-8B Layer 0 q_proj from {REPO_ID}...")
    safetensors_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors", local_files_only=True)

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        layer0_q = build_packed_linear(
            f=f,
            prefix="model.layers.0.self_attn.q_proj",
            in_features=4096,
            out_features=4096,
            device="cuda",
        ).to("cuda")

    W_real_f32 = reconstruct_fp32_littlebit_matrix(layer0_q)
    print(f"      Layer 0 q_proj loaded. Reconstructed FP32 matrix shape: {W_real_f32.shape}")

    torch.manual_seed(42)
    W_syn_f32 = torch.randn((4096, 4096), device="cuda", dtype=torch.float32) * 0.02
    print("      Synthetic Gaussian matrix (4096 x 4096) initialized.")

    # 3. Quantize and pack matrices
    print("\n[3/5] Quantizing and packing weights for Stage 1 and Stage 2 (G=32, G=16)...")

    # Real model weights
    W_real_st1_packed, s_real_st1 = pack_int2_weight(W_real_f32)
    W_real_g32_packed, s_real_g32, z_real_g32 = pack_int2_group_weight(W_real_f32, group_size=32)
    W_real_g16_packed, s_real_g16, z_real_g16 = pack_int2_group_weight(W_real_f32, group_size=16)

    # Synthetic weights
    W_syn_st1_packed, s_syn_st1 = pack_int2_weight(W_syn_f32)
    W_syn_g32_packed, s_syn_g32, z_syn_g32 = pack_int2_group_weight(W_syn_f32, group_size=32)
    W_syn_g16_packed, s_syn_g16, z_syn_g16 = pack_int2_group_weight(W_syn_f32, group_size=16)

    # Calculate buffer memory sizes
    mb_fp16 = (4096 * 4096 * 2) / (1024 ** 2)
    mb_st1 = (W_real_st1_packed.numel() * 4 + s_real_st1.numel() * 2) / (1024 ** 2)
    mb_g32 = (W_real_g32_packed.numel() * 4 + s_real_g32.numel() * 2 + z_real_g32.numel() * 2) / (1024 ** 2)
    mb_g16 = (W_real_g16_packed.numel() * 4 + s_real_g16.numel() * 2 + z_real_g16.numel() * 2) / (1024 ** 2)

    print(f"      FP16 Reference Buffer Size  : {mb_fp16:.2f} MB (16.0 bpw)")
    print(f"      Stage 1 INT2 (Single Scale) : {mb_st1:.2f} MB (2.00 bpw)")
    print(f"      Stage 2 INT2 (Group 32)     : {mb_g32:.2f} MB (3.00 bpw)")
    print(f"      Stage 2 INT2 (Group 16)     : {mb_g16:.2f} MB (4.00 bpw)")

    # 4. Verify numerical correctness of Stage 2 CUDA kernel against dequantized PyTorch math
    print("\n[4/5] Verifying Stage 2 CUDA kernel numerical correctness (Max Abs Error < 2e-3)...")
    x_test = torch.randn((1, 4096), device="cuda", dtype=torch.float16)

    # Test G=32
    y_cuda_g32 = unpack_int2_group_gemv(x_test, W_real_g32_packed, s_real_g32, z_real_g32, group_size=32)
    W_dequant_g32 = unpack_int2_group_weight(W_real_g32_packed, s_real_g32, z_real_g32, group_size=32)
    y_math_g32 = torch.matmul(x_test.to(torch.float32), W_dequant_g32.t()).to(torch.float16)
    err_g32 = (y_cuda_g32.to(torch.float32) - y_math_g32.to(torch.float32)).abs().max().item()
    print(f"      Group 32 CUDA kernel vs dequantized math Max Abs Error: {err_g32:.6f}")
    assert err_g32 < 2e-3, f"Group 32 kernel numerical check failed: {err_g32}"

    # Test G=16
    y_cuda_g16 = unpack_int2_group_gemv(x_test, W_real_g16_packed, s_real_g16, z_real_g16, group_size=16)
    W_dequant_g16 = unpack_int2_group_weight(W_real_g16_packed, s_real_g16, z_real_g16, group_size=16)
    y_math_g16 = torch.matmul(x_test.to(torch.float32), W_dequant_g16.t()).to(torch.float16)
    err_g16 = (y_cuda_g16.to(torch.float32) - y_math_g16.to(torch.float32)).abs().max().item()
    print(f"      Group 16 CUDA kernel vs dequantized math Max Abs Error: {err_g16:.6f}")
    assert err_g16 < 2e-3, f"Group 16 kernel numerical check failed: {err_g16}"
    print("      Stage 2 CUDA kernel numerical verification PASSED!")

    # 5. Benchmarking across Real and Synthetic weights
    print("\n[5/5] Running side-by-side latency & accuracy benchmarks (1,000 runs per configuration)...")

    test_cases = [
        (
            "Real Model (Qwen3-8B Layer 0 q_proj)",
            W_real_f32,
            W_real_st1_packed, s_real_st1,
            W_real_g32_packed, s_real_g32, z_real_g32,
            W_real_g16_packed, s_real_g16, z_real_g16,
        ),
        (
            "Synthetic Matrix (4096 x 4096 Gaussian)",
            W_syn_f32,
            W_syn_st1_packed, s_syn_st1,
            W_syn_g32_packed, s_syn_g32, z_syn_g32,
            W_syn_g16_packed, s_syn_g16, z_syn_g16,
        ),
    ]

    all_results = []

    for name, W_ref, st1_w, st1_s, g32_w, g32_s, g32_z, g16_w, g16_s, g16_z in test_cases:
        x = torch.randn((1, 4096), device="cuda", dtype=torch.float16)
        x_f32 = x.to(torch.float32)
        y_ref = torch.matmul(x_f32, W_ref.t()).squeeze(0)  # [4096] in FP32

        # 1. FP16 cuBLAS Baseline
        W_fp16 = W_ref.to(torch.float16)
        y_fp16 = torch.matmul(x, W_fp16.t()).squeeze(0).to(torch.float32)
        med_fp16, mean_fp16, p95_fp16 = benchmark_gemv_latency(lambda: torch.matmul(x, W_fp16.t()))
        max_err_fp16 = (y_fp16 - y_ref).abs().max().item()
        mean_err_fp16 = (y_fp16 - y_ref).abs().mean().item()
        cos_fp16 = torch.cosine_similarity(y_fp16.unsqueeze(0), y_ref.unsqueeze(0)).item()

        # 2. Stage 1 INT2 (Single Scale)
        y_st1 = unpack_int2_gemv(x, st1_w, st1_s).squeeze(0).to(torch.float32)
        med_st1, mean_st1, p95_st1 = benchmark_gemv_latency(lambda: unpack_int2_gemv(x, st1_w, st1_s))
        max_err_st1 = (y_st1 - y_ref).abs().max().item()
        mean_err_st1 = (y_st1 - y_ref).abs().mean().item()
        cos_st1 = torch.cosine_similarity(y_st1.unsqueeze(0), y_ref.unsqueeze(0)).item()
        from cuda_int2_ops import unpack_int2_weight
        W_st1_recon = unpack_int2_weight(st1_w, st1_s)
        w_cos_st1 = torch.cosine_similarity(W_st1_recon.view(1, -1), W_ref.view(1, -1)).item()

        # 3. Stage 2 INT2 (Group 32)
        y_g32 = unpack_int2_group_gemv(x, g32_w, g32_s, g32_z, group_size=32).squeeze(0).to(torch.float32)
        med_g32, mean_g32, p95_g32 = benchmark_gemv_latency(lambda: unpack_int2_group_gemv(x, g32_w, g32_s, g32_z, group_size=32))
        max_err_g32 = (y_g32 - y_ref).abs().max().item()
        mean_err_g32 = (y_g32 - y_ref).abs().mean().item()
        cos_g32 = torch.cosine_similarity(y_g32.unsqueeze(0), y_ref.unsqueeze(0)).item()
        W_g32_recon = unpack_int2_group_weight(g32_w, g32_s, g32_z, group_size=32)
        w_cos_g32 = torch.cosine_similarity(W_g32_recon.view(1, -1), W_ref.view(1, -1)).item()

        # 4. Stage 2 INT2 (Group 16)
        y_g16 = unpack_int2_group_gemv(x, g16_w, g16_s, g16_z, group_size=16).squeeze(0).to(torch.float32)
        med_g16, mean_g16, p95_g16 = benchmark_gemv_latency(lambda: unpack_int2_group_gemv(x, g16_w, g16_s, g16_z, group_size=16))
        max_err_g16 = (y_g16 - y_ref).abs().max().item()
        mean_err_g16 = (y_g16 - y_ref).abs().mean().item()
        cos_g16 = torch.cosine_similarity(y_g16.unsqueeze(0), y_ref.unsqueeze(0)).item()
        W_g16_recon = unpack_int2_group_weight(g16_w, g16_s, g16_z, group_size=16)
        w_cos_g16 = torch.cosine_similarity(W_g16_recon.view(1, -1), W_ref.view(1, -1)).item()

        all_results.append({
            "test_name": name,
            "st1": {
                "mb": mb_st1,
                "latency_med": med_st1,
                "latency_mean": mean_st1,
                "latency_p95": p95_st1,
                "max_err": max_err_st1,
                "mean_err": mean_err_st1,
                "cos_sim": cos_st1,
                "w_cos_sim": w_cos_st1,
            },
            "g32": {
                "mb": mb_g32,
                "latency_med": med_g32,
                "latency_mean": mean_g32,
                "latency_p95": p95_g32,
                "max_err": max_err_g32,
                "mean_err": mean_err_g32,
                "cos_sim": cos_g32,
                "w_cos_sim": w_cos_g32,
            },
            "g16": {
                "mb": mb_g16,
                "latency_med": med_g16,
                "latency_mean": mean_g16,
                "latency_p95": p95_g16,
                "max_err": max_err_g16,
                "mean_err": mean_err_g16,
                "cos_sim": cos_g16,
                "w_cos_sim": w_cos_g16,
            },
            "fp16": {
                "mb": mb_fp16,
                "latency_med": med_fp16,
                "latency_mean": mean_fp16,
                "latency_p95": p95_fp16,
                "max_err": max_err_fp16,
                "mean_err": mean_err_fp16,
                "cos_sim": cos_fp16,
                "w_cos_sim": 1.0,
            },
        })

    # 6. Print Markdown Results Table
    print("\n" + "=" * 80)
    print("STAGE 2 BENCHMARK RESULTS")
    print("=" * 80 + "\n")

    md_lines = []
    md_lines.append(f"### Stage 2: Asymmetric Per-Group INT2 GEMV Benchmark")
    md_lines.append(f"**Hardware:** NVIDIA {gpu_name} (sm_{cc[0]}{cc[1]}, 4.0 GB VRAM)")
    md_lines.append(f"**Matrix Dimensions:** $N = 4096, K = 4096$ (GEMV batch size $M = 1$)")
    md_lines.append(f"**Timing Methodology:** Median over 1,000 timed runs (200 warmup runs) using `torch.cuda.Event`\n")

    for res in all_results:
        md_lines.append(f"#### Test Case: {res['test_name']}\n")
        md_lines.append("| Precision / Architecture | Packed Buffer Size | Latency (Median) | Latency (Mean) | Latency (P95) | Max Abs Error | Mean Abs Error | Output Cosine Sim | Weight Cosine Sim |")
        md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        # Stage 1
        st1 = res["st1"]
        md_lines.append(
            f"| **Stage 1 INT2 (Single Scale)** | {st1['mb']:.2f} MB | "
            f"**{st1['latency_med']:.1f} µs** | {st1['latency_mean']:.1f} µs | {st1['latency_p95']:.1f} µs | "
            f"{st1['max_err']:.4f} | {st1['mean_err']:.6f} | **{st1['cos_sim']:.6f}** | **{st1['w_cos_sim']:.6f}** |"
        )

        # Stage 2 (Group 32)
        g32 = res["g32"]
        md_lines.append(
            f"| **Stage 2 INT2 (Group 32 Asym)** | {g32['mb']:.2f} MB | "
            f"**{g32['latency_med']:.1f} µs** | {g32['latency_mean']:.1f} µs | {g32['latency_p95']:.1f} µs | "
            f"{g32['max_err']:.4f} | {g32['mean_err']:.6f} | **{g32['cos_sim']:.6f}** | **{g32['w_cos_sim']:.6f}** |"
        )

        # Stage 2 (Group 16)
        g16 = res["g16"]
        md_lines.append(
            f"| **Stage 2 INT2 (Group 16 Asym)** | {g16['mb']:.2f} MB | "
            f"**{g16['latency_med']:.1f} µs** | {g16['latency_mean']:.1f} µs | {g16['latency_p95']:.1f} µs | "
            f"{g16['max_err']:.4f} | {g16['mean_err']:.6f} | **{g16['cos_sim']:.6f}** | **{g16['w_cos_sim']:.6f}** |"
        )

        # FP16 cuBLAS Reference
        fp16 = res["fp16"]
        md_lines.append(
            f"| *FP16 cuBLAS Reference* | {fp16['mb']:.2f} MB | "
            f"{fp16['latency_med']:.1f} µs | {fp16['latency_mean']:.1f} µs | {fp16['latency_p95']:.1f} µs | "
            f"{fp16['max_err']:.4f} | {fp16['mean_err']:.6f} | {fp16['cos_sim']:.6f} | {fp16['w_cos_sim']:.6f} |"
        )
        md_lines.append("")

    markdown_str = "\n".join(md_lines)
    print(markdown_str)


if __name__ == "__main__":
    run_benchmark()
