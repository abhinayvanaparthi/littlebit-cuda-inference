"""
benchmark_int2_stage1.py
Standalone Stage 1 prototype and benchmark comparing:
1. 1-bit (0.55 bpw) LittleBit baseline (rank-128 factorized + residual)
2. Stage 1 2-bit (INT2, single per-row FP16 scale) register-unpacking GEMV
3. FP16 cuBLAS baseline (for reference)

Evaluates:
- Real model weights: Qwen3-8B Layer 0 self_attn.q_proj (4096 x 4096)
- Synthetic weights: Gaussian random matrix (4096 x 4096)
- Accuracy metrics: Max Abs Error, Mean Abs Error, Cosine Similarity vs FP32 reference
- Buffer memory footprint: Packed weight + scale buffer size in MB
- Latency metrics: Median, Mean, and P95 latency (microseconds) over 1,000 runs on RTX 3050 Ti
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from safetensors import safe_open
from huggingface_hub import hf_hub_download

# Local modules
import cuda_ops
from cuda_int2_ops import (
    pack_int2_weight,
    unpack_int2_weight,
    unpack_int2_gemv,
    get_cuda_int2_ext,
)
from run_qwen3_littlebit import (
    PackedLittleBitLinear,
    build_packed_linear,
    _unpack_to_fp32,
)

REPO_ID = "LittleBitLLM/littlebit-qwen3-8b-0.55bpw"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def reconstruct_fp32_littlebit_matrix(layer: PackedLittleBitLinear) -> torch.Tensor:
    """
    Reconstructs the full [out_features, in_features] FP32 matrix represented
    by a PackedLittleBitLinear layer:
      W = (U * u1) @ diag(v1 * u2) @ (V * v2)
      if residual:
        W += (U_R * u1_R) @ diag(v1_R * u2_R) @ (V_R * v2_R)
    """
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
    """
    Measures median, mean, and p95 latency in microseconds using CUDA events.
    """
    # Warmup
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
        latencies.append(start_event.elapsed_time(end_event) * 1000.0)  # ms -> us

    arr = np.array(latencies)
    return float(np.median(arr)), float(np.mean(arr)), float(np.percentile(arr, 95))


def run_benchmark():
    if DEVICE != "cuda":
        print("ERROR: CUDA is not available. Please run with run_with_msvc.bat.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print("=" * 80)
    print(f"STAGE 1 2-BIT (INT2) GEMV BENCHMARK — NVIDIA {gpu_name} (sm_{cc[0]}{cc[1]})")
    print("=" * 80)

    # 1. Compile/load CUDA extensions
    print("\n[1/5] Compiling and loading CUDA extensions...")
    _ = get_cuda_int2_ext()
    _ = cuda_ops._get_ext()
    print("      Both CUDA extensions (1-bit LittleBit and 2-bit INT2) loaded successfully.")

    # 2. Load Layer 0 self_attn.q_proj from Hugging Face checkpoint
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

    # Reconstruct ground truth FP32 weight matrix
    W_real_f32 = reconstruct_fp32_littlebit_matrix(layer0_q)
    print(f"      Layer 0 q_proj loaded. Reconstructed FP32 matrix shape: {W_real_f32.shape}")

    # Synthetic matrix for comparison
    torch.manual_seed(42)
    W_syn_f32 = torch.randn((4096, 4096), device="cuda", dtype=torch.float32) * 0.02

    # 3. Pack with Stage 1 INT2
    print("\n[3/5] Packing weights into 2-bit (INT2) representation (16 weights/uint32)...")
    W_real_int2_packed, scales_real_int2 = pack_int2_weight(W_real_f32)
    W_syn_int2_packed, scales_syn_int2 = pack_int2_weight(W_syn_f32)

    # Calculate buffer memory
    # 1-bit buffer: V_packed, U_packed, V_R_packed, U_R_packed, scale vectors
    bytes_1bit = (
        layer0_q.V_packed.element_size() * layer0_q.V_packed.numel() +
        layer0_q.U_packed.element_size() * layer0_q.U_packed.numel() +
        layer0_q.V_R_packed.element_size() * layer0_q.V_R_packed.numel() +
        layer0_q.U_R_packed.element_size() * layer0_q.U_R_packed.numel() +
        (layer0_q.v2.numel() + layer0_q.v1.numel() + layer0_q.u2.numel() + layer0_q.u1.numel() +
         layer0_q.v2_R.numel() + layer0_q.v1_R.numel() + layer0_q.u2_R.numel() + layer0_q.u1_R.numel()) * 2
    )
    mb_1bit = bytes_1bit / (1024 ** 2)

    # 2-bit buffer: W_packed (int32) + scales (float16)
    bytes_int2 = (
        W_real_int2_packed.element_size() * W_real_int2_packed.numel() +
        scales_real_int2.element_size() * scales_real_int2.numel()
    )
    mb_int2 = bytes_int2 / (1024 ** 2)

    # FP16 buffer
    mb_fp16 = (4096 * 4096 * 2) / (1024 ** 2)

    print(f"      1-bit LittleBit Buffer Size : {mb_1bit:.3f} MB ({mb_1bit*1024:.1f} KB)")
    print(f"      Stage 1 INT2 Buffer Size    : {mb_int2:.3f} MB")
    print(f"      FP16 Reference Buffer Size  : {mb_fp16:.3f} MB")

    # 4. Verify numerical correctness of INT2 CUDA kernel against CPU/FP32 math
    print("\n[4/5] Verifying INT2 CUDA kernel correctness...")
    x_test = torch.randn((1, 4096), device="cuda", dtype=torch.float16)
    y_cuda = unpack_int2_gemv(x_test, W_real_int2_packed, scales_real_int2)

    W_dequant = unpack_int2_weight(W_real_int2_packed, scales_real_int2)
    y_math = torch.matmul(x_test.to(torch.float32), W_dequant.t()).to(torch.float16)
    kernel_error = (y_cuda.to(torch.float32) - y_math.to(torch.float32)).abs().max().item()
    print(f"      Max absolute error between CUDA kernel & dequantized math: {kernel_error:.6f}")
    assert kernel_error < 0.02, f"Kernel error too high: {kernel_error}"
    print("      INT2 kernel numerical accuracy check PASSED!")

    # 5. Benchmarking
    print("\n[5/5] Running side-by-side latency & accuracy benchmarks (1,000 runs per test)...")

    results = []
    test_cases = [
        ("Real Model (Qwen3-8B Layer 0 q_proj)", W_real_f32, layer0_q, W_real_int2_packed, scales_real_int2),
        ("Synthetic Matrix (4096 x 4096 Gaussian)", W_syn_f32, None, W_syn_int2_packed, scales_syn_int2),
    ]

    for name, W_ref, layer_1bit, W_int2, scales_int2 in test_cases:
        x = torch.randn((1, 4096), device="cuda", dtype=torch.float16)
        x_f32 = x.to(torch.float32)
        y_ref = torch.matmul(x_f32, W_ref.t()).squeeze(0)  # [4096] in FP32

        # --- FP16 cuBLAS Baseline ---
        W_fp16 = W_ref.to(torch.float16)
        y_fp16 = torch.matmul(x, W_fp16.t()).squeeze(0).to(torch.float32)
        med_fp16, mean_fp16, p95_fp16 = benchmark_gemv_latency(lambda: torch.matmul(x, W_fp16.t()))

        max_err_fp16 = (y_fp16 - y_ref).abs().max().item()
        mean_err_fp16 = (y_fp16 - y_ref).abs().mean().item()
        cos_fp16 = torch.cosine_similarity(y_fp16.unsqueeze(0), y_ref.unsqueeze(0)).item()

        # --- 1-bit LittleBit Baseline ---
        if layer_1bit is not None:
            y_1bit = layer_1bit(x).squeeze(0).to(torch.float32)
            med_1bit, mean_1bit, p95_1bit = benchmark_gemv_latency(lambda: layer_1bit(x))

            # Accuracy vs FP32 ground truth
            # Note: For Layer 0 q_proj, W_real_f32 was reconstructed from the LittleBit factorization,
            # so the error here isolates FP16 CUDA kernel precision vs explicit FP32 math.
            max_err_1bit = (y_1bit - y_ref).abs().max().item()
            mean_err_1bit = (y_1bit - y_ref).abs().mean().item()
            cos_1bit = torch.cosine_similarity(y_1bit.unsqueeze(0), y_ref.unsqueeze(0)).item()
        else:
            med_1bit = mean_1bit = p95_1bit = 0.0
            max_err_1bit = mean_err_1bit = 0.0
            cos_1bit = 0.0

        # --- Stage 1 2-bit (INT2) Kernel ---
        y_int2 = unpack_int2_gemv(x, W_int2, scales_int2).squeeze(0).to(torch.float32)
        med_int2, mean_int2, p95_int2 = benchmark_gemv_latency(lambda: unpack_int2_gemv(x, W_int2, scales_int2))

        max_err_int2 = (y_int2 - y_ref).abs().max().item()
        mean_err_int2 = (y_int2 - y_ref).abs().mean().item()
        cos_int2 = torch.cosine_similarity(y_int2.unsqueeze(0), y_ref.unsqueeze(0)).item()

        # Weight-level reconstruction accuracy
        W_int2_recon = unpack_int2_weight(W_int2, scales_int2)
        w_cos_int2 = torch.cosine_similarity(W_int2_recon.view(1, -1), W_ref.view(1, -1)).item()
        w_mae_int2 = (W_int2_recon - W_ref).abs().mean().item()

        results.append({
            "test_name": name,
            "1bit": {
                "mb": mb_1bit if layer_1bit is not None else 0.0,
                "latency_med": med_1bit,
                "latency_mean": mean_1bit,
                "latency_p95": p95_1bit,
                "max_err": max_err_1bit,
                "mean_err": mean_err_1bit,
                "cos_sim": cos_1bit,
            },
            "int2": {
                "mb": mb_int2,
                "latency_med": med_int2,
                "latency_mean": mean_int2,
                "latency_p95": p95_int2,
                "max_err": max_err_int2,
                "mean_err": mean_err_int2,
                "cos_sim": cos_int2,
                "w_cos_sim": w_cos_int2,
                "w_mae": w_mae_int2,
            },
            "fp16": {
                "mb": mb_fp16,
                "latency_med": med_fp16,
                "latency_mean": mean_fp16,
                "latency_p95": p95_fp16,
                "max_err": max_err_fp16,
                "mean_err": mean_err_fp16,
                "cos_sim": cos_fp16,
            }
        })

    # 6. Format Markdown Report
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80 + "\n")

    md_output = []
    md_output.append(f"### Stage 1: 2-bit (INT2) Register-Unpacking GEMV Benchmark")
    md_output.append(f"**Hardware:** NVIDIA {gpu_name} (sm_{cc[0]}{cc[1]}, 4.0 GB VRAM)")
    md_output.append(f"**Matrix Dimensions:** $N = 4096, K = 4096$ (GEMV batch size $M = 1$)")
    md_output.append(f"**Timing Methodology:** Median over 1,000 timed runs (200 warmup runs) using `torch.cuda.Event`\n")

    for r in results:
        md_output.append(f"#### Test Case: {r['test_name']}\n")
        md_output.append("| Precision / Architecture | Packed Buffer Size | Latency (Median) | Latency (Mean) | Latency (P95) | Max Abs Error | Mean Abs Error | Cosine Similarity |")
        md_output.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        if r["1bit"]["mb"] > 0:
            md_output.append(
                f"| **1-bit Baseline (0.55 bpw LittleBit)** | {r['1bit']['mb']:.2f} MB | "
                f"**{r['1bit']['latency_med']:.1f} µs** | {r['1bit']['latency_mean']:.1f} µs | {r['1bit']['latency_p95']:.1f} µs | "
                f"{r['1bit']['max_err']:.4f} | {r['1bit']['mean_err']:.6f} | **{r['1bit']['cos_sim']:.6f}** |"
            )

        md_output.append(
            f"| **Stage 1: 2-bit INT2 (Single Scale)** | {r['int2']['mb']:.2f} MB | "
            f"**{r['int2']['latency_med']:.1f} µs** | {r['int2']['latency_mean']:.1f} µs | {r['int2']['latency_p95']:.1f} µs | "
            f"{r['int2']['max_err']:.4f} | {r['int2']['mean_err']:.6f} | **{r['int2']['cos_sim']:.6f}** |"
        )

        md_output.append(
            f"| *FP16 cuBLAS Reference (Unquantized)* | {r['fp16']['mb']:.2f} MB | "
            f"{r['fp16']['latency_med']:.1f} µs | {r['fp16']['latency_mean']:.1f} µs | {r['fp16']['latency_p95']:.1f} µs | "
            f"{r['fp16']['max_err']:.4f} | {r['fp16']['mean_err']:.6f} | {r['fp16']['cos_sim']:.6f} |"
        )
        md_output.append("")

        if "w_cos_sim" in r["int2"]:
            md_output.append(f"* **INT2 Weight Reconstruction:** Cosine Similarity = **{r['int2']['w_cos_sim']:.6f}**, MAE = **{r['int2']['w_mae']:.6f}**\n")

    markdown_text = "\n".join(md_output)
    print(markdown_text)


if __name__ == "__main__":
    run_benchmark()
