"""
quantize_qwen3_int2_gptq.py
Stage 3 Quantizer for Qwen3-8B:
- Second-Order Hessian Error Compensation (GPTQ) with actorder=True and blocksize=128
- Activation-Weighted Per-Group (G=16) Scale & Zero-Point Search
- Uniform G=16 for ALL 7 projections per layer (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- Sequential layer-by-layer calibration (< 1.8 GB GPU VRAM) using 64 x 512 calibration sequences
- Produces ./qwen3-8b-int2-stage3/ safetensors checkpoint matching exact schema and metadata format
"""

import os
import sys
import json
import time
import shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

LINEAR_PROJECTIONS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

# Rich multi-domain calibration corpus (ML systems, CUDA, math, architecture, algorithms)
CALIBRATION_TEXTS = [
    """Deep learning models based on the Transformer architecture rely heavily on multi-head attention and multi-layer perceptron blocks.
The attention mechanism computes Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.
In modern large language models like Qwen, LLaMA, and Mistral, Rotary Position Embedding (RoPE) is applied to query and key representations.
RoPE encodes relative position information by rotating 2D sub-vectors of queries and keys using complex multiplication.
The rotational matrix preserves inner products between tokens as a function of their relative distance, allowing efficient extrapolation.
During generation, causal masking guarantees that token i can only attend to previous tokens j <= i.""",

    """Inference acceleration on consumer GPUs with limited VRAM (such as 4GB NVIDIA RTX 3050 Ti) requires extreme weight compression.
Standard formats like FP16 or BF16 require 2 bytes (16 bits) per weight parameter.
Quantization compresses weights into lower bit-widths: INT8 (1 byte), INT4 (4 bits), INT3 (3 bits), INT2 (2 bits), or sub-1-bit.
Post-training quantization (PTQ) techniques such as GPTQ (Generalized Post-Training Quantization) use second-order Taylor expansion
of the task loss to compensate for quantization errors across remaining columns using the inverse Hessian matrix H^(-1).
By minimizing the squared activation reconstruction error ||W X - W_quant X||_F^2, GPTQ preserves model perplexity even at low bit-widths.""",

    """To achieve high decode throughput (tokens/sec) during LLM generation, kernel developers implement fused GEMV CUDA kernels.
In memory-bandwidth-bound single-batch decoding (M = 1), compute intensity is very low.
The GPU execution latency is strictly bounded by the rate at which weight data can be fetched from VRAM into registers.
By packing 16 2-bit weights into a single 32-bit unsigned integer (uint32), memory bandwidth traffic is reduced by 8x compared to FP16.
Inside the CUDA kernel, 32-thread warps collaboratively load coalesced words and unpack 2-bit integers directly into register accumulators.
Warp reduction instructions (__shfl_down_sync) sum dot products across threads without writing intermediate activations to global memory.""",

    """Consider the singular value decomposition (SVD) of a real matrix A in R^(m x n): A = U Sigma V^T.
Here U is an m x m orthogonal matrix, Sigma is an m x n rectangular diagonal matrix with non-negative real numbers on the diagonal,
and V is an n x n orthogonal matrix. The singular values sigma_1 >= sigma_2 >= ... >= sigma_r >= 0 are the square roots of the eigenvalues
of A^T A and A A^T. The Eckart-Young-Mirsky theorem states that the best rank-k approximation of A with respect to the Frobenius norm
is obtained by truncating the SVD at the top k singular values: A_k = sum_{i=1}^k sigma_i u_i v_i^T.
In factorized quantization, weights are decomposed into low-rank binary or ternary basis matrices scaled by diagonal vectors.""",

    """The Python PyTorch library exposes an extensible C++ and CUDA extension mechanism through torch.utils.cpp_extension.load.
Custom C++ binding code connects PyTorch ATen tensors to native CUDA kernels via PyBind11.
Tensors allocated on GPU provide raw device pointers via tensor.data_ptr<scalar_t>().
The PyTorch caching allocator manages virtual memory addresses to prevent costly cudaMalloc and cudaFree syscalls during iteration.
When executing CUDA kernels inside PyTorch, passing streams via at::cuda::getCurrentCUDAStream() ensures asynchronous operations
are properly ordered on the execution queue without unnecessary CPU host synchronizations.""",

    """Computer systems execute machine instructions through deep instruction pipelines, branch predictors, and cache hierarchies.
A modern CPU features L1 instruction and data caches (typically 32KB to 64KB per core with 4-5 cycle latency),
L2 cache (512KB to 1MB per core with 12-14 cycle latency), and a shared L3 cache (16MB to 96MB with 40-60 cycle latency).
Main memory access (DRAM) requires 60 to 100 nanoseconds, corresponding to hundreds of CPU clock cycles.
SIMD vector extensions (AVX-512, AVX2, ARM NEON) execute parallel operations on 128-bit, 256-bit, or 512-bit registers.
GPU architectures (NVIDIA Ampere, Ada Lovelace, Hopper, Blackwell) exploit massive thread-level parallelism (TLP) with Streaming Multiprocessors (SMs).""",

    """Differential equations describe continuous systems in physics and engineering.
Newton's second law F = m * a leads to second-order ordinary differential equations such as the harmonic oscillator:
m * d^2x/dt^2 + c * dx/dt + k * x = F(t).
When damping c is zero and forcing F(t) is zero, the system oscillates at its natural frequency omega_0 = sqrt(k / m).
Numerical integration schemes such as Runge-Kutta 4th order (RK4) approximate solutions step by step:
y_{n+1} = y_n + h/6 * (k_1 + 2*k_2 + 2*k_3 + k_4),
providing fourth-order local truncation error O(h^5) and global error O(h^4).""",

    """Distributed training of deep neural networks relies on data parallelism, tensor model parallelism, pipeline parallelism, and sequence parallelism.
In Data Parallelism (DDP), each worker holds an identical replica of the model and processes a distinct shard of the batch.
Gradients are synchronized across GPUs using the AllReduce communication primitive via NVIDIA NCCL (Collective Communications Library).
Ring-AllReduce and Tree-AllReduce algorithms minimize communication overhead: for N GPUs and tensor size S, Ring-AllReduce communicates 2(N-1)/N * S bytes.
ZeRO (Zero Redundancy Optimizer) partitions optimizer states (ZeRO-1), gradients (ZeRO-2), and model parameters (ZeRO-3) across nodes."""
]


def build_calibration_dataset(tokenizer: AutoTokenizer, num_sequences: int = 64, seq_len: int = 512) -> torch.Tensor:
    """Builds exactly num_sequences of length seq_len from multi-domain texts."""
    total_tokens_needed = num_sequences * seq_len
    full_text = "\n\n".join(CALIBRATION_TEXTS)

    encoded = tokenizer.encode(full_text, add_special_tokens=False)
    # Loop to fill the required token budget
    tokens = []
    while len(tokens) < total_tokens_needed:
        tokens.extend(encoded)

    tokens = tokens[:total_tokens_needed]
    tensor_tokens = torch.tensor(tokens, dtype=torch.long).view(num_sequences, seq_len)
    return tensor_tokens


def find_optimal_scales_and_zeros(
    W: torch.Tensor,
    H_diag: torch.Tensor,
    group_size: int = 16,
    num_clip_steps: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finds per-group (scale, zero_point) minimizing activation-weighted MSE:
    sum_j a_j^2 * (w_j - w_q_j)^2 where a_j^2 = diag(H)_j.
    W: [N, K], H_diag: [K]
    Returns: best_scale [N, K // group_size], best_zp [N, K // group_size]
    """
    N, K = W.shape
    num_groups = K // group_size
    Wg = W.view(N, num_groups, group_size)
    a2 = H_diag.clamp(min=1e-8).view(1, num_groups, group_size)

    w_min = Wg.amin(dim=-1, keepdim=True)
    w_max = Wg.amax(dim=-1, keepdim=True)

    best_loss = torch.full((N, num_groups, 1), float("inf"), device=W.device, dtype=torch.float32)
    best_scale = torch.zeros((N, num_groups), device=W.device, dtype=torch.float32)
    best_zp = torch.zeros((N, num_groups), device=W.device, dtype=torch.float32)

    clip_ratios = torch.linspace(0.70, 1.00, num_clip_steps, device=W.device, dtype=torch.float32)

    for alpha in clip_ratios:
        c_min = w_min * alpha
        c_max = w_max * alpha
        scale = torch.clamp((c_max - c_min) / 3.0, min=1e-5)
        zp = c_min

        q = torch.clamp(torch.round((Wg - zp) / scale), 0, 3)
        w_hat = scale * q + zp
        # Activation-weighted error
        loss = ((Wg - w_hat).pow(2) * a2).sum(dim=-1, keepdim=True)

        mask = loss < best_loss
        best_loss = torch.where(mask, loss, best_loss)
        best_scale = torch.where(mask.squeeze(-1), scale.squeeze(-1), best_scale)
        best_zp = torch.where(mask.squeeze(-1), zp.squeeze(-1), best_zp)

    return best_scale, best_zp


@torch.no_grad()
def quantize_projection_gptq(
    W: torch.Tensor,
    H: torch.Tensor,
    group_size: int = 16,
    blocksize: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs Cholesky-damped Blocked GPTQ with actorder=True and G=16.
    Returns:
      W_packed:    [N, K // 16] (int32)
      scales:      [N, K // 16] (float16)
      zero_points: [N, K // 16] (float16)
      W_quant:     [N, K] (float32 dequantized weight matrix)
    """
    N, K = W.shape
    device = W.device

    # 1. Activation-weighted scale & zero-point search
    diag_H = torch.diag(H)
    best_scale, best_zp = find_optimal_scales_and_zeros(W, diag_H, group_size=group_size)

    # 2. Damping & actorder permutation
    diag_H = torch.diag(H)
    best_scale, best_zp = find_optimal_scales_and_zeros(W, diag_H, group_size=group_size)

    damp = 0.01 * torch.mean(diag_H)
    H_damped = H + damp * torch.eye(K, device=device, dtype=torch.float32)

    perm = torch.argsort(diag_H, descending=True)
    invperm = torch.argsort(perm)

    H_p = H_damped[perm, :][:, perm]

    try:
        L = torch.linalg.cholesky(H_p)
        H_inv = torch.cholesky_inverse(L)
    except Exception:
        H_inv = torch.linalg.pinv(H_p)

    try:
        U = torch.linalg.cholesky(H_inv, upper=True)
    except Exception:
        U = torch.linalg.cholesky(H_inv + 1e-6 * torch.eye(K, device=device), upper=True)

    W_p = W[:, perm].clone().detach()
    group_idx = perm // group_size

    # 3. Blocked GPTQ column compensation loop
    for i in range(0, K, blocksize):
        i_end = min(i + blocksize, K)
        count = i_end - i
        W1 = W_p[:, i:i_end]
        U1 = U[i:i_end, i:i_end]
        Err = torch.zeros_like(W1)

        for j in range(count):
            k = i + j
            g = group_idx[k]
            s = best_scale[:, g]
            z = best_zp[:, g]
            w = W1[:, j]
            d = U1[j, j]

            q = s * torch.clamp(torch.round((w - z) / s), 0, 3) + z
            err = (w - q) / d
            W1[:, j:] -= err.unsqueeze(1) * U1[j, j:].unsqueeze(0)
            Err[:, j] = err

        if i_end < K:
            W_p[:, i_end:] -= Err @ U[i:i_end, i_end:]

    # 4. Invert permutation back to original column order
    W_quant = W_p[:, invperm]

    # 5. Pack codes into uint32 (int32 view)
    s_exp = best_scale.unsqueeze(-1).repeat(1, 1, group_size).view(N, K)
    z_exp = best_zp.unsqueeze(-1).repeat(1, 1, group_size).view(N, K)
    Q = torch.clamp(torch.round((W_quant - z_exp) / s_exp), 0, 3).to(torch.int32)

    codes_16 = Q.view(N, K // 16, 16)
    shifts = torch.arange(0, 32, 2, dtype=torch.int32, device=device)
    shifted = codes_16 << shifts.unsqueeze(0).unsqueeze(0)
    W_packed = shifted.sum(dim=-1, dtype=torch.int32)

    scales_fp16 = best_scale.to(torch.float16)
    zero_points_fp16 = best_zp.to(torch.float16)

    return W_packed, scales_fp16, zero_points_fp16, W_quant


def run_stage3_gptq(
    repo_id: str = "Qwen/Qwen3-8B",
    output_dir: str = "./qwen3-8b-int2-stage3",
    num_sequences: int = 64,
    seq_len: int = 512,
    micro_batch: int = 4,
    blocksize: int = 128,
    max_layers: int = None,
):
    print("=" * 80)
    print("STAGE 3: CHOLEKSY-DAMPED BLOCKED GPTQ QUANTIZER (QWEN3-8B)")
    print(f"Target Checkpoint:  {output_dir}")
    print(f"Group Layout:       Uniform G=16 for all 7 linear projections")
    print(f"Calibration Budget: {num_sequences} sequences x {seq_len} tokens = {num_sequences * seq_len} tokens")
    print(f"Micro-batch:        {micro_batch} sequences per accumulation step")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 1. Download & copy metadata
    print("\n[1/5] Setting up configuration and tokenizer metadata...")
    metadata_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]
    for fn in metadata_files:
        try:
            fpath = hf_hub_download(repo_id=repo_id, filename=fn, local_files_only=True)
            shutil.copy(fpath, os.path.join(output_dir, fn))
            print(f"      Saved {fn}")
        except Exception as e:
            print(f"      Note: {fn} ({e})")

    config = AutoConfig.from_pretrained(output_dir, local_files_only=True)
    config._attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)

    # 2. Build calibration dataset
    print("\n[2/5] Building multi-domain calibration dataset...")
    calib_tokens = build_calibration_dataset(tokenizer, num_sequences=num_sequences, seq_len=seq_len)
    print(f"      Generated {calib_tokens.shape[0]} sequences of {calib_tokens.shape[1]} tokens ({calib_tokens.numel()} tokens total).")

    # 3. Stream embed_tokens to compute Layer 0 inps on CPU RAM
    print("\n[3/5] Streaming embed_tokens to produce initial calibration inputs (inps)...")
    index_file = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json", local_files_only=True)
    with open(index_file, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    embed_shard = hf_hub_download(repo_id=repo_id, filename=weight_map["model.embed_tokens.weight"], local_files_only=True)
    with safe_open(embed_shard, framework="pt", device="cpu") as f:
        embed_weight = f.get_tensor("model.embed_tokens.weight").to(torch.float16)

    # Compute inps in micro-batches
    inps = torch.empty((num_sequences, seq_len, config.hidden_size), dtype=torch.float16, device="cpu")
    num_micro_batches = num_sequences // micro_batch

    for mb_idx in range(num_micro_batches):
        batch_ids = calib_tokens[mb_idx * micro_batch : (mb_idx + 1) * micro_batch]
        batch_embeds = F.embedding(batch_ids, embed_weight)
        inps[mb_idx * micro_batch : (mb_idx + 1) * micro_batch].copy_(batch_embeds)

    del embed_weight
    torch.cuda.empty_cache()
    print(f"      Initial inps shape: {tuple(inps.shape)} (CPU RAM: {inps.numel() * 2 / (1024**2):.1f} MB)")

    # Rotary embedding and position IDs setup
    rotary_emb = Qwen3RotaryEmbedding(config).to(device)
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    dummy_x = torch.empty((micro_batch, seq_len, config.hidden_size), dtype=torch.float16, device=device)
    pos_emb = rotary_emb(dummy_x, pos_ids)
    del dummy_x

    # 4. Sequential Layer-by-Layer Calibration & GPTQ Execution
    num_layers_to_run = config.num_hidden_layers if max_layers is None else min(max_layers, config.num_hidden_layers)
    print(f"\n[4/5] Running sequential layer-by-layer GPTQ across {num_layers_to_run} decoder layers...")

    all_quantized_tensors: Dict[str, torch.Tensor] = {}
    saved_shards_map: Dict[str, str] = {}
    start_time = time.perf_counter()

    for layer_idx in range(num_layers_to_run):
        t_layer_start = time.perf_counter()
        print(f"\n---> [Layer {layer_idx:2d}/{num_layers_to_run}] Streaming layer weights onto CUDA...")

        # Instantiate layer skeleton on CUDA
        layer = Qwen3DecoderLayer(config, layer_idx=layer_idx).to(device=device, dtype=torch.float16)

        # Load weights for this layer from safetensors shards
        layer_prefix = f"model.layers.{layer_idx}."
        needed_keys = [k for k in weight_map.keys() if k.startswith(layer_prefix)]
        shard_files = set(weight_map[k] for k in needed_keys)

        for sf in shard_files:
            shard_path = hf_hub_download(repo_id=repo_id, filename=sf, local_files_only=True)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for k in needed_keys:
                    if weight_map[k] == sf:
                        t = f.get_tensor(k)
                        sub_k = k[len(layer_prefix):]
                        parts = sub_k.split(".")
                        mod = layer
                        for p in parts[:-1]:
                            mod = getattr(mod, p)
                        param = getattr(mod, parts[-1])
                        param.data.copy_(t.to(device=device, dtype=torch.float16))

        # --- Sub-step A: Accumulate H for Q, K, V ---
        H_qkv = torch.zeros((config.hidden_size, config.hidden_size), device=device, dtype=torch.float32)
        for mb in range(num_micro_batches):
            x_mb = inps[mb * micro_batch : (mb + 1) * micro_batch].to(device)
            normed = layer.input_layernorm(x_mb).view(-1, config.hidden_size).float()
            H_qkv += normed.T @ normed
            del x_mb, normed

        # Quantize q_proj, k_proj, v_proj
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            proj = getattr(layer.self_attn, proj_name)
            w = proj.weight.data.float()
            W_packed, scales, zp, W_quant = quantize_projection_gptq(w, H_qkv, group_size=16, blocksize=blocksize)

            prefix = f"model.layers.{layer_idx}.self_attn.{proj_name}"
            all_quantized_tensors[f"{prefix}.W_packed"] = W_packed.cpu()
            all_quantized_tensors[f"{prefix}.scales"] = scales.cpu()
            all_quantized_tensors[f"{prefix}.zero_points"] = zp.cpu()
            all_quantized_tensors[f"{prefix}.group_size"] = torch.tensor([16], dtype=torch.int32)

            proj.weight.data.copy_(W_quant.half())
            del w, W_packed, scales, zp, W_quant

        del H_qkv
        torch.cuda.empty_cache()

        # --- Sub-step B: Accumulate H for O ---
        H_o = torch.zeros((config.hidden_size, config.hidden_size), device=device, dtype=torch.float32)

        def o_hook(mod, inp, out):
            x_act = inp[0].view(-1, config.hidden_size).float()
            H_o.add_(x_act.T @ x_act)

        h_hook = layer.self_attn.o_proj.register_forward_hook(o_hook)
        with torch.no_grad():
            for mb in range(num_micro_batches):
                x_mb = inps[mb * micro_batch : (mb + 1) * micro_batch].to(device)
                _ = layer.self_attn(
                    hidden_states=layer.input_layernorm(x_mb),
                    attention_mask=None,
                    position_embeddings=pos_emb,
                    position_ids=pos_ids,
                )
                del x_mb
        h_hook.remove()

        # Quantize o_proj
        w = layer.self_attn.o_proj.weight.data.float()
        W_packed, scales, zp, W_quant = quantize_projection_gptq(w, H_o, group_size=16, blocksize=blocksize)
        prefix = f"model.layers.{layer_idx}.self_attn.o_proj"
        all_quantized_tensors[f"{prefix}.W_packed"] = W_packed.cpu()
        all_quantized_tensors[f"{prefix}.scales"] = scales.cpu()
        all_quantized_tensors[f"{prefix}.zero_points"] = zp.cpu()
        all_quantized_tensors[f"{prefix}.group_size"] = torch.tensor([16], dtype=torch.int32)

        layer.self_attn.o_proj.weight.data.copy_(W_quant.half())
        del w, H_o, W_packed, scales, zp, W_quant
        torch.cuda.empty_cache()

        # --- Sub-step C: Accumulate H for Gate & Up ---
        H_gate_up = torch.zeros((config.hidden_size, config.hidden_size), device=device, dtype=torch.float32)
        with torch.no_grad():
            for mb in range(num_micro_batches):
                x_mb = inps[mb * micro_batch : (mb + 1) * micro_batch].to(device)
                attn_out, _ = layer.self_attn(
                    hidden_states=layer.input_layernorm(x_mb),
                    attention_mask=None,
                    position_embeddings=pos_emb,
                    position_ids=pos_ids,
                )
                h_post = layer.post_attention_layernorm(x_mb + attn_out).view(-1, config.hidden_size).float()
                H_gate_up += h_post.T @ h_post
                del x_mb, attn_out, h_post

        # Quantize gate_proj & up_proj
        for proj_name in ["gate_proj", "up_proj"]:
            proj = getattr(layer.mlp, proj_name)
            w = proj.weight.data.float()
            W_packed, scales, zp, W_quant = quantize_projection_gptq(w, H_gate_up, group_size=16, blocksize=blocksize)

            prefix = f"model.layers.{layer_idx}.mlp.{proj_name}"
            all_quantized_tensors[f"{prefix}.W_packed"] = W_packed.cpu()
            all_quantized_tensors[f"{prefix}.scales"] = scales.cpu()
            all_quantized_tensors[f"{prefix}.zero_points"] = zp.cpu()
            all_quantized_tensors[f"{prefix}.group_size"] = torch.tensor([16], dtype=torch.int32)

            proj.weight.data.copy_(W_quant.half())
            del w, W_packed, scales, zp, W_quant

        del H_gate_up
        torch.cuda.empty_cache()

        # --- Sub-step D: Accumulate H for Down ---
        H_down = torch.zeros((config.intermediate_size, config.intermediate_size), device=device, dtype=torch.float32)

        def down_hook(mod, inp, out):
            x_act = inp[0].view(-1, config.intermediate_size).float()
            H_down.add_(x_act.T @ x_act)

        h_hook = layer.mlp.down_proj.register_forward_hook(down_hook)
        with torch.no_grad():
            for mb in range(num_micro_batches):
                x_mb = inps[mb * micro_batch : (mb + 1) * micro_batch].to(device)
                attn_out, _ = layer.self_attn(
                    hidden_states=layer.input_layernorm(x_mb),
                    attention_mask=None,
                    position_embeddings=pos_emb,
                    position_ids=pos_ids,
                )
                h_post = layer.post_attention_layernorm(x_mb + attn_out)
                _ = layer.mlp(h_post)
                del x_mb, attn_out, h_post
        h_hook.remove()

        # Quantize down_proj
        w = layer.mlp.down_proj.weight.data.float()
        W_packed, scales, zp, W_quant = quantize_projection_gptq(w, H_down, group_size=16, blocksize=blocksize)
        prefix = f"model.layers.{layer_idx}.mlp.down_proj"
        all_quantized_tensors[f"{prefix}.W_packed"] = W_packed.cpu()
        all_quantized_tensors[f"{prefix}.scales"] = scales.cpu()
        all_quantized_tensors[f"{prefix}.zero_points"] = zp.cpu()
        all_quantized_tensors[f"{prefix}.group_size"] = torch.tensor([16], dtype=torch.int32)

        layer.mlp.down_proj.weight.data.copy_(W_quant.half())
        del w, H_down, W_packed, scales, zp, W_quant
        torch.cuda.empty_cache()

        # Save norms
        all_quantized_tensors[f"model.layers.{layer_idx}.input_layernorm.weight"] = layer.input_layernorm.weight.data.cpu()
        all_quantized_tensors[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight.data.cpu()
        all_quantized_tensors[f"model.layers.{layer_idx}.self_attn.q_norm.weight"] = layer.self_attn.q_norm.weight.data.cpu()
        all_quantized_tensors[f"model.layers.{layer_idx}.self_attn.k_norm.weight"] = layer.self_attn.k_norm.weight.data.cpu()

        # --- Sub-step E: Update inps for next layer and measure SQNR ---
        outs = torch.empty_like(inps)
        signal_pwr, noise_pwr = 0.0, 0.0

        with torch.no_grad():
            for mb in range(num_micro_batches):
                x_mb = inps[mb * micro_batch : (mb + 1) * micro_batch].to(device)
                out_mb = layer(
                    x_mb,
                    attention_mask=None,
                    position_embeddings=pos_emb,
                    position_ids=pos_ids,
                )
                outs[mb * micro_batch : (mb + 1) * micro_batch].copy_(out_mb.cpu())

                signal_pwr += (x_mb.float() ** 2).mean().item()
                noise_pwr += ((out_mb.float() - x_mb.float()) ** 2).mean().item()
                del x_mb, out_mb

        inps = outs
        dt_layer = time.perf_counter() - t_layer_start
        sqnr_val = 10.0 * torch.log10(torch.tensor(max(signal_pwr, 1e-6) / max(noise_pwr, 1e-6))).item()
        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        print(f"      Layer {layer_idx:2d} Done in {dt_layer:.1f}s | Output Activation std={inps.float().std():.4f} (min={inps.float().min():.2f}, max={inps.float().max():.2f}) | Peak VRAM: {vram_mb:.1f} MB")

        del layer
        torch.cuda.empty_cache()

    # 5. Save all shards, non-linear weights, and index.json
    print("\n[5/5] Packaging and saving safetensors shards to disk...")

    # Load embed_tokens, lm_head, model.norm
    for sf in set(weight_map[k] for k in ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]):
        shard_path = hf_hub_download(repo_id=repo_id, filename=sf, local_files_only=True)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]:
                if weight_map[k] == sf:
                    all_quantized_tensors[k] = f.get_tensor(k).to(torch.float16)

    # Shard assignment (5 shards)
    shard_count = 5
    out_shards: List[Dict[str, torch.Tensor]] = [{} for _ in range(shard_count)]
    new_weight_map: Dict[str, str] = {}

    for k, tensor in all_quantized_tensors.items():
        if k == "model.embed_tokens.weight":
            s_idx = 0
        elif k in ("lm_head.weight", "model.norm.weight"):
            s_idx = 4
        elif "model.layers." in k:
            layer_num = int(k.split(".")[2])
            s_idx = min(layer_num // 8, 4)
        else:
            s_idx = 0

        shard_name = f"model-{s_idx+1:05d}-of-{shard_count:05d}.safetensors"
        out_shards[s_idx][k] = tensor
        new_weight_map[k] = shard_name

    for s_idx, shard_dict in enumerate(out_shards):
        shard_name = f"model-{s_idx+1:05d}-of-{shard_count:05d}.safetensors"
        shard_path = os.path.join(output_dir, shard_name)
        print(f"      Writing {len(shard_dict)} tensors to {shard_name}...")
        save_file(shard_dict, shard_path)

    # Save index.json
    index_dict = {
        "metadata": {
            "total_size": 0,
            "quantization": "int2_asymmetric_group_gptq",
            "uniform_g16": True,
        },
        "weight_map": new_weight_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(index_dict, f, indent=2)

    total_elapsed = time.perf_counter() - start_time
    print("=" * 80)
    print(f"STAGE 3 GPTQ QUANTIZATION COMPLETE in {total_elapsed:.1f}s ({total_elapsed/60.0:.2f} min)!")
    print(f"Total Tensors Saved: {len(new_weight_map)}")
    print(f"Target Directory:    {os.path.abspath(output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 GPTQ Quantizer for Qwen3-8B")
    parser.add_argument("--repo-id", type=str, default="Qwen/Qwen3-8B", help="Base model repo ID")
    parser.add_argument("--output-dir", type=str, default="./qwen3-8b-int2-stage3", help="Output directory")
    parser.add_argument("--num-sequences", type=int, default=64, help="Number of calibration sequences")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length per calibration sample")
    parser.add_argument("--micro-batch", type=int, default=4, help="Micro-batch size for GPU execution")
    parser.add_argument("--blocksize", type=int, default=128, help="GPTQ column blocksize")
    parser.add_argument("--max-layers", type=int, default=None, help="Quantize only first N layers (for testing)")
    args = parser.parse_args()

    run_stage3_gptq(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        micro_batch=args.micro_batch,
        blocksize=args.blocksize,
        max_layers=args.max_layers,
    )
