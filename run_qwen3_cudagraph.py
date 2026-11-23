"""
run_qwen3_cudagraph.py - CUDA Graph Accelerated 4 GB VRAM Inference for LittleBit Qwen3-8B (0.55 bpw)

Key Features:
  1. Reuses PackedLittleBitLinear, CPUEmbeddingWrapper, rotary materialization,
     zero-meta-tensor assertion, and layer-0 accuracy check from run_qwen3_littlebit.py.
  2. Uses StaticCache(config=cfg, max_cache_len=512) for fixed KV cache shapes on CUDA
     with zero dynamic allocations during decode.
  3. Runs prefill using CPU-looked-up embeddings with explicit cache_position and position_ids.
  4. Captures the single-token decode forward pass into torch.cuda.CUDAGraph() on a side stream
     using pre-allocated static tensors (static_embeds, static_cache_pos, static_pos_ids).
  5. Replays the CUDA Graph during autoregressive generation with nucleus sampling
     (temperature=0.7, top_p=0.9, repetition_penalty=1.18, max_new_tokens=96).
  6. Prints peak VRAM, prefill time (ms), CUDA-Graphed tokens/sec, and the generated text.
"""

from __future__ import annotations
import os
import sys
import time
import gc
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import StaticCache
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# Ensure local modules (cuda_ops, packer) can be imported
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from cuda_ops import littlebit_forward_cuda

REPO_ID = "LittleBitLLM/littlebit-qwen3-8b-0.55bpw"


# ---------------------------------------------------------------------------
# 1. CPU Embedding Wrapper
# ---------------------------------------------------------------------------

class CPUEmbeddingWrapper(nn.Module):
    """
    Holds the 151936 x 4096 embedding table (~1.24 GB) in CPU system RAM.
    During each forward step, looks up the token embeddings on CPU and
    transfers only the small activation tensor to CUDA in FP16.
    """
    def __init__(self, weight: torch.Tensor, device: str = "cuda:0", dtype: torch.dtype = torch.float16):
        super().__init__()
        self.weight = nn.Parameter(weight.to("cpu", dtype=dtype), requires_grad=False)
        self.target_device = device
        self.target_dtype = dtype

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids_cpu = input_ids.to("cpu")
        embeds = F.embedding(ids_cpu, self.weight)
        return embeds.to(device=self.target_device, dtype=self.target_dtype)


# ---------------------------------------------------------------------------
# 2. PackedLittleBitLinear Module
# ---------------------------------------------------------------------------

class PackedLittleBitLinear(nn.Module):
    """
    PyTorch nn.Module representing a bit-packed LittleBit linear layer.
    Holds pre-packed int32 binary weights and FP16 scale vectors.
    Executes register-level unpacking GEMM via cuda_ops.littlebit_forward_cuda.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        split_dim: int,
        residual: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.split_dim = split_dim
        self.residual = residual
        self.has_bias = bias

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in PackedLittleBitLinear")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self):
        return [k for k, _ in self.named_buffers()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return littlebit_forward_cuda(x, self)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"split_dim={self.split_dim}, residual={self.residual}, has_bias={self.has_bias}"
        )


def build_packed_linear(
    f: safe_open,
    prefix: str,
    in_features: int,
    out_features: int,
    device: str = "cpu",
) -> PackedLittleBitLinear:
    """
    Constructs a PackedLittleBitLinear from safetensors tensors on CPU.
    Loads pre-packed int32 matrices and casts scale vectors to float16.
    """
    split_dim_key = f"{prefix}._split_dim_final"
    if split_dim_key in f.keys():
        split_dim = int(f.get_tensor(split_dim_key).item())
    elif f"{prefix}.U_shape" in f.keys():
        split_dim = int(f.get_tensor(f"{prefix}.U_shape")[1].item())
    else:
        split_dim = int(f.get_tensor(f"{prefix}.V_packed").shape[0])

    has_residual = f"{prefix}.V_R_packed" in f.keys()
    has_bias = f"{prefix}.bias" in f.keys()

    layer = PackedLittleBitLinear(
        in_features=in_features,
        out_features=out_features,
        split_dim=split_dim,
        residual=has_residual,
        bias=has_bias,
    )

    # Main branch: pre-packed int32 weights and float16 scales
    V_packed = f.get_tensor(f"{prefix}.V_packed").to(device=device)
    U_packed = f.get_tensor(f"{prefix}.U_packed").to(device=device)
    v2 = f.get_tensor(f"{prefix}.v2").to(device=device, dtype=torch.float16)
    v1 = f.get_tensor(f"{prefix}.v1").to(device=device, dtype=torch.float16)
    u2 = f.get_tensor(f"{prefix}.u2").to(device=device, dtype=torch.float16)
    u1 = f.get_tensor(f"{prefix}.u1").to(device=device, dtype=torch.float16)

    layer.register_buffer("V_packed", V_packed)
    layer.register_buffer("U_packed", U_packed)
    layer.register_buffer("v2", v2)
    layer.register_buffer("v1", v1)
    layer.register_buffer("u2", u2)
    layer.register_buffer("u1", u1)

    # Residual branch if present
    if has_residual:
        V_R_packed = f.get_tensor(f"{prefix}.V_R_packed").to(device=device)
        U_R_packed = f.get_tensor(f"{prefix}.U_R_packed").to(device=device)
        v2_R = f.get_tensor(f"{prefix}.v2_R").to(device=device, dtype=torch.float16)
        v1_R = f.get_tensor(f"{prefix}.v1_R").to(device=device, dtype=torch.float16)
        u2_R = f.get_tensor(f"{prefix}.u2_R").to(device=device, dtype=torch.float16)
        u1_R = f.get_tensor(f"{prefix}.u1_R").to(device=device, dtype=torch.float16)

        layer.register_buffer("V_R_packed", V_R_packed)
        layer.register_buffer("U_R_packed", U_R_packed)
        layer.register_buffer("v2_R", v2_R)
        layer.register_buffer("v1_R", v1_R)
        layer.register_buffer("u2_R", u2_R)
        layer.register_buffer("u1_R", u1_R)

    if has_bias:
        bias_tensor = f.get_tensor(f"{prefix}.bias").to(device=device, dtype=torch.float16)
        layer.register_buffer("bias", bias_tensor)

    return layer


# ---------------------------------------------------------------------------
# 3. Single-Layer Numerical Accuracy Verification
# ---------------------------------------------------------------------------

def _unpack_to_fp32(packed_tensor: torch.Tensor, orig_cols: int) -> torch.Tensor:
    """
    Unpacks an int32/uint32 matrix of shape (n_rows, words_per_row)
    into a full FP32 matrix of shape (n_rows, orig_cols) with values in {-1.0, +1.0}.
    Bit 0 = +1.0, Bit 1 = -1.0 (LSB first).
    """
    n_rows, words_per_row = packed_tensor.shape
    device = packed_tensor.device
    packed_i64 = packed_tensor.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(32, dtype=torch.int64, device=device)
    bits = (packed_i64.unsqueeze(-1) >> shifts) & 1
    bits_flat = bits.view(n_rows, words_per_row * 32)[:, :orig_cols]
    return (1.0 - 2.0 * bits_flat.to(torch.float32)).contiguous()


def explicit_fp32_unpacked_forward(
    x: torch.Tensor,
    layer: PackedLittleBitLinear,
) -> torch.Tensor:
    """
    Computes ground-truth LittleBit forward pass using explicit FP32 unpacked matrices.
    """
    x_f32 = x.to(torch.float32)
    in_feat = layer.in_features
    split_dim = layer.split_dim

    V_f32 = _unpack_to_fp32(layer.V_packed, in_feat)      # (split_dim, in_features)
    U_f32 = _unpack_to_fp32(layer.U_packed, split_dim)    # (out_features, split_dim)

    v2 = layer.v2.to(device=x.device, dtype=torch.float32)
    v1 = layer.v1.to(device=x.device, dtype=torch.float32)
    u2 = layer.u2.to(device=x.device, dtype=torch.float32)
    u1 = layer.u1.to(device=x.device, dtype=torch.float32)

    a = x_f32 * v2
    h = torch.matmul(a, V_f32.t())
    g = h * (v1 * u2)
    z = torch.matmul(g, U_f32.t())
    y = z * u1

    if layer.residual and hasattr(layer, "V_R_packed"):
        V_R_f32 = _unpack_to_fp32(layer.V_R_packed, in_feat)
        U_R_f32 = _unpack_to_fp32(layer.U_R_packed, split_dim)
        v2_R = layer.v2_R.to(device=x.device, dtype=torch.float32)
        v1_R = layer.v1_R.to(device=x.device, dtype=torch.float32)
        u2_R = layer.u2_R.to(device=x.device, dtype=torch.float32)
        u1_R = layer.u1_R.to(device=x.device, dtype=torch.float32)

        a_R = x_f32 * v2_R
        h_R = torch.matmul(a_R, V_R_f32.t())
        g_R = h_R * (v1_R * u2_R)
        z_R = torch.matmul(g_R, U_R_f32.t())
        y_R = z_R * u1_R
        y = y + y_R

    if hasattr(layer, "bias") and layer.bias is not None:
        y = y + layer.bias.to(device=x.device, dtype=torch.float32)

    return y


def verify_layer_accuracy(
    layer: PackedLittleBitLinear,
    layer_name: str = "model.layers[0].self_attn.q_proj",
    batch_size: int = 1,
    seq_len: int = 16,
    device: str = "cuda:0",
):
    """
    Compares PackedLittleBitLinear against explicit FP32 unpacked reference math.
    """
    print("\n" + "=" * 75)
    print(f"  Single-Layer Numerical Accuracy Check: {layer_name}")
    print("=" * 75)
    print(f"  Configuration: in_features={layer.in_features}, out_features={layer.out_features}, "
          f"split_dim={layer.split_dim}, residual={layer.residual}")

    torch.manual_seed(42)
    x_test = torch.randn(batch_size, seq_len, layer.in_features, device=device, dtype=torch.float16)

    torch.cuda.synchronize()
    out_cuda = layer(x_test).to(torch.float32)
    torch.cuda.synchronize()

    out_ref = explicit_fp32_unpacked_forward(x_test, layer)

    diff = (out_cuda - out_ref).abs()
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()

    print(f"  Test Tensor Shape   : {tuple(x_test.shape)} (FP16)")
    print(f"  Max Absolute Error  : {max_abs_err:.6e}")
    print(f"  Mean Absolute Error : {mean_abs_err:.6e}")

    status = "PASSED" if max_abs_err < 1e-2 else "WARNING: High difference"
    print(f"  Accuracy Status     : [{status}] (threshold: 1e-2)")
    print("=" * 75 + "\n")


# ---------------------------------------------------------------------------
# 4. Model Streaming Loader
# ---------------------------------------------------------------------------

def load_littlebit_qwen3(
    repo_id: str = REPO_ID,
    cuda_device: str = "cuda:0",
) -> tuple[nn.Module, AutoTokenizer]:
    """
    Loads Qwen3-8B 0.55bpw architecture onto meta device, loads pre-packed weights,
    and moves transformer blocks + lm_head to CUDA (FP16) while keeping embed_tokens on CPU.
    """
    print(f"Loading configuration and tokenizer from local cache ({repo_id})...")
    config = AutoConfig.from_pretrained(repo_id, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, local_files_only=True)

    safetensors_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", local_files_only=True)
    print(f"Checkpoint file: {safetensors_path}")

    # 1. Instantiate skeleton model on meta device
    print("Instantiating Qwen3 skeleton model on meta device...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    # 2. Materialize rotary embedding on CUDA (avoids meta tensor errors)
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        rot_cls = type(model.model.rotary_emb)
        model.model.rotary_emb = rot_cls(config=config).to(cuda_device)
        print("Materialized rotary embedding on CUDA.")

    # 3. Stream-load weights layer-by-layer
    print("Stream-loading weights and moving layers to CUDA...")
    num_layers = len(model.model.layers)

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        # Load embedding table on CPU
        print("Loading embed_tokens table onto CPU (saving ~1.24 GB VRAM)...")
        embed_weight = f.get_tensor("model.embed_tokens.weight").to(torch.float16)
        model.model.embed_tokens = CPUEmbeddingWrapper(embed_weight, device=cuda_device, dtype=torch.float16)

        # Stream-load each of the 36 decoder layers
        for i in range(num_layers):
            layer = model.model.layers[i]
            prefix = f"model.layers.{i}"

            # Attention projections
            layer.self_attn.q_proj = build_packed_linear(
                f, f"{prefix}.self_attn.q_proj", config.hidden_size, config.hidden_size
            )
            layer.self_attn.k_proj = build_packed_linear(
                f, f"{prefix}.self_attn.k_proj", config.hidden_size,
                (config.hidden_size // config.num_attention_heads) * config.num_key_value_heads
            )
            layer.self_attn.v_proj = build_packed_linear(
                f, f"{prefix}.self_attn.v_proj", config.hidden_size,
                (config.hidden_size // config.num_attention_heads) * config.num_key_value_heads
            )
            layer.self_attn.o_proj = build_packed_linear(
                f, f"{prefix}.self_attn.o_proj", config.hidden_size, config.hidden_size
            )

            # Attention Q/K RMSNorms
            layer.self_attn.q_norm.weight = nn.Parameter(
                f.get_tensor(f"{prefix}.self_attn.q_norm.weight").to(torch.float16)
            )
            layer.self_attn.k_norm.weight = nn.Parameter(
                f.get_tensor(f"{prefix}.self_attn.k_norm.weight").to(torch.float16)
            )

            # MLP projections
            layer.mlp.gate_proj = build_packed_linear(
                f, f"{prefix}.mlp.gate_proj", config.hidden_size, config.intermediate_size
            )
            layer.mlp.up_proj = build_packed_linear(
                f, f"{prefix}.mlp.up_proj", config.hidden_size, config.intermediate_size
            )
            layer.mlp.down_proj = build_packed_linear(
                f, f"{prefix}.mlp.down_proj", config.intermediate_size, config.hidden_size
            )

            # Layer norms
            layer.input_layernorm.weight = nn.Parameter(
                f.get_tensor(f"{prefix}.input_layernorm.weight").to(torch.float16)
            )
            layer.post_attention_layernorm.weight = nn.Parameter(
                f.get_tensor(f"{prefix}.post_attention_layernorm.weight").to(torch.float16)
            )

            # Move layer directly to CUDA
            layer.to(cuda_device)

            if (i + 1) % 6 == 0 or (i + 1) == num_layers:
                vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                print(f"  Layer {i+1:2d}/{num_layers} loaded to CUDA | VRAM: {vram_mb:7.1f} MB")

        # Final RMSNorm
        model.model.norm.weight = nn.Parameter(
            f.get_tensor("model.norm.weight").to(torch.float16)
        )
        model.model.norm.to(cuda_device)

        # LM Head (keep on CUDA if under 3.2 GB budget)
        current_vram = torch.cuda.memory_allocated()
        lm_head_weight = f.get_tensor("lm_head.weight").to(torch.float16)
        lm_head_bytes = lm_head_weight.nelement() * lm_head_weight.element_size()

        if current_vram + lm_head_bytes <= 3.2 * (1024 ** 3):
            print(f"Placing lm_head on CUDA ({lm_head_bytes / (1024**2):.1f} MB)...")
            model.lm_head.weight = nn.Parameter(lm_head_weight.to(cuda_device))
        else:
            print("Warning: VRAM over 3.2 GB, offloading lm_head to CPU.")
            model.lm_head.weight = nn.Parameter(lm_head_weight)

    # 4. Strict assertions: verify zero parameters or buffers remain on meta device
    meta_params = [name for name, p in model.named_parameters() if p.device.type == "meta"]
    meta_buffers = [name for name, b in model.named_buffers() if b.device.type == "meta"]

    for i in range(num_layers):
        qn = model.model.layers[i].self_attn.q_norm.weight
        kn = model.model.layers[i].self_attn.k_norm.weight
        assert qn.device.type != "meta", f"Layer {i} q_norm.weight is still on meta device!"
        assert kn.device.type != "meta", f"Layer {i} k_norm.weight is still on meta device!"

    assert len(meta_params) == 0, f"Found parameters still on meta device: {meta_params}"
    assert len(meta_buffers) == 0, f"Found buffers still on meta device: {meta_buffers}"
    print(f"Assertion passed: All {num_layers} layers (including q_norm and k_norm) and model buffers are verified loaded (0 meta tensors remain).")

    torch.cuda.empty_cache()
    gc.collect()

    allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"\nModel loading complete! Total VRAM allocated: {allocated_mb:.1f} MB ({allocated_gb:.2f} GB / 4.00 GB)")

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# 5. Repetition Penalty & Nucleus Sampling Helpers
# ---------------------------------------------------------------------------

def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: List[int],
    penalty: float = 1.18,
) -> torch.Tensor:
    """
    Applies repetition penalty to logits in-place.
    For token_id in token_ids:
      if logit > 0: logit /= penalty
      if logit < 0: logit *= penalty
    """
    if penalty == 1.0 or not token_ids:
        return logits

    unique_tokens = list(set(token_ids))
    unique_tensor = torch.tensor(unique_tokens, dtype=torch.long, device=logits.device)
    scores = logits[0, unique_tensor]
    scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    logits[0, unique_tensor] = scores
    return logits


def sample_next_token(
    logits: torch.Tensor,
    context_tokens: List[int],
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.18,
) -> int:
    """
    Applies repetition penalty, temperature scaling, and nucleus (top-p) sampling.
    """
    # 1. Repetition penalty
    logits = apply_repetition_penalty(logits, context_tokens, penalty=repetition_penalty)

    # 2. Temperature scaling
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1).item()

    logits = logits / temperature

    # 3. Softmax to probability distribution
    probs = F.softmax(logits, dim=-1)

    # 4. Top-p (nucleus) filtering
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    sorted_probs[sorted_indices_to_remove] = 0.0
    prob_sum = sorted_probs.sum(dim=-1, keepdim=True)
    if prob_sum.item() > 0:
        sorted_probs = sorted_probs / prob_sum
    else:
        return sorted_indices[0, 0].item()

    # 5. Multinomial sample
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token_id = torch.gather(sorted_indices, -1, sampled_sorted_idx).item()

    return next_token_id


# ---------------------------------------------------------------------------
# 6. CUDA Graph Accelerated Inference and Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_and_benchmark_cudagraph(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 96,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.18,
    max_cache_len: int = 512,
    device: str = "cuda:0",
):
    print("\n" + "=" * 75)
    print("  LittleBit Qwen3-8B CUDA-Graph Accelerated Generation Benchmark")
    print("=" * 75)
    print(f"Direct Completion Prompt:\n  {prompt!r}\n")

    # 1. Encode prompt
    input_ids_cpu = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids_cpu.shape[-1]
    print(f"Prompt length: {seq_len} tokens")

    # 2. Initialize StaticCache with fixed max_cache_len to prevent dynamic VRAM reallocations
    print(f"Initializing StaticCache(max_cache_len={max_cache_len})...")
    cache = StaticCache(config=model.config, max_cache_len=max_cache_len)

    # Lookup prompt embeddings on CPU and transfer to CUDA
    inputs_embeds = model.model.embed_tokens(input_ids_cpu)  # (1, seq_len, 4096) on CUDA

    cache_position = torch.arange(seq_len, device=device)
    position_ids = cache_position.unsqueeze(0)

    # 3. Run prefill pass
    print("Running prefill pass...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t_prefill_start = time.perf_counter()
    outputs = model(
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=cache_position,
        position_ids=position_ids,
        use_cache=True,
    )
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t_prefill_start

    # Sample first token from prefill logits
    prefill_logits = outputs.logits[:, -1, :].clone()
    context_tokens = input_ids_cpu[0].tolist()
    next_token_id = sample_next_token(
        prefill_logits,
        context_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    generated_tokens: List[int] = [next_token_id]
    context_tokens.append(next_token_id)

    # Stop tokens
    eos_ids: Set[int] = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    for token_str in ["<|im_end|>", "<|endoftext|>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(token_str)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.add(tid)
        except Exception:
            pass

    # 4. Pre-allocate static CUDA tensors for single-token decode
    hidden_size = model.config.hidden_size
    static_embeds = torch.empty((1, 1, hidden_size), device=device, dtype=torch.float16)
    static_cache_pos = torch.empty((1,), device=device, dtype=torch.long)
    static_pos_ids = torch.empty((1, 1), device=device, dtype=torch.long)

    # Initial capture positions
    initial_decode_pos = seq_len
    static_cache_pos.fill_(initial_decode_pos)
    static_pos_ids.fill_(initial_decode_pos)
    first_token_embed = model.model.embed_tokens(torch.tensor([[next_token_id]]))
    static_embeds.copy_(first_token_embed)

    # Warm up decode pass 3 times on a side stream before capture
    print("Warming up decode pass 3x on side stream for CUDA Graph capture...")
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            _ = model(
                inputs_embeds=static_embeds,
                past_key_values=cache,
                cache_position=static_cache_pos,
                position_ids=static_pos_ids,
                use_cache=True,
            )
    torch.cuda.current_stream().wait_stream(side_stream)

    # Capture decode forward pass into CUDA Graph
    print("Capturing decode pass into torch.cuda.CUDAGraph()...")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side_stream):
        static_outputs = model(
            inputs_embeds=static_embeds,
            past_key_values=cache,
            cache_position=static_cache_pos,
            position_ids=static_pos_ids,
            use_cache=True,
        )
    static_logits = static_outputs.logits  # shape: (1, 1, vocab_size)
    print("CUDA Graph capture successful!")

    # 5. Autoregressive token generation loop using CUDA Graph replay
    torch.cuda.synchronize()
    t_gen_start = time.perf_counter()

    for step in range(max_new_tokens - 1):
        cur_pos = seq_len + step
        if cur_pos >= max_cache_len:
            print(f"Reached max_cache_len ({max_cache_len}), stopping generation.")
            break

        # Look up embedding of next_token_id on CPU and copy into static_embeds
        token_embed = model.model.embed_tokens(torch.tensor([[next_token_id]]))
        static_embeds.copy_(token_embed)

        # Update static_cache_pos and static_pos_ids in-place via .copy_()
        cur_pos_tensor = torch.tensor([cur_pos], device=device, dtype=torch.long)
        static_cache_pos.copy_(cur_pos_tensor)
        static_pos_ids.copy_(cur_pos_tensor.unsqueeze(0))

        # Replay the CUDA Graph (zero CPU kernel-launch overhead)
        g.replay()

        # Sample next token from static_logits
        step_logits = static_logits[:, -1, :].clone()
        next_token_id = sample_next_token(
            step_logits,
            context_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        generated_tokens.append(next_token_id)
        context_tokens.append(next_token_id)

        if next_token_id in eos_ids:
            break

    torch.cuda.synchronize()
    gen_time = time.perf_counter() - t_gen_start

    num_generated = len(generated_tokens)
    tokens_per_sec = (num_generated - 1) / gen_time if gen_time > 0 and num_generated > 1 else num_generated / max(gen_time, 1e-6)

    # 6. Memory statistics
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # 7. Decode text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_output = prompt + generated_text

    print("\n" + "=" * 75)
    print("  CUDA Graph Benchmark Results:")
    print("=" * 75)
    print(f"  1. Peak VRAM Allocated       : {peak_vram_mb:8.1f} MB  ({peak_vram_gb:.2f} GB / 4.00 GB)")
    print(f"  2. Prefill Time              : {prefill_time*1000:8.2f} ms")
    print(f"  3. CUDA-Graphed Decode Speed : {tokens_per_sec:8.2f} tokens/sec ({num_generated} tokens in {gen_time:.3f} s)")
    print(f"  4. Sampling Parameters       : temp={temperature}, top_p={top_p}, rep_penalty={repetition_penalty}")
    print("=" * 75)
    print("\nGenerated Text Output:\n")
    print("-" * 75)
    print(full_output)
    print("-" * 75)


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA device not available.")
        print("Please run this script inside your active Windows MSVC/CUDA shell via:")
        print("  .\\run_with_msvc.bat run_qwen3_cudagraph.py")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"CUDA Device: {device_name} ({total_mem_gb:.2f} GB VRAM)")

    # 1. Load Model and Tokenizer
    model, tokenizer = load_littlebit_qwen3(REPO_ID, cuda_device="cuda:0")

    # 2. Single-layer numerical accuracy check on layer 0 q_proj
    verify_layer_accuracy(
        layer=model.model.layers[0].self_attn.q_proj,
        layer_name="model.layers[0].self_attn.q_proj",
        device="cuda:0",
    )

    # 3. Direct completion prompt
    prompt = "Sub-1-bit quantization compresses large language models by"

    # 4. Run CUDA-Graphed generation and benchmark
    generate_and_benchmark_cudagraph(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=96,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.18,
        max_cache_len=512,
        device="cuda:0",
    )


if __name__ == "__main__":
    main()
