"""
run_qwen3_littlebit.py - 4 GB VRAM Inference for LittleBit Qwen3-8B (0.55 bpw)

Loads LittleBitLLM/littlebit-qwen3-8b-0.55bpw from local Hugging Face cache:
  1. Initializes Qwen3ForCausalLM (36 decoder blocks) on meta device.
  2. Replaces all linear projections with PackedLittleBitLinear.
  3. Stream-loads pre-packed int32 tensors (U_packed, V_packed, U_R_packed, V_R_packed)
     and float16 scale vectors directly on CPU without re-packing.
  4. Keeps embed_tokens (~1.24 GB) in CPU system RAM via CPUEmbeddingWrapper,
     moving only looked-up activations to CUDA per step.
  5. Moves all 36 packed decoder blocks, RMSNorms, and lm_head (FP16) to CUDA.
  6. Asserts 0 parameters or buffers remain on meta device, explicitly verifying
     q_norm.weight and k_norm.weight across all 36 layers.
  7. Runs a single-layer numerical accuracy check on layer 0 q_proj comparing CUDA
     PackedLittleBitLinear against explicit FP32 unpacked reference math.
  8. Prompts the model with direct completion prefix:
     "Sub-1-bit quantization compresses large language models by"
  9. Runs nucleus sampling (temperature=0.7, top_p=0.9, repetition_penalty=1.18, max_new_tokens=96)
     with 1-token CUDA warmup and prints peak VRAM, prefill time, generation tokens/sec, and output.
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
    Computes ground-truth LittleBit forward pass using explicit FP32 unpacked matrices:
      y = ((((x * v2) @ V.T) * (v1 * u2)) @ U.T) * u1
      if residual:
        y += ((((x * v2_R) @ V_R.T) * (v1_R * u2_R)) @ U_R.T) * u1_R
    """
    x_f32 = x.to(torch.float32)
    in_feat = layer.in_features
    split_dim = layer.split_dim

    # Unpack main branch to FP32 {-1.0, +1.0}
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

    # Residual branch
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
    Compares PackedLittleBitLinear (running custom CUDA kernel) against
    explicit FP32 unpacked reference math on the exact layer weights.
    """
    print("\n" + "=" * 75)
    print(f"  Single-Layer Numerical Accuracy Check: {layer_name}")
    print("=" * 75)
    print(f"  Configuration: in_features={layer.in_features}, out_features={layer.out_features}, "
          f"split_dim={layer.split_dim}, residual={layer.residual}")

    torch.manual_seed(42)
    x_test = torch.randn(batch_size, seq_len, layer.in_features, device=device, dtype=torch.float16)

    # 1. Run custom CUDA kernel via PackedLittleBitLinear
    torch.cuda.synchronize()
    out_cuda = layer(x_test).to(torch.float32)
    torch.cuda.synchronize()

    # 2. Run explicit FP32 unpacked reference math
    out_ref = explicit_fp32_unpacked_forward(x_test, layer)

    # 3. Compute absolute error metrics
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

    # Specifically check q_norm.weight and k_norm.weight across all 36 layers
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
    Prevents narrow 0.55 bpw logit margins from locking into single-token loops.
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

    # Mask out tokens with cumulative probability above top_p (keep at least the first token)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    sorted_probs[sorted_indices_to_remove] = 0.0
    prob_sum = sorted_probs.sum(dim=-1, keepdim=True)
    if prob_sum.item() > 0:
        sorted_probs = sorted_probs / prob_sum
    else:
        # Fallback to top token
        return sorted_indices[0, 0].item()

    # 5. Multinomial sample from filtered distribution
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token_id = torch.gather(sorted_indices, -1, sampled_sorted_idx).item()

    return next_token_id


# ---------------------------------------------------------------------------
# 6. Text Generation and Benchmarking
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_and_benchmark(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 96,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.18,
    device: str = "cuda:0",
):
    print("\n" + "=" * 75)
    print("  LittleBit Qwen3-8B Inference & Generation Benchmark")
    print("=" * 75)
    print(f"Direct Completion Prompt:\n  {prompt!r}\n")

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    prompt_len = input_ids.shape[-1]
    print(f"Prompt length: {prompt_len} tokens")

    # 1. One-token CUDA warmup pass
    print("Executing 1-token CUDA warmup pass...")
    warmup_ids = input_ids[:, :1].clone()
    _ = model(input_ids=warmup_ids, use_cache=False)
    torch.cuda.synchronize()
    print("CUDA warmup complete.")

    # Reset peak memory stats before actual generation
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # 2. Prefill pass
    t_prefill_start = time.perf_counter()
    outputs = model(input_ids=input_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t_prefill_start

    # Sample next token from prefill logits using nucleus sampling
    prefill_logits = outputs.logits[:, -1, :].clone()
    context_tokens = input_ids[0].tolist()
    next_token_id = sample_next_token(
        prefill_logits,
        context_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    next_token = torch.tensor([[next_token_id]], device=device)
    past_key_values = outputs.past_key_values

    generated_tokens: List[int] = [next_token_id]
    context_tokens.append(next_token_id)

    # Collect EOS and pad token IDs
    eos_ids: Set[int] = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    for token_str in ["<|im_end|>", "<|endoftext|>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(token_str)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.add(tid)
        except Exception:
            pass

    # 3. Autoregressive token generation loop
    t_gen_start = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        step_logits = outputs.logits[:, -1, :].clone()
        next_token_id = sample_next_token(
            step_logits,
            context_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        next_token = torch.tensor([[next_token_id]], device=device)
        past_key_values = outputs.past_key_values

        generated_tokens.append(next_token_id)
        context_tokens.append(next_token_id)

        if next_token_id in eos_ids:
            break

    torch.cuda.synchronize()
    gen_time = time.perf_counter() - t_gen_start

    num_generated = len(generated_tokens)
    tokens_per_sec = (num_generated - 1) / gen_time if gen_time > 0 and num_generated > 1 else num_generated / max(gen_time, 1e-6)

    # 4. Memory statistics
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # 5. Decode text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_output = prompt + generated_text

    print("\n" + "=" * 75)
    print("  Benchmark Results:")
    print("=" * 75)
    print(f"  1. Peak VRAM Allocated : {peak_vram_mb:8.1f} MB  ({peak_vram_gb:.2f} GB / 4.00 GB)")
    print(f"  2. Prefill Time        : {prefill_time*1000:8.2f} ms")
    print(f"  3. Generation Speed    : {tokens_per_sec:8.2f} tokens/sec ({num_generated} tokens in {gen_time:.3f} s)")
    print(f"  4. Sampling Parameters : temp={temperature}, top_p={top_p}, rep_penalty={repetition_penalty}")
    print("=" * 75)
    print("\nGenerated Text Output:\n")
    print("-" * 75)
    print(full_output)
    print("-" * 75)


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA device not available.")
        print("Please run this script inside your active Windows MSVC/CUDA shell via:")
        print("  .\\run_with_msvc.bat run_qwen3_littlebit.py")
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

    # 3. Direct completion prompt (clean prefix, avoiding chat template tokens that destabilize low-bit weights)
    prompt = "Sub-1-bit quantization compresses large language models by"

    # 4. Run generation and benchmark with nucleus sampling
    generate_and_benchmark(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=96,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.18,
        device="cuda:0",
    )


if __name__ == "__main__":
    main()
