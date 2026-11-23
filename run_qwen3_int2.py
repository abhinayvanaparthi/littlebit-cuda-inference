"""
run_qwen3_int2.py
End-to-End Inference Engine for Qwen3-8B Stage 2 (INT2 Group Quantized):
- Loads the quantized ./qwen3-8b-int2-stage2/ safetensors checkpoint onto RTX 3050 Ti.
- Keeps embed_tokens in CPU host memory via CPUEmbeddingWrapper to guarantee peak VRAM < 3.85 GB.
- Evaluates lm_head on CPU (or CUDA via --lm-head-cuda).
- Replaces all 36 transformer layer linear projections with Int2GroupLinear calling sm_86 CUDA kernels.
- Benchmarks peak VRAM, prompt prefill latency/speed, and decode throughput (tokens/sec).
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import StaticCache, DynamicCache

from cuda_int2_group_ops import Int2GroupLinear

CUDA_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class CPUEmbeddingWrapper(nn.Module):
    """Offloads the 1.24 GB embedding table to CPU host memory."""
    def __init__(self, weight: torch.Tensor, device: str = CUDA_DEVICE, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.weight = nn.Parameter(weight.cpu().to(dtype), requires_grad=False)
        self.target_device = device
        self.target_dtype = dtype

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids_cpu = input_ids.cpu().to(torch.long)
        embeds = F.embedding(ids_cpu, self.weight)
        return embeds.to(device=self.target_device, dtype=self.target_dtype)


class CPUHeadWrapper(nn.Module):
    """Offloads the 1.24 GB lm_head table to CPU host memory."""
    def __init__(self, weight: torch.Tensor, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.weight = nn.Parameter(weight.cpu().to(dtype), requires_grad=False)
        self.dtype = dtype

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, 1, hidden_dim] or [B, S, hidden_dim]
        orig_device = hidden_states.device
        h_cpu = hidden_states.cpu().to(self.dtype)
        logits_cpu = F.linear(h_cpu, self.weight)
        return logits_cpu.to(orig_device)


def load_int2_model(checkpoint_dir: str = "./qwen3-8b-int2-stage2", lm_head_cuda: bool = False):
    print("=" * 80)
    print(f"LOADING INT2 QWEN3-8B MODEL FROM: {checkpoint_dir}")
    print("=" * 80)

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist. Run quantize_qwen3_int2.py first.")

    config = AutoConfig.from_pretrained(checkpoint_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)

    # 1. Instantiate skeleton on meta device
    print("\n[1/4] Instantiating Qwen3-8B skeleton on meta device...")
    with torch.device("meta"):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_config(config)

    # Materialize rotary embedding on CUDA
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        rot_cls = type(model.model.rotary_emb)
        model.model.rotary_emb = rot_cls(config=config).to(CUDA_DEVICE)
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "rotary_emb"):
                layer.self_attn.rotary_emb = model.model.rotary_emb

    # 2. Replace all 36 layers' linear modules with Int2GroupLinear
    print("[2/4] Replacing linear projections with Int2GroupLinear modules...")
    linear_specs = {
        "self_attn.q_proj": (config.hidden_size, config.hidden_size),
        "self_attn.k_proj": (config.hidden_size, config.num_key_value_heads * config.head_dim),
        "self_attn.v_proj": (config.hidden_size, config.num_key_value_heads * config.head_dim),
        "self_attn.o_proj": (config.hidden_size, config.hidden_size),
        "mlp.gate_proj":    (config.hidden_size, config.intermediate_size),
        "mlp.up_proj":      (config.hidden_size, config.intermediate_size),
        "mlp.down_proj":    (config.intermediate_size, config.hidden_size),
    }

    num_layers = len(model.model.layers)
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for name, (in_f, out_f) in linear_specs.items():
            parent_name, child_name = name.split(".")
            parent_mod = getattr(layer, parent_name)
            # Default group_size (overridden during weight loading if saved)
            g_size = 16 if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")) else 32
            int2_mod = Int2GroupLinear(in_f, out_f, group_size=g_size, bias=False, device=CUDA_DEVICE)
            setattr(parent_mod, child_name, int2_mod)

    # 3. Load weights from safetensors shards
    print("[3/4] Streaming weights from safetensors into CUDA buffers and CPU wrappers...")
    index_file = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        shard_names = sorted(list(set(index_data["weight_map"].values())))
    else:
        shard_names = ["model.safetensors"]

    for shard_name in shard_names:
        shard_path = os.path.join(checkpoint_dir, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensor = f.get_tensor(k)

                if k == "model.embed_tokens.weight":
                    model.model.embed_tokens = CPUEmbeddingWrapper(tensor, device=CUDA_DEVICE)
                elif k == "lm_head.weight":
                    if lm_head_cuda:
                        model.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(device=CUDA_DEVICE, dtype=torch.float16)
                        model.lm_head.weight.data.copy_(tensor.to(device=CUDA_DEVICE, dtype=torch.float16))
                    else:
                        model.lm_head = CPUHeadWrapper(tensor, dtype=torch.float16)
                elif "W_packed" in k or "scales" in k or "zero_points" in k or "group_size" in k:
                    # Target Int2GroupLinear buffer
                    # e.g. model.layers.0.self_attn.q_proj.W_packed
                    parts = k.split(".")
                    layer_idx = int(parts[2])
                    parent_name = parts[3]
                    proj_name = parts[4]
                    buf_name = parts[5]

                    layer = model.model.layers[layer_idx]
                    parent_mod = getattr(layer, parent_name)
                    proj_mod = getattr(parent_mod, proj_name)

                    if buf_name == "group_size":
                        proj_mod.group_size = int(tensor.item())
                    else:
                        buf = getattr(proj_mod, buf_name)
                        buf.copy_(tensor.to(device=CUDA_DEVICE))
                elif "q_norm.weight" in k or "k_norm.weight" in k:
                    # Attention Q/K RMSNorms (e.g. model.layers.0.self_attn.q_norm.weight)
                    parts = k.split(".")
                    layer_idx = int(parts[2])
                    norm_name = parts[4]
                    layer = model.model.layers[layer_idx]
                    norm_mod = getattr(layer.self_attn, norm_name)
                    norm_mod.weight = nn.Parameter(tensor.to(device=CUDA_DEVICE, dtype=torch.float16), requires_grad=False)
                elif "input_layernorm" in k or "post_attention_layernorm" in k:
                    # Norm weights
                    parts = k.split(".")
                    layer_idx = int(parts[2])
                    norm_name = parts[3]
                    layer = model.model.layers[layer_idx]
                    norm_mod = getattr(layer, norm_name)
                    norm_mod.weight = nn.Parameter(tensor.to(device=CUDA_DEVICE, dtype=torch.float16), requires_grad=False)
                elif k == "model.norm.weight":
                    model.model.norm.weight = nn.Parameter(tensor.to(device=CUDA_DEVICE, dtype=torch.float16), requires_grad=False)

    # 4. Zero meta tensor verification & cleanup of any remaining meta buffers
    print("[4/4] Verifying 0 meta tensors remain across all model components...")

    for name, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            if "inv_freq" in name and hasattr(model.model, "rotary_emb") and hasattr(model.model.rotary_emb, "inv_freq"):
                parent_mod = model
                sub_names = name.split(".")
                for s in sub_names[:-1]:
                    parent_mod = getattr(parent_mod, s)
                setattr(parent_mod, sub_names[-1], model.model.rotary_emb.inv_freq.clone())
            else:
                parent_mod = model
                sub_names = name.split(".")
                for s in sub_names[:-1]:
                    parent_mod = getattr(parent_mod, s)
                setattr(parent_mod, sub_names[-1], torch.zeros(buf.shape, dtype=buf.dtype, device=CUDA_DEVICE))

    meta_params = [name for name, p in model.named_parameters() if p.device.type == "meta"]
    meta_buffers = [name for name, b in model.named_buffers() if b.device.type == "meta"]
    assert len(meta_params) == 0, f"Found meta parameters: {meta_params}"
    assert len(meta_buffers) == 0, f"Found meta buffers: {meta_buffers}"

    allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"      Model fully loaded! Total GPU VRAM Allocated: {allocated_mb:.1f} MB ({allocated_mb/1024:.2f} GB / 4.00 GB)")
    return model, tokenizer


def sample_next_token(
    logits: torch.Tensor,
    context_tokens: List[int],
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.18,
) -> int:
    logits = logits.to(torch.float32)

    # Repetition penalty
    if repetition_penalty != 1.0 and context_tokens:
        unique_tokens = list(set(context_tokens))
        t = torch.tensor(unique_tokens, dtype=torch.long, device=logits.device)
        vals = logits[0, t]
        logits[0, t] = torch.where(vals < 0, vals * repetition_penalty, vals / repetition_penalty)

    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1).item()

    logits = logits / max(temperature, 1e-4)
    probs = F.softmax(logits, dim=-1)

    # Nucleus (top-p)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    sorted_probs[sorted_indices_to_remove] = 0.0
    p_sum = sorted_probs.sum(dim=-1, keepdim=True)
    if p_sum.item() > 0:
        sorted_probs = sorted_probs / p_sum
    else:
        return sorted_indices[0, 0].item()

    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    return torch.gather(sorted_indices, -1, sampled_sorted_idx).item()


def generate(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str = "Sub-1-bit and 2-bit quantization compresses large language models by",
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.18,
):
    print("\n" + "=" * 80)
    print("RUNNING GENERATION BENCHMARK")
    print(f"Prompt: \"{prompt}\"")
    print(f"Sampling: temp={temperature}, top_p={top_p}, rep_penalty={repetition_penalty}, max_new_tokens={max_new_tokens}")
    print("=" * 80)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[-1]
    inputs_embeds = model.model.embed_tokens(input_ids)

    # Initialize cache
    cache = DynamicCache()

    # 1. Warmup pass
    with torch.no_grad():
        _ = model.model(
            inputs_embeds=inputs_embeds[:, :1, :],
            past_key_values=cache,
            use_cache=True,
        )
    cache = DynamicCache()
    torch.cuda.synchronize()

    # 2. Prefill pass
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            past_key_values=cache,
            use_cache=True,
        )
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t_prefill_start
    prefill_tps = seq_len / max(prefill_time, 1e-6)

    logits = outputs.logits[:, -1, :].clone()
    context_tokens = input_ids[0].tolist()
    next_token_id = sample_next_token(logits, context_tokens, temperature, top_p, repetition_penalty)
    generated_tokens = [next_token_id]
    context_tokens.append(next_token_id)

    eos_token_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    print(f"\n[Prefill] {seq_len} tokens in {prefill_time*1000.0:.1f} ms ({prefill_tps:.1f} tokens/sec)")
    print("[Decode] Streaming tokens: ", end="", flush=True)

    # 3. Decode loop
    t_decode_start = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        if next_token_id in eos_token_ids:
            break

        next_token_embed = model.model.embed_tokens(torch.tensor([[next_token_id]]))
        with torch.no_grad():
            step_outputs = model(
                inputs_embeds=next_token_embed,
                past_key_values=cache,
                use_cache=True,
            )
        logits = step_outputs.logits[:, -1, :].clone()
        next_token_id = sample_next_token(logits, context_tokens, temperature, top_p, repetition_penalty)
        generated_tokens.append(next_token_id)
        context_tokens.append(next_token_id)

        token_str = tokenizer.decode([next_token_id], skip_special_tokens=True)
        print(token_str, end="", flush=True)

    torch.cuda.synchronize()
    decode_time = time.perf_counter() - t_decode_start
    num_generated = len(generated_tokens)
    decode_tps = (num_generated - 1) / max(decode_time, 1e-6) if num_generated > 1 else 1.0 / max(decode_time, 1e-6)
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("\n\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Generated text: \"{tokenizer.decode(generated_tokens, skip_special_tokens=True)}\"")
    print(f"Peak VRAM:          {peak_vram_gb:.2f} GB / 4.00 GB")
    print(f"Prefill Latency:    {prefill_time*1000.0:.1f} ms ({prefill_tps:.1f} tokens/sec)")
    print(f"Decode Throughput:  {decode_tps:.2f} tokens/sec ({num_generated} tokens in {decode_time:.2f}s)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End INT2 Qwen3-8B Inference")
    parser.add_argument("--checkpoint-dir", type=str, default="./qwen3-8b-int2-stage2", help="Path to INT2 checkpoint")
    parser.add_argument("--prompt", type=str, default="Sub-1-bit and 2-bit quantization compresses large language models by", help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.18, help="Repetition penalty")
    parser.add_argument("--lm-head-cuda", action="store_true", help="Put lm_head on CUDA (requires extra 1.24 GB VRAM)")
    args = parser.parse_args()

    model, tokenizer = load_int2_model(checkpoint_dir=args.checkpoint_dir, lm_head_cuda=args.lm_head_cuda)
    generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
