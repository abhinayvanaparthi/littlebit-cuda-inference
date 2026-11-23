"""
server_littlebit.py - FastAPI Web Server with CUDA Graph Inference for LittleBit Qwen3-8B

Features:
  1. Preloads LittleBit Qwen3-8B, StaticCache(max_cache_len=512), and captures
     torch.cuda.CUDAGraph() once on startup (skipping layer-0 accuracy check).
  2. Guarded by asyncio.Lock() to prevent concurrent collisions on static CUDA buffers.
  3. Zeroes out all 36 layers' StaticCache before each request to prevent context leakage.
  4. SSE StreamingResponse (text/event-stream) yielding live tokens and final metrics (TTFT, TPS, VRAM).
  5. Dark-mode single-page interface at GET / with live metrics, parameter sliders,
     preset prompts, and generation abort support.
  6. Starts on 127.0.0.1:8000 via Uvicorn.
"""

from __future__ import annotations
import os
import sys
import time
import gc
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import StaticCache
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Ensure local modules (cuda_ops) can be imported
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from cuda_ops import littlebit_forward_cuda

REPO_ID = "LittleBitLLM/littlebit-qwen3-8b-0.55bpw"
MAX_CACHE_LEN = 2048
CUDA_DEVICE = "cuda:0"


# ---------------------------------------------------------------------------
# 1. Model Components (CPUEmbeddingWrapper & PackedLittleBitLinear)
# ---------------------------------------------------------------------------

class CPUEmbeddingWrapper(nn.Module):
    def __init__(self, weight: torch.Tensor, device: str = CUDA_DEVICE, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.weight = nn.Parameter(weight.to("cpu", dtype=dtype), requires_grad=False)
        self.target_device = device
        self.target_dtype = dtype

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids_cpu = input_ids.to("cpu")
        embeds = F.embedding(ids_cpu, self.weight)
        return embeds.to(device=self.target_device, dtype=self.target_dtype)


class PackedLittleBitLinear(nn.Module):
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


def build_packed_linear(
    f: safe_open,
    prefix: str,
    in_features: int,
    out_features: int,
    device: str = "cpu",
) -> PackedLittleBitLinear:
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

    layer.register_buffer("V_packed", f.get_tensor(f"{prefix}.V_packed").to(device=device))
    layer.register_buffer("U_packed", f.get_tensor(f"{prefix}.U_packed").to(device=device))
    layer.register_buffer("v2", f.get_tensor(f"{prefix}.v2").to(device=device, dtype=torch.float16))
    layer.register_buffer("v1", f.get_tensor(f"{prefix}.v1").to(device=device, dtype=torch.float16))
    layer.register_buffer("u2", f.get_tensor(f"{prefix}.u2").to(device=device, dtype=torch.float16))
    layer.register_buffer("u1", f.get_tensor(f"{prefix}.u1").to(device=device, dtype=torch.float16))

    if has_residual:
        layer.register_buffer("V_R_packed", f.get_tensor(f"{prefix}.V_R_packed").to(device=device))
        layer.register_buffer("U_R_packed", f.get_tensor(f"{prefix}.U_R_packed").to(device=device))
        layer.register_buffer("v2_R", f.get_tensor(f"{prefix}.v2_R").to(device=device, dtype=torch.float16))
        layer.register_buffer("v1_R", f.get_tensor(f"{prefix}.v1_R").to(device=device, dtype=torch.float16))
        layer.register_buffer("u2_R", f.get_tensor(f"{prefix}.u2_R").to(device=device, dtype=torch.float16))
        layer.register_buffer("u1_R", f.get_tensor(f"{prefix}.u1_R").to(device=device, dtype=torch.float16))

    if has_bias:
        layer.register_buffer("bias", f.get_tensor(f"{prefix}.bias").to(device=device, dtype=torch.float16))

    return layer


def load_littlebit_qwen3(repo_id: str = REPO_ID, cuda_device: str = CUDA_DEVICE):
    config = AutoConfig.from_pretrained(repo_id, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, local_files_only=True)
    safetensors_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", local_files_only=True)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        rot_cls = type(model.model.rotary_emb)
        model.model.rotary_emb = rot_cls(config=config).to(cuda_device)

    num_layers = len(model.model.layers)
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        embed_weight = f.get_tensor("model.embed_tokens.weight").to(torch.float16)
        model.model.embed_tokens = CPUEmbeddingWrapper(embed_weight, device=cuda_device, dtype=torch.float16)

        for i in range(num_layers):
            layer = model.model.layers[i]
            prefix = f"model.layers.{i}"
            layer.self_attn.q_proj = build_packed_linear(f, f"{prefix}.self_attn.q_proj", config.hidden_size, config.hidden_size)
            layer.self_attn.k_proj = build_packed_linear(f, f"{prefix}.self_attn.k_proj", config.hidden_size, (config.hidden_size // config.num_attention_heads) * config.num_key_value_heads)
            layer.self_attn.v_proj = build_packed_linear(f, f"{prefix}.self_attn.v_proj", config.hidden_size, (config.hidden_size // config.num_attention_heads) * config.num_key_value_heads)
            layer.self_attn.o_proj = build_packed_linear(f, f"{prefix}.self_attn.o_proj", config.hidden_size, config.hidden_size)
            layer.self_attn.q_norm.weight = nn.Parameter(f.get_tensor(f"{prefix}.self_attn.q_norm.weight").to(torch.float16))
            layer.self_attn.k_norm.weight = nn.Parameter(f.get_tensor(f"{prefix}.self_attn.k_norm.weight").to(torch.float16))
            layer.mlp.gate_proj = build_packed_linear(f, f"{prefix}.mlp.gate_proj", config.hidden_size, config.intermediate_size)
            layer.mlp.up_proj = build_packed_linear(f, f"{prefix}.mlp.up_proj", config.hidden_size, config.intermediate_size)
            layer.mlp.down_proj = build_packed_linear(f, f"{prefix}.mlp.down_proj", config.intermediate_size, config.hidden_size)
            layer.input_layernorm.weight = nn.Parameter(f.get_tensor(f"{prefix}.input_layernorm.weight").to(torch.float16))
            layer.post_attention_layernorm.weight = nn.Parameter(f.get_tensor(f"{prefix}.post_attention_layernorm.weight").to(torch.float16))
            layer.to(cuda_device)

        model.model.norm.weight = nn.Parameter(f.get_tensor("model.norm.weight").to(torch.float16))
        model.model.norm.to(cuda_device)
        model.lm_head.weight = nn.Parameter(f.get_tensor("lm_head.weight").to(torch.float16).to(cuda_device))

    # Assert zero parameters or buffers remain on meta device
    meta_params = [name for name, p in model.named_parameters() if p.device.type == "meta"]
    meta_buffers = [name for name, b in model.named_buffers() if b.device.type == "meta"]
    for i in range(num_layers):
        assert model.model.layers[i].self_attn.q_norm.weight.device.type != "meta"
        assert model.model.layers[i].self_attn.k_norm.weight.device.type != "meta"
    assert len(meta_params) == 0 and len(meta_buffers) == 0

    torch.cuda.empty_cache()
    gc.collect()
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# 2. Global Inference Engine State
# ---------------------------------------------------------------------------

class InferenceEngine:
    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.cache: Optional[StaticCache] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_embeds: Optional[torch.Tensor] = None
        self.static_cache_pos: Optional[torch.Tensor] = None
        self.static_pos_ids: Optional[torch.Tensor] = None
        self.static_logits: Optional[torch.Tensor] = None
        self.eos_ids: Set[int] = set()
        self.lock = asyncio.Lock()

    def load(self):
        print("[InferenceEngine] Loading LittleBit Qwen3-8B into 4 GB VRAM budget...")
        self.model, self.tokenizer = load_littlebit_qwen3(REPO_ID, cuda_device=CUDA_DEVICE)

        # Collect EOS IDs
        self.eos_ids = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        for token_str in ["<|im_end|>", "<|endoftext|>"]:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(token_str)
                if isinstance(tid, int) and tid >= 0:
                    self.eos_ids.add(tid)
            except Exception:
                pass

        # Fixed StaticCache
        print(f"[InferenceEngine] Initializing StaticCache(max_cache_len={MAX_CACHE_LEN})...")
        self.cache = StaticCache(config=self.model.config, max_cache_len=MAX_CACHE_LEN)

        # Pre-allocate static decode tensors
        hidden_size = self.model.config.hidden_size
        self.static_embeds = torch.empty((1, 1, hidden_size), device=CUDA_DEVICE, dtype=torch.float16)
        self.static_cache_pos = torch.empty((1,), device=CUDA_DEVICE, dtype=torch.long)
        self.static_pos_ids = torch.empty((1, 1), device=CUDA_DEVICE, dtype=torch.long)

        # Warmup and capture CUDA Graph
        print("[InferenceEngine] Warming up and capturing decode CUDA Graph...")
        self.static_cache_pos.fill_(16)
        self.static_pos_ids.fill_(16)
        self.static_embeds.copy_(self.model.model.embed_tokens(torch.tensor([[1]])))

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                _ = self.model(
                    inputs_embeds=self.static_embeds,
                    past_key_values=self.cache,
                    cache_position=self.static_cache_pos,
                    position_ids=self.static_pos_ids,
                    use_cache=True,
                )
        torch.cuda.current_stream().wait_stream(side_stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=side_stream):
            static_outputs = self.model(
                inputs_embeds=self.static_embeds,
                past_key_values=self.cache,
                cache_position=self.static_cache_pos,
                position_ids=self.static_pos_ids,
                use_cache=True,
            )
        self.static_logits = static_outputs.logits

        # Reset cache buffers to 0
        self.reset_cache()
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"[InferenceEngine] Ready! VRAM allocated: {allocated_mb:.1f} MB / 4.00 GB")

    def reset_cache(self):
        if self.cache is None:
            return

        if hasattr(self.cache, "layers"):
            has_explicit_cache = False
            for layer in self.cache.layers:
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if keys is not None and values is not None:
                    has_explicit_cache = True
                    keys.zero_()
                    values.zero_()
                elif getattr(layer, "key_cache", None) is not None and getattr(layer, "value_cache", None) is not None:
                    has_explicit_cache = True
                    layer.key_cache.zero_()
                    layer.value_cache.zero_()

                if hasattr(layer, "cumulative_length"):
                    if isinstance(layer.cumulative_length, int):
                        layer.cumulative_length = 0
                    elif hasattr(layer.cumulative_length, "zero_"):
                        layer.cumulative_length.zero_()

            if not has_explicit_cache and hasattr(self.cache, "reset"):
                self.cache.reset()
        elif hasattr(self.cache, "key_cache") and hasattr(self.cache, "value_cache"):
            for k in self.cache.key_cache:
                if k is not None:
                    k.zero_()
            for v in self.cache.value_cache:
                if v is not None:
                    v.zero_()
        elif hasattr(self.cache, "reset"):
            self.cache.reset()


engine = InferenceEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not torch.cuda.is_available():
        print("ERROR: CUDA device not available.")
        sys.exit(1)
    engine.load()
    yield


app = FastAPI(title="LittleBit Qwen3-8B Inference Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# 3. Sampling Logic
# ---------------------------------------------------------------------------

def apply_repetition_penalty(logits: torch.Tensor, token_ids: List[int], penalty: float = 1.18) -> torch.Tensor:
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
    logits = apply_repetition_penalty(logits, context_tokens, penalty=repetition_penalty)
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1).item()

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

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

    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    return torch.gather(sorted_indices, -1, sampled_sorted_idx).item()


# ---------------------------------------------------------------------------
# 4. API Endpoints
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = "Sub-1-bit quantization compresses large language models by"
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.18
    max_new_tokens: int = 256


@app.post("/generate")
async def generate_stream(req_body: GenerateRequest, request: Request):
    async def sse_generator():
        # Ensure only one request uses static CUDA Graph & StaticCache at a time
        async with engine.lock:
            # 1. Zero out StaticCache buffers before request
            engine.reset_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # 2. Tokenize prompt & clamp to max cache length (2048)
            input_ids_cpu = engine.tokenizer.encode(req_body.prompt, return_tensors="pt")
            seq_len = input_ids_cpu.shape[-1]
            if seq_len >= MAX_CACHE_LEN - 16:
                input_ids_cpu = input_ids_cpu[:, -(MAX_CACHE_LEN - 16):]
                seq_len = input_ids_cpu.shape[-1]

            max_tokens = min(req_body.max_new_tokens, MAX_CACHE_LEN - seq_len)
            if max_tokens <= 0:
                max_tokens = 1

            # 3. Prefill pass
            t_prefill_start = time.perf_counter()
            inputs_embeds = engine.model.model.embed_tokens(input_ids_cpu)
            cache_pos = torch.arange(seq_len, device=CUDA_DEVICE)
            pos_ids = cache_pos.unsqueeze(0)

            outputs = engine.model(
                inputs_embeds=inputs_embeds,
                past_key_values=engine.cache,
                cache_position=cache_pos,
                position_ids=pos_ids,
                use_cache=True,
            )
            torch.cuda.synchronize()
            ttft_ms = (time.perf_counter() - t_prefill_start) * 1000.0

            # Sample first token from prefill
            prefill_logits = outputs.logits[:, -1, :].clone()
            context_tokens = input_ids_cpu[0].tolist()
            next_token_id = sample_next_token(
                prefill_logits,
                context_tokens,
                temperature=req_body.temperature,
                top_p=req_body.top_p,
                repetition_penalty=req_body.repetition_penalty,
            )
            generated_count = 1
            context_tokens.append(next_token_id)

            first_text = engine.tokenizer.decode([next_token_id], skip_special_tokens=True)
            yield f"data: {json.dumps({'token': first_text, 'done': False})}\n\n"

            # 4. CUDA Graph decode loop
            t_gen_start = time.perf_counter()
            for step in range(max_tokens - 1):
                if await request.is_disconnected():
                    break
                if next_token_id in engine.eos_ids:
                    break

                cur_pos = seq_len + step
                if cur_pos >= MAX_CACHE_LEN:
                    break

                # Update static buffers in-place
                token_embed = engine.model.model.embed_tokens(torch.tensor([[next_token_id]]))
                engine.static_embeds.copy_(token_embed)

                cur_pos_t = torch.tensor([cur_pos], device=CUDA_DEVICE, dtype=torch.long)
                engine.static_cache_pos.copy_(cur_pos_t)
                engine.static_pos_ids.copy_(cur_pos_t.unsqueeze(0))

                # Replay CUDA Graph
                engine.graph.replay()

                # Sample from static logits
                step_logits = engine.static_logits[:, -1, :].clone()
                next_token_id = sample_next_token(
                    step_logits,
                    context_tokens,
                    temperature=req_body.temperature,
                    top_p=req_body.top_p,
                    repetition_penalty=req_body.repetition_penalty,
                )
                generated_count += 1
                context_tokens.append(next_token_id)

                token_text = engine.tokenizer.decode([next_token_id], skip_special_tokens=True)
                yield f"data: {json.dumps({'token': token_text, 'done': False})}\n\n"
                await asyncio.sleep(0)  # Flush SSE chunk to client

                if next_token_id in engine.eos_ids:
                    break

            torch.cuda.synchronize()
            gen_time = time.perf_counter() - t_gen_start
            tps = (generated_count - 1) / gen_time if gen_time > 0 and generated_count > 1 else generated_count / max(gen_time, 1e-6)
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

            yield f"data: {json.dumps({'done': True, 'ttft_ms': round(ttft_ms, 1), 'tps': round(tps, 1), 'vram_mb': round(peak_vram_mb, 1), 'tokens': generated_count})}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@app.post("/reset")
async def reset_cache_endpoint():
    async with engine.lock:
        engine.reset_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    return {"status": "ok", "message": "StaticCache zeroed successfully"}


# ---------------------------------------------------------------------------
# 5. Dark-Mode Web Client Interface (GET /)
# ---------------------------------------------------------------------------

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LittleBit Qwen3-8B | 4GB VRAM Inference Server</title>
  <style>
    :root {
      --bg: #0c0e14;
      --card-bg: #141824;
      --panel-bg: #1c2234;
      --border: #28314a;
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --accent-glow: rgba(99, 102, 241, 0.25);
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --success: #10b981;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 24px;
    }
    .container { width: 100%; max-width: 1050px; display: flex; flex-direction: column; gap: 20px; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 24px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .badge {
      background: var(--panel-bg);
      padding: 6px 12px;
      border-radius: 20px;
      font-size: 13px;
      color: var(--success);
      border: 1px solid rgba(16, 185, 129, 0.25);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--success); box-shadow: 0 0 8px var(--success); }
    .metrics-bar {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
    }
    .metric-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px 20px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .metric-title { font-size: 13px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 24px; font-weight: 700; color: #fff; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .main-grid {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 20px;
    }
    @media (max-width: 860px) { .main-grid { grid-template-columns: 1fr; } }
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    textarea {
      width: 100%;
      height: 110px;
      background: var(--panel-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
      color: var(--text);
      font-size: 14px;
      line-height: 1.5;
      resize: vertical;
      outline: none;
    }
    textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-glow); }
    .presets { display: flex; flex-wrap: wrap; gap: 8px; }
    .preset-btn {
      background: var(--panel-bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
      font-size: 12px;
      padding: 6px 10px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .preset-btn:hover { color: #fff; border-color: var(--accent); }
    .btn-row { display: flex; gap: 12px; }
    button.action {
      flex: 1;
      padding: 12px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 14px;
      border: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.2s;
    }
    .btn-gen { background: var(--accent); color: #fff; }
    .btn-gen:hover { background: var(--accent-hover); box-shadow: 0 0 12px var(--accent-glow); }
    .btn-stop { background: var(--panel-bg); color: var(--danger); border: 1px solid var(--border); }
    .btn-stop:hover { background: rgba(239, 68, 68, 0.15); border-color: var(--danger); }
    .btn-clear { background: var(--panel-bg); color: var(--text-muted); border: 1px solid var(--border); }
    .btn-clear:hover { background: rgba(255, 255, 255, 0.08); color: var(--text); border-color: var(--text-muted); }
    .output-box {
      min-height: 220px;
      max-height: 480px;
      overflow-y: auto;
      background: var(--panel-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      font-size: 14px;
      line-height: 1.6;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      color: #e2e8f0;
      position: relative;
    }
    .cursor { display: inline-block; width: 8px; height: 16px; background: var(--accent); margin-left: 2px; vertical-align: middle; animation: blink 0.8s infinite; }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
    .slider-group { display: flex; flex-direction: column; gap: 6px; }
    .slider-header { display: flex; justify-content: space-between; font-size: 13px; color: var(--text-muted); }
    .slider-header span.val { color: var(--text); font-weight: 600; font-family: monospace; }
    input[type=range] { width: 100%; accent-color: var(--accent); }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h2>LittleBit Qwen3-8B (0.55 bpw)</h2>
        <p style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">NVIDIA RTX 3050 Ti (4GB VRAM) &bull; CUDA Graph &bull; StaticCache(2048)</p>
      </div>
      <div class="badge"><div class="pulse"></div> 4.0 GB VRAM Active</div>
    </header>

    <div class="metrics-bar">
      <div class="metric-card">
        <div class="metric-title">Time to First Token</div>
        <div class="metric-value" id="val-ttft">-- ms</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">Decode Throughput</div>
        <div class="metric-value" id="val-tps" style="color: var(--success);">-- tok/s</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">Peak VRAM Allocated</div>
        <div class="metric-value" id="val-vram">-- MB</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">Tokens Generated</div>
        <div class="metric-value" id="val-tokens">0</div>
      </div>
    </div>

    <div class="main-grid">
      <div class="panel">
        <div>
          <label style="font-size: 13px; color: var(--text-muted); font-weight: 600; display: block; margin-bottom: 8px;">Direct Completion Prompt:</label>
          <textarea id="prompt-input">Sub-1-bit quantization compresses large language models by</textarea>
        </div>
        <div class="presets">
          <button class="preset-btn" onclick="setPreset('Sub-1-bit quantization compresses large language models by')">Sub-1-bit Compression</button>
          <button class="preset-btn" onclick="setPreset('The key advantage of register-level bit unpacking over VRAM materialization is')">Register Unpacking</button>
          <button class="preset-btn" onclick="setPreset('To run an 8B parameter model in under 4 GB of VRAM, LittleBit uses')">4GB VRAM Architecture</button>
          <button class="preset-btn" onclick="setPreset('Low-bit LLMs achieve high compression ratios while retaining reasoning by')">Low-Bit Reasoning</button>
        </div>
        <div class="btn-row">
          <button class="action btn-gen" id="btn-generate" onclick="startGeneration()">&#9654; Generate (CUDA Graph)</button>
          <button class="action btn-stop" id="btn-stop" onclick="stopGeneration()" disabled>&#9632; Stop</button>
          <button class="action btn-clear" id="btn-clear" onclick="clearAll()">&#8634; Reset Cache / Clear</button>
        </div>
        <div>
          <label style="font-size: 13px; color: var(--text-muted); font-weight: 600; display: block; margin-bottom: 8px;">Live Streamed Output:</label>
          <div class="output-box" id="output-box"><span id="output-text"></span><span class="cursor" id="cursor" style="display: none;"></span></div>
        </div>
      </div>

      <div class="panel">
        <h3 style="font-size: 15px; border-bottom: 1px solid var(--border); padding-bottom: 8px;">Sampling Controls</h3>
        <div class="slider-group">
          <div class="slider-header"><span>Temperature</span><span class="val" id="disp-temp">0.70</span></div>
          <input type="range" id="param-temp" min="0.0" max="1.5" step="0.05" value="0.70" oninput="updateVal('temp')">
        </div>
        <div class="slider-group">
          <div class="slider-header"><span>Top-P (Nucleus)</span><span class="val" id="disp-top-p">0.90</span></div>
          <input type="range" id="param-top-p" min="0.1" max="1.0" step="0.05" value="0.90" oninput="updateVal('top-p')">
        </div>
        <div class="slider-group">
          <div class="slider-header"><span>Repetition Penalty</span><span class="val" id="disp-rep">1.18</span></div>
          <input type="range" id="param-rep" min="1.0" max="2.0" step="0.02" value="1.18" oninput="updateVal('rep')">
        </div>
        <div class="slider-group">
          <div class="slider-header"><span>Max New Tokens</span><span class="val" id="disp-max">256</span></div>
          <input type="range" id="param-max" min="32" max="1024" step="32" value="256" oninput="updateVal('max')">
        </div>
        <div style="font-size: 12px; color: var(--text-muted); line-height: 1.5; margin-top: 10px; background: var(--panel-bg); padding: 12px; border-radius: 8px;">
          <strong>Tip:</strong> Nucleus sampling (top_p=0.9, temp=0.7) prevents low-bit 0.55 bpw quantized models from locking into repetitive cycles.
        </div>
      </div>
    </div>
  </div>

  <script>
    let abortCtrl = null;

    function setPreset(txt) {
      document.getElementById('prompt-input').value = txt;
    }

    function updateVal(name) {
      const val = document.getElementById('param-' + name).value;
      document.getElementById('disp-' + name).innerText = (name === 'max') ? val : parseFloat(val).toFixed(2);
    }

    async function startGeneration() {
      const prompt = document.getElementById('prompt-input').value;
      const temp = parseFloat(document.getElementById('param-temp').value);
      const topP = parseFloat(document.getElementById('param-top-p').value);
      const rep = parseFloat(document.getElementById('param-rep').value);
      const maxTokens = parseInt(document.getElementById('param-max').value);

      const outText = document.getElementById('output-text');
      const cursor = document.getElementById('cursor');
      const btnGen = document.getElementById('btn-generate');
      const btnStop = document.getElementById('btn-stop');

      outText.innerText = prompt;
      cursor.style.display = 'inline-block';
      btnGen.disabled = true;
      btnStop.disabled = false;

      document.getElementById('val-ttft').innerText = '-- ms';
      document.getElementById('val-tps').innerText = '-- tok/s';
      document.getElementById('val-tokens').innerText = '0';

      abortCtrl = new AbortController();

      try {
        const resp = await fetch('/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: prompt,
            temperature: temp,
            top_p: topP,
            repetition_penalty: rep,
            max_new_tokens: maxTokens
          }),
          signal: abortCtrl.signal
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split('\\n');
          buffer = lines.pop();

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.slice(6));
              if (data.token) {
                outText.innerText += data.token;
                document.getElementById('output-box').scrollTop = document.getElementById('output-box').scrollHeight;
              }
              if (data.done) {
                document.getElementById('val-ttft').innerText = data.ttft_ms + ' ms';
                document.getElementById('val-tps').innerText = data.tps + ' tok/s';
                document.getElementById('val-vram').innerText = data.vram_mb + ' MB';
                document.getElementById('val-tokens').innerText = data.tokens;
              }
            }
          }
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          outText.innerText += '\\n[Error: ' + err.message + ']';
        }
      } finally {
        cursor.style.display = 'none';
        btnGen.disabled = false;
        btnStop.disabled = true;
        abortCtrl = null;
      }
    }

    function stopGeneration() {
      if (abortCtrl) {
        abortCtrl.abort();
      }
    }

    async function clearAll() {
      stopGeneration();
      document.getElementById('output-text').innerText = '';
      document.getElementById('val-ttft').innerText = '-- ms';
      document.getElementById('val-tps').innerText = '-- tok/s';
      document.getElementById('val-vram').innerText = '-- MB';
      document.getElementById('val-tokens').innerText = '0';
      try {
        await fetch('/reset', { method: 'POST' });
      } catch (err) {
        console.warn('Failed to reset server cache:', err);
      }
    }
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(content=HTML_CONTENT)


# ---------------------------------------------------------------------------
# 6. Main Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server_littlebit:app", host="127.0.0.1", port=8000, reload=False)
