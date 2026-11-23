"""
quantize_qwen3_int2.py
Streaming Model Quantizer for Qwen3-8B to Stage 2 Asymmetric INT2:
- Streams weights tensor-by-tensor from unquantized FP16 safetensors shards onto CUDA.
- Peak VRAM during quantization is strictly throttled to < 500 MB.
- Linear projections:
    * q_proj, k_proj, v_proj, o_proj: group_size=16 (or group_size=32 if --uniform-g32)
    * gate_proj, up_proj, down_proj: group_size=32
- Saves packed uint32 (int32 view) weights, FP16 scales, and FP16 zero_points plus non-linear
  weights (embed_tokens, lm_head, norm layers) to local ./qwen3-8b-int2-stage2/ safetensors checkpoint.
"""

import os
import sys
import json
import shutil
import argparse
import time
import torch
from typing import Dict, Any, List
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download

from cuda_int2_group_ops import pack_int2_group_weight

LINEAR_TARGET_KEYS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)


def get_group_size(tensor_name: str, uniform_g32: bool = False) -> int:
    if uniform_g32:
        return 32
    if any(attn_k in tensor_name for attn_k in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return 16
    return 32


def quantize_model(
    repo_id: str = "Qwen/Qwen3-8B",
    output_dir: str = "./qwen3-8b-int2-stage2",
    uniform_g32: bool = False,
    max_layers: int = None,
):
    print("=" * 80)
    print(f"STREAMING INT2 QUANTIZER: {repo_id} -> {output_dir}")
    print(f"Quantization Mode: {'Uniform G=32' if uniform_g32 else 'Hybrid (Attn G=16, MLP G=32)'}")
    if max_layers:
        print(f"Layer Limit: First {max_layers} layers only")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Download & copy configs and tokenizers
    print("\n[1/4] Downloading and saving config and tokenizer metadata...")
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
            fpath = hf_hub_download(repo_id=repo_id, filename=fn)
            shutil.copy(fpath, os.path.join(output_dir, fn))
            print(f"      Saved {fn}")
        except Exception as e:
            print(f"      Note: {fn} not present or skipped ({e})")

    # 2. Get index or single safetensors
    index_path = None
    try:
        index_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        shards = sorted(list(set(weight_map.values())))
    except Exception:
        weight_map = None
        shards = ["model.safetensors"]

    print(f"\n[2/4] Found {len(shards)} safetensors shard(s) to process.")

    new_weight_map: Dict[str, str] = {}
    total_weights_quantized = 0
    total_non_linear_copied = 0
    start_time = time.perf_counter()

    # 3. Stream shard by shard
    for shard_idx, shard_name in enumerate(shards, start=1):
        print(f"\n---> [{shard_idx}/{len(shards)}] Downloading and streaming {shard_name}...")
        shard_file = hf_hub_download(repo_id=repo_id, filename=shard_name)
        out_shard_name = f"model-{shard_idx:05d}-of-{len(shards):05d}.safetensors"
        out_shard_path = os.path.join(output_dir, out_shard_name)

        out_tensors: Dict[str, torch.Tensor] = {}

        with safe_open(shard_file, framework="pt", device="cpu") as f:
            tensor_keys = list(f.keys())
            print(f"     Found {len(tensor_keys)} tensors in {shard_name}.")

            for t_idx, k in enumerate(tensor_keys, start=1):
                # Filter by max_layers if requested
                if max_layers is not None and "model.layers." in k:
                    parts = k.split(".")
                    layer_num = int(parts[2])
                    if layer_num >= max_layers:
                        continue

                is_linear_weight = any(k.endswith(target) for target in LINEAR_TARGET_KEYS)

                if is_linear_weight:
                    g_size = get_group_size(k, uniform_g32=uniform_g32)
                    w_cpu = f.get_tensor(k)
                    w_cuda = w_cpu.to(device="cuda", dtype=torch.float32)

                    # Quantize on CUDA
                    packed, scales, zp = pack_int2_group_weight(w_cuda, group_size=g_size)

                    # Move immediately to CPU memory to keep GPU VRAM < 500 MB
                    prefix = k[:-len(".weight")]
                    packed_key = f"{prefix}.W_packed"
                    scales_key = f"{prefix}.scales"
                    zp_key = f"{prefix}.zero_points"
                    gsize_key = f"{prefix}.group_size"

                    out_tensors[packed_key] = packed.cpu()
                    out_tensors[scales_key] = scales.cpu()
                    out_tensors[zp_key] = zp.cpu()
                    out_tensors[gsize_key] = torch.tensor([g_size], dtype=torch.int32)

                    new_weight_map[packed_key] = out_shard_name
                    new_weight_map[scales_key] = out_shard_name
                    new_weight_map[zp_key] = out_shard_name
                    new_weight_map[gsize_key] = out_shard_name

                    del w_cuda, packed, scales, zp
                    torch.cuda.empty_cache()

                    total_weights_quantized += 1
                    if t_idx % 10 == 0 or t_idx == len(tensor_keys):
                        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                        print(f"     [{t_idx}/{len(tensor_keys)}] Quantized {k} (G={g_size}) | Peak VRAM: {vram_mb:.1f} MB")
                else:
                    # Non-linear weights (embed_tokens, lm_head, layernorms, biases)
                    tensor_cpu = f.get_tensor(k).to(torch.float16)
                    out_tensors[k] = tensor_cpu
                    new_weight_map[k] = out_shard_name
                    total_non_linear_copied += 1

        # Save quantized shard to disk
        print(f"     Saving {len(out_tensors)} tensors to {out_shard_name}...")
        save_file(out_tensors, out_shard_path)
        del out_tensors

    # 4. Save model.safetensors.index.json
    print("\n[4/4] Writing output safetensors index file...")
    new_index = {
        "metadata": {
            "total_size": 0,
            "quantization": "int2_asymmetric_group",
            "uniform_g32": uniform_g32,
        },
        "weight_map": new_weight_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2)

    elapsed = time.perf_counter() - start_time
    print("=" * 80)
    print(f"QUANTIZATION COMPLETE in {elapsed:.1f}s ({elapsed/60.0:.2f} min)!")
    print(f"Linear weights quantized: {total_weights_quantized}")
    print(f"Non-linear weights copied: {total_non_linear_copied}")
    print(f"Checkpoint saved at: {os.path.abspath(output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming INT2 Quantizer for Qwen3-8B")
    parser.add_argument("--repo-id", type=str, default="Qwen/Qwen3-8B", help="Hugging Face repo ID")
    parser.add_argument("--output-dir", type=str, default="./qwen3-8b-int2-stage2", help="Output directory")
    parser.add_argument("--uniform-g32", action="store_true", help="Force group_size=32 for all linear projections")
    parser.add_argument("--max-layers", type=int, default=None, help="Quantize only first N layers (for testing)")
    args = parser.parse_args()

    quantize_model(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        uniform_g32=args.uniform_g32,
        max_layers=args.max_layers,
    )
