"""
download_weights.py - Pre-download checkpoint files to local Hugging Face cache.

Ensures required model files exist locally so that local_files_only=True
in inference runners (server_littlebit.py, run_qwen3_littlebit.py, run_qwen3_cudagraph.py)
succeeds without runtime download failures.

Usage:
  python download_weights.py                  # Downloads LittleBit 0.55 bpw weights (~2.1 GB)
  python download_weights.py --qwen3-base     # Also downloads Qwen/Qwen3-8B unquantized weights (~16 GB)
"""

import sys
import argparse
from huggingface_hub import hf_hub_download

LITTLEBIT_REPO = "LittleBitLLM/littlebit-qwen3-8b-0.55bpw"
LITTLEBIT_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
]

QWEN3_BASE_REPO = "Qwen/Qwen3-8B"


def download_littlebit(repo_id: str = LITTLEBIT_REPO):
    print("=" * 70)
    print(f"Downloading LittleBit Qwen3-8B (0.55 bpw) from {repo_id}")
    print("=" * 70)
    for filename in LITTLEBIT_FILES:
        print(f"Fetching {filename}...")
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"  -> Cached at: {path}")
    print("\n[OK] 0.55 bpw checkpoint cached successfully.")


def download_qwen3_base(repo_id: str = QWEN3_BASE_REPO):
    print("\n" + "=" * 70)
    print(f"Downloading Qwen3-8B base model from {repo_id} (~16 GB)")
    print("=" * 70)
    from transformers import AutoConfig, AutoTokenizer
    print("Fetching config and tokenizer...")
    AutoConfig.from_pretrained(repo_id)
    AutoTokenizer.from_pretrained(repo_id)

    # Download index and shards
    index_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")
    print(f"  -> Index cached at: {index_path}")
    import json
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    shards = sorted(list(set(data["weight_map"].values())))
    for s in shards:
        print(f"Fetching shard {s}...")
        p = hf_hub_download(repo_id=repo_id, filename=s)
        print(f"  -> Cached at: {p}")
    print("\n[OK] Qwen3-8B base model cached successfully.")


def main():
    parser = argparse.ArgumentParser(description="Download model weights for LittleBit and INT2.")
    parser.add_argument("--qwen3-base", action="store_true", help="Download unquantized Qwen/Qwen3-8B (needed for quantize_qwen3_int2_gptq.py)")
    args = parser.parse_args()

    download_littlebit()

    if args.qwen3-base:
        download_qwen3_base()


if __name__ == "__main__":
    main()
