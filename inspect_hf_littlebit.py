"""
inspect_hf_littlebit.py - Inspects LittleBitLLM/littlebit-qwen3-8b-0.55bpw
Downloads/uses cached config and safetensors file without loading model onto GPU.
Prints exact parameter names, shapes, and dtypes for layer 0, embed_tokens, and lm_head.
"""

import os
import json
from huggingface_hub import hf_hub_download
from safetensors import safe_open

REPO_ID = "LittleBitLLM/littlebit-qwen3-8b-0.55bpw"

def main():
    print(f"=== LittleBit Model Inspection: {REPO_ID} ===")
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"Architecture:       {config.get('architectures', ['Unknown'])[0]}")
    print(f"Model Type:         {config.get('model_type')}")
    print(f"Hidden Size:        {config.get('hidden_size')}")
    print(f"Num Layers:         {config.get('num_hidden_layers')}")
    print(f"Num Attention Heads:{config.get('num_attention_heads')}")
    print(f"Num KV Heads:       {config.get('num_key_value_heads')}")
    print(f"Intermediate Size:  {config.get('intermediate_size')}")
    print(f"Vocab Size:         {config.get('vocab_size')}")

    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors")
    print(f"\nSafetensors file:   {model_path} ({os.path.getsize(model_path) / (1024**3):.2f} GB)")

    print("\n" + "="*85)
    print(f"{'Parameter Name':<55} | {'Shape':<18} | {'Dtype'}")
    print("="*85)

    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = f.keys()

        # 1. Global parameters
        for target in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]:
            if target in keys:
                slice_ = f.get_slice(target)
                print(f"{target:<55} | {str(slice_.get_shape()):<18} | {slice_.get_dtype()}")

        print("-" * 85)
        # 2. Layer 0 parameters
        layer0_keys = sorted([k for k in keys if "model.layers.0." in k])
        for k in layer0_keys:
            slice_ = f.get_slice(k)
            print(f"{k:<55} | {str(slice_.get_shape()):<18} | {slice_.get_dtype()}")

    print("="*85)

if __name__ == "__main__":
    main()
