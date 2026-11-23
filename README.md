# LittleBit Qwen3-8B (0.55 bpw) — Custom CUDA Inference Engine & Streaming Server

A custom CUDA C++ / PyTorch inference runtime and FastAPI Server-Sent Events (SSE) web interface built from scratch to run **Qwen3-8B compressed to 0.55 bits per weight** (`LittleBitLLM/littlebit-qwen3-8b-0.55bpw`) on a **4 GB VRAM laptop GPU** (NVIDIA GeForce RTX 3050 Ti) at **>20 tokens/sec**.

Standard inference engines (Ollama, llama.cpp, vLLM) do not support sub-1-bit latent factorized checkpoints. This project implements custom register-level bit-unpacking GEMM/GEMV CUDA kernels, zero-materialization weight execution, static KV caching, and `torch.cuda.CUDAGraph` replay.

---

## Key Engineering Highlights

* **Sub-2 GB Total VRAM Footprint:** Fits an 8B-parameter transformer architecture inside **1,984.6 MB (1.94 GB / 4.00 GB)** with a **2,048-token** context window.
* **Register-Level On-the-Fly Bit Unpacking:** Never materializes full FP16 weight matrices in VRAM; unpacks 32-bit packed `+1 / -1` latent factors (`V`, `U`, and residual `V_R`, `U_R`) directly inside GPU registers.
* **Dual-Path CUDA Kernels (`sm_86`):**
  * **Prefill (`M > 1`):** Tiled shared-memory GEMM kernel (`packed_binary_gemm_fp16_kernel`) with FP32 inner accumulators.
  * **Decode (`M == 1`):** Dedicated warp-per-row GEMV kernel (`packed_binary_gemv_fp16_kernel`) with 8 warps/block, `smem_x[1024]` activation caching, coalesced 128-byte reads, and intra-warp `__shfl_down_sync` reduction.
* **Zero-Overhead CUDA Graph Decode:** Eliminates Windows WDDM driver launch latency during single-token autoregressive generation using `StaticCache(max_cache_len=2048)` and `torch.cuda.CUDAGraph`.
* **Live Streaming Web UI:** FastAPI backend streaming token deltas over SSE (`/generate`) with real-time telemetry (TTFT, `tokens/sec`, peak VRAM) and interactive nucleus sampling controls.

---

## Benchmark Progression (RTX 3050 Ti Laptop GPU — 4.00 GB VRAM)

| Optimization Stage | KV Cache | Decode Speed | Peak VRAM | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| 1. Initial Eager PyTorch + Unoptimized Kernel | Dynamic | 3.53 tok/s | ~1.70 GB | 1.00× |
| 2. FP32 Accumulators + Tiled GEMM (`M > 1`) | Dynamic | 5.04 tok/s | 1,685.0 MB | 1.43× |
| 3. `StaticCache` KV Pre-Allocation | 512 tokens | 8.32 tok/s | 1,750.2 MB | 2.36× |
| 4. `torch.cuda.CUDAGraph` Decode Replay | 512 tokens | 18.02 tok/s | 1,750.2 MB | 5.10× |
| 5. Warp-Per-Row `M == 1` GEMV + FastAPI SSE | 512 tokens | **22.20 tok/s** | **1,752.4 MB** | **6.29×** |
| 6. Expanded Context Window (`StaticCache(2048)`) | 2048 tokens | **20.80 tok/s** | **1,984.6 MB** | **5.89×** |

---

## VRAM Memory Budget (`StaticCache(2048)`)

| Component | Device | Precision | Memory |
| :--- | :---: | :---: | :---: |
| `model.embed_tokens` | System RAM (CPU) | FP16 | ~1,240.0 MB *(offloaded)* |
| 36 Packed Transformer Layers (`V, U, V_R, U_R`) | CUDA (VRAM) | 0.55 bpw + FP16 scales | 464.4 MB |
| `lm_head` Projection | CUDA (VRAM) | FP16 | 1,187.0 MB |
| `StaticCache` (`max_cache_len=2048`) | CUDA (VRAM) | FP16 | ~288.0 MB |
| Activations & CUDA Graph Workspace | CUDA (VRAM) | FP16 / FP32 | ~45.2 MB |
| **Total Peak VRAM Allocated** | **CUDA (VRAM)** | — | **1,984.6 MB / 4,096 MB** |

---

## Project Structure

```text
Little Bit/
├── cuda_unpack.cu           # Custom CUDA kernels (tiled GEMM + warp-per-row GEMV)
├── cuda_binding.cpp         # PyTorch C++ ATen extension bindings
├── cuda_ops.py              # JIT compiler wrapper & BitLinear forward dispatch
├── run_qwen3_littlebit.py   # Eager-mode CLI runner + Layer-0 FP32 numerical accuracy test
├── run_qwen3_cudagraph.py   # CUDA Graph CLI benchmark runner
├── server_littlebit.py      # FastAPI SSE server + embedded dark-mode web UI
└── run_with_msvc.bat        # Visual Studio 2022 x64 environment wrapper for nvcc/cl.exe
```

---

## Prerequisites & Quickstart (Windows)

### Requirements
* **GPU:** NVIDIA GPU with Compute Capability `sm_86` or newer (`>= 4 GB` VRAM)
* **Toolchain:** CUDA Toolkit 12.x + Visual Studio 2022 Build Tools (`cl.exe` + `ninja`)
* **Python:** Python 3.11+ with PyTorch (CUDA), `transformers`, `safetensors`, `fastapi`, and `uvicorn`
* **Weights:** Hugging Face checkpoint `LittleBitLLM/littlebit-qwen3-8b-0.55bpw` cached locally

### 1. Launch the Web UI & Streaming Server
```powershell
.\run_with_msvc.bat server_littlebit.py
```
Open **`http://127.0.0.1:8000`** in your browser once Uvicorn starts.

### 2. Run the Standalone CUDA Graph CLI Benchmark
```powershell
.\run_with_msvc.bat run_qwen3_cudagraph.py
```
Runs single-layer numerical verification against unpacked FP32 weights (`Max Absolute Error < 1e-2`), captures the decode CUDA Graph, and benchmarks prefill latency and decode throughput.
