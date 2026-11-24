# LittleBit Qwen3-8B (0.55 bpw) & 2-Bit GPTQ — Custom CUDA Inference Engine & Streaming Server

A custom CUDA C++ and PyTorch inference runtime and FastAPI Server-Sent Events (SSE) web interface designed to serve **Qwen3-8B compressed to 0.55 bits per weight** (`LittleBitLLM/littlebit-qwen3-8b-0.55bpw`) and **uniform 2-bit GPTQ** on an entry-level **4 GB VRAM laptop GPU** (NVIDIA GeForce RTX 3050 Ti Laptop).

Standard LLM inference frameworks (vLLM, Ollama, llama.cpp) do not support sub-1-bit latent-factorized matrix formats or custom 2-bit group quantization out of the box. This repository implements dedicated `sm_86` register-level bit-unpacking GEMM/GEMV CUDA kernels, host CPU parameter offloading wrappers, static KV caching, `torch.cuda.CUDAGraph` execution, and a layer-by-layer second-order Hessian Blocked GPTQ quantizer.

---

## Benchmark Results (RTX 3050 Ti Laptop GPU — 4.00 GB VRAM)

### 1. Measured Results from Repository Logs

The following benchmarks were measured on a Windows 11 machine equipped with an NVIDIA GeForce RTX 3050 Ti Laptop GPU (4.00 GB VRAM, 75W TGP ceiling). All runs are recorded in [`logs/`](file:///d:/Personal%20Projects/Little%20Bit/logs).

| Evaluation / Configuration | Runner Script | Context / KV Cache | Decode Speed | Prefill Time / Latency | Peak VRAM (GPU) | Reference Log |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **0.55 bpw CUDA Graph Decode (Run 1)** | `run_qwen3_cudagraph.py` | `StaticCache(512)` | **26.65 tok/s** | 542.45 ms | **1,750.2 MB** *(1.71 GB)* | [`logs/cudagraph_run1.txt`](file:///d:/Personal%20Projects/Little%20Bit/logs/cudagraph_run1.txt) |
| **0.55 bpw CUDA Graph Decode (Run 2)** | `run_qwen3_cudagraph.py` | `StaticCache(512)` | **18.01 tok/s** | 605.15 ms | **1,750.2 MB** *(1.71 GB)* | [`logs/cudagraph_run2.txt`](file:///d:/Personal%20Projects/Little%20Bit/logs/cudagraph_run2.txt) |
| **INT2 Stage 3 GPTQ Decode ($G=16$)** | `eval_int2.py` | `DynamicCache` | **5.18 – 6.01 tok/s** | — | **3.28 GB** *(3,312.6 MB load)* | [`logs/eval_int2.txt`](file:///d:/Personal%20Projects/Little%20Bit/logs/eval_int2.txt) |
| **Layer-0 Numerical Accuracy Check** | `run_qwen3_cudagraph.py` | 16 tokens ($B=1$) | — | Max Abs Err: **4.83e-3** | — | [`logs/cudagraph_run1.txt`](file:///d:/Personal%20Projects/Little%20Bit/logs/cudagraph_run1.txt) |
| **Stage 3 GPTQ Quantization (36 Layers)** | `quantize_qwen3_int2_gptq.py` | 32,768 calib tokens | — | 3,169.3 s *(52.8 min)* | **392.4 MB** | [`logs/task-954.log`](file:///d:/Personal%20Projects/Little%20Bit/logs/task-954.log) |

> [!NOTE]
> **Performance Variance on Laptop Hardware:**
> Autoregressive decode speed for the 0.55 bpw CUDA Graph engine varies significantly between runs (**18.01 to 26.65 tokens/sec**) on the same 12-token prompt (`"Sub-1-bit quantization compresses large language models by"`, 96 tokens generated). This variance is primarily caused by dynamic laptop thermal throttling, GPU clock scaling, and power-limit constraints (75W cap, documented in [`logs/gpu.txt`](file:///d:/Personal%20Projects/Little%20Bit/logs/gpu.txt)).

---

### 2. Development Progression (From Developer Screenshots, Not Re-Measured)

The table below reflects development milestone progression recorded from developer screenshots during earlier iterative engineering phases. Early stages and the 2,048-token context window have not been re-benchmarked in the current log set.

| Optimization Stage | KV Cache | Decode Speed | Peak VRAM | Speedup | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 1. Initial Eager PyTorch + Unoptimized Kernel | Dynamic | 3.53 tok/s | ~1.70 GB | 1.00× | Dev screenshot |
| 2. FP32 Accumulators + Tiled GEMM ($M > 1$) | Dynamic | 5.04 tok/s | 1,685.0 MB | 1.43× | Dev screenshot |
| 3. `StaticCache` KV Pre-Allocation | 512 tokens | 8.32 tok/s | 1,750.2 MB | 2.36× | Dev screenshot |
| 4. `torch.cuda.CUDAGraph` Decode Replay | 512 tokens | 18.02 tok/s | 1,750.2 MB | 5.10× | Dev screenshot |
| 5. Warp-Per-Row $M == 1$ GEMV + FastAPI SSE | 512 tokens | 22.20 tok/s | 1,752.4 MB | 6.29× | Dev screenshot |
| 6. Expanded Context Window (`StaticCache(2048)`) | 2048 tokens | 20.80 tok/s | 1,984.6 MB | 5.89× | Calculated total / dev screenshot; not re-measured in current logs |

---

## VRAM Memory Budget (0.55 bpw Architecture)

All values are reported in **Mebibytes (MiB)** based on exact tensor byte counts:

| Component | Target Device | Precision / Format | Memory (MiB) | Engineering Role |
| :--- | :---: | :---: | :---: | :--- |
| `model.embed_tokens` | System RAM (CPU) | FP16 | **1,187.0 MiB** | Offloaded via `CPUEmbeddingWrapper`; lookups run on CPU, transferring only slice activations to CUDA |
| 36 Packed Transformer Blocks (`V, U, V_R, U_R`) | CUDA (VRAM) | 0.55 bpw + FP16 scales | **456.8 MiB** | Packed 32-bit uint32 weights ($478,940,112\text{ bytes}$); ~464 MiB including PyTorch layer buffers and RMSNorms |
| `lm_head` Projection | CUDA (VRAM) | FP16 | **1,187.0 MiB** | Vocabulary projection ($151,936 \times 4096 \times 2\text{ bytes}$) retained in VRAM for zero-PCIe decode logits |
| `StaticCache(max_cache_len=512)` | CUDA (VRAM) | FP16 | **72.0 MiB** | $36\text{ layers} \times 2\text{ (K+V)} \times 1\text{ (B)} \times 8\text{ (heads)} \times 512 \times 128 \times 2\text{ bytes}$ |
| **Measured Model Load Base VRAM** | **CUDA (VRAM)** | — | **1,652.4 MB** | Measured during model loading before cache allocation ([`logs/cudagraph_run1.txt`](file:///d:/Personal%20Projects/Little%20Bit/logs/cudagraph_run1.txt)) |
| **Measured Peak VRAM (`StaticCache(512)`)** | **CUDA (VRAM)** | — | **1,750.2 MB** *(~1,669.1 MiB)* | Measured peak during generation including CUDA Graph buffers ([`logs/cudagraph_run1.txt`](file:///d:/Personal%20Projects/Little%20Bit/logs/cudagraph_run1.txt)) |
| `StaticCache(max_cache_len=2048)` *(Optional / Server)* | CUDA (VRAM) | FP16 | *288.0 MiB* | $36 \times 2 \times 1 \times 8 \times 2048 \times 128 \times 2\text{ bytes}$ (extended server context) |
| **Total VRAM with 2048 Cache** | **CUDA (VRAM)** | — | **~1,984.6 MB** *(~1,892.7 MiB)* | **Calculated total, not logged in current benchmark runs** |

---

## Custom CUDA Kernels & Architecture

```text
                  Input Activation Token X [1, 4096]
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       Branch 1 (Main Factor)          Branch 2 (Residual Factor)
         Scale vector v2                 Scale vector v2_R
                 │                               │
                 ▼                               ▼
      Register Bit-Unpack V           Register Bit-Unpack V_R
        (int32 -> {-1.0, +1.0})         (int32 -> {-1.0, +1.0})
                 │                               │
                 ▼                               ▼
         Inner Scale (v1 * u2)           Inner Scale (v1_R * u2_R)
                 │                               │
                 ▼                               ▼
      Register Bit-Unpack U           Register Bit-Unpack U_R
        (int32 -> {-1.0, +1.0})         (int32 -> {-1.0, +1.0})
                 │                               │
                 ▼                               ▼
         Scale vector u1                 Scale vector u1_R
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                     FP32 Accumulation + Add
                                 │
                                 ▼
                    Output Projection Y [1, 4096]
```

1. **Register-Level On-the-Fly Unpacking:**
   Packed 32-bit words are loaded from global VRAM and unpacked directly inside GPU thread registers via bit-shifting and masking (`(word >> bit) & 1`). Weights are never materialized as full FP16 or FP32 matrices in global memory.
2. **Dual Execution Paths (`sm_86`):**
   * **Multi-Token Prefill (`M > 1`):** Tiled shared-memory GEMM kernel (`packed_binary_gemm_fp16_kernel` in [`cuda_unpack.cu`](file:///d:/Personal%20Projects/Little%20Bit/cuda_unpack.cu)) with FP32 accumulators.
   * **Single-Token Decode (`M == 1`):** Dedicated warp-per-row GEMV kernel (`packed_binary_gemv_fp16_kernel`) featuring 8 warps per block, `smem_x[1024]` input activation caching, coalesced 128-byte global memory reads, and intra-warp `__shfl_down_sync` register reduction.
3. **Driver Dispatch Latency Reduction via CUDA Graph:**
   Autoregressive generation on Windows WDDM drivers suffers from substantial CPU-side kernel dispatch overhead. By capturing the single-token decode pass into `torch.cuda.CUDAGraph` using fixed static buffers, CPU launch latency is eliminated during decode.

---

## Stage 3: Second-Order Hessian Blocked GPTQ Quantizer (INT2)

In addition to serving pre-factorized 0.55 bpw weights, this repository includes an end-to-end 2-bit quantization pipeline (`quantize_qwen3_int2_gptq.py`):

1. **Sequential Streaming Quantization:**
   * Quantizes `Qwen/Qwen3-8B` layer by layer directly onto CUDA, maintaining a peak GPU memory footprint of **392.4 MB** throughout quantization ([`logs/task-954.log`](file:///d:/Personal%20Projects/Little%20Bit/logs/task-954.log)).
   * Accumulates exact float32 Hessian covariance matrices ($H = X^T X$) on CPU RAM using 32,768 calibration tokens (64 sequences $\times$ 512 tokens) streamed in micro-batches of 4.
2. **Cholesky-Damped Blocked GPTQ:**
   * Applies second-order error compensation using the inverse Hessian $H^{-1}$ stabilized with Cholesky damping ($\lambda = 0.01$).
   * Quantizes all 7 linear projections per layer (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with uniform **`group_size = 16`**, packing 16 2-bit weights into each `uint32` word with per-group FP16 scale and zero-point tensors.
   * Final quantized model is packaged into 5 safetensors shards (~5.96 GB total) in `./qwen3-8b-int2-stage3/`.
3. **INT2 Kernel Verification:**
   * [`test_int2_shapes.py`](file:///d:/Personal%20Projects/Little%20Bit/test_int2_shapes.py) verifies `Int2GroupLinear` against dequantized PyTorch FP32 matrix multiplication across all four Qwen3-8B projection matrix shapes:
     * 1D decode ($M=1$): $\text{Max Abs Error} < 2 \times 10^{-3}$
     * 2D decode ($M=1$): $\text{Max Abs Error} < 2 \times 10^{-3}$
     * 3D prefill ($M=16$): $\text{Max Abs Error} < 6 \times 10^{-3}$ and $\text{Cosine Similarity} > 0.9999$
   * [`benchmark_int2_stage1.py`](file:///d:/Personal%20Projects/Little%20Bit/benchmark_int2_stage1.py) and [`benchmark_int2_stage2.py`](file:///d:/Personal%20Projects/Little%20Bit/benchmark_int2_stage2.py) benchmark single-scale vs per-group CUDA kernels against FP16 baselines.
4. **INT2 Evaluation Results ([`logs/eval_int2.txt`](file:///d:/Personal%20Projects/Little%20Bit/logs/eval_int2.txt)):**
   * **Memory & Speed:** Loads into 3,312.6 MB VRAM with a peak generation memory of **3.28 GB** (without CUDA Graph, with `lm_head` evaluated on CPU via `CPUHeadWrapper`). Decode throughput ranges from **5.18 to 6.01 tokens/sec**.
   * **Coherence:** Evaluated across three prompt regimes:
     * *Prompt 1 (CPU vs GPU difference):* Generated fluent, structured bullet points (5.71 tok/s).
     * *Prompt 2 (Python palindrome check):* Produced valid, functioning Python code and explanation (6.01 tok/s).
     * *Prompt 3 (Sheep math riddle):* Coherent output, but failed basic arithmetic reasoning (concluded $17 - 9 = 8$ instead of the correct answer 9) (5.18 tok/s).
     * *Repetition Bug Status:* The severe repetition collapse observed in Stage 2 RTN (`'>'>>'>'`) did not occur; output text remained coherent.
   * **Scope:** The INT2 GPTQ path is currently CLI-only (`run_qwen3_int2.py`, `eval_int2.py`) and is **not wired into `server_littlebit.py`**.

---

## Project Structure

```text
Little Bit/
├── cuda_unpack.cu               # 0.55 bpw CUDA kernels (tiled GEMM + warp-per-row GEMV)
├── cuda_binding.cpp             # PyTorch ATen C++ extension bindings for 0.55 bpw kernels
├── cuda_ops.py                  # JIT loader and forward dispatch for PackedLittleBitLinear
├── cuda_int2_group_unpack.cu   # INT2 sm_86 CUDA register-unpacking GEMV kernel (G=16 / G=32)
├── cuda_int2_group_binding.cpp # PyTorch ATen bindings for INT2 group-quantized GEMV
├── cuda_int2_group_ops.py       # JIT compiler and Int2GroupLinear PyTorch module
├── quantize_qwen3_int2_gptq.py  # Second-order Hessian Blocked GPTQ streaming quantizer
├── run_qwen3_littlebit.py       # Eager-mode 0.55 bpw CLI runner + numerical accuracy test
├── run_qwen3_cudagraph.py       # CUDA Graph benchmark runner (StaticCache(512))
├── run_qwen3_int2.py            # End-to-end INT2 inference runner (supports Stage 2 & 3)
├── eval_int2.py                 # Multi-prompt evaluation script for INT2 Stage 3 checkpoint
├── download_weights.py          # Helper script to pre-fetch Hugging Face checkpoints
├── server_littlebit.py          # FastAPI SSE streaming web server + dark-mode UI
├── test_int2_shapes.py          # Numerical shape correctness test suite for Int2GroupLinear
├── test_runtime.py              # Verification suite against Samsung LittleBit reference math
├── packer.py                    # Binary packing utilities (LSB-first, +1.0->0, -1.0->1)
├── run_with_msvc.bat            # MSVC + CUDA developer environment wrapper
├── .env.example                 # Template for environment configuration
└── logs/                        # Raw benchmark and evaluation output logs
    ├── cudagraph_run1.txt       # Measured 26.65 tok/s benchmark run
    ├── cudagraph_run2.txt       # Measured 18.01 tok/s benchmark run
    ├── eval_int2.txt            # Measured INT2 Stage 3 evaluation run
    ├── task-954.log             # Measured 392.4 MB peak VRAM GPTQ quantizer log
    └── gpu.txt                  # NVIDIA-SMI GPU hardware status
```

---

## Prerequisites & Quickstart (Windows)

### Requirements

* **Operating System:** Windows 11
* **GPU:** NVIDIA GPU with Compute Capability `sm_86` or newer (Ampere / Ada / Blackwell) with $\ge 4.0\text{ GB}$ VRAM.
* **Compiler Toolchain:** Visual Studio 2022 (Build Tools, Community, or Professional with MSVC `cl.exe`) + CUDA Toolkit 12.x + `ninja`.
* **Python:** Python 3.11+ virtual environment (`./.venv`) with PyTorch (CUDA enabled), `transformers`, `huggingface_hub`, `safetensors`, `fastapi`, and `uvicorn`.
* **Disk Space Requirements:**
  * ~2.1 GB to store `LittleBitLLM/littlebit-qwen3-8b-0.55bpw`
  * ~6.0 GB if using the Stage 3 INT2 checkpoint
  * ~30 GB if downloading unquantized `Qwen/Qwen3-8B` and running the GPTQ quantizer locally

---

### Step 1: Pre-Download Weights

Inference scripts use `local_files_only=True` to guarantee deterministic local runs. Download the weights beforehand to avoid runtime cache misses:

```powershell
# Download the 0.55 bpw LittleBit checkpoint (~2.1 GB)
.\run_with_msvc.bat download_weights.py

# Optional: Also download full Qwen3-8B unquantized shards if running GPTQ (~16 GB)
.\run_with_msvc.bat download_weights.py --qwen3-base
```

---

### Step 2: Launch Web UI & Streaming Server (0.55 bpw)

```powershell
.\run_with_msvc.bat server_littlebit.py
```

Open **`http://127.0.0.1:8000`** once Uvicorn finishes warming up and capturing the CUDA Graph.

---

### Step 3: Run Standalone CLI Benchmarks

```powershell
# 1. Benchmark 0.55 bpw with StaticCache(512) and CUDA Graph replay
.\run_with_msvc.bat run_qwen3_cudagraph.py

# 2. Benchmark INT2 Stage 3 GPTQ inference on multi-turn prompts
.\run_with_msvc.bat eval_int2.py

# 3. Run unit tests across all INT2 projection dimensions
.\run_with_msvc.bat test_int2_shapes.py
```

> [!TIP]
> **First-Run JIT Compile Time:**
> On the first run, PyTorch's `torch.utils.cpp_extension.load` compiles the CUDA C++ extensions (`cuda_unpack.cu` or `cuda_int2_group_unpack.cu`) via MSVC and `nvcc`. This compilation takes 60–120 seconds; subsequent executions load cached binaries in `<1` second.

> [!NOTE]
> **Running `test_runtime.py`:**
> `test_runtime.py` validates bit-packing against Samsung's original reference classes. Running this test requires cloning Samsung's repository into the project root:
> ```powershell
> git clone https://github.com/SamsungLabs/LittleBit
> .\run_with_msvc.bat test_runtime.py
> ```
> `LittleBit/` is gitignored and is not bundled with this repository.

---

## Known Limitations

1. **No Standardized Quality Evaluation:** Neither the 0.55 bpw factorized model nor the INT2 GPTQ model has undergone formal perplexity benchmarking (e.g. WikiText-2 PPL) or academic evaluation suites (MMLU, GSM8K).
2. **Output Quality at Extreme Compression:** At 0.55 bpw (sub-1-bit), model outputs are grammatically fluent English but lack factual and semantic grounding (sample outputs discuss unrelated concepts rather than directly answering prompts).
3. **INT2 Speed & Server Support:** The INT2 Stage 3 implementation runs at 5–6 tokens/sec (due to un-graphed execution and CPU-offloaded `lm_head`) and is currently accessible via CLI only; it is not supported in `server_littlebit.py`.
4. **Single-Client Server Architecture:** `server_littlebit.py` guards static CUDA Graph memory with an `asyncio.Lock()`. Concurrent client requests are serialized.
5. **Network & Disconnect Handling:** The server does not configure CORS middleware, and client disconnects during SSE streaming do not abort in-flight CUDA kernel execution on the GPU.
6. **Platform Scope:** Designed and verified specifically for Windows 11 with MSVC and `sm_86` or newer NVIDIA GPUs.

---

## Credits & Licensing

* **LittleBit Method:** Samsung Research (*"LittleBit: Ultra-Low Bit Factorization for LLMs"*, NeurIPS 2025; [github.com/SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit)), licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
* **Model Checkpoint:** `LittleBitLLM/littlebit-qwen3-8b-0.55bpw` produced and published by the independent open-source **LittleBitLLM** community on Hugging Face. Model weights are downloaded directly from Hugging Face Hub and are not included in this repository.
* **Base Architecture:** `Qwen/Qwen3-8B` by the Qwen Team (Alibaba), licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
* **Code Provenance:**
  * `packer.py` adheres to the same bit-packing format and binary conventions (LSB-first, bit 0 = +1.0, bit 1 = -1.0) as Samsung's `binary_packer.py` to ensure compatibility with pre-quantized LittleBit weights.
  * `test_runtime.py` dynamically imports modules from a separate, unvendored local clone of `SamsungLabs/LittleBit` (gitignored) for validation against reference PyTorch math.
  * **Original Components:** All custom CUDA C++ kernels (`cuda_unpack.cu`, `cuda_int2_unpack.cu`, `cuda_int2_group_unpack.cu`), ATen extension bindings (`cuda_binding.cpp`, `cuda_int2_binding.cpp`, `cuda_int2_group_binding.cpp`), JIT loader modules (`cuda_ops.py`, `cuda_int2_ops.py`, `cuda_int2_group_ops.py`), inference engines and memory wrappers (`run_qwen3_littlebit.py`, `run_qwen3_cudagraph.py`, `run_qwen3_int2.py`), the FastAPI SSE server (`server_littlebit.py`), and the second-order Hessian Blocked GPTQ quantizer (`quantize_qwen3_int2_gptq.py`).
