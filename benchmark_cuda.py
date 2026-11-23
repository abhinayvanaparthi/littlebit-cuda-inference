"""
benchmark_cuda.py - Accuracy verification and latency/VRAM benchmark for
the custom register-unpacking CUDA kernel.

Comparisons:
  (a) Dense FP16 nn.Linear            - baseline (cublas GEMM)
  (b) kernel_ref.py littlebit_forward_ref  - VRAM-unpack reference
  (c) cuda_ops.py  littlebit_forward_cuda  - register-unpack (this work)

Accuracy tests:
  - B=1 and B=16, standard 4096x4096/R=128, non-multiple-of-32 1024x300/R=100,
    residual=True variants. max_abs_error must be < 1e-3 vs reference.

Benchmark:
  - Config: M=4096, K=4096, R=128, residual=True, B=1 (token-generation regime)
  - Metric: latency (ms, median over 200 warmup + 500 measured iterations)
  - Metric: peak VRAM (torch.cuda.max_memory_allocated)
"""

from __future__ import annotations
import os
import sys
import gc
import time
import statistics
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup: ensure packer/kernel_ref/cuda_ops are importable
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "LittleBit"))

from packer import pack_binary_matrix, PackedLittleBitLayer
from kernel_ref import littlebit_forward_ref, unpack_and_matvec_ref

# ---------------------------------------------------------------------------
# Synthetic packed-layer builder (no Samsung dependency required)
# ---------------------------------------------------------------------------

def make_packed_layer(
    in_features: int,
    out_features: int,
    split_dim: int,
    residual: bool = False,
    bias: bool = True,
    seed: int = 42,
    device: str = "cuda",
) -> PackedLittleBitLayer:
    """
    Creates a PackedLittleBitLayer with random {-1,+1} weights and
    random FP16 scale vectors. Used for standalone benchmarking without
    Samsung's codebase.
    """
    torch.manual_seed(seed)
    rng = lambda shape: (torch.randint(0, 2, shape, device="cpu").float() * 2.0 - 1.0)

    V = rng((split_dim, in_features))
    U = rng((out_features, split_dim))
    V_packed = pack_binary_matrix(V)   # (split_dim,  ceil(in_features/32))  uint32
    U_packed = pack_binary_matrix(U)   # (out_features, ceil(split_dim/32))   uint32

    # Scale vectors as FP16 (non-unit so errors would surface)
    torch.manual_seed(seed + 1)
    v2 = torch.randn(in_features,  dtype=torch.float16)
    v1 = torch.randn(split_dim,    dtype=torch.float16)
    u2 = torch.randn(split_dim,    dtype=torch.float16)
    u1 = torch.randn(out_features, dtype=torch.float16)

    tensors = {
        "V_packed": V_packed,
        "U_packed": U_packed,
        "v2": v2, "v1": v1, "u2": u2, "u1": u1,
    }

    if residual:
        torch.manual_seed(seed + 10)
        V_R = rng((split_dim, in_features))
        U_R = rng((out_features, split_dim))
        tensors["V_R_packed"] = pack_binary_matrix(V_R)
        tensors["U_R_packed"] = pack_binary_matrix(U_R)
        tensors["v2_R"] = torch.randn(in_features,  dtype=torch.float16)
        tensors["v1_R"] = torch.randn(split_dim,    dtype=torch.float16)
        tensors["u2_R"] = torch.randn(split_dim,    dtype=torch.float16)
        tensors["u1_R"] = torch.randn(out_features, dtype=torch.float16)

    if bias:
        tensors["bias"] = torch.randn(out_features, dtype=torch.float16)

    layer = PackedLittleBitLayer(
        in_features=in_features,
        out_features=out_features,
        split_dim=split_dim,
        residual=residual,
        tensors=tensors,
    )
    return layer.to(device)


# ---------------------------------------------------------------------------
# Accuracy verification
# ---------------------------------------------------------------------------

ACCURACY_TESTS = [
    # (name,                                   M,     K,    R,   B,  residual)
    ("Standard 4096x4096/R=128  B=1",        4096,  4096, 128,  1, False),
    ("Standard 4096x4096/R=128  B=16",       4096,  4096, 128, 16, False),
    ("Non-mul32 1024x300/R=96   B=1",        1024,   300,  96,  1, False),
    ("Non-mul32 1024x300/R=96   B=16",       1024,   300,  96, 16, False),
    ("Residual  4096x4096/R=128 B=1",        4096,  4096, 128,  1, True),
    ("Residual  1024x300/R=96   B=1",        1024,   300,  96,  1, True),
]


def _littlebit_forward_fp64_ref(x: torch.Tensor, layer) -> torch.Tensor:
    """
    High-precision float64 reference forward pass.
    Computes the SAME operations as littlebit_forward_ref but in float64,
    giving a true mathematical reference independent of cuBLAS fp32 order.
    """
    from kernel_ref import unpack_and_matvec_ref

    orig_shape  = x.shape
    in_features  = int(getattr(layer, "in_features",  layer["in_features"]))
    out_features = int(getattr(layer, "out_features", layer["out_features"]))
    split_dim    = int(getattr(layer, "split_dim",    layer["split_dim"]))
    residual     = bool(getattr(layer, "residual",    layer["residual"]))

    x_flat = x.view(-1, in_features).double()
    dev = x.device

    def _g(k):
        t = getattr(layer, k, None) if hasattr(layer, k) else layer.get(k, None)
        return t.to(device=dev, dtype=torch.float64)

    def _gp(k):
        t = getattr(layer, k, None) if hasattr(layer, k) else layer.get(k, None)
        return t.to(device=dev)

    v2 = _g("v2"); v1 = _g("v1"); u2 = _g("u2"); u1 = _g("u1")
    V_packed = _gp("V_packed"); U_packed = _gp("U_packed")

    a = x_flat * v2
    h = unpack_and_matvec_ref(a, V_packed, in_features, dtype=torch.float64)
    g = h * (v1 * u2)
    z = unpack_and_matvec_ref(g, U_packed, split_dim, dtype=torch.float64)
    y = z * u1

    has_res = hasattr(layer, "V_R_packed") or (isinstance(layer, dict) and "V_R_packed" in layer)
    if residual and has_res:
        v2R = _g("v2_R"); v1R = _g("v1_R"); u2R = _g("u2_R"); u1R = _g("u1_R")
        VR = _gp("V_R_packed"); UR = _gp("U_R_packed")
        aR = x_flat * v2R
        hR = unpack_and_matvec_ref(aR, VR, in_features, dtype=torch.float64)
        gR = hR * (v1R * u2R)
        zR = unpack_and_matvec_ref(gR, UR, split_dim, dtype=torch.float64)
        y = y + zR * u1R

    bias = getattr(layer, "bias", None)
    if bias is None and isinstance(layer, dict):
        bias = layer.get("bias", None)
    if bias is not None:
        y = y + bias.to(device=dev, dtype=torch.float64)

    return y.float().view(*orig_shape[:-1], out_features)


def run_accuracy_checks(forward_cuda_fn):
    print("\n" + "=" * 70)
    print("ACCURACY VERIFICATION  (max_abs_error < 1e-3 vs fp64 reference)")
    print("  Reference: same forward pass computed in float64 (mathematical truth)")
    print("=" * 70)

    all_passed = True
    for name, M, K, R, B, residual in ACCURACY_TESTS:
        layer = make_packed_layer(
            in_features=K, out_features=M, split_dim=R,
            residual=residual, bias=True, device="cuda"
        )
        torch.manual_seed(77)
        x = torch.randn(B, K, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            # Compare both fp32 ref and cuda kernel against the fp64 mathematical truth
            y_fp64 = _littlebit_forward_fp64_ref(x, layer)
            y_ref  = littlebit_forward_ref(x, layer)
            y_cuda = forward_cuda_fn(x, layer)

        err_cuda = (y_cuda.float() - y_fp64.float()).abs().max().item()
        err_ref  = (y_ref.float()  - y_fp64.float()).abs().max().item()
        status = "PASS" if err_cuda < 1e-3 else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] {name:<42}  cuda_err={err_cuda:.3e}  ref_err={err_ref:.3e}")

        del layer, x, y_fp64, y_ref, y_cuda
        torch.cuda.empty_cache()

    print()
    if all_passed:
        print("  All accuracy checks PASSED.")
    else:
        print("  ONE OR MORE ACCURACY CHECKS FAILED!")
    return all_passed


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _cuda_timed_ms(fn, warmup: int = 200, iters: int = 500) -> tuple[float, float]:
    """Returns (median_ms, min_ms) using CUDA events."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return statistics.median(times_ms), min(times_ms)


def _peak_vram_mb(fn, label: str) -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
        fn()
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()
    return peak / (1024 ** 2)


# ---------------------------------------------------------------------------
# Dense FP16 baseline
# ---------------------------------------------------------------------------

def make_dense_linear(K: int, M: int, device: str = "cuda") -> nn.Linear:
    lin = nn.Linear(K, M, bias=False, dtype=torch.float16, device=device)
    return lin.eval()


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

BM_K        = 4096
BM_M        = 4096
BM_R        = 128
BM_BATCH    = 1
BM_RESIDUAL = True


def run_benchmark(forward_cuda_fn):
    print("\n" + "=" * 70)
    print(f"BENCHMARK  (M={BM_M}, K={BM_K}, R={BM_R}, residual={BM_RESIDUAL}, B={BM_BATCH})")
    print("=" * 70)

    device = "cuda"
    torch.manual_seed(0)
    x_fp32 = torch.randn(BM_BATCH, BM_K, device=device, dtype=torch.float32)
    x_fp16 = x_fp32.half()

    # ---- (a) Dense FP16 nn.Linear ----------------------------------------
    dense = make_dense_linear(BM_K, BM_M)

    def dense_fn():
        with torch.no_grad():
            return dense(x_fp16)

    med_dense, min_dense = _cuda_timed_ms(dense_fn)
    vram_dense = _peak_vram_mb(dense_fn, "dense")
    print(f"\n  (a) Dense FP16 nn.Linear")
    print(f"      Latency : {med_dense:.3f} ms (median)  {min_dense:.3f} ms (min)")
    print(f"      Peak VRAM: {vram_dense:.2f} MB")

    del dense
    gc.collect()
    torch.cuda.empty_cache()

    # ---- (b) kernel_ref  VRAM-unpack reference ----------------------------
    layer = make_packed_layer(
        in_features=BM_K, out_features=BM_M, split_dim=BM_R,
        residual=BM_RESIDUAL, bias=True, device=device
    )

    def ref_fn():
        with torch.no_grad():
            return littlebit_forward_ref(x_fp32, layer)

    med_ref, min_ref = _cuda_timed_ms(ref_fn)
    vram_ref = _peak_vram_mb(ref_fn, "ref")
    print(f"\n  (b) kernel_ref (VRAM unpack)")
    print(f"      Latency : {med_ref:.3f} ms (median)  {min_ref:.3f} ms (min)")
    print(f"      Peak VRAM: {vram_ref:.2f} MB")

    # ---- (c) CUDA register-unpack kernel ----------------------------------
    def cuda_fn():
        with torch.no_grad():
            return forward_cuda_fn(x_fp32, layer)

    med_cuda, min_cuda = _cuda_timed_ms(cuda_fn)
    vram_cuda = _peak_vram_mb(cuda_fn, "cuda")
    print(f"\n  (c) littlebit_forward_cuda (register unpack)")
    print(f"      Latency : {med_cuda:.3f} ms (median)  {min_cuda:.3f} ms (min)")
    print(f"      Peak VRAM: {vram_cuda:.2f} MB")

    # ---- Summary table ----------------------------------------------------
    print(f"\n  {'Method':<40}  {'Median (ms)':>12}  {'Min (ms)':>10}  {'Peak VRAM':>10}")
    print(f"  {'-'*40}  {'-'*12}  {'-'*10}  {'-'*10}")
    print(f"  {'(a) Dense FP16 nn.Linear':<40}  {med_dense:>12.3f}  {min_dense:>10.3f}  {vram_dense:>8.2f} MB")
    print(f"  {'(b) kernel_ref (VRAM unpack)':<40}  {med_ref:>12.3f}  {min_ref:>10.3f}  {vram_ref:>8.2f} MB")
    print(f"  {'(c) register-unpack CUDA':<40}  {med_cuda:>12.3f}  {min_cuda:>10.3f}  {vram_cuda:>8.2f} MB")

    speedup_vs_dense = med_dense / med_cuda if med_cuda > 0 else float("inf")
    speedup_vs_ref   = med_ref   / med_cuda if med_cuda > 0 else float("inf")
    vram_saved_vs_ref = vram_ref - vram_cuda
    print(f"\n  Speedup  cuda vs dense : {speedup_vs_dense:.2f}x")
    print(f"  Speedup  cuda vs ref   : {speedup_vs_ref:.2f}x")
    print(f"  VRAM saved vs ref      : {vram_saved_vs_ref:.2f} MB")

    del layer, x_fp32, x_fp16
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Exiting.")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    print("=" * 70)
    print(f"  LittleBit CUDA Register-Unpack Benchmark")
    print(f"  GPU : {gpu}")
    print(f"  PyTorch : {torch.__version__}  |  CUDA : {torch.version.cuda}")
    print("=" * 70)

    # Import CUDA ops (triggers JIT compile on first run)
    from cuda_ops import littlebit_forward_cuda

    # Step 1: accuracy
    passed = run_accuracy_checks(littlebit_forward_cuda)
    if not passed:
        print("\nABORTING BENCHMARK: accuracy checks failed.")
        sys.exit(1)

    # Step 2: benchmark
    run_benchmark(littlebit_forward_cuda)

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
