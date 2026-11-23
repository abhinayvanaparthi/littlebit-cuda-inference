/*
 * cuda_binding.cpp - PyTorch/pybind11 bindings for packed binary GEMM & GEMV.
 *
 * Separated from cuda_unpack.cu so that MSVC compiles this file (which
 * includes torch/extension.h) while nvcc compiles the pure CUDA file.
 * This avoids Windows header conflicts between MSVC STL and CUDA toolchain.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdexcept>
#include <string>

// Forward declarations of CUDA launch helpers defined in cuda_unpack.cu
extern "C" {
void launch_packed_binary_gemm_fp32(
    const float*    X,
    const uint32_t* W_packed,
    float*          Y,
    int B, int orig_cols, int out_features, int words_per_row,
    cudaStream_t stream
);

void launch_packed_binary_gemm_fp16(
    const __half*   X,
    const uint32_t* W_packed,
    __half*         Y,
    int B, int orig_cols, int out_features, int words_per_row,
    cudaStream_t stream
);

void launch_packed_binary_gemv_fp16(
    const __half*   X,
    const uint32_t* W_packed,
    __half*         Y,
    int orig_cols, int out_features, int words_per_row,
    cudaStream_t stream
);
} // extern "C"


// ---------------------------------------------------------------------------
// packed_matvec_cuda
//   X        : (B, orig_cols)  FP32 or FP16
//   W_packed : (out_features, words_per_row)  int32 view of uint32 packed weights
//   orig_cols: true (unpadded) number of input columns
//   returns  : Y (B, out_features), same dtype as X
// ---------------------------------------------------------------------------
torch::Tensor packed_matvec_cuda(
    torch::Tensor X,
    torch::Tensor W_packed,
    int64_t       orig_cols
) {
    TORCH_CHECK(X.is_cuda(),        "X must be a CUDA tensor");
    TORCH_CHECK(W_packed.is_cuda(), "W_packed must be a CUDA tensor");
    TORCH_CHECK(X.dim() == 2,       "X must be 2D (B, orig_cols)");
    TORCH_CHECK(W_packed.dim() == 2,"W_packed must be 2D (out_features, words_per_row)");
    TORCH_CHECK(W_packed.scalar_type() == torch::kInt32,
                "W_packed must be int32 (uint32 view)");
    TORCH_CHECK(X.is_contiguous(),        "X must be contiguous");
    TORCH_CHECK(W_packed.is_contiguous(), "W_packed must be contiguous");

    int B            = (int)X.size(0);
    int out_features = (int)W_packed.size(0);
    int words_per_row= (int)W_packed.size(1);

    // Reinterpret packed int32 storage as uint32 (safe bitwise alias)
    const uint32_t* w_ptr = reinterpret_cast<const uint32_t*>(
        W_packed.data_ptr<int32_t>()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (X.scalar_type() == torch::kFloat32) {
        auto Y = torch::empty({B, out_features}, X.options());
        launch_packed_binary_gemm_fp32(
            X.data_ptr<float>(), w_ptr, Y.data_ptr<float>(),
            B, (int)orig_cols, out_features, words_per_row, stream
        );
        return Y;

    } else if (X.scalar_type() == torch::kFloat16) {
        auto Y = torch::empty({B, out_features}, X.options());
        if (B == 1) {
            launch_packed_binary_gemv_fp16(
                reinterpret_cast<const __half*>(X.data_ptr<at::Half>()),
                w_ptr,
                reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
                (int)orig_cols, out_features, words_per_row, stream
            );
        } else {
            launch_packed_binary_gemm_fp16(
                reinterpret_cast<const __half*>(X.data_ptr<at::Half>()),
                w_ptr,
                reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
                B, (int)orig_cols, out_features, words_per_row, stream
            );
        }
        return Y;

    } else {
        TORCH_CHECK(false, "packed_matvec_cuda: X dtype must be float32 or float16, got ",
                    X.scalar_type());
    }
}


// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "packed_matvec",
        &packed_matvec_cuda,
        "Register-level packed binary GEMM & GEMV: Y = X @ W^T  (no VRAM unpack).\n"
        "Args:\n"
        "  X        (Tensor): shape (B, orig_cols), float32 or float16, CUDA\n"
        "  W_packed (Tensor): shape (out_features, words_per_row), int32 (uint32 view), CUDA\n"
        "  orig_cols (int)  : unpadded inner dimension\n"
        "Returns:\n"
        "  Y (Tensor): shape (B, out_features), same dtype and device as X"
    );
}
