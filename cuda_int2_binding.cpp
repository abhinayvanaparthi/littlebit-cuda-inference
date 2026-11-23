#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" {
void launch_packed_int2_gemv_fp16(
    const __half*   X,
    const uint32_t* W_packed,
    const __half*   scales,
    __half*         Y,
    int orig_cols,
    int out_features,
    int words_per_row,
    cudaStream_t    stream
);
}

torch::Tensor packed_int2_gemv_cuda(
    torch::Tensor x,           // [K] or [1, K] (FP16)
    torch::Tensor w_packed,    // [N, K / 16] (int32)
    torch::Tensor scales       // [N] (FP16)
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w_packed.is_cuda(), "w_packed must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(w_packed.scalar_type() == torch::kInt32, "w_packed must be int32");

    auto x_contig = x.contiguous();
    auto w_contig = w_packed.contiguous();
    auto s_contig = scales.contiguous();

    int orig_cols = x_contig.numel();
    int out_features = w_contig.size(0);
    int words_per_row = w_contig.size(1);

    TORCH_CHECK(orig_cols == words_per_row * 16, "orig_cols must equal words_per_row * 16");

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(x.device());
    torch::Tensor y;
    if (x.dim() == 2) {
        y = torch::empty({1, out_features}, options);
    } else {
        y = torch::empty({out_features}, options);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_packed_int2_gemv_fp16(
        reinterpret_cast<const __half*>(x_contig.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(w_contig.data_ptr<int32_t>()),
        reinterpret_cast<const __half*>(s_contig.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        orig_cols,
        out_features,
        words_per_row,
        stream
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("packed_int2_gemv_cuda", &packed_int2_gemv_cuda, "Packed INT2 GEMV CUDA (FP16)");
}
