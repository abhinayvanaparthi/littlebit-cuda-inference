#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" {
void launch_packed_int2_group_gemv_fp16(
    const __half*   X,
    const uint32_t* W_packed,
    const __half*   scales,
    const __half*   zero_points,
    __half*         Y,
    int orig_cols,
    int out_features,
    int words_per_row,
    int group_size,
    cudaStream_t    stream
);
}

torch::Tensor packed_int2_group_gemv_cuda(
    torch::Tensor x,             // [K] or [1, K] (FP16)
    torch::Tensor w_packed,      // [N, K / 16] (int32)
    torch::Tensor scales,        // [N, num_groups] (FP16)
    torch::Tensor zero_points,   // [N, num_groups] (FP16)
    int64_t       group_size     // 16 or 32
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w_packed.is_cuda(), "w_packed must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(zero_points.is_cuda(), "zero_points must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(zero_points.scalar_type() == torch::kFloat16, "zero_points must be float16");
    TORCH_CHECK(w_packed.scalar_type() == torch::kInt32, "w_packed must be int32");

    auto x_contig = x.contiguous();
    auto w_contig = w_packed.contiguous();
    auto s_contig = scales.contiguous();
    auto z_contig = zero_points.contiguous();

    int orig_cols = x_contig.numel();
    int out_features = w_contig.size(0);
    int words_per_row = w_contig.size(1);

    TORCH_CHECK(orig_cols == words_per_row * 16, "orig_cols must equal words_per_row * 16");

    int expected_groups = (orig_cols + group_size - 1) / group_size;
    TORCH_CHECK(s_contig.size(1) == expected_groups, "scales num_groups mismatch");
    TORCH_CHECK(z_contig.size(1) == expected_groups, "zero_points num_groups mismatch");

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(x.device());
    torch::Tensor y;
    if (x.dim() == 2) {
        y = torch::empty({1, out_features}, options);
    } else {
        y = torch::empty({out_features}, options);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_packed_int2_group_gemv_fp16(
        reinterpret_cast<const __half*>(x_contig.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(w_contig.data_ptr<int32_t>()),
        reinterpret_cast<const __half*>(s_contig.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(z_contig.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        orig_cols,
        out_features,
        words_per_row,
        static_cast<int>(group_size),
        stream
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("packed_int2_group_gemv_cuda", &packed_int2_group_gemv_cuda, "Packed INT2 Group GEMV CUDA (FP16)");
}
