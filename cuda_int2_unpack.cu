#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// CUDA Kernel: packed_int2_gemv_fp16_kernel
//
// Optimized for sm_86 architecture (RTX 3050 Ti):
// - Single-token decode (M == 1): computes Y = X @ W.T * scale
// - Block structure: 256 threads = 8 warps of 32 threads.
// - Grid: gridDim.x = (out_features + 7) / 8 blocks.
// - Each 32-thread warp independently computes 1 output row (out_features).
// - Shared memory tile: GEMV_INT2_TILE_COLS = 1024 half elements (2048 bytes).
// - In INT2, each uint32 holds 16 weights (2 bits each: codes 0..3 mapping to {-1.5, -0.5, +0.5, +1.5}).
// - In 1024 columns, there are 1024 / 16 = 64 uint32 words per row.
// - 32 lanes in each warp process 2 uint32 words each per tile:
//     Word 0: tile_word_start + lane_id (coalesced 128-byte warp load)
//     Word 1: tile_word_start + 32 + lane_id (coalesced 128-byte warp load)
// - Intra-warp __shfl_down_sync reduction across 32 lanes.
// - Single per-row scale multiplication performed by lane 0 before writing to Y.
// ---------------------------------------------------------------------------

#define GEMV_INT2_TILE_COLS 1024
#define GEMV_INT2_TILE_WORDS (GEMV_INT2_TILE_COLS / 16) // 64

__global__ void packed_int2_gemv_fp16_kernel(
    const __half*   __restrict__ X,          // [orig_cols]
    const uint32_t* __restrict__ W_packed,   // [out_features, words_per_row]
    const __half*   __restrict__ scales,     // [out_features] per-row scale
    __half*         __restrict__ Y,          // [out_features]
    int orig_cols,
    int out_features,
    int words_per_row
) {
    int warp_id = threadIdx.x / 32;          // 0..7
    int lane_id = threadIdx.x % 32;          // 0..31
    int row_idx = blockIdx.x * 8 + warp_id;  // Global output row index

    const uint32_t* w_row = (row_idx < out_features)
        ? (W_packed + (size_t)row_idx * words_per_row)
        : nullptr;

    __shared__ __half smem_x[GEMV_INT2_TILE_COLS];

    float acc = 0.0f;
    int num_tiles = (orig_cols + GEMV_INT2_TILE_COLS - 1) / GEMV_INT2_TILE_COLS;

    for (int t = 0; t < num_tiles; t++) {
        int tile_col_start  = t * GEMV_INT2_TILE_COLS;
        int tile_word_start = t * GEMV_INT2_TILE_WORDS;

        // 256 threads collaboratively load 1024 half elements (4 elements per thread)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int col_offset = threadIdx.x + i * 256;
            int global_col = tile_col_start + col_offset;
            smem_x[col_offset] = (global_col < orig_cols) ? X[global_col] : __float2half(0.0f);
        }

        __syncthreads();

        if (row_idx < out_features) {
            // Word 0: 32 lanes load words [tile_word_start .. tile_word_start + 31]
            int word_idx0 = tile_word_start + lane_id;
            // Word 1: 32 lanes load words [tile_word_start + 32 .. tile_word_start + 63]
            int word_idx1 = tile_word_start + 32 + lane_id;

            uint32_t reg0 = (word_idx0 < words_per_row) ? w_row[word_idx0] : 0u;
            uint32_t reg1 = (word_idx1 < words_per_row) ? w_row[word_idx1] : 0u;

            int base_col0 = lane_id * 16;
            int base_col1 = 512 + lane_id * 16;

            // Unpack 16 weights from word 0
            #pragma unroll
            for (int b = 0; b < 16; b++) {
                uint32_t code = (reg0 >> (2 * b)) & 0x3;
                float w_val = (float)code - 1.5f;
                float x_val = __half2float(smem_x[base_col0 + b]);
                acc += w_val * x_val;
            }

            // Unpack 16 weights from word 1
            #pragma unroll
            for (int b = 0; b < 16; b++) {
                uint32_t code = (reg1 >> (2 * b)) & 0x3;
                float w_val = (float)code - 1.5f;
                float x_val = __half2float(smem_x[base_col1 + b]);
                acc += w_val * x_val;
            }
        }

        __syncthreads();
    }

    // Warp-level reduction across all 32 lanes
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Lane 0 of each warp multiplies by the single per-row FP16 scale and writes output
    if (lane_id == 0 && row_idx < out_features) {
        float row_scale = (scales != nullptr) ? __half2float(scales[row_idx]) : 1.0f;
        Y[row_idx] = __float2half(acc * row_scale);
    }
}

// ---------------------------------------------------------------------------
// C++ Launcher wrapper
// ---------------------------------------------------------------------------

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
) {
    int threads_per_block = 256; // 8 warps
    int warps_per_block = 8;
    int blocks = (out_features + warps_per_block - 1) / warps_per_block;

    packed_int2_gemv_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
        X, W_packed, scales, Y, orig_cols, out_features, words_per_row
    );
}
} // extern "C"
