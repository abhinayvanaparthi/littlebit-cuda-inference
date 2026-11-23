#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// CUDA Kernel: packed_int2_group_gemv_fp16_kernel
//
// Optimized for sm_86 architecture (RTX 3050 Ti):
// - Supports both single-token decode (B == 1) and prompt prefill (B > 1):
//     Computes Y[b, :] = X[b, :] @ W.T
// - Asymmetric per-group FP16 scale and zero_point:
//     W_hat[i, j] = scale[i, g] * q[i, j] + zero_point[i, g]
// - Block structure: 256 threads = 8 warps of 32 threads.
// - Grid: gridDim.x = (out_features + 7) / 8, gridDim.y = B.
// - Each 32-thread warp independently computes 1 output row for 1 batch token.
// - Shared memory tile: GEMV_TILE_COLS = 1024 half elements (2048 bytes).
// - In INT2, each uint32 holds 16 weights (2 bits each: q in {0, 1, 2, 3}).
// - In 1024 columns, there are 1024 / 16 = 64 uint32 words per row.
// - 32 lanes in each warp process 2 uint32 words each per tile:
//     Word 0: tile_word_start + lane_id (coalesced 128-byte warp load)
//     Word 1: tile_word_start + 32 + lane_id (coalesced 128-byte warp load)
// - Supports template specialization for GROUP_SIZE = 16 and GROUP_SIZE = 32.
// - Accumulates in FP32 registers; warp reduction via __shfl_down_sync.
// ---------------------------------------------------------------------------

#define GEMV_TILE_COLS 1024
#define GEMV_TILE_WORDS (GEMV_TILE_COLS / 16) // 64

template <int GROUP_SIZE>
__global__ void packed_int2_group_gemv_fp16_kernel(
    const __half*   __restrict__ X,            // [B, orig_cols]
    const uint32_t* __restrict__ W_packed,     // [out_features, words_per_row]
    const __half*   __restrict__ scales,       // [out_features, num_groups]
    const __half*   __restrict__ zero_points,  // [out_features, num_groups]
    __half*         __restrict__ Y,            // [B, out_features]
    int B,
    int orig_cols,
    int out_features,
    int words_per_row,
    int num_groups
) {
    int b_idx = blockIdx.y;
    if (b_idx >= B) return;

    int warp_id = threadIdx.x / 32;          // 0..7
    int lane_id = threadIdx.x % 32;          // 0..31
    int row_idx = blockIdx.x * 8 + warp_id;  // Global output row index

    const __half* x_row = X + (size_t)b_idx * orig_cols;
    __half*       y_row = Y + (size_t)b_idx * out_features;

    const uint32_t* w_row = (row_idx < out_features)
        ? (W_packed + (size_t)row_idx * words_per_row)
        : nullptr;
    const __half* s_row = (row_idx < out_features)
        ? (scales + (size_t)row_idx * num_groups)
        : nullptr;
    const __half* z_row = (row_idx < out_features)
        ? (zero_points + (size_t)row_idx * num_groups)
        : nullptr;

    __shared__ __half smem_x[GEMV_TILE_COLS];

    float acc = 0.0f;
    int num_tiles = (orig_cols + GEMV_TILE_COLS - 1) / GEMV_TILE_COLS;

    for (int t = 0; t < num_tiles; t++) {
        int tile_col_start  = t * GEMV_TILE_COLS;
        int tile_word_start = t * GEMV_TILE_WORDS;

        // 256 threads collaboratively load 1024 half elements into shared memory
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int col_offset = threadIdx.x + i * 256;
            int global_col = tile_col_start + col_offset;
            smem_x[col_offset] = (global_col < orig_cols) ? x_row[global_col] : __float2half(0.0f);
        }

        __syncthreads();

        if (row_idx < out_features) {
            // Word 0: 32 lanes load words [tile_word_start .. tile_word_start + 31]
            int word_idx0 = tile_word_start + lane_id;
            // Word 1: 32 lanes load words [tile_word_start + 32 .. tile_word_start + 63]
            int word_idx1 = tile_word_start + 32 + lane_id;

            uint32_t reg0 = (word_idx0 < words_per_row) ? w_row[word_idx0] : 0u;
            uint32_t reg1 = (word_idx1 < words_per_row) ? w_row[word_idx1] : 0u;

            // Group indices
            int g0, g1;
            if constexpr (GROUP_SIZE == 16) {
                g0 = word_idx0;
                g1 = word_idx1;
            } else if constexpr (GROUP_SIZE == 32) {
                g0 = word_idx0 >> 1;
                g1 = word_idx1 >> 1;
            } else {
                g0 = (word_idx0 * 16) / GROUP_SIZE;
                g1 = (word_idx1 * 16) / GROUP_SIZE;
            }

            float s0 = (g0 < num_groups && s_row != nullptr) ? __half2float(s_row[g0]) : 0.0f;
            float z0 = (g0 < num_groups && z_row != nullptr) ? __half2float(z_row[g0]) : 0.0f;

            float s1 = (g1 < num_groups && s_row != nullptr) ? __half2float(s_row[g1]) : 0.0f;
            float z1 = (g1 < num_groups && z_row != nullptr) ? __half2float(z_row[g1]) : 0.0f;

            int base_col0 = lane_id * 16;
            int base_col1 = 512 + lane_id * 16;

            // Unpack 16 weights from word 0
            #pragma unroll
            for (int b = 0; b < 16; b++) {
                uint32_t q = (reg0 >> (2 * b)) & 0x3;
                float w_val = s0 * (float)q + z0;
                float x_val = __half2float(smem_x[base_col0 + b]);
                acc += w_val * x_val;
            }

            // Unpack 16 weights from word 1
            #pragma unroll
            for (int b = 0; b < 16; b++) {
                uint32_t q = (reg1 >> (2 * b)) & 0x3;
                float w_val = s1 * (float)q + z1;
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

    // Lane 0 of each warp writes accumulated result to y_row
    if (lane_id == 0 && row_idx < out_features) {
        y_row[row_idx] = __float2half(acc);
    }
}

// ---------------------------------------------------------------------------
// C++ Launcher wrapper
// ---------------------------------------------------------------------------

extern "C" {
void launch_packed_int2_group_gemv_fp16(
    const __half*   X,
    const uint32_t* W_packed,
    const __half*   scales,
    const __half*   zero_points,
    __half*         Y,
    int B,
    int orig_cols,
    int out_features,
    int words_per_row,
    int group_size,
    cudaStream_t    stream
) {
    int threads_per_block = 256; // 8 warps
    int warps_per_block = 8;
    int blocks_x = (out_features + warps_per_block - 1) / warps_per_block;
    dim3 grid(blocks_x, B);
    int num_groups = (orig_cols + group_size - 1) / group_size;

    if (group_size == 16) {
        packed_int2_group_gemv_fp16_kernel<16><<<grid, threads_per_block, 0, stream>>>(
            X, W_packed, scales, zero_points, Y, B, orig_cols, out_features, words_per_row, num_groups
        );
    } else if (group_size == 32) {
        packed_int2_group_gemv_fp16_kernel<32><<<grid, threads_per_block, 0, stream>>>(
            X, W_packed, scales, zero_points, Y, B, orig_cols, out_features, words_per_row, num_groups
        );
    } else {
        packed_int2_group_gemv_fp16_kernel<0><<<grid, threads_per_block, 0, stream>>>(
            X, W_packed, scales, zero_points, Y, B, orig_cols, out_features, words_per_row, num_groups
        );
    }
}
} // extern "C"
