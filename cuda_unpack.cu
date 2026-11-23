/*
 * cuda_unpack.cu - Pure CUDA kernels for register-level bit-unpacking GEMM & GEMV.
 *
 * Convention (matches packer.py / kernel_ref.py):
 *   packed_weight shape: (out_features, ceil(orig_cols / 32)), dtype uint32
 *   Bit 0 (LSB) = column offset 0 within chunk
 *   Bit 31 (MSB) = column offset 31 within chunk
 *   bit == 0 -> +1.0f,  bit == 1 -> -1.0f   ( 1.0f - 2.0f * bit )
 *   Columns >= orig_cols are zero-padded, ignored via masking.
 *
 * Two specialized kernel pathways:
 *   1. packed_binary_gemv_fp16_kernel (M == 1 decode):
 *      - Warp-per-row architecture: 8 warps (256 threads) per block.
 *      - Each 32-thread warp processes 1 output row (gridDim.x = (out_features + 7) / 8).
 *      - Input X cached in __shared__ memory in tiles of TILE_COLS = 1024 (2 KB).
 *      - The 32 lanes in each warp load coalesced 128-byte int32 words (w_row[tile_word + lane_id]).
 *      - Unrolls all 32 bits into FP32 accumulator.
 *      - Warp reduction via __shfl_down_sync down to lane 0.
 *
 *   2. packed_binary_gemm_fp16_kernel (M > 1 prefill):
 *      - Shared-memory tiled 2D GEMM over batch dimension B.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

constexpr int TILE_COLS = 256;
constexpr int TILE_WORDS = TILE_COLS / 32; // 8 words

constexpr int GEMV_TILE_COLS = 1024;
constexpr int GEMV_TILE_WORDS = GEMV_TILE_COLS / 32; // 32 words
constexpr int WARPS_PER_BLOCK = 8;
constexpr int WARP_SIZE = 32;

// ---------------------------------------------------------------------------
// Kernel 1: packed_binary_gemv_fp16 (Dedicated M == 1 Warp-Per-Row GEMV)
//   Launch: 8 warps (256 threads) per block.
//   Each 32-thread warp computes 1 output feature: Y[0, row_idx].
//   Lanes in each warp read coalesced 128-byte words, unpack against smem_x,
//   and warp-reduce via __shfl_down_sync.
// ---------------------------------------------------------------------------

__global__ void packed_binary_gemv_fp16_kernel(
    const __half*   __restrict__ X,          // [1, orig_cols]
    const uint32_t* __restrict__ W_packed,   // [out_features, words_per_row]
    __half*         __restrict__ Y,          // [1, out_features]
    int orig_cols,
    int out_features,
    int words_per_row
) {
    int warp_id = threadIdx.x / WARP_SIZE; // 0..7
    int lane_id = threadIdx.x % WARP_SIZE; // 0..31
    int row_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    const uint32_t* w_row = (row_idx < out_features) ? (W_packed + (size_t)row_idx * words_per_row) : nullptr;

    __shared__ __half smem_x[GEMV_TILE_COLS];

    float acc = 0.0f;
    int num_tiles = (orig_cols + GEMV_TILE_COLS - 1) / GEMV_TILE_COLS;

    for (int t = 0; t < num_tiles; t++) {
        int tile_col_start  = t * GEMV_TILE_COLS;
        int tile_word_start = t * GEMV_TILE_WORDS;

        // Collaboratively load GEMV_TILE_COLS (1024) elements into shared memory
        // 256 threads in block: each thread loads 4 elements
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int col_offset = threadIdx.x + i * 256;
            int global_col = tile_col_start + col_offset;
            smem_x[col_offset] = (global_col < orig_cols) ? X[global_col] : __float2half(0.0f);
        }

        __syncthreads();

        if (row_idx < out_features) {
            int word_idx = tile_word_start + lane_id;
            uint32_t reg = (word_idx < words_per_row) ? w_row[word_idx] : 0u;
            int base_col = lane_id * 32;

            #pragma unroll
            for (int b = 0; b < 32; b++) {
                float x_val = __half2float(smem_x[base_col + b]);
                float term = (reg & (1u << b)) ? -x_val : x_val;
                acc += term;
            }
        }

        __syncthreads();
    }

    // Warp-level reduction across all 32 lanes
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Lane 0 of each warp writes the reduced dot product to Y
    if (lane_id == 0 && row_idx < out_features) {
        Y[row_idx] = __float2half(acc);
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: packed_binary_gemm_fp16 (Shared-Memory Tiled for M > 1 Prefill)
//   One thread per output element: Y[b, out] = dot(X[b,:], W[out,:])
// ---------------------------------------------------------------------------

__global__ void packed_binary_gemm_fp16_kernel(
    const __half*   __restrict__ X,          // [B, orig_cols]
    const uint32_t* __restrict__ W_packed,   // [out_features, words_per_row]
    __half*         __restrict__ Y,          // [B, out_features]
    int B,
    int orig_cols,
    int out_features,
    int words_per_row
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx   = blockIdx.y;

    if (b_idx >= B) return;

    const __half*   x_row = X + (size_t)b_idx * orig_cols;
    const uint32_t* w_row = (out_idx < out_features) ? (W_packed + (size_t)out_idx * words_per_row) : nullptr;

    __shared__ __half smem_x[TILE_COLS];

    float acc = 0.0f;
    int num_tiles = (orig_cols + TILE_COLS - 1) / TILE_COLS;

    for (int t = 0; t < num_tiles; t++) {
        int tile_col_start  = t * TILE_COLS;
        int tile_word_start = t * TILE_WORDS;

        int load_col = tile_col_start + threadIdx.x;
        smem_x[threadIdx.x] = (load_col < orig_cols) ? x_row[load_col] : __float2half(0.0f);

        __syncthreads();

        if (out_idx < out_features) {
            #pragma unroll
            for (int w = 0; w < TILE_WORDS; w++) {
                int word_idx = tile_word_start + w;
                if (word_idx < words_per_row) {
                    uint32_t reg = w_row[word_idx];
                    int base_col = w * 32;

                    #pragma unroll
                    for (int b = 0; b < 32; b++) {
                        float x_val = __half2float(smem_x[base_col + b]);
                        float term = (reg & (1u << b)) ? -x_val : x_val;
                        acc += term;
                    }
                }
            }
        }

        __syncthreads();
    }

    if (out_idx < out_features) {
        Y[(size_t)b_idx * out_features + out_idx] = __float2half(acc);
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: packed_binary_gemm_fp32 (Shared-Memory Tiled)
// ---------------------------------------------------------------------------

__global__ void packed_binary_gemm_fp32_kernel(
    const float*    __restrict__ X,          // [B, orig_cols]
    const uint32_t* __restrict__ W_packed,   // [out_features, words_per_row]
    float*          __restrict__ Y,          // [B, out_features]
    int B,
    int orig_cols,
    int out_features,
    int words_per_row
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx   = blockIdx.y;

    if (b_idx >= B) return;

    const float*    x_row = X + (size_t)b_idx * orig_cols;
    const uint32_t* w_row = (out_idx < out_features) ? (W_packed + (size_t)out_idx * words_per_row) : nullptr;

    __shared__ float smem_x[TILE_COLS];

    float acc = 0.0f;
    int num_tiles = (orig_cols + TILE_COLS - 1) / TILE_COLS;

    for (int t = 0; t < num_tiles; t++) {
        int tile_col_start  = t * TILE_COLS;
        int tile_word_start = t * TILE_WORDS;

        int load_col = tile_col_start + threadIdx.x;
        smem_x[threadIdx.x] = (load_col < orig_cols) ? x_row[load_col] : 0.0f;

        __syncthreads();

        if (out_idx < out_features) {
            #pragma unroll
            for (int w = 0; w < TILE_WORDS; w++) {
                int word_idx = tile_word_start + w;
                if (word_idx < words_per_row) {
                    uint32_t reg = w_row[word_idx];
                    int base_col = w * 32;

                    #pragma unroll
                    for (int b = 0; b < 32; b++) {
                        float x_val = smem_x[base_col + b];
                        float term = (reg & (1u << b)) ? -x_val : x_val;
                        acc += term;
                    }
                }
            }
        }

        __syncthreads();
    }

    if (out_idx < out_features) {
        Y[(size_t)b_idx * out_features + out_idx] = acc;
    }
}

// ---------------------------------------------------------------------------
// C dispatch functions (called from cuda_binding.cpp)
// ---------------------------------------------------------------------------

extern "C" {

void launch_packed_binary_gemv_fp16(
    const __half*   X,
    const uint32_t* W_packed,
    __half*         Y,
    int orig_cols, int out_features, int words_per_row,
    cudaStream_t stream
) {
    int blocks = (out_features + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int threads = WARPS_PER_BLOCK * WARP_SIZE; // 256
    packed_binary_gemv_fp16_kernel<<<blocks, threads, 0, stream>>>(
        X, W_packed, Y, orig_cols, out_features, words_per_row
    );
}

void launch_packed_binary_gemm_fp16(
    const __half*   X,
    const uint32_t* W_packed,
    __half*         Y,
    int B, int orig_cols, int out_features, int words_per_row,
    cudaStream_t stream
) {
    if (B == 1) {
        launch_packed_binary_gemv_fp16(
            X, W_packed, Y, orig_cols, out_features, words_per_row, stream
        );
        return;
    }
    const int THREADS = 256;
    dim3 grid((out_features + THREADS - 1) / THREADS, B);
    packed_binary_gemm_fp16_kernel<<<grid, THREADS, 0, stream>>>(
        X, W_packed, Y, B, orig_cols, out_features, words_per_row
    );
}

void launch_packed_binary_gemm_fp32(
    const float*    X,
    const uint32_t* W_packed,
    float*          Y,
    int B, int orig_cols, int out_features, int words_per_row,
    cudaStream_t stream
) {
    const int THREADS = 256;
    dim3 grid((out_features + THREADS - 1) / THREADS, B);
    packed_binary_gemm_fp32_kernel<<<grid, THREADS, 0, stream>>>(
        X, W_packed, Y, B, orig_cols, out_features, words_per_row
    );
}

} // extern "C"
