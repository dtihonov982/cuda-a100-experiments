#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); return 1; } \
} while(0)

// ----------------------------------------------------------------------------
// Warp reduction helpers (same pattern as layernorm / softmax)
// ----------------------------------------------------------------------------

__device__ float warp_reduce_max(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));
    return val;
}

// ----------------------------------------------------------------------------
// Quantize kernel: float → int8, per-row scales
//
// For each row:
//   1. Find max absolute value (reduction) → compute scale
//   2. Divide every element by scale and round to nearest integer
//
// Why per-row?
//   A weight matrix might have one row with values in [-10, 10] and another
//   in [-0.01, 0.01]. A single global scale would waste 10 bits of precision
//   on the small row. Per-row scales let each row use the full [-127, 127]
//   range independently.
// ----------------------------------------------------------------------------
__global__ void quantize(const float *X, int8_t *Q, float *scales,
                         int rows, int cols) {
    int row    = blockIdx.x;
    int tid    = threadIdx.x;
    int warp   = tid / 32;
    int lane   = tid % 32;
    int nwarps = blockDim.x / 32;

    extern __shared__ float smem[];  // one slot per warp

    // --- Step 1: find max absolute value in this row ---
    float local_max = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        local_max = fmaxf(local_max, fabsf(X[row * cols + i]));

    local_max = warp_reduce_max(local_max);

    if (lane == 0) smem[warp] = local_max;
    __syncthreads();

    float row_max = 0.0f;
    if (warp == 0) {
        float v = (lane < nwarps) ? smem[lane] : 0.0f;
        row_max = warp_reduce_max(v);
        if (lane == 0) smem[0] = row_max;
    }
    __syncthreads();
    row_max = smem[0];

    // scale maps the float range to [-127, 127]
    // We avoid ±128 to keep symmetric quantization (easier dequant, no special cases)
    float scale     = row_max / 127.0f;
    float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;

    if (tid == 0) scales[row] = scale;

    // --- Step 2: quantize ---
    // __float2int_rn: convert float to int with round-to-nearest
    // Clamp to [-127, 127] to guard against floating point edge cases
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = X[row * cols + i] * inv_scale;
        int   ival = __float2int_rn(val);
        // clamp
        ival = ival >  127 ?  127 : ival;
        ival = ival < -127 ? -127 : ival;
        Q[row * cols + i] = (int8_t)ival;
    }
}

// ----------------------------------------------------------------------------
// Dequantize kernel: int8 + scales → float
//
// Simply multiply each element by its row's scale.
// This is what a W8A16 GEMM kernel does on-the-fly to each weight tile
// before accumulating into the output.
// ----------------------------------------------------------------------------
__global__ void dequantize(const int8_t *Q, const float *scales, float *X,
                            int rows, int cols) {
    int row = blockIdx.x;
    float scale = scales[row];

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        X[row * cols + i] = (float)Q[row * cols + i] * scale;
}

int main() {
    // Typical weight matrix: one linear layer in a 7B model
    // e.g. LLaMA-7B: hidden=4096, intermediate=11008
    const int rows = 4096;
    const int cols = 4096;

    size_t sz       = (size_t)rows * cols * sizeof(float);
    size_t sz_q     = (size_t)rows * cols * sizeof(int8_t);
    size_t sz_scale = (size_t)rows       * sizeof(float);

    float   *h_X      = (float*)  malloc(sz);
    int8_t  *h_Q      = (int8_t*) malloc(sz_q);
    float   *h_scales = (float*)  malloc(sz_scale);
    float   *h_deq    = (float*)  malloc(sz);  // dequantized result

    // Simulate a weight matrix: normal-ish distribution with occasional outliers
    // (outliers are common in LLM weights, especially in attention projections)
    srand(42);
    for (int i = 0; i < rows * cols; i++) {
        float v = (float)rand()/RAND_MAX * 2.0f - 1.0f;
        // Add 1% outliers (10× larger) — common in transformer weights
        if (rand() % 100 == 0) v *= 10.0f;
        h_X[i] = v;
    }

    float   *d_X, *d_deq, *d_scales;
    int8_t  *d_Q;
    CUDA_CHECK(cudaMalloc(&d_X,      sz));
    CUDA_CHECK(cudaMalloc(&d_Q,      sz_q));
    CUDA_CHECK(cudaMalloc(&d_scales, sz_scale));
    CUDA_CHECK(cudaMalloc(&d_deq,    sz));

    CUDA_CHECK(cudaMemcpy(d_X, h_X, sz, cudaMemcpyHostToDevice));

    int threads = 256;
    int nwarps  = threads / 32;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Quantize ---
    cudaEventRecord(start);
    quantize<<<rows, threads, nwarps * sizeof(float)>>>(d_X, d_Q, d_scales, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_q = 0;
    cudaEventElapsedTime(&ms_q, start, stop);

    // --- Dequantize (to measure roundtrip error) ---
    cudaEventRecord(start);
    dequantize<<<rows, threads>>>(d_Q, d_scales, d_deq, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_dq = 0;
    cudaEventElapsedTime(&ms_dq, start, stop);

    CUDA_CHECK(cudaMemcpy(h_Q,      d_Q,      sz_q,     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_scales, d_scales, sz_scale, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_deq,    d_deq,    sz,       cudaMemcpyDeviceToHost));

    // --- Measure quantization error ---
    double sum_err = 0.0, max_err = 0.0, sum_sq_orig = 0.0;
    for (int i = 0; i < rows * cols; i++) {
        double err = fabs((double)h_deq[i] - (double)h_X[i]);
        sum_err   += err;
        max_err    = err > max_err ? err : max_err;
        sum_sq_orig += (double)h_X[i] * (double)h_X[i];
    }
    double mean_err = sum_err / ((double)rows * cols);
    // Relative error: mean absolute error / RMS of original weights
    double rms_orig = sqrt(sum_sq_orig / ((double)rows * cols));
    double rel_err  = mean_err / rms_orig * 100.0;

    printf("Quantization error (float -> int8 -> float):\n");
    printf("  Max absolute error:  %.6f\n", max_err);
    printf("  Mean absolute error: %.6f\n", mean_err);
    printf("  Relative error:      %.4f%%  (of weight RMS)\n", rel_err);

    // --- Memory savings ---
    double mb_fp32 = (double)rows * cols * sizeof(float) / 1024.0 / 1024.0;
    double mb_int8 = (double)rows * cols * sizeof(int8_t) / 1024.0 / 1024.0;
    double mb_scales = (double)rows * sizeof(float) / 1024.0 / 1024.0;

    printf("\nMemory for %dx%d weight matrix:\n", rows, cols);
    printf("  FP32:              %.1f MB\n", mb_fp32);
    printf("  INT8 + scales:     %.1f MB  (%.1fx smaller)\n",
           mb_int8 + mb_scales, mb_fp32 / (mb_int8 + mb_scales));

    // --- Bandwidth ---
    // Quantize: reads FP32, writes INT8 + scales ≈ reads + 1/4 writes
    double q_bytes = (double)rows * cols * (sizeof(float) + sizeof(int8_t));
    printf("\nQuantize time:   %.3f ms  (%.1f GB/s effective)\n",
           ms_q, q_bytes / (ms_q * 1e-3) / 1e9);
    printf("Dequantize time: %.3f ms\n", ms_dq);

    // --- Show a few sample values ---
    printf("\nSample row 0 (first 8 weights):\n");
    printf("  scale = %.6f\n", h_scales[0]);
    printf("  orig:  ");
    for (int i = 0; i < 8; i++) printf("%7.4f ", h_X[i]);
    printf("\n  int8:  ");
    for (int i = 0; i < 8; i++) printf("%7d ", (int)h_Q[i]);
    printf("\n  deq:   ");
    for (int i = 0; i < 8; i++) printf("%7.4f ", h_deq[i]);
    printf("\n");

    cudaFree(d_X); cudaFree(d_Q); cudaFree(d_scales); cudaFree(d_deq);
    free(h_X); free(h_Q); free(h_scales); free(h_deq);
    return 0;
}
