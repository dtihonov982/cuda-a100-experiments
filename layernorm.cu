#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); return 1; } \
} while(0)

// ----------------------------------------------------------------------------
// Warp reductions (same pattern as softmax)
// ----------------------------------------------------------------------------

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ----------------------------------------------------------------------------
// Layer norm kernel
//
// One block per row. Each block:
//   Pass 1 — compute sum and sum-of-squares in one sweep → mean, variance
//   Pass 2 — normalize and apply learned scale (gamma) and shift (beta)
//
// Why gamma and beta?
//   Pure normalization destroys the layer's ability to represent the identity
//   function. gamma and beta let the network "undo" normalization if needed.
//   In transformers, gamma and beta are vectors of length cols (hidden dim).
// ----------------------------------------------------------------------------
__global__ void layernorm(const float *X, float *Y,
                          const float *gamma, const float *beta,
                          int rows, int cols, float eps) {
    int row    = blockIdx.x;
    if (row >= rows) return;

    const float *x = X + row * cols;
    float       *y = Y + row * cols;

    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane    = tid % 32;
    int nwarps  = blockDim.x / 32;

    // Shared memory: two arrays (sum, sum_sq), one slot per warp
    extern __shared__ float smem[];
    float *smem_sum   = smem;
    float *smem_sumsq = smem + nwarps;

    // -----------------------------------------------------------------------
    // Pass 1: accumulate sum and sum-of-squares in a single loop
    //
    // sum   = Σ x[i]
    // sumsq = Σ x[i]²
    //
    // From these two scalars we can derive:
    //   mean = sum / N
    //   var  = sumsq / N - mean²   (variance via E[X²] - E[X]²)
    //
    // This avoids a second pass to compute (x - mean)² for every element.
    // -----------------------------------------------------------------------
    float local_sum = 0.0f, local_sumsq = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = x[i];
        local_sum   += v;
        local_sumsq += v * v;
    }

    // Warp-level reduction
    local_sum   = warp_reduce_sum(local_sum);
    local_sumsq = warp_reduce_sum(local_sumsq);

    if (lane == 0) {
        smem_sum[warp_id]   = local_sum;
        smem_sumsq[warp_id] = local_sumsq;
    }
    __syncthreads();

    // Cross-warp reduction in first warp
    float row_mean = 0.0f, row_var = 0.0f;
    if (warp_id == 0) {
        float s  = (lane < nwarps) ? smem_sum[lane]   : 0.0f;
        float sq = (lane < nwarps) ? smem_sumsq[lane] : 0.0f;
        s  = warp_reduce_sum(s);
        sq = warp_reduce_sum(sq);
        if (lane == 0) {
            smem_sum[0]   = s  / cols;            // mean
            smem_sumsq[0] = sq / cols - (s / cols) * (s / cols);  // var
        }
    }
    __syncthreads();
    row_mean = smem_sum[0];
    row_var  = smem_sumsq[0];

    // Precompute 1 / sqrt(var + eps) — shared across all elements in this row
    float inv_std = rsqrtf(row_var + eps);

    // -----------------------------------------------------------------------
    // Pass 2: normalize and apply gamma / beta
    //
    // y[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i]
    //
    // gamma and beta are per-feature (per-column), not per-row.
    // -----------------------------------------------------------------------
    for (int i = tid; i < cols; i += blockDim.x)
        y[i] = gamma[i] * (x[i] - row_mean) * inv_std + beta[i];
}

// CPU reference
void layernorm_cpu(const float *X, float *Y,
                   const float *gamma, const float *beta,
                   int rows, int cols, float eps) {
    for (int r = 0; r < rows; r++) {
        const float *x = X + r * cols;
        float       *y = Y + r * cols;
        float sum = 0.0f, sumsq = 0.0f;
        for (int i = 0; i < cols; i++) { sum += x[i]; sumsq += x[i] * x[i]; }
        float mean    = sum / cols;
        float var     = sumsq / cols - mean * mean;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < cols; i++)
            y[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    }
}

int main() {
    // Typical transformer hidden dim: 4096 (LLaMA-7B), 8192 (LLaMA-70B)
    // Rows: batch_size * seq_len
    const int rows = 1024;
    const int cols = 4096;
    const float eps = 1e-5f;

    size_t sz     = (size_t)rows * cols * sizeof(float);
    size_t sz_col = cols * sizeof(float);

    float *h_X     = (float*)malloc(sz);
    float *h_Y     = (float*)malloc(sz);
    float *h_ref   = (float*)malloc(sz);
    float *h_gamma = (float*)malloc(sz_col);
    float *h_beta  = (float*)malloc(sz_col);

    srand(42);
    for (int i = 0; i < rows * cols; i++)
        h_X[i] = (float)rand() / RAND_MAX * 4.0f - 2.0f;
    // Init gamma=1, beta=0 (identity — easy to verify)
    for (int i = 0; i < cols; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }

    float *d_X, *d_Y, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_X,     sz));
    CUDA_CHECK(cudaMalloc(&d_Y,     sz));
    CUDA_CHECK(cudaMalloc(&d_gamma, sz_col));
    CUDA_CHECK(cudaMalloc(&d_beta,  sz_col));

    CUDA_CHECK(cudaMemcpy(d_X,     h_X,     sz,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, sz_col, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta,  h_beta,  sz_col, cudaMemcpyHostToDevice));

    int threads = 256;
    int nwarps  = threads / 32;
    int smem    = 2 * nwarps * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    layernorm<<<rows, threads, smem>>>(d_X, d_Y, d_gamma, d_beta, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, sz, cudaMemcpyDeviceToHost));

    // Verify first 4 rows
    layernorm_cpu(h_X, h_ref, h_gamma, h_beta, 4, cols, eps);
    float max_err = 0.0f;
    for (int i = 0; i < 4 * cols; i++) {
        float err = fabsf(h_Y[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max error vs CPU: %e %s\n", max_err, max_err < 1e-4f ? "(OK)" : "(FAIL)");

    // Memory traffic: X read twice (pass1 + pass2), gamma+beta read once each, Y written once
    // = (2*rows*cols + 2*cols + rows*cols) * 4 bytes ≈ 3 * rows * cols * 4 bytes
    double bytes = (3.0 * rows * cols + 2.0 * cols) * sizeof(float);
    double bw    = bytes / (ms * 1e-3) / 1e9;

    printf("\nShape:  %d rows x %d cols (hidden dim)\n", rows, cols);
    printf("Time:   %.3f ms\n", ms);
    printf("BW:     %.1f GB/s  (A100 peak ~2000 GB/s => %.1f%%)\n",
           bw, bw / 2000.0 * 100.0);

    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_gamma); cudaFree(d_beta);
    free(h_X); free(h_Y); free(h_ref); free(h_gamma); free(h_beta);
    return 0;
}
