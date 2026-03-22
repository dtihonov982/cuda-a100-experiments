#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); return 1; } \
} while(0)

// ----------------------------------------------------------------------------
// Online softmax reduction
//
// The naive version made 3 passes over X: one for max, one for exp-sum,
// one for output. Here we fuse the first two into one pass.
//
// Key idea: maintain a (max, sum) pair and update it as we see new values.
// When a new element exceeds the current max, we rescale the running sum:
//
//   new_sum = old_sum * exp(old_max - new_max) + exp(x - new_max)
//
// This is mathematically equivalent to the two-pass version. The rescaling
// factor exp(old_max - new_max) corrects all previously seen elements.
//
// To reduce (max, sum) pairs across threads we use the same butterfly
// pattern as before, but combining two pairs instead of two scalars.
// ----------------------------------------------------------------------------

// Combine two (max, sum) pairs from different threads into one.
// This is the core operation of the online reduction.
__device__ void combine(float m1, float s1, float m2, float s2,
                        float *m_out, float *s_out) {
    if (m1 >= m2) {
        *m_out = m1;
        *s_out = s1 + s2 * expf(m2 - m1);
    } else {
        *m_out = m2;
        *s_out = s2 + s1 * expf(m1 - m2);
    }
}

// Warp-level reduction over (max, sum) pairs using shuffle
__device__ void warp_reduce_online(float *m, float *s) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_xor_sync(0xffffffff, *m, offset);
        float s2 = __shfl_xor_sync(0xffffffff, *s, offset);
        combine(*m, *s, m2, s2, m, s);
    }
}

__global__ void softmax_online(float *X, float *Y, int rows, int cols) {
    int row    = blockIdx.x;
    if (row >= rows) return;

    float *x = X + row * cols;
    float *y = Y + row * cols;

    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane    = tid % 32;
    int nwarps  = blockDim.x / 32;

    // Shared memory: two arrays (max and sum), one slot per warp
    extern __shared__ float smem[];
    float *smem_max = smem;
    float *smem_sum = smem + nwarps;

    // -----------------------------------------------------------------------
    // Pass 1: single-pass online reduction to get (row_max, row_sum)
    //
    // Each thread maintains its own (local_max, local_sum) pair and updates
    // it as it strides through the row. This replaces the two separate loops
    // in the naive version (one for max, one for exp-sum).
    // -----------------------------------------------------------------------
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    for (int i = tid; i < cols; i += blockDim.x) {
        float v = x[i];
        // Update running (max, sum) with the new value v.
        // exp(v - v) = exp(0) = 1, so the new element contributes 1 to sum
        // after normalization by the new max.
        combine(local_max, local_sum, v, 1.0f, &local_max, &local_sum);
    }

    // Reduce within warp
    warp_reduce_online(&local_max, &local_sum);

    // Warp 0 of each warp writes result to shared memory
    if (lane == 0) {
        smem_max[warp_id] = local_max;
        smem_sum[warp_id] = local_sum;
    }
    __syncthreads();

    // First warp reduces across all warps
    float row_max = -INFINITY, row_sum = 0.0f;
    if (warp_id == 0) {
        float m = (lane < nwarps) ? smem_max[lane] : -INFINITY;
        float s = (lane < nwarps) ? smem_sum[lane] : 0.0f;
        warp_reduce_online(&m, &s);
        if (lane == 0) { smem_max[0] = m; smem_sum[0] = s; }
    }
    __syncthreads();
    row_max = smem_max[0];
    row_sum = smem_sum[0];

    // -----------------------------------------------------------------------
    // Pass 2: write output (one read of X, one write of Y)
    // expf is now called once per element instead of twice.
    // -----------------------------------------------------------------------
    for (int i = tid; i < cols; i += blockDim.x)
        y[i] = expf(x[i] - row_max) / row_sum;
}

void softmax_cpu(float *X, float *Y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *x = X + r * cols;
        float *y = Y + r * cols;
        float mx = x[0];
        for (int i = 1; i < cols; i++) mx = fmaxf(mx, x[i]);
        float s = 0.0f;
        for (int i = 0; i < cols; i++) s += expf(x[i] - mx);
        for (int i = 0; i < cols; i++) y[i] = expf(x[i] - mx) / s;
    }
}

int main() {
    const int rows = 1024;
    const int cols = 1024;

    size_t sz = rows * cols * sizeof(float);
    float *h_X   = (float*)malloc(sz);
    float *h_Y   = (float*)malloc(sz);
    float *h_ref = (float*)malloc(sz);

    srand(42);
    for (int i = 0; i < rows * cols; i++)
        h_X[i] = (float)rand() / RAND_MAX * 4.0f - 2.0f;

    float *d_X, *d_Y;
    CUDA_CHECK(cudaMalloc(&d_X, sz));
    CUDA_CHECK(cudaMalloc(&d_Y, sz));
    CUDA_CHECK(cudaMemcpy(d_X, h_X, sz, cudaMemcpyHostToDevice));

    int threads = 256;
    int nwarps  = threads / 32;
    int smem    = 2 * nwarps * sizeof(float);  // max array + sum array

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    softmax_online<<<rows, threads, smem>>>(d_X, d_Y, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, sz, cudaMemcpyDeviceToHost));

    softmax_cpu(h_X, h_ref, 4, cols);
    float max_err = 0.0f;
    for (int i = 0; i < 4 * cols; i++) {
        float err = fabsf(h_Y[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max error vs CPU: %e %s\n", max_err, max_err < 1e-5f ? "(OK)" : "(FAIL)");

    // Theoretical minimum: 2 passes (1 read for reduction + 1 read+write for output)
    // = 3 * rows * cols * 4 bytes
    double bytes = 3.0 * rows * cols * sizeof(float);
    double bw    = bytes / (ms * 1e-3) / 1e9;

    printf("\nShape:  %d rows x %d cols\n", rows, cols);
    printf("Time:   %.3f ms\n", ms);
    printf("BW:     %.1f GB/s  (A100 peak ~2000 GB/s => %.1f%%)\n",
           bw, bw / 2000.0 * 100.0);
    printf("\nnaive was 128 GB/s (4 passes) — compare here.\n");

    cudaFree(d_X); cudaFree(d_Y);
    free(h_X); free(h_Y); free(h_ref);
    return 0;
}
