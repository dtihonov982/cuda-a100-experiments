#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); return 1; } \
} while(0)

// ----------------------------------------------------------------------------
// Warp reduction helpers
//
// A "warp" is 32 threads that execute in lockstep on the GPU.
// __shfl_xor_sync lets a thread read a register from another thread in the
// same warp — no shared memory, no synchronization needed.
//
// The mask 0xffffffff means "all 32 threads participate".
// XOR with offset 16, 8, 4, 2, 1 is the butterfly pattern that reduces
// 32 values to 1 in 5 steps.
// ----------------------------------------------------------------------------

__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;  // all threads in the warp now hold the warp-wide max
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;  // all threads in the warp now hold the warp-wide sum
}

// ----------------------------------------------------------------------------
// Naive softmax kernel
//
// One thread block per row. Threads cooperate to compute:
//   1. max of the row  (for numerical stability — prevents exp() overflow)
//   2. sum of exp(x - max)
//   3. divide each element by the sum
//
// This version handles rows up to blockDim.x * 32 elements wide.
// We use shared memory to communicate partial results between warps.
// ----------------------------------------------------------------------------
__global__ void softmax_naive(float *X, float *Y, int rows, int cols) {
    // Each block handles one row
    int row = blockIdx.x;
    if (row >= rows) return;

    float *x = X + row * cols;
    float *y = Y + row * cols;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane    = tid % 32;     // position within the warp (0..31)
    int nwarps  = blockDim.x / 32;

    // Shared memory: one slot per warp for inter-warp communication
    extern __shared__ float smem[];  // size = nwarps floats

    // -----------------------------------------------------------------------
    // Step 1: find row max (needed for numerically stable exp)
    //
    // Why stability matters: exp(1000) = inf, but exp(1000 - 1000) = 1.
    // Subtracting the max before exp() keeps values in a safe range.
    // The result is mathematically identical: softmax(x) = softmax(x - c).
    // -----------------------------------------------------------------------
    float thread_max = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x)
        thread_max = fmaxf(thread_max, x[i]);

    // Reduce within each warp
    thread_max = warp_reduce_max(thread_max);

    // Lane 0 of each warp writes its warp's max to shared memory
    if (lane == 0) smem[warp_id] = thread_max;
    __syncthreads();

    // First warp reduces across all warp-maxes to get the block-wide max
    float row_max = -INFINITY;
    if (warp_id == 0) {
        float v = (lane < nwarps) ? smem[lane] : -INFINITY;
        row_max = warp_reduce_max(v);
        // Broadcast result so all threads can read it
        if (lane == 0) smem[0] = row_max;
    }
    __syncthreads();
    row_max = smem[0];

    // -----------------------------------------------------------------------
    // Step 2: compute sum of exp(x - max)
    // -----------------------------------------------------------------------
    float thread_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        thread_sum += expf(x[i] - row_max);

    thread_sum = warp_reduce_sum(thread_sum);

    if (lane == 0) smem[warp_id] = thread_sum;
    __syncthreads();

    float row_sum = 0.0f;
    if (warp_id == 0) {
        float v = (lane < nwarps) ? smem[lane] : 0.0f;
        row_sum = warp_reduce_sum(v);
        if (lane == 0) smem[0] = row_sum;
    }
    __syncthreads();
    row_sum = smem[0];

    // -----------------------------------------------------------------------
    // Step 3: write normalized output
    // -----------------------------------------------------------------------
    for (int i = tid; i < cols; i += blockDim.x)
        y[i] = expf(x[i] - row_max) / row_sum;
}

// CPU reference
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
    // Typical attention score shape: (batch*heads*seq, seq)
    // e.g. batch=8, heads=16, seq=1024 → 131072 rows of length 1024
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

    // One block per row, 256 threads per block
    // Shared memory: one float per warp = 256/32 = 8 floats
    int threads = 256;
    int nwarps  = threads / 32;
    int smem    = nwarps * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    softmax_naive<<<rows, threads, smem>>>(d_X, d_Y, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, sz, cudaMemcpyDeviceToHost));

    // Verify against CPU (first 4 rows)
    softmax_cpu(h_X, h_ref, 4, cols);
    float max_err = 0.0f;
    for (int i = 0; i < 4 * cols; i++) {
        float err = fabsf(h_Y[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max error vs CPU: %e %s\n", max_err, max_err < 1e-5f ? "(OK)" : "(FAIL)");

    // Bandwidth: we read X once, write Y once = 2 * rows * cols * 4 bytes
    double bytes = 2.0 * rows * cols * sizeof(float);
    double bw    = bytes / (ms * 1e-3) / 1e9;

    // A100 peak memory bandwidth: ~2000 GB/s
    printf("\nShape:  %d rows x %d cols\n", rows, cols);
    printf("Time:   %.3f ms\n", ms);
    printf("BW:     %.1f GB/s  (A100 peak ~2000 GB/s => %.1f%%)\n",
           bw, bw / 2000.0 * 100.0);

    cudaFree(d_X); cudaFree(d_Y);
    free(h_X); free(h_Y); free(h_ref);
    return 0;
}
