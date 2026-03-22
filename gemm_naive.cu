#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); return 1; } \
} while(0)

// Naive GEMM: C = A * B
// A is (M x K), B is (K x N), C is (M x N)
//
// Each thread computes ONE output element C[row][col].
// To do that it walks the full K dimension, loading one value from A and one from B per step.
// Every load goes to global memory — that's what makes this naive.
__global__ void gemm_naive(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // which output row
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // which output col

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// CPU reference so we can verify correctness
void gemm_cpu(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

int main() {
    // Square matrices — typical for transformer weight shapes
    const int M = 1024, N = 1024, K = 1024;

    size_t sA = M * K * sizeof(float);
    size_t sB = K * N * sizeof(float);
    size_t sC = M * N * sizeof(float);

    float *h_A = (float*)malloc(sA);
    float *h_B = (float*)malloc(sB);
    float *h_C = (float*)malloc(sC);      // GPU result
    float *h_ref = (float*)malloc(sC);    // CPU reference

    // Small random values so the dot products stay finite
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX - 0.5f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sA));
    CUDA_CHECK(cudaMalloc(&d_B, sB));
    CUDA_CHECK(cudaMalloc(&d_C, sC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sB, cudaMemcpyHostToDevice));

    // 16x16 thread block = 256 threads, same as hello_cuda but now 2D
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);

    // Time with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gemm_naive<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sC, cudaMemcpyDeviceToHost));

    // Verify against CPU (only check a sample — full check is O(N^3) slow)
    printf("Running CPU reference on 64x64 corner...\n");
    const int CHECK = 64;
    gemm_cpu(h_A, h_B, h_ref, CHECK, CHECK, K);
    float max_err = 0.0f;
    for (int i = 0; i < CHECK * CHECK; i++) {
        float err = fabsf(h_C[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max error vs CPU: %e %s\n", max_err, max_err < 1e-3f ? "(OK)" : "(FAIL)");

    // How fast are we vs peak?
    // Each output element does K multiplies + K adds = 2K FLOPs
    // Total: 2 * M * N * K FLOPs
    double flops = 2.0 * M * N * K;
    double tflops = flops / (ms * 1e-3) / 1e12;

    // A100 peak FP32: ~19.5 TFLOPS
    printf("\nMatrix: %dx%d x %dx%d\n", M, K, K, N);
    printf("Time:   %.2f ms\n", ms);
    printf("Perf:   %.2f TFLOPS\n", tflops);
    printf("A100 peak FP32 ~19.5 TFLOPS  =>  %.1f%% utilization\n", tflops / 19.5 * 100);
    printf("\nThis is intentionally bad. Shared memory tiling fixes it.\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
