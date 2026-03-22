#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); return 1; } \
} while(0)

// Tile size: each thread block cooperates to load a TILE x TILE chunk
// of A and B into shared memory before computing.
#define TILE 16

// Tiled GEMM: C = A * B
// A is (M x K), B is (K x N), C is (M x N)
//
// Key idea: instead of each thread independently fetching from global memory,
// the whole block loads a TILE×TILE patch of A and B into shared memory,
// then all threads compute partial sums from the fast on-chip data.
// This is repeated for each tile along the K dimension.
__global__ void gemm_tiled(float *A, float *B, float *C, int M, int N, int K) {
    // Each thread is responsible for one output element C[row][col]
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // Shared memory tiles — live on-chip for the duration of this block's execution.
    // __shared__ means all 256 threads in this block share these arrays.
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    float sum = 0.0f;

    // Slide a TILE-wide window across the K dimension
    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t++) {

        // --- Phase 1: cooperatively load one tile of A and one tile of B ---
        // Each of the 256 threads loads exactly one element. Together they
        // fill the entire TILE×TILE shared memory array in one step.

        int aCol = t * TILE + threadIdx.x;  // column index into A
        int bRow = t * TILE + threadIdx.y;  // row index into B

        // Guard against out-of-bounds when M, N, K aren't multiples of TILE
        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        // --- Barrier: wait until ALL threads have written their element ---
        // Without this, a fast thread might start reading tileB before a slow
        // thread has written its value. __syncthreads() prevents that race.
        __syncthreads();

        // --- Phase 2: each thread accumulates its partial dot product ---
        // All reads are now from shared memory (~100x faster than global).
        for (int k = 0; k < TILE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // --- Barrier: wait before overwriting shared memory on next iteration ---
        // Without this, a fast thread could start loading the next tile and
        // clobber tileA/tileB before slow threads finish reading this tile.
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

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
    const int M = 1024, N = 1024, K = 1024;

    size_t sA = M * K * sizeof(float);
    size_t sB = K * N * sizeof(float);
    size_t sC = M * N * sizeof(float);

    float *h_A = (float*)malloc(sA);
    float *h_B = (float*)malloc(sB);
    float *h_C = (float*)malloc(sC);
    float *h_ref = (float*)malloc(sC);

    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX - 0.5f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sA));
    CUDA_CHECK(cudaMalloc(&d_B, sB));
    CUDA_CHECK(cudaMalloc(&d_C, sC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sB, cudaMemcpyHostToDevice));

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gemm_tiled<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sC, cudaMemcpyDeviceToHost));

    // Verify correctness against CPU
    printf("Running CPU reference on 64x64 corner...\n");
    const int CHECK = 64;
    gemm_cpu(h_A, h_B, h_ref, CHECK, CHECK, K);
    float max_err = 0.0f;
    for (int i = 0; i < CHECK; i++)
        for (int j = 0; j < CHECK; j++) {
            float err = fabsf(h_C[i * N + j] - h_ref[i * CHECK + j]);
            if (err > max_err) max_err = err;
        }
    printf("Max error vs CPU: %e %s\n", max_err, max_err < 1e-3f ? "(OK)" : "(FAIL)");

    double flops = 2.0 * M * N * K;
    double tflops = flops / (ms * 1e-3) / 1e12;

    printf("\nMatrix: %dx%d x %dx%d  (tile size: %dx%d)\n", M, K, K, N, TILE, TILE);
    printf("Time:   %.2f ms\n", ms);
    printf("Perf:   %.2f TFLOPS\n", tflops);
    printf("A100 peak FP32 ~19.5 TFLOPS  =>  %.1f%% utilization\n", tflops / 19.5 * 100);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
