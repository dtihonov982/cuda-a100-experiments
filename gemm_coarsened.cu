#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); return 1; } \
} while(0)

// Block tile: each thread block covers a BM x BN patch of C
// Thread tile: each thread computes a WM x WN patch (using registers)
// Shared tile: BK is the tile width along the K dimension
#define BM 64   // block rows
#define BN 64   // block cols
#define BK 16   // tile depth along K
#define WM 4    // each thread's output rows
#define WN 4    // each thread's output cols
// Threads per block: (BM/WM) x (BN/WN) = 16 x 16 = 256

__global__ void gemm_coarsened(float *A, float *B, float *C, int M, int N, int K) {
    // Which output patch this thread is responsible for
    int outRow = threadIdx.y * WM;  // first row this thread writes (within the block tile)
    int outCol = threadIdx.x * WN;  // first col this thread writes

    // Registers: WM x WN accumulators — these stay in fast registers the entire kernel.
    // Unlike shared memory, registers are private to each thread.
    float sum[WM][WN] = {};

    // Shared memory tiles — same as before, but now bigger (64x16 each)
    __shared__ float tileA[BM][BK];
    __shared__ float tileB[BK][BN];

    // Linear thread id (0..255) used for cooperative loading
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int numTiles = (K + BK - 1) / BK;
    for (int t = 0; t < numTiles; t++) {

        // --- Cooperative load: all 256 threads fill tileA (64x16 = 1024 elements) ---
        // Each thread loads 4 elements by striding through the flat index.
        for (int i = 0; i < BM * BK / 256; i++) {  // = 4 iterations
            int idx = tid + i * 256;
            int r = idx / BK, c = idx % BK;
            int gRow = blockIdx.y * BM + r;
            int gCol = t * BK + c;
            tileA[r][c] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
        }

        // --- Cooperative load: all 256 threads fill tileB (16x64 = 1024 elements) ---
        for (int i = 0; i < BK * BN / 256; i++) {  // = 4 iterations
            int idx = tid + i * 256;
            int r = idx / BN, c = idx % BN;
            int gRow = t * BK + r;
            int gCol = blockIdx.x * BN + c;
            tileB[r][c] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }

        __syncthreads();  // wait for all loads before computing

        // --- Register tiling: for each position along BK ---
        // This is the key loop. Each iteration:
        //   1. Loads WM values from tileA into a small register array (a_reg)
        //   2. Loads WN values from tileB into a small register array (b_reg)
        //   3. Computes the outer product a_reg x b_reg and adds to sum[][]
        //
        // All reads are from shared memory, all writes to registers.
        // The outer product gives WM*WN = 16 FLOPs per shared memory read pair.
        for (int k = 0; k < BK; k++) {
            float a_reg[WM], b_reg[WN];

            for (int i = 0; i < WM; i++) a_reg[i] = tileA[outRow + i][k];
            for (int j = 0; j < WN; j++) b_reg[j] = tileB[k][outCol + j];

            // Outer product: every (i,j) pair gets one multiply-add
            for (int i = 0; i < WM; i++)
                for (int j = 0; j < WN; j++)
                    sum[i][j] += a_reg[i] * b_reg[j];
        }

        __syncthreads();  // wait before overwriting shared memory next iteration
    }

    // --- Write the WM x WN register tile back to global memory ---
    for (int i = 0; i < WM; i++) {
        for (int j = 0; j < WN; j++) {
            int gRow = blockIdx.y * BM + outRow + i;
            int gCol = blockIdx.x * BN + outCol + j;
            if (gRow < M && gCol < N)
                C[gRow * N + gCol] = sum[i][j];
        }
    }
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

    dim3 threads(BN / WN, BM / WM);  // 16 x 16 = 256
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gemm_coarsened<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sC, cudaMemcpyDeviceToHost));

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

    printf("\nMatrix: %dx%d x %dx%d\n", M, K, K, N);
    printf("Block tile: %dx%d  Thread tile: %dx%d  K-tile: %d\n", BM, BN, WM, WN, BK);
    printf("Time:   %.2f ms\n", ms);
    printf("Perf:   %.2f TFLOPS\n", tflops);
    printf("A100 peak FP32 ~19.5 TFLOPS  =>  %.1f%% utilization\n", tflops / 19.5 * 100);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
