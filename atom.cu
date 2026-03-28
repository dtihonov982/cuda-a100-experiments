// atom.cu — The smallest kernel we can read completely.
//
// Goal: see what ONE line of CUDA C++ becomes in PTX and SASS.
// This is the entry point to the rabbit hole.
//
// We will compile this THREE ways:
//   1. Normal:       nvcc -O2 -arch=sm_80 -o atom atom.cu
//   2. To PTX only:  nvcc -O2 -arch=sm_80 -ptx -o atom.ptx atom.cu
//   3. To SASS:      cuobjdump --dump-sass atom   (after normal compile)

#include <stdio.h>

// ─────────────────────────────────────────────────────────────
// The "atom" kernel — dead simple on purpose.
// Each thread adds 1.0 to ONE element.
//
// threadIdx.x  — the thread's index within its warp/block
// blockIdx.x   — which block this thread belongs to
// blockDim.x   — how many threads per block
//
// Together: each thread gets a unique global index → unique memory address.
// ─────────────────────────────────────────────────────────────
__global__ void atom(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}

int main() {
    const int N    = 1024;
    const int TPB  = 128;   // threads per block

    // Allocate and fill with 0.0f
    float *h = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h[i] = (float)i;

    float *d;
    cudaMalloc(&d, N * sizeof(float));
    cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice);

    atom<<<(N + TPB - 1) / TPB, TPB>>>(d, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h, d, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify: each h[i] should now be i + 1.0
    int errors = 0;
    for (int i = 0; i < N; i++)
        if (h[i] != (float)i + 1.0f) errors++;

    printf("atom kernel: %s (%d errors)\n", errors == 0 ? "CORRECT" : "WRONG", errors);

    cudaFree(d);
    free(h);
    return 0;
}
