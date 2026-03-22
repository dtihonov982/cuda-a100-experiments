#include <stdio.h>

__global__ void add(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Memory: %.0f GB\n", prop.totalGlobalMem / 1e9);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Compute: %d.%d\n\n", prop.major, prop.minor);

    // Vector addition: c = a + b
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);

    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);

    for (int i = 0; i < N; i++) { h_a[i] = i; h_b[i] = i * 2; }

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    add<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("FAIL at %d: got %d\n", i, h_c[i]);
            return 1;
        }
    }
    printf("Vector addition OK (%dM elements)\n", N >> 20);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
