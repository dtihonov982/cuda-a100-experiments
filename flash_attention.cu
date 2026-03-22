#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); return 1; } \
} while(0)

// Simplified single-head FlashAttention.
// Q, K, V, O: (seq_len, head_dim)
//
// Design constraints (to keep the code teachable):
//   - TILE == HEAD_DIM: each thread handles one head dimension AND one score per tile.
//   - seq_len must be divisible by TILE.
//   - No batch dimension, no masking, no dropout.
//
// Each thread block handles one query vector.
// Thread d (0..HEAD_DIM-1) is responsible for output dimension d.

#define HEAD_DIM 64
#define TILE     64   // must equal HEAD_DIM in this implementation

__global__ void flash_attention(
        const float *Q, const float *K, const float *V, float *O,
        int seq_len, float scale) {

    int qi = blockIdx.x;   // which query position this block handles
    int d  = threadIdx.x;  // head dimension index (0..HEAD_DIM-1)

    int warp = d / 32, lane = d % 32;

    // -----------------------------------------------------------------------
    // Shared memory layout
    //   q_s           — query vector (broadcast to all threads)
    //   k_s[j][d]     — key tile:   row j, dim d
    //   v_s[j][d]     — value tile: row j, dim d
    //   p_s[j]        — softmax numerators exp(s[j] - tile_max) for this tile
    //   sm[2]         — scratch for 2-warp cross-warp reductions
    //
    // k_s has +1 column padding to break shared memory bank conflicts.
    // Without it, all 64 threads reading column d of the same row would
    // hit the same bank (stride 64 × 4 bytes = 256 bytes = 8 × 32 banks).
    // -----------------------------------------------------------------------
    __shared__ float q_s[HEAD_DIM];
    __shared__ float k_s[TILE][HEAD_DIM + 1];
    __shared__ float v_s[TILE][HEAD_DIM];
    __shared__ float p_s[TILE];
    __shared__ float sm[2];

    // Load this query into shared memory once — reused across all K/V tiles
    q_s[d] = Q[qi * HEAD_DIM + d];
    __syncthreads();

    // -----------------------------------------------------------------------
    // Running state (private registers — never touch global or shared memory)
    //   o — unnormalized output for dimension d:  Σ exp(sⱼ - m) * V[j][d]
    //   m — running max of all scores seen so far
    //   l — running normalizer:                   Σ exp(sⱼ - m)
    // -----------------------------------------------------------------------
    float o = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    int num_tiles = seq_len / TILE;

    for (int t = 0; t < num_tiles; t++) {

        // --- Load one tile of K and V into shared memory ---
        // Thread d loads column d for every row in the tile.
        // Access pattern: consecutive threads read consecutive floats → coalesced.
        for (int j = 0; j < TILE; j++) {
            k_s[j][d] = K[(t * TILE + j) * HEAD_DIM + d];
            v_s[j][d] = V[(t * TILE + j) * HEAD_DIM + d];
        }
        __syncthreads();

        // --- Score: thread d computes s[d] = (q · k[d]) * scale ---
        // Each thread owns one score (for the d-th key in this tile).
        // The dot product loops serially over HEAD_DIM — all reads from shared memory.
        float s = 0.0f;
        for (int i = 0; i < HEAD_DIM; i++)
            s += q_s[i] * k_s[d][i];
        s *= scale;

        // --- Tile max: 2-warp reduction ---
        float wmax = s;
        for (int off = 16; off > 0; off >>= 1)
            wmax = fmaxf(wmax, __shfl_xor_sync(0xffffffff, wmax, off));
        if (lane == 0) sm[warp] = wmax;
        __syncthreads();
        float tile_max = fmaxf(sm[0], sm[1]);

        // --- Softmax numerators ---
        float e = expf(s - tile_max);
        p_s[d] = e;   // store for the output accumulation loop below

        // --- Tile sum: 2-warp reduction ---
        float wsum = e;
        for (int off = 16; off > 0; off >>= 1)
            wsum += __shfl_xor_sync(0xffffffff, wsum, off);
        if (lane == 0) sm[warp] = wsum;
        __syncthreads();                  // also guarantees p_s is fully written
        float tile_sum = sm[0] + sm[1];

        // --- Online softmax update ---
        //
        // All scores seen so far were computed relative to old max m.
        // The new tile has local max tile_max. Combined max is new_m.
        //
        // Correction factors rescale old and new contributions to the same base:
        //   c_old = exp(m       - new_m)   → scale down old running state
        //   c_new = exp(tile_max - new_m)   → scale down new tile's contributions
        //
        // This is the same trick as softmax_online.cu, applied tile-by-tile.
        float new_m = fmaxf(m, tile_max);
        float c_old = expf(m        - new_m);
        float c_new = expf(tile_max - new_m);

        // Update normalizer
        float new_l = c_old * l + c_new * tile_sum;

        // Update output for dimension d:
        //   new_o = c_old * o  +  c_new * Σⱼ p_s[j] * V[j][d]
        //                                   ^^^^^^^^^^^^^^^^^^^
        //                           this tile's weighted V contribution
        float new_o = c_old * o;
        for (int j = 0; j < TILE; j++)
            new_o += c_new * p_s[j] * v_s[j][d];

        m = new_m;
        l = new_l;
        o = new_o;

        __syncthreads();  // before k_s / v_s / p_s are overwritten next iteration
    }

    // Divide by the normalizer to get the final softmax-weighted output
    O[qi * HEAD_DIM + d] = o / l;
}

// CPU reference: standard 3-step attention (materializes the N×N matrix)
void attention_cpu(const float *Q, const float *K, const float *V, float *O,
                   int seq_len, int head_dim, float scale) {
    float *S = (float*)malloc(seq_len * seq_len * sizeof(float));

    // S = Q @ Kᵀ * scale
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < seq_len; j++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += Q[i*head_dim+d] * K[j*head_dim+d];
            S[i*seq_len+j] = dot * scale;
        }

    // Softmax each row of S, then O = softmax(S) @ V
    for (int i = 0; i < seq_len; i++) {
        float mx = S[i*seq_len];
        for (int j = 1; j < seq_len; j++) mx = fmaxf(mx, S[i*seq_len+j]);
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) { S[i*seq_len+j] = expf(S[i*seq_len+j]-mx); sum += S[i*seq_len+j]; }
        for (int j = 0; j < seq_len; j++) S[i*seq_len+j] /= sum;

        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int j = 0; j < seq_len; j++)
                acc += S[i*seq_len+j] * V[j*head_dim+d];
            O[i*head_dim+d] = acc;
        }
    }
    free(S);
}

int main() {
    const int seq_len  = 512;
    const int head_dim = HEAD_DIM;
    const float scale  = 1.0f / sqrtf((float)head_dim);

    size_t sz = seq_len * head_dim * sizeof(float);

    float *h_Q   = (float*)malloc(sz);
    float *h_K   = (float*)malloc(sz);
    float *h_V   = (float*)malloc(sz);
    float *h_O   = (float*)malloc(sz);
    float *h_ref = (float*)malloc(sz);

    srand(42);
    for (int i = 0; i < seq_len * head_dim; i++) {
        h_Q[i] = (float)rand()/RAND_MAX - 0.5f;
        h_K[i] = (float)rand()/RAND_MAX - 0.5f;
        h_V[i] = (float)rand()/RAND_MAX - 0.5f;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, sz));
    CUDA_CHECK(cudaMalloc(&d_K, sz));
    CUDA_CHECK(cudaMalloc(&d_V, sz));
    CUDA_CHECK(cudaMalloc(&d_O, sz));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, sz, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // seq_len blocks, HEAD_DIM threads each
    cudaEventRecord(start);
    flash_attention<<<seq_len, HEAD_DIM>>>(d_Q, d_K, d_V, d_O, seq_len, scale);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    CUDA_CHECK(cudaMemcpy(h_O, d_O, sz, cudaMemcpyDeviceToHost));

    // Verify against CPU reference (first 8 rows)
    attention_cpu(h_Q, h_K, h_V, h_ref, seq_len, head_dim, scale);
    float max_err = 0.0f;
    for (int i = 0; i < seq_len * head_dim; i++) {
        float err = fabsf(h_O[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max error vs CPU: %e %s\n", max_err, max_err < 1e-4f ? "(OK)" : "(FAIL)");

    // Memory traffic comparison
    // Flash:  reads Q+K+V (3×seq×d), writes O (1×seq×d) = 4×seq×d floats
    // Naive:  additionally reads+writes the seq×seq attention matrix (×2 = 2×seq²)
    double flash_bytes = 4.0 * seq_len * head_dim * sizeof(float);
    double naive_extra = 2.0 * seq_len * seq_len * sizeof(float);
    double bw = flash_bytes / (ms * 1e-3) / 1e9;

    printf("\nseq=%d  head_dim=%d\n", seq_len, head_dim);
    printf("Time:         %.3f ms\n", ms);
    printf("Flash BW:     %.1f GB/s  (only Q/K/V/O traffic)\n", bw);
    printf("\nMemory flash avoids vs naive:\n");
    printf("  seq×seq attention matrix = %.1f KB  (at seq=%d)\n",
           naive_extra/1024.0, seq_len);
    printf("  At seq=2048 this would be %.1f MB per layer\n",
           2.0*2048*2048*sizeof(float)/1024.0/1024.0);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_ref);
    return 0;
}
