# Project: CUDA Learning for LLM Research

## Who the user is
Complete beginner to CUDA. Background is in LLM research (not systems/GPU programming).
The goal is to build practical GPU kernel knowledge relevant to transformer workloads.

## Learning goal
Work up from first principles to writing and optimizing CUDA kernels that matter for LLMs:
matrix multiply (GEMM), softmax, attention, layer norm, quantization, etc.

The path is: **understand → implement naive → profile → optimize**.
Never skip to an optimized version without first understanding why the naive one is slow.

## How to assist
- One small step at a time. Don't introduce multiple new concepts at once.
- Always explain *why* something matters for LLMs specifically.
- After writing a kernel, always show how to measure its performance (CUDA events or `ncu`).
- Prefer correctness + clarity over cleverness. Comments should explain GPU concepts, not restate code.
- When introducing a new concept (shared memory, warp, occupancy, etc.), define it plainly before using it.

## Current progress

### GEMM (matrix multiply)
- [x] gemm_naive.cu — 1 thread = 1 output, global memory only. 14.7% A100 utilization.
- [x] gemm_tiled.cu — 16×16 shared memory tile. 20.9% utilization.
- [x] gemm_coarsened.cu — 4×4 register tile per thread (thread coarsening). 42.5% utilization.

### Reductions (per-row)
- [x] softmax_naive.cu — 3-pass: max, exp-sum, output. Warp shuffle reduction. 128 GB/s.
- [x] softmax_online.cu — Online softmax: fused max+sum in one pass. 152 GB/s.
- [x] layernorm.cu — Single-pass mean+variance (sum + sum_sq), rsqrtf, γ/β. 315 GB/s.

### Attention
- [x] flash_attention.cu — Tiled attention with online softmax, avoids N×N matrix. Single head, seq=512, head_dim=64.

### Quantization
- [x] quantize.cu — Per-row INT8 quantization + dequantization. 4× memory reduction. ~0.5% relative error.
- [ ] gemm_w8a16.cu — W8A16 GEMM: int8 weights × fp32 activations, dequantize on-the-fly.

## Concepts introduced so far
- 2D thread/block indexing, cudaEvents timing
- Global memory coalescing, shared memory, bank conflicts (+1 padding)
- Register tiling, #pragma unroll, arithmetic intensity, roofline model
- Warp shuffles (__shfl_xor_sync), parallel reduction, 2-warp cross-reduction
- Online softmax (running max/sum correction), Welford-style single-pass stats
- rsqrtf, __float2int_rn, per-row quantization scales

## Environment
- GPU: NVIDIA A100 80GB, compute capability 8.0
- CUDA toolkit: 12.5
- Compile: `nvcc -O2 -arch=sm_80 -o <name> <name>.cu`
- nvcc is not on PATH by default — check README or ask user how to invoke it

## Key A100 numbers to keep in mind
- FP32 peak: ~19.5 TFLOPS
- FP16/BF16 peak: ~77.5 TFLOPS (Tensor Cores)
- Memory bandwidth: ~2 TB/s
- Shared memory per SM: 164 KB (configurable up to 164 KB)
- SMs: 108
