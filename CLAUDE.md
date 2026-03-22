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
- [x] hello_cuda.cu — device query, vector add, CUDA_CHECK macro, cudaEvents
- [ ] gemm_naive.cu — naive matrix multiply, 2D thread indexing, global memory bottleneck

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
