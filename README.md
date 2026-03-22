# CUDA A100 Experiments

## Build

Always compile with an explicit architecture flag to target the A100 and avoid PTX JIT issues:

```sh
nvcc -arch=sm_80 -o hello_cuda hello_cuda.cu
```

Without `-arch=sm_80`, nvcc emits PTX that gets JIT-compiled by the driver at runtime. If the toolkit and driver versions are mismatched, this fails with:

> the provided PTX was compiled with an unsupported toolchain

`sm_80` emits native cubin for the A100 (compute capability 8.0), bypassing JIT entirely.

## Environment

- GPU: NVIDIA A100 80GB
- Compute capability: 8.0 (`sm_80`)
- CUDA toolkit: 12.5
