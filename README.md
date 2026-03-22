# CUDA A100 Experiments

CUDA kernels for LLM-relevant GPU operations, built from first principles on an A100 80GB.

## Kernels

### GEMM
| File | Technique | A100 FP32 util |
|---|---|---|
| gemm_naive.cu | 1 thread = 1 output, all global memory | 14.7% |
| gemm_tiled.cu | 16×16 shared memory tile | 20.9% |
| gemm_coarsened.cu | 4×4 register tile per thread | 42.5% |

### Softmax / Layer Norm
| File | Technique | Bandwidth |
|---|---|---|
| softmax_naive.cu | 3-pass, warp shuffle reduction | 128 GB/s |
| softmax_online.cu | Online softmax, fused max+sum pass | 152 GB/s |
| layernorm.cu | Single-pass mean+variance, γ/β | 315 GB/s |

### Attention
| File | Technique | Note |
|---|---|---|
| flash_attention.cu | Tiled attention, online softmax, no N×N matrix | seq=512, head_dim=64 |

### Quantization
| File | Technique | Note |
|---|---|---|
| quantize.cu | Per-row INT8 quant + dequant | ~0.5% rel error, 4× memory reduction |

## Build

Binaries go into `build/`. Uses `make`:

```sh
make          # build all
make <name>   # build one, e.g. make gemm_naive
```

nvcc is not on PATH by default on this machine. Set it explicitly:

```sh
NVCC=/usr/local/cuda/bin/nvcc make
```

Always compile with `-arch=sm_80` to target the A100 and avoid PTX JIT issues. Without it, nvcc emits PTX that gets JIT-compiled at runtime — if toolkit/driver versions are mismatched this fails with:

> the provided PTX was compiled with an unsupported toolchain

## Environment

- GPU: NVIDIA A100 80GB
- Compute capability: 8.0 (`sm_80`)
- CUDA toolkit: 12.5
- FP32 peak: ~19.5 TFLOPS
- Memory bandwidth: ~2 TB/s
- Shared memory per SM: 164 KB
