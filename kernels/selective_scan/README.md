# Selective scan CUDA extension

This directory contains the oflex implementation used by both the STMamba encoder and decoder. Unused core/ndstate/nrow variants and benchmark scripts are omitted.

```bash
pip install --no-build-isolation ./kernels/selective_scan
```

Install a CUDA-compatible PyTorch build and CUDA toolkit first. Set `SELECTIVE_SCAN_CXX_STANDARD` to override the compiler standard. See the root README and LICENSES for environment details and upstream attribution.
