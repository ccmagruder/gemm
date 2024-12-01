# Generalized Matrix Multiply Benchmarking

This repo exists to create a reproducible benchmarking suite for harware-accelerated matrix multiplies.

## Hardware Specifications

* Alienware Aurora R16
* Ubuntu 24.04 LTS (Noble)
* 13th Gen Intel Core i9-13900F (68MB, 24 cores, 32 threads, up to 5.60 GHz P-Core Thermal Velocity)
* NVIDIA GeForce RTX 4070, 12 GB GDDR6X
* 64 GB: 2 x 32 GB, DDR5, 5200 MT/s
* 1 TB (2 x 512 GB), M.2, PCIe, SSD


## Dependencies

1. CUDA 12.6.2
1. Intel MKL

## Contributing

```
git clone https://github.com/ccmagruder/gemm.git
```

### Branching

Feature branches should be prefixed with `feature/`, e.g. `feature/add_f16_benchmark`.
Hotfix branches should be prefixed with `hotfix/`, e.g. `hotfix/patch_ci_build_failure`.
Hotfix PRs should only be used to fix `main` when unstable; all other changes are features.

### Merging

The `main` branch is protected and linear history is enforced by squash-rebasing branches on the trunk.
