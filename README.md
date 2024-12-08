# Generalized Matrix Multiply Benchmarking

This repo exists to create a reproducible benchmarking suite for harware-accelerated matrix multiplies.

## Host Hardware Specifications

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

### Build and Test
```
git clone https://github.com/ccmagruder/gemm.git /root
cd /root/gemm
docker compose run --rm --build devcontainer
cmake -S gemm -B build
cmake --build build
ctest --test-dir build
```

### IDE
```
docker compose build devcontainer
docker commpose build ide
docker compose up -d ide
docker exec ide zsh
...
exit
docker compose down ide
```

### Commit

Feature branches should be prefixed with `feature/`, e.g. `feature/add_f16_benchmark`.
Hotfix branches should be prefixed with `hotfix/`, e.g. `hotfix/patch_ci_build_failure`.
Hotfix PRs should only be used to fix `main` when unstable; all other changes are features.

```
git clone https://github.com/ccmagruder/gemm.git
cd gemm
git checkout -b feature/{feature description}
git push --set-upstream origin feature/{feature description}
git add {files}
git commit -S -m "{commit message}"
git push
```

### Merging

The `main` branch is protected and linear history is enforced by squash-rebasing branches on the trunk.

## GitHub Actions Runner

The GitHub Actions Runner is hosted within a docker instance running on the Host

```
echo $TOKEN > .gh_pat
docker compose build devcontainer
docker compose build runner
docker compose up runner
```
