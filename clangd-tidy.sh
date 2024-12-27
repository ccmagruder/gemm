#! /bin/bash

# Clangd by default will not run "expensive" checks. So we need two config
# files: one for the language server; the other for the CI.
# Because clangd-tidy does not support separate paths for config files,
# we break these into separate files and do this moving hack.
mv .clangd .clangd-orig
mv .clangd-ci .clangd

clangd-tidy -p build \
    gemm/**/*.h \
    gemm/**/*.cpp \
    gemm/**/*.cuh \
    gemm/**/*.cu

mv .clangd .clangd-ci
mv .clangd-orig .clangd
