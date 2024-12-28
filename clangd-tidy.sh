#! /bin/bash

# Clangd by default will not run "expensive" checks. So we need two config
# files: one for the language server; the other for the CI.
# Because clangd-tidy does not support separate paths for config files,
# we break these into separate files and do this moving hack.
cp .clangd .clangd-orig
echo "Diagnostics:" >> .clangd
echo "  ClangTidy:" >> .clangd 
echo "    FastCheckFilter: None" >> .clangd

clangd-tidy -p build --fail-on-severity warn \
    gemm/**/*.h \
    gemm/**/*.cpp \
    gemm/**/*.cuh \
    gemm/**/*.cu
EXIT=$?

mv .clangd-orig .clangd
exit $EXIT
