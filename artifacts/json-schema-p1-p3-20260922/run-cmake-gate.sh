#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
task_dir=/home/ajmalrasi/json-schema-p1-p3-20260922
source_dir="$task_dir/source"
build_dir="$task_dir/cmake-check"
xgrammar_dir=/home/ajmalrasi/xgrammar-phase0-20260922/xgrammar
exec > >(tee -a "$task_dir/cmake-gate.log") 2>&1
export TRT_PACKAGE_DIR=/usr
date -Is
cmake -E remove_directory "$build_dir"
cmake -S "$source_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDA_CTK_VERSION=13.2 \
  -DCUDA_DIR=/usr/local/cuda/targets/sbsa-linux \
  -DTRT_PACKAGE_DIR=/usr \
  -DENABLE_XGRAMMAR=ON \
  -DXGRAMMAR_ROOT="$xgrammar_dir" \
  -DBUILD_UNIT_TESTS=OFF
cmake --build "$build_dir" --target edgellmXGrammar --parallel 2
test -f "$build_dir/cpp/libedgellmXGrammar.a"
echo CMAKE_XGRAMMAR_GATE=passed
date -Is
