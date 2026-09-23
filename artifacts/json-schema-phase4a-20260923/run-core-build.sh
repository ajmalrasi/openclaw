#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

task_dir=/home/ajmalrasi/json-schema-phase4a-20260923
source_dir="$task_dir/source"
build_dir="$task_dir/build"
xgrammar_dir=/home/ajmalrasi/xgrammar-phase0-20260922/xgrammar

log_name=${PHASE4A_LOG:-core-build.log}
exec > >(tee -a "$task_dir/$log_name") 2>&1
date -Is
echo "SOURCE_COMMIT=2cb8bb85086b6eb26bfa6fbb7f7703730d66abac"
echo "PHASE4A_SCOPE=isolated-core-build-only"
test "$(systemctl --user show -p MainPID --value openclaw-tensorrt-edgellm.service)" = 0
test -f "$source_dir/cpp/runtime/guidedDecoding.cpp"
test -f "$xgrammar_dir/include/xgrammar/xgrammar.h"
free -m
available_mib=$(awk '/MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
build_jobs=1
if ((available_mib >= 4096)) && (( $(nproc) >= 2 )); then
  build_jobs=2
fi
echo "BUILD_JOBS=$build_jobs AVAILABLE_MIB=$available_mib"

export TRT_PACKAGE_DIR=/usr
cmake -S "$source_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDA_CTK_VERSION=13.2 \
  -DCUDA_DIR=/usr/local/cuda/targets/sbsa-linux \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUTE_DSL_ARTIFACT_TAG=sm_87 \
  -DTRT_PACKAGE_DIR=/usr \
  -DENABLE_XGRAMMAR=ON \
  -DXGRAMMAR_ROOT="$xgrammar_dir" \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_PYTHON_BINDINGS=OFF
cmake --build "$build_dir" --target edgellmCore --parallel "$build_jobs"
test -s "$build_dir/cpp/libedgellmCore.a"
echo PHASE4A_CORE_BUILD=passed
date -Is
