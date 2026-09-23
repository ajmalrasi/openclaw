#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

task_dir=/home/ajmalrasi/json-schema-phase4b-20260923
source_dir=/home/ajmalrasi/json-schema-phase4a-20260923/source
build_dir=/home/ajmalrasi/json-schema-phase4a-20260923/build
reference_dir=/home/ajmalrasi/TensorRT-Edge-LLM
xgrammar_dir=/home/ajmalrasi/xgrammar-phase0-20260922/xgrammar

exec > >(tee -a "$task_dir/binding-build.log") 2>&1
date -Is
echo "SOURCE_COMMIT=2cb8bb85086b6eb26bfa6fbb7f7703730d66abac"
echo "PHASE4B_SCOPE=isolated-binding-build-and-import"
test "$(systemctl --user show -p MainPID --value openclaw-tensorrt-edgellm.service)" = 0
test -s "$build_dir/cpp/libedgellmCore.a"

available_mib=$(awk '/MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
cores=$(nproc)
if ((available_mib < 4096 || cores < 2)); then
  echo "Insufficient headroom for requested multicore build: available_mib=$available_mib cores=$cores" >&2
  exit 2
fi
build_jobs=2
if ((available_mib >= 6144 && cores >= 3)); then
  build_jobs=3
fi
echo "BUILD_JOBS=$build_jobs AVAILABLE_MIB=$available_mib CORES=$cores"
free -m

export TRT_PACKAGE_DIR=/usr
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/targets/sbsa-linux/lib
pybind11_dir=$("$reference_dir/.venv/bin/python" -m pybind11 --cmakedir)
cmake -S "$source_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDA_CTK_VERSION=13.2 \
  -DCUDA_DIR=/usr/local/cuda/targets/sbsa-linux \
  -DTRT_PACKAGE_DIR=/usr \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUTE_DSL_ARTIFACT_TAG=sm_87 \
  -DENABLE_XGRAMMAR=ON \
  -DXGRAMMAR_ROOT="$xgrammar_dir" \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPython_EXECUTABLE="$reference_dir/.venv/bin/python" \
  -Dpybind11_DIR="$pybind11_dir"
cmake --build "$build_dir" --target _edgellm_runtime --parallel "$build_jobs"
test -s "$build_dir/pybind/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so"
EDGELLM_PYBIND_DIR="$build_dir/pybind" PYTHONPATH="$source_dir" \
  "$reference_dir/.venv/bin/python" -c 'from experimental.server.runtime.engine import _import_runtime; r = _import_runtime(); assert hasattr(r.ContinuousSequenceOptions(), "json_schema"); print("BINDING_IMPORT=passed", r.__file__)'
echo PHASE4B_BINDING_BUILD=passed
date -Is
