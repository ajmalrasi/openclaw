#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

task_dir=/home/ajmalrasi/json-schema-p1-p3-20260922
source_dir="$task_dir/source"
build_dir="$task_dir/build"
reference_dir=/home/ajmalrasi/TensorRT-Edge-LLM
xgrammar_dir=/home/ajmalrasi/xgrammar-phase0-20260922/xgrammar
xgrammar_build=/home/ajmalrasi/xgrammar-phase0-20260922/build
log="$task_dir/build.log"
export TRT_PACKAGE_DIR=/usr
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/targets/sbsa-linux/lib

exec > >(tee -a "$log") 2>&1
date -Is
systemctl --user is-active openclaw-tensorrt-edgellm.service || true
test "$(systemctl --user show -p MainPID --value openclaw-tensorrt-edgellm.service)" = 0
test -f "$xgrammar_build/libxgrammar.a"
mkdir -p "$build_dir"
cp "$reference_dir/build/cpp/libedgellmCore.a" "$build_dir/libedgellmCore.a"
cd "$build_dir"

common=(
  -DTRT_EDGELLM_CUDA_LIBRARY_T_COMPAT
  -DTRT_EDGELLM_ENABLE_XGRAMMAR=1
  -Wno-deprecated-declarations -Wall -Werror -Wno-error=unused-parameter
  -O2 -DNDEBUG -std=gnu++17 -fPIC
  -I"$source_dir/cpp"
  -I"$reference_dir/3rdParty/nlohmannJson/include"
  -I"$reference_dir/3rdParty/stb"
  -I"$reference_dir/3rdParty/miniaudio"
  -I"$xgrammar_dir/include"
  -I"$xgrammar_dir/3rdparty/dlpack/include"
  -isystem /usr/local/cuda/targets/sbsa-linux/include
)

sources=(
  runtime/llmInferenceRuntime
  runtime/sequenceStepRuntime
  runtime/state/sequenceSlots
  runtime/state/sequencePolicy
  runtime/state/sequenceChannel
  runtime/continuousScheduler
  runtime/greedySchedulerBackend
  runtime/guidedDecoding
  runtime/exec/engineExecutor
)
for source in "${sources[@]}"; do
  object="$(basename "$source").cpp.o"
  /usr/bin/aarch64-linux-gnu-g++ "${common[@]}" \
    -c "$source_dir/cpp/$source.cpp" -o "$object"
  ar r libedgellmCore.a "$object"
done

/usr/bin/aarch64-linux-gnu-g++ "${common[@]}" -O1 -fvisibility=hidden \
  -I"$reference_dir/examples/multimodal" \
  -isystem /usr/include/python3.12 \
  -isystem "$reference_dir/.venv/lib/python3.12/site-packages/pybind11/include" \
  -c "$source_dir/experimental/pybind/edgellm_pybind.cpp" -o edgellm_pybind.cpp.o

/usr/bin/aarch64-linux-gnu-g++ -fPIC -shared \
  -L/usr/local/cuda/targets/sbsa-linux/lib \
  -L/usr/local/cuda/targets/sbsa-linux/lib/stubs \
  -Wl,-rpath,/usr/local/cuda/targets/sbsa-linux/lib \
  -o _edgellm_runtime.cpython-312-aarch64-linux-gnu.so edgellm_pybind.cpp.o \
  "$reference_dir/build/experimental/pybind/CMakeFiles/_edgellm_runtime.dir/device_link_stub.cu.o" \
  "$reference_dir/build/experimental/pybind/CMakeFiles/_edgellm_runtime.dir/cmake_device_link.o" \
  -Wl,--whole-archive libedgellmCore.a "$xgrammar_build/libxgrammar.a" -Wl,--no-whole-archive \
  "$reference_dir/build/cpp/libedgellmBuilder.a" \
  /usr/lib/aarch64-linux-gnu/libnvinfer.so /usr/lib/aarch64-linux-gnu/libnvonnxparser.so \
  -lcuda /usr/local/cuda/targets/sbsa-linux/lib/libcudart.so \
  "$reference_dir/cpp/kernels/cuteDSLArtifact/aarch64/sm_87/libcutedsl_aarch64.a" \
  -ldl -lcudadevrt -lcudart_static -lrt -lpthread -ldl

/usr/bin/aarch64-linux-gnu-g++ "${common[@]}" \
  -I"$xgrammar_dir/3rdparty/googletest/googletest/include" \
  "$source_dir/unittests/cpp/runtime/continuousSchedulerTest.cpp" \
  "$source_dir/unittests/cpp/runtime/state/sequencePolicyTest.cpp" \
  "$source_dir/cpp/runtime/continuousScheduler.cpp" \
  "$source_dir/cpp/runtime/state/sequenceSlots.cpp" \
  "$source_dir/cpp/runtime/state/sequencePolicy.cpp" \
  "$source_dir/cpp/runtime/state/sequenceChannel.cpp" \
  "$xgrammar_build/lib/libgtest_main.a" "$xgrammar_build/lib/libgtest.a" \
  -pthread -o json_schema_scheduler_tests
./json_schema_scheduler_tests --gtest_output=xml:"$task_dir/scheduler-tests.xml"

export EDGELLM_PYBIND_DIR="$build_dir"
cd "$source_dir"
"$reference_dir/.venv/bin/python" -c \
  'from experimental.server.runtime.engine import _import_runtime; r = _import_runtime(); print(r.__file__); assert hasattr(r.ContinuousSequenceOptions(), "json_schema")'
LLM_SDK_DIR="$source_dir" "$reference_dir/.venv/bin/python" -m pytest -q \
  -o addopts= \
  tests/python-unittests/test_structured_output.py \
  tests/python-unittests/test_continuous_http.py \
  tests/python-unittests/test_server_requests.py \
  tests/python-unittests/test_server_runtime.py \
  --junitxml="$task_dir/python-tests.xml"
echo BUILD_GATE=passed
date -Is
