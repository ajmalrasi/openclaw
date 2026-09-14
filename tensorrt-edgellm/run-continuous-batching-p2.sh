#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

task_dir=/home/ajmalrasi/continuous-batching-p2-20260914
reference_dir=/home/ajmalrasi/TensorRT-Edge-LLM
engine_dir=/home/ajmalrasi/Qwen3.5-4B/engines/llm-b2-input6144-kv8192-vanilla
checkpoint_dir=/home/ajmalrasi/Qwen3.5-4B/quantized
model_unit=openclaw-tensorrt-edgellm.service
timer_unit=openclaw-tensorrt-watchdog.timer
watchdog_unit=openclaw-tensorrt-watchdog.service
export TRT_PACKAGE_DIR=/usr
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/targets/sbsa-linux/lib
export EDGELLM_PLUGIN_PATH="$reference_dir/build/libNvInfer_edgellm_plugin.so"

test "$(git -C "$reference_dir" rev-parse HEAD)" = e8b29522938901f6df19ebeedd4b69bc8edbcd97
test -f "$task_dir/include/runtime/llmRankRuntime.h"
test -f "$task_dir/continuousBatchingProbe.cpp"
systemctl --user is-active --quiet "$model_unit"
timer_active=false
if systemctl --user is-active --quiet "$timer_unit"; then
    timer_active=true
fi
curl --fail --silent --max-time 10 http://127.0.0.1:11434/v1/models |
    jq -e '.data[] | select(.id == "openclaw" and .owned_by == "tensorrt-edgellm")'
curl --fail --silent --max-time 10 http://127.0.0.1:11434/health |
    jq -e '.status == "healthy" and .active_requests == 0 and .queued_requests == 0'
date -Is
git -C "$reference_dir" diff -- experimental/server/config.py experimental/server/runtime/engine.py > "$task_dir/preserved-engine-dir.patch"
sha256sum "$engine_dir/config.json" "$task_dir/continuousBatchingProbe.cpp" "$task_dir/include/runtime/llmRankRuntime.h"
free -m

restore() {
    task_result=$?
    trap - EXIT INT TERM
    set +e
    echo "RESTORE begin task_exit=$task_result"
    systemctl --user start "$model_unit"
    restored=false
    for ((attempt = 0; attempt < 90; ++attempt)); do
        if curl --fail --silent --max-time 2 http://127.0.0.1:11434/health > "$task_dir/restored-health.json"; then
            if jq -e '.status == "healthy"' "$task_dir/restored-health.json"; then
                restored=true
                break
            fi
        fi
        sleep 2
    done
    if "$timer_active"; then
        systemctl --user start "$timer_unit"
    fi
    systemctl --user is-active "$model_unit" "$timer_unit"
    free -m
    date -Is
    echo "RESTORE healthy=$restored task_exit=$task_result"
    if ! "$restored"; then
        exit 90
    fi
    exit "$task_result"
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

systemctl --user stop "$timer_unit" "$watchdog_unit"
systemctl --user stop "$model_unit"
test "$(systemctl --user show -p MainPID --value "$model_unit")" = 0
echo 'MAINTENANCE model stopped; watchdog paused'
free -m

# Runtime layout is unchanged. Compile the new step/state implementations beside the pinned archive.
# Reuse its CUDA device-link object without changing the production runtime.
/usr/bin/aarch64-linux-gnu-g++ -DTRT_EDGELLM_CUDA_LIBRARY_T_COMPAT \
    -Wno-deprecated-declarations -Wall -Werror -Wno-error=unused-parameter -O2 -DNDEBUG -std=gnu++17 \
    -I"$task_dir/include" -I"$reference_dir/cpp" -I"$reference_dir/3rdParty/nlohmannJson/include" \
    -I"$reference_dir/3rdParty/stb" -I"$reference_dir/3rdParty/miniaudio" \
    -isystem /usr/local/cuda/targets/sbsa-linux/include \
    -c "$task_dir/continuousBatchingProbe.cpp" -o "$task_dir/continuousBatchingProbe.o"
for task_source in sequenceStepRuntime sequenceSlots; do
    /usr/bin/aarch64-linux-gnu-g++ -DTRT_EDGELLM_CUDA_LIBRARY_T_COMPAT \
        -Wno-deprecated-declarations -Wall -Werror -Wno-error=unused-parameter -O2 -DNDEBUG -std=gnu++17 \
        -I"$task_dir/include" -I"$reference_dir/cpp" -I"$reference_dir/3rdParty/nlohmannJson/include" \
        -I"$reference_dir/3rdParty/stb" -I"$reference_dir/3rdParty/miniaudio" \
        -isystem /usr/local/cuda/targets/sbsa-linux/include \
        -c "$task_dir/$task_source.cpp" -o "$task_dir/$task_source.o"
done
/usr/bin/aarch64-linux-gnu-g++ \
    -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/cuda/targets/sbsa-linux/lib/stubs \
    -Wl,--unresolved-symbols=ignore-in-shared-libs -Wl,-rpath,/usr/local/cuda/targets/sbsa-linux/lib \
    "$task_dir/continuousBatchingProbe.o" \
    "$task_dir/sequenceStepRuntime.o" "$task_dir/sequenceSlots.o" \
    "$reference_dir/build/examples/llm/CMakeFiles/llm_inference.dir/cmake_device_link.o" \
    "$reference_dir/build/cpp/libedgellmCore.a" "$reference_dir/build/examples/utils/libexampleUtils.a" \
    "$reference_dir/build/cpp/libedgellmCore.a" \
    "$reference_dir/cpp/kernels/cuteDSLArtifact/aarch64/sm_87/libcutedsl_aarch64.a" \
    -lcuda /usr/lib/aarch64-linux-gnu/libnvonnxparser.so /usr/lib/aarch64-linux-gnu/libnvinfer.so \
    /usr/local/cuda/targets/sbsa-linux/lib/libcudart.so -ldl -lcudadevrt -lcudart_static -lrt -lpthread -ldl \
    -o "$task_dir/continuous_batching_probe"
echo 'BUILD passed'
date -Is
# One deadline includes loading and every functional fixture; this is not a throughput benchmark.
timeout --signal=TERM --kill-after=5s 260s \
    "$task_dir/continuous_batching_probe" "$engine_dir" "$checkpoint_dir" --steps
