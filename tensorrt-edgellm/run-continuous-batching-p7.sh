#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

task_dir=/home/ajmalrasi/continuous-batching-p7-20260915
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
test -f "$task_dir/source/cpp/runtime/llmInferenceRuntime.cpp"
for previous in continuous-batching-p6-attempt7.service; do
    if systemctl --user is-active --quiet "$previous"; then
        echo "Previous maintenance is still active"
        exit 2
    fi
done
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
sha256sum "$engine_dir/config.json"
free -m

restore() {
    task_result=$?
    trap - EXIT INT TERM
    set +e
    echo "RESTORE begin task_exit=$task_result"
    if [[ -f "$task_dir/candidate.pid" ]]; then
        candidate_pid=$(cat "$task_dir/candidate.pid")
        if [[ -r "/proc/$candidate_pid/cmdline" ]] && tr '\0' ' ' < "/proc/$candidate_pid/cmdline" | grep -q 'experimental.server'; then
            kill -TERM "$candidate_pid"
            for ((n = 0; n < 50; ++n)); do
                kill -0 "$candidate_pid" 2>/dev/null || break
                sleep .2
            done
            kill -KILL "$candidate_pid" 2>/dev/null || true
        fi
    fi
    if [[ "$(systemctl --user show -p Result --value "$model_unit")" == start-limit-hit ]]; then
        systemctl --user reset-failed "$model_unit"
    fi
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

mkdir -p "$task_dir/build"
cd "$task_dir/build"
cp "$reference_dir/build/cpp/libedgellmCore.a" libedgellmCore.a
for task_source in runtime/llmInferenceRuntime runtime/sequenceStepRuntime runtime/state/sequenceSlots runtime/state/sequencePolicy runtime/state/sequenceChannel runtime/continuousScheduler runtime/greedySchedulerBackend runtime/exec/engineExecutor; do
    task_name=$(basename "$task_source")
    if [[ ! -f "$task_name.cpp.o" || "$task_dir/source/cpp/$task_source.cpp" -nt "$task_name.cpp.o" ]]; then
    /usr/bin/aarch64-linux-gnu-g++ -DTRT_EDGELLM_CUDA_LIBRARY_T_COMPAT \
      -Wno-deprecated-declarations -Wall -Werror -Wno-error=unused-parameter -O2 -DNDEBUG -std=gnu++17 -fPIC \
      -I"$task_dir/source/cpp" -I"$reference_dir/3rdParty/nlohmannJson/include" \
      -I"$reference_dir/3rdParty/stb" -I"$reference_dir/3rdParty/miniaudio" \
      -isystem /usr/local/cuda/targets/sbsa-linux/include \
      -c "$task_dir/source/cpp/$task_source.cpp" -o "$task_name.cpp.o"
    fi
    ar r libedgellmCore.a "$task_name.cpp.o"
done
if [[ ! -f edgellm_pybind.cpp.o || "$task_dir/source/experimental/pybind/edgellm_pybind.cpp" -nt edgellm_pybind.cpp.o ]]; then
/usr/bin/aarch64-linux-gnu-g++ -DTRT_EDGELLM_CUDA_LIBRARY_T_COMPAT \
  -Wno-deprecated-declarations -Wall -Werror -Wno-error=unused-parameter -O1 -DNDEBUG -std=gnu++17 -fPIC -fvisibility=hidden \
  -I"$task_dir/source/cpp" -I"$reference_dir/examples/multimodal" \
  -I"$reference_dir/3rdParty/nlohmannJson/include" -I"$reference_dir/3rdParty/stb" \
  -isystem /usr/local/cuda/targets/sbsa-linux/include -isystem /usr/include/python3.12 \
  -isystem "$reference_dir/.venv/lib/python3.12/site-packages/pybind11/include" \
  -c "$task_dir/source/experimental/pybind/edgellm_pybind.cpp" -o edgellm_pybind.cpp.o
fi
/usr/bin/aarch64-linux-gnu-g++ -fPIC -shared \
  -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/cuda/targets/sbsa-linux/lib/stubs \
  -Wl,-rpath,/usr/local/cuda/targets/sbsa-linux/lib \
  -o _edgellm_runtime.cpython-312-aarch64-linux-gnu.so edgellm_pybind.cpp.o \
  "$reference_dir/build/experimental/pybind/CMakeFiles/_edgellm_runtime.dir/device_link_stub.cu.o" \
  "$reference_dir/build/experimental/pybind/CMakeFiles/_edgellm_runtime.dir/cmake_device_link.o" \
  -Wl,--whole-archive libedgellmCore.a -Wl,--no-whole-archive "$reference_dir/build/cpp/libedgellmBuilder.a" \
  /usr/lib/aarch64-linux-gnu/libnvinfer.so /usr/lib/aarch64-linux-gnu/libnvonnxparser.so \
  -lcuda /usr/local/cuda/targets/sbsa-linux/lib/libcudart.so \
  "$reference_dir/cpp/kernels/cuteDSLArtifact/aarch64/sm_87/libcutedsl_aarch64.a" \
  -ldl -lcudadevrt -lcudart_static -lrt -lpthread -ldl
/usr/bin/aarch64-linux-gnu-g++ -DTRT_EDGELLM_CUDA_LIBRARY_T_COMPAT \
  -Wno-deprecated-declarations -Wall -Werror -Wno-error=unused-parameter -O2 -DNDEBUG -std=gnu++17 \
  -I"$task_dir/source/cpp" -I"$reference_dir/3rdParty/nlohmannJson/include" \
  -isystem /usr/local/cuda/targets/sbsa-linux/include \
  -c "$task_dir/source/examples/llm/continuousBatchingProbe.cpp" -o probe.o
/usr/bin/aarch64-linux-gnu-g++ \
  -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/cuda/targets/sbsa-linux/lib/stubs \
  -Wl,-rpath,/usr/local/cuda/targets/sbsa-linux/lib -Wl,--unresolved-symbols=ignore-in-shared-libs \
  probe.o "$reference_dir/build/examples/llm/CMakeFiles/llm_inference.dir/cmake_device_link.o" \
  libedgellmCore.a "$reference_dir/build/examples/utils/libexampleUtils.a" libedgellmCore.a \
  "$reference_dir/cpp/kernels/cuteDSLArtifact/aarch64/sm_87/libcutedsl_aarch64.a" \
  -lcuda /usr/lib/aarch64-linux-gnu/libnvonnxparser.so /usr/lib/aarch64-linux-gnu/libnvinfer.so \
  /usr/local/cuda/targets/sbsa-linux/lib/libcudart.so -ldl -lcudadevrt -lcudart_static -lrt -lpthread -ldl \
  -o continuous_batching_probe
echo 'BUILD available'
date -Is
export EDGELLM_PYBIND_DIR="$task_dir/build"
export EDGELLM_CONTINUOUS_BATCHING=1
cd "$task_dir/source"
# Import test does not load the model.
"$reference_dir/.venv/bin/python" -c 'from experimental.server.runtime.engine import _import_runtime; r = _import_runtime(); print(r.__file__); assert hasattr(r, "ContinuousTicket")'
if [[ "${1:---build-only}" == --build-only ]]; then
    exit 0
fi
if [[ "${1:---build-only}" == --graphs ]]; then
    timeout --signal=TERM --kill-after=5s 90s "$task_dir/build/continuous_batching_probe" "$engine_dir" "$checkpoint_dir" --graphs
else
    timeout --signal=TERM --kill-after=5s 300s "$reference_dir/.venv/bin/python" "$task_dir/benchmark.py"
fi
