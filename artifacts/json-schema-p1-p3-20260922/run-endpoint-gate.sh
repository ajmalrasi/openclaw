#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

task_dir=/home/ajmalrasi/json-schema-p1-p3-20260922
source_dir="$task_dir/source"
build_dir="$task_dir/build"
reference_dir=/home/ajmalrasi/TensorRT-Edge-LLM
checkpoint_dir=/home/ajmalrasi/Qwen3.5-4B/quantized
engine_dir=${ENGINE_DIR:-/home/ajmalrasi/Qwen3.5-4B/engines/llm-b2-input6144-kv8192-vanilla}
log="$task_dir/endpoint-gate.log"

exec > >(tee -a "$log") 2>&1
date -Is
grep -q 'BUILD_GATE=passed' "$task_dir/build.log"
test "$(systemctl --user show -p MainPID --value openclaw-tensorrt-edgellm.service)" = 0
test ! -e /proc/"$(cat "$task_dir/candidate.pid" 2>/dev/null || echo 0)"
export PYTHONUNBUFFERED=1
export EDGELLM_PYBIND_DIR="$build_dir"
export EDGELLM_CONTINUOUS_BATCHING=1
export EDGELLM_CONTINUOUS_GRAPHS=0
export EDGELLM_PLUGIN_PATH="$reference_dir/build/libNvInfer_edgellm_plugin.so"
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/targets/sbsa-linux/lib

cleanup() {
  result=$?
  trap - EXIT INT TERM
  if [[ -f "$task_dir/candidate.pid" ]]; then
    candidate_pid=$(cat "$task_dir/candidate.pid")
    kill -TERM "$candidate_pid" 2>/dev/null || true
    for ((attempt = 0; attempt < 50; ++attempt)); do
      kill -0 "$candidate_pid" 2>/dev/null || break
      sleep .2
    done
    kill -KILL "$candidate_pid" 2>/dev/null || true
  fi
  echo "ENDPOINT_GATE_EXIT=$result"
  free -m
  date -Is
  exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$source_dir"
timeout --signal=TERM --kill-after=5s 300s \
  "$reference_dir/.venv/bin/python" -m experimental.server "$checkpoint_dir" \
  --engine-dir "$engine_dir" --host 127.0.0.1 --port 11435 \
  --served-model-name openclaw-json-candidate --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml --enable-auto-tool-choice \
  --max-queued-requests 8 --queue-timeout 30 \
  > "$task_dir/candidate-server.log" 2>&1 &
echo $! > "$task_dir/candidate.pid"

healthy=false
for ((attempt = 0; attempt < 120; ++attempt)); do
  if curl --fail --silent --max-time 2 http://127.0.0.1:11435/health > "$task_dir/candidate-health.json"; then
    if jq -e '.status == "healthy"' "$task_dir/candidate-health.json"; then
      healthy=true
      break
    fi
  fi
  sleep 1
done
test "$healthy" = true
timeout --signal=TERM --kill-after=5s 240s \
  "$reference_dir/.venv/bin/python" "$task_dir/endpoint_gate.py"
curl --fail --silent --max-time 5 http://127.0.0.1:11435/health | tee "$task_dir/final-health.json"
echo ENDPOINT_GATE=passed
