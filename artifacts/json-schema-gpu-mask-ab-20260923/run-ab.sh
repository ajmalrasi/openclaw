#!/usr/bin/env bash
set -euo pipefail
task_dir=/home/ajmalrasi/json-schema-gpu-mask-ab-20260923
old_source=/home/ajmalrasi/tensorrt-json-schema-release-2cb8bb8/source
old_binding=/home/ajmalrasi/tensorrt-json-schema-release-2cb8bb8/pybind
new_source=/home/ajmalrasi/json-schema-phase4a-20260923/source
new_binding=/home/ajmalrasi/json-schema-phase4a-20260923/build/pybind
reference_dir=/home/ajmalrasi/TensorRT-Edge-LLM
checkpoint_dir=/home/ajmalrasi/Qwen3.5-4B/quantized
engine_dir=/home/ajmalrasi/Qwen3.5-4B/engines/llm-b2-input6144-kv8192-vanilla
service=openclaw-tensorrt-edgellm.service
timer=openclaw-tensorrt-watchdog.timer
exec > >(tee -a "$task_dir/ab.log") 2>&1
date -Is
echo GPU_MASK_AB_SCOPE=same-engine-same-workload
test -s "$new_binding/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so"
test -s "$old_binding/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so"
test -f "$task_dir/benchmark.py"
test -f "$task_dir/endpoint_gate.py"
curl --fail --silent --max-time 5 http://127.0.0.1:11434/health | jq -e '.status == "healthy" and .active_requests == 0 and .queued_requests == 0'
service_was_active=false
timer_was_active=false
candidate_pid=
if systemctl --user is-active --quiet "$service"; then service_was_active=true; fi
if systemctl --user is-active --quiet "$timer"; then timer_was_active=true; fi
stop_candidate() {
  if [[ -n "$candidate_pid" ]]; then
    kill -TERM "$candidate_pid" 2>/dev/null || true
    for ((attempt = 0; attempt < 50; ++attempt)); do
      kill -0 "$candidate_pid" 2>/dev/null || break
      sleep .2
    done
    kill -KILL "$candidate_pid" 2>/dev/null || true
    candidate_pid=
  fi
}
restore() {
  result=$?
  trap - EXIT INT TERM
  stop_candidate
  if [[ "$service_was_active" == true ]]; then systemctl --user start "$service" || result=1; fi
  if [[ "$timer_was_active" == true ]]; then systemctl --user start "$timer" || result=1; fi
  if [[ "$service_was_active" == true ]]; then
    healthy=false
    for ((attempt = 0; attempt < 90; ++attempt)); do
      if curl --fail --silent --max-time 2 http://127.0.0.1:11434/health | jq -e '.status == "healthy"' >/dev/null; then
        healthy=true
        break
      fi
      sleep 1
    done
    if [[ "$healthy" != true ]]; then result=1; fi
  fi
  systemctl --user show "$service" -p ActiveState -p MainPID --no-pager
  systemctl --user show "$timer" -p ActiveState --no-pager
  free -m
  echo "GPU_MASK_AB_EXIT=$result"
  date -Is
  exit "$result"
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
systemctl --user stop "$timer"
systemctl --user stop "$service"
export PYTHONUNBUFFERED=1
export EDGELLM_CONTINUOUS_BATCHING=1
export EDGELLM_CONTINUOUS_GRAPHS=0
export EDGELLM_PLUGIN_PATH="$reference_dir/build/libNvInfer_edgellm_plugin.so"
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/targets/sbsa-linux/lib
run_mode() {
  mode=$1
  source_dir=$2
  binding_dir=$3
  export EDGELLM_PYBIND_DIR="$binding_dir"
  cd "$source_dir"
  timeout --signal=TERM --kill-after=5s 110s \
    "$reference_dir/.venv/bin/python" -m experimental.server "$checkpoint_dir" \
    --engine-dir "$engine_dir" --host 127.0.0.1 --port 11435 \
    --served-model-name openclaw-json-candidate --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml --enable-auto-tool-choice \
    --max-queued-requests 8 --queue-timeout 30 \
    > "$task_dir/$mode-server.log" 2>&1 &
  candidate_pid=$!
  healthy=false
  for ((attempt = 0; attempt < 60; ++attempt)); do
    if curl --fail --silent --max-time 2 http://127.0.0.1:11435/health | jq -e '.status == "healthy"' >/dev/null; then
      healthy=true
      break
    fi
    if ! kill -0 "$candidate_pid" 2>/dev/null; then break; fi
    sleep 1
  done
  test "$healthy" = true
  server_pid=$(pgrep -P "$candidate_pid" -f 'python -m experimental.server' | head -n 1)
  test -n "$server_pid"
  export CANDIDATE_SERVER_PID="$server_pid"
  echo "GPU_MASK_MODE=$mode SERVER_PID=$server_pid"
  if [[ "$mode" == gpu ]]; then
    timeout --signal=TERM --kill-after=5s 60s \
      "$reference_dir/.venv/bin/python" "$task_dir/endpoint_gate.py"
  fi
  timeout --signal=TERM --kill-after=5s 65s \
    "$reference_dir/.venv/bin/python" "$task_dir/benchmark.py" | tee "$task_dir/$mode-benchmark.jsonl"
  curl --fail --silent --max-time 5 http://127.0.0.1:11435/health | jq -e '.status == "healthy" and .active_requests == 0'
  stop_candidate
  sleep 1
}
run_mode cpu "$old_source" "$old_binding"
run_mode gpu "$new_source" "$new_binding"
echo GPU_MASK_AB_GATE=passed
