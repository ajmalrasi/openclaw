#!/usr/bin/env bash
set -euo pipefail
task_dir=/home/ajmalrasi/json-schema-phase4h-20260923
source_dir=/home/ajmalrasi/TensorRT-Edge-LLM
reference_dir=/home/ajmalrasi/TensorRT-Edge-LLM
checkpoint_dir=/home/ajmalrasi/Qwen3.5-4B/quantized
engine_dir=/home/ajmalrasi/Qwen3.5-4B/engines/llm-b2-input6144-kv8192-vanilla
exec > >(tee -a "$task_dir/performance.log") 2>&1
date -Is
echo PHASE4H_SCOPE=existing-binding-memory-baseline
test "$(systemctl --user show -p MainPID --value openclaw-tensorrt-edgellm.service)" = 0
export PYTHONUNBUFFERED=1
unset EDGELLM_PYBIND_DIR
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
  echo "PHASE4H_EXIT=$result"
  free -m
  date -Is
  exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cd "$source_dir"
timeout --signal=TERM --kill-after=5s 285s \
  "$reference_dir/.venv/bin/python" -m experimental.server "$checkpoint_dir" \
  --engine-dir "$engine_dir" --host 127.0.0.1 --port 11435 \
  --served-model-name openclaw-json-candidate --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml --enable-auto-tool-choice \
  --max-queued-requests 8 --queue-timeout 30 \
  > "$task_dir/candidate-server.log" 2>&1 &
echo $! > "$task_dir/candidate.pid"
healthy=false
for ((attempt = 0; attempt < 90; ++attempt)); do
  if curl --fail --silent --max-time 2 http://127.0.0.1:11435/health | jq -e '.status == "healthy"' >/dev/null; then
    healthy=true
    break
  fi
  sleep 1
done
test "$healthy" = true
server_pid=$(pgrep -P "$(cat "$task_dir/candidate.pid")" -f 'python -m experimental.server' | head -n 1)
test -n "$server_pid"
export CANDIDATE_SERVER_PID="$server_pid"
timeout --signal=TERM --kill-after=5s 180s "$reference_dir/.venv/bin/python" "$task_dir/baseline_memory.py"
curl --fail --silent --max-time 5 http://127.0.0.1:11435/health | jq -e '.status == "healthy" and .active_requests == 0'
echo PHASE4H_BASELINE_GATE=passed
