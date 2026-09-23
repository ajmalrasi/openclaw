#!/usr/bin/env bash
set -euo pipefail
task_dir=/home/ajmalrasi/json-schema-phase4i-20260923
dropin_dir=/home/ajmalrasi/.config/systemd/user/openclaw-tensorrt-edgellm.service.d
dropin_file="$dropin_dir/zz-json-schema-experiment.conf"
service=openclaw-tensorrt-edgellm.service
timer=openclaw-tensorrt-watchdog.timer
exec > >(tee -a "$task_dir/cutover.log") 2>&1
date -Is
echo PHASE4I_SCOPE=reversible-experimental-cutover
test -s "$task_dir/zz-json-schema-experiment.conf"
test -s /home/ajmalrasi/json-schema-phase4a-20260923/build/pybind/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so
test -s /home/ajmalrasi/Qwen3.5-4B/engines/llm-b2-input6144-kv8192-vanilla/llm.engine
test ! -e "$dropin_file"
test "$(systemctl --user show -p MainPID --value "$service")" = 0
if ss -ltn | grep -q ':11434 '; then
  echo PORT_11434_OCCUPIED=1
  exit 1
fi
systemctl --user cat "$service" > "$task_dir/pre-cutover-unit.txt"
systemctl --user show "$service" -p ActiveState -p UnitFileState > "$task_dir/pre-cutover-state.txt"
systemctl --user show "$timer" -p ActiveState -p UnitFileState > "$task_dir/pre-cutover-timer.txt"
free -m
committed=false
timer_was_active=false
if systemctl --user is-active --quiet "$timer"; then
  timer_was_active=true
  systemctl --user stop "$timer"
fi
cleanup() {
  result=$?
  trap - EXIT INT TERM
  if [[ "$committed" != true ]]; then
    echo PHASE4I_ROLLBACK=starting
    systemctl --user stop "$service" || true
    if [[ -e "$dropin_file" ]]; then
      mv "$dropin_file" "$task_dir/rolled-back-zz-json-schema-experiment.conf"
    fi
    systemctl --user daemon-reload || true
    systemctl --user reset-failed "$service" || true
    echo PHASE4I_ROLLBACK=completed
  fi
  if [[ "$timer_was_active" == true ]]; then
    systemctl --user start "$timer" || true
  fi
  systemctl --user show "$service" -p ActiveState -p MainPID -p ExecMainStatus --no-pager
  systemctl --user show "$timer" -p ActiveState --no-pager
  free -m
  echo "PHASE4I_EXIT=$result"
  date -Is
  exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
install -m 0644 "$task_dir/zz-json-schema-experiment.conf" "$dropin_file"
systemctl --user daemon-reload
systemctl --user start "$service"
healthy=false
for ((attempt = 0; attempt < 90; ++attempt)); do
  if curl --fail --silent --max-time 2 http://127.0.0.1:11434/health | jq -e '.status == "healthy"' > "$task_dir/health.json"; then
    healthy=true
    break
  fi
  if ! systemctl --user is-active --quiet "$service"; then
    echo PHASE4I_SERVICE_EXITED_BEFORE_HEALTH=1
    break
  fi
  sleep 1
done
test "$healthy" = true
curl --fail --silent --max-time 5 http://127.0.0.1:11434/v1/models | tee "$task_dir/models.json"
timeout --signal=TERM --kill-after=5s 120s \
  /home/ajmalrasi/TensorRT-Edge-LLM/.venv/bin/python "$task_dir/live_endpoint_gate.py"
curl --fail --silent --max-time 5 http://127.0.0.1:11434/health | tee "$task_dir/final-health.json"
test "$(jq -r '.active_requests' "$task_dir/final-health.json")" = 0
server_pid=$(systemctl --user show -p MainPID --value "$service")
grep -E '^(VmRSS|VmHWM|VmSwap):' "/proc/$server_pid/status"
echo PHASE4I_EXPERIMENTAL_CUTOVER=passed
committed=true
