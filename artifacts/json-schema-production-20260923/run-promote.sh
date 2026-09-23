#!/usr/bin/env bash
set -euo pipefail
task_dir=/home/ajmalrasi/json-schema-production-20260923
release_dir=/home/ajmalrasi/tensorrt-json-schema-release-2cb8bb8
source_dir=/home/ajmalrasi/json-schema-phase4a-20260923/source
binding_file=/home/ajmalrasi/json-schema-phase4a-20260923/build/pybind/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so
dropin_dir=/home/ajmalrasi/.config/systemd/user/openclaw-tensorrt-edgellm.service.d
production_dropin="$dropin_dir/zz-json-schema-production.conf"
experiment_dropin="$dropin_dir/zz-json-schema-experiment.conf"
service=openclaw-tensorrt-edgellm.service
timer=openclaw-tensorrt-watchdog.timer
exec > >(tee -a "$task_dir/promotion.log") 2>&1
date -Is
echo JSON_SCHEMA_PROMOTION_SCOPE=persistent-production
test -s "$task_dir/zz-json-schema-production.conf"
test -s "$binding_file"
test -d "$source_dir"
test -f "$experiment_dropin"
test ! -e "$production_dropin"
test ! -e "$release_dir"
systemctl --user is-active --quiet "$service"
systemctl --user is-active --quiet "$timer"
test "$(systemctl --user is-enabled "$service")" = enabled
test "$(systemctl --user is-enabled "$timer")" = enabled
test "$(loginctl show-user ajmalrasi -p Linger --value)" = yes
systemctl --user cat "$service" > "$task_dir/pre-promotion-unit.txt"
mkdir -p "$release_dir"
cp -a "$source_dir" "$release_dir/source"
mkdir -p "$release_dir/pybind"
cp -a "$binding_file" "$release_dir/pybind/"
test "$(sha256sum "$binding_file" | cut -d' ' -f1)" = "$(sha256sum "$release_dir/pybind/$(basename "$binding_file")" | cut -d' ' -f1)"
echo "RELEASE_BINDING_SHA256=$(sha256sum "$release_dir/pybind/$(basename "$binding_file")" | cut -d' ' -f1)"
promoted=false
cleanup() {
  result=$?
  trap - EXIT INT TERM
  if [[ "$promoted" != true ]]; then
    echo JSON_SCHEMA_PROMOTION_RECOVERY=starting
    if [[ ! -e "$experiment_dropin" && -e "$task_dir/retired-zz-json-schema-experiment.conf" ]]; then
      mv "$task_dir/retired-zz-json-schema-experiment.conf" "$experiment_dropin"
    fi
    if [[ -e "$production_dropin" ]]; then
      mv "$production_dropin" "$task_dir/failed-zz-json-schema-production.conf"
    fi
    systemctl --user daemon-reload || true
    systemctl --user restart "$service" || true
    echo JSON_SCHEMA_PROMOTION_RECOVERY=previous-experiment-config
  fi
  systemctl --user show "$service" -p ActiveState -p MainPID -p UnitFileState --no-pager
  systemctl --user show "$timer" -p ActiveState -p UnitFileState --no-pager
  free -m
  echo "JSON_SCHEMA_PROMOTION_EXIT=$result"
  date -Is
  exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
install -m 0644 "$task_dir/zz-json-schema-production.conf" "$production_dropin"
systemctl --user daemon-reload
systemctl --user restart "$service"
healthy=false
for ((attempt = 0; attempt < 90; ++attempt)); do
  if curl --fail --silent --max-time 2 http://127.0.0.1:11434/health | jq -e '.status == "healthy" and .model == "openclaw"' > "$task_dir/health.json"; then
    healthy=true
    break
  fi
  if ! systemctl --user is-active --quiet "$service"; then
    break
  fi
  sleep 1
done
test "$healthy" = true
timeout --signal=TERM --kill-after=5s 120s \
  /home/ajmalrasi/TensorRT-Edge-LLM/.venv/bin/python "$task_dir/live_endpoint_gate.py"
systemctl --user start openclaw-tensorrt-watchdog.service
test "$(systemctl --user show openclaw-tensorrt-watchdog.service -p ExecMainStatus --value)" = 0
server_pid=$(systemctl --user show "$service" -p MainPID --value)
test "$(readlink "/proc/$server_pid/cwd")" = "$release_dir/source"
grep -E '^(VmRSS|VmHWM|VmSwap):' "/proc/$server_pid/status"
mv "$experiment_dropin" "$task_dir/retired-zz-json-schema-experiment.conf"
systemctl --user daemon-reload
systemctl --user cat "$service" > "$task_dir/final-unit.txt"
test "$(systemctl --user show "$service" -p WorkingDirectory --value)" = "$release_dir/source"
test "$(systemctl --user is-enabled "$service")" = enabled
test "$(systemctl --user is-enabled "$timer")" = enabled
test "$(loginctl show-user ajmalrasi -p Linger --value)" = yes
echo JSON_SCHEMA_PERSISTENT_PRODUCTION=passed
promoted=true
