#!/usr/bin/env bash
set -euo pipefail
task_dir=/home/ajmalrasi/json-schema-gpu-mask-final-build-20260923
source_dir=/home/ajmalrasi/json-schema-phase4a-20260923/source
build_dir=/home/ajmalrasi/json-schema-phase4a-20260923/build
service=openclaw-tensorrt-edgellm.service
timer=openclaw-tensorrt-watchdog.timer
exec > >(tee -a "$task_dir/incremental-build.log") 2>&1
date -Is
echo GPU_MASK_SCOPE=final-host-shape-rebuild
test -s "$build_dir/pybind/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so"
test "$(systemctl --user show "$service" -p WorkingDirectory --value)" = /home/ajmalrasi/tensorrt-json-schema-release-2cb8bb8/source
curl --fail --silent --max-time 5 http://127.0.0.1:11434/health | jq -e '.status == "healthy" and .active_requests == 0 and .queued_requests == 0'
service_was_active=false
timer_was_active=false
if systemctl --user is-active --quiet "$service"; then service_was_active=true; fi
if systemctl --user is-active --quiet "$timer"; then timer_was_active=true; fi
restore() {
  result=$?
  trap - EXIT INT TERM
  if [[ "$service_was_active" == true ]]; then
    systemctl --user start "$service" || result=1
  fi
  if [[ "$timer_was_active" == true ]]; then
    systemctl --user start "$timer" || result=1
  fi
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
  echo "GPU_MASK_FINAL_BUILD_EXIT=$result"
  date -Is
  exit "$result"
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
systemctl --user stop "$timer"
systemctl --user stop "$service"
available_mib=$(awk '/MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
cores=$(nproc)
if ((available_mib < 4096 || cores < 2)); then
  echo "Insufficient memory or cores for multicore build: available_mib=$available_mib cores=$cores" >&2
  exit 2
fi
build_jobs=2
if ((available_mib >= 6144 && cores >= 3)); then build_jobs=3; fi
echo "BUILD_JOBS=$build_jobs AVAILABLE_MIB=$available_mib CORES=$cores"
free -m
export TRT_PACKAGE_DIR=/usr
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/targets/sbsa-linux/lib
timeout --signal=TERM --kill-after=10s 3000s \
  cmake --build "$build_dir" --target _edgellm_runtime --parallel "$build_jobs"
test -s "$build_dir/pybind/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so"
EDGELLM_PYBIND_DIR="$build_dir/pybind" PYTHONPATH="$source_dir" \
  /home/ajmalrasi/TensorRT-Edge-LLM/.venv/bin/python -c \
  'from experimental.server.runtime.engine import _import_runtime; r = _import_runtime(); assert hasattr(r.ContinuousSequenceOptions(), "json_schema"); print("GPU_MASK_BINDING_IMPORT=passed", r.__file__)'
echo GPU_MASK_FINAL_BUILD=passed
