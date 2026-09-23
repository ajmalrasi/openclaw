#!/usr/bin/env bash
set -euo pipefail
task_dir=/home/ajmalrasi/json-schema-gpu-mask-unit-20260923
source_dir=/home/ajmalrasi/json-schema-phase4a-20260923/source
build_dir=/home/ajmalrasi/json-schema-phase4a-20260923/build
reference_dir=/home/ajmalrasi/TensorRT-Edge-LLM
xgrammar_dir=/home/ajmalrasi/xgrammar-phase0-20260922/xgrammar
service=openclaw-tensorrt-edgellm.service
timer=openclaw-tensorrt-watchdog.timer
exec > >(tee -a "$task_dir/unit-test.log") 2>&1
date -Is
echo GPU_MASK_UNIT_SCOPE=focused-cuda-kernel-test
test -f "$source_dir/unittests/cpp/sampler/samplingTests.cpp"
test -f "$source_dir/3rdParty/googletest/CMakeLists.txt"
curl --fail --silent --max-time 5 http://127.0.0.1:11434/health | jq -e '.status == "healthy" and .active_requests == 0 and .queued_requests == 0'
service_was_active=false
timer_was_active=false
if systemctl --user is-active --quiet "$service"; then service_was_active=true; fi
if systemctl --user is-active --quiet "$timer"; then timer_was_active=true; fi
restore() {
  result=$?
  trap - EXIT INT TERM
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
  echo "GPU_MASK_UNIT_EXIT=$result"
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
  echo "Insufficient memory or cores: available_mib=$available_mib cores=$cores" >&2
  exit 2
fi
build_jobs=2
if ((available_mib >= 6144 && cores >= 3)); then build_jobs=3; fi
echo "BUILD_JOBS=$build_jobs AVAILABLE_MIB=$available_mib CORES=$cores"
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
  -DBUILD_UNIT_TESTS=ON \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPython_EXECUTABLE="$reference_dir/.venv/bin/python" \
  -Dpybind11_DIR="$pybind11_dir"
timeout --signal=TERM --kill-after=10s 3000s \
  cmake --build "$build_dir" --target unitTestCommon --parallel "$build_jobs"
timeout --signal=TERM --kill-after=5s 120s \
  "$build_dir/unittests/unitTestCommon" \
  --gtest_filter=SamplingTest.AllowedTokenMaskSeparatesRowsAndHandlesPartialWord \
  --gtest_output="xml:$task_dir/gpu-mask-unit.xml"
echo GPU_MASK_FOCUSED_UNIT=passed
