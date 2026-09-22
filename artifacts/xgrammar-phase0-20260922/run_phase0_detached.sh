#!/usr/bin/env bash
set -euo pipefail

root=/home/ajmalrasi/xgrammar-phase0-20260922
source_dir="$root/xgrammar"
build_dir="$root/build"
fixture_dir="$root/fixtures"
log="$root/phase0.log"
model_dir=/home/ajmalrasi/Qwen3.5-4B/engines/llm-b2-input6144-kv8192-vanilla
python=/home/ajmalrasi/TensorRT-Edge-LLM/.venv/bin/python

exec >"$log" 2>&1
trap 'status=$?; echo "EXIT_STATUS $status $(date --iso-8601=seconds)"' EXIT
echo "START $(date --iso-8601=seconds)"
echo "SOURCE_COMMIT $(git -C "$source_dir" rev-parse HEAD)"
echo "ARCH $(uname -m)"
echo "KERNEL $(uname -r)"
echo "MEMORY_BEFORE"
free -b

"$python" "$root/extract_qwen_tokenizer.py" "$model_dir" "$fixture_dir"
rm -rf "$build_dir"
mkdir -p "$build_dir"
cp "$root/config.cmake" "$build_dir/config.cmake"
cmake -S "$source_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --parallel 2
ctest --test-dir "$build_dir" --output-on-failure

g++ -std=c++17 -O2 \
  -I"$source_dir/include" \
  -I"$source_dir/3rdparty/picojson" \
  -I"$source_dir/3rdparty/dlpack/include" \
  "$root/xgrammar_phase0_smoke.cc" "$build_dir/libxgrammar.a" \
  -pthread -o "$root/xgrammar_phase0_smoke"

/usr/bin/time -v "$root/xgrammar_phase0_smoke" "$fixture_dir"
echo "MEMORY_AFTER"
free -b
echo "END $(date --iso-8601=seconds)"
