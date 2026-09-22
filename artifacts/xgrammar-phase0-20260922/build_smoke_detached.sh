#!/usr/bin/env bash
set -euo pipefail

root=/home/ajmalrasi/xgrammar-phase0-20260922
source_dir="$root/xgrammar"
build_dir="$root/build"
log="$root/build-smoke.log"

exec >"$log" 2>&1
trap 'status=$?; echo "EXIT_STATUS $status $(date --iso-8601=seconds)"' EXIT
echo "START $(date --iso-8601=seconds)"
g++ -std=c++17 -O2 \
  -I"$source_dir/include" \
  -I"$source_dir/3rdparty/picojson" \
  -I"$source_dir/3rdparty/dlpack/include" \
  "$root/xgrammar_phase0_smoke.cc" "$build_dir/libxgrammar.a" \
  -pthread -o "$root/xgrammar_phase0_smoke"
echo "END $(date --iso-8601=seconds)"
