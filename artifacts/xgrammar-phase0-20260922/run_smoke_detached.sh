#!/usr/bin/env bash
set -euo pipefail

root=/home/ajmalrasi/xgrammar-phase0-20260922
log="$root/smoke.log"

exec >"$log" 2>&1
trap 'status=$?; echo "EXIT_STATUS $status $(date --iso-8601=seconds)"' EXIT
echo "START $(date --iso-8601=seconds)"
echo "MEMORY_BEFORE"
free -b
"$root/xgrammar_phase0_smoke" "$root/fixtures"
echo "MEMORY_AFTER"
free -b
echo "END $(date --iso-8601=seconds)"
