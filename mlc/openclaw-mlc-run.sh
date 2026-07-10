#!/bin/bash
# openclaw-mlc-run.sh — serve openclaw (MLC-LLM / Qwen3-4B q4f16) on :11434.
# Tries the full 4096 context first; if the KV won't allocate OR the context is
# too tight to actually generate (working-memory OOM on fragmented/busy memory),
# falls back to progressively smaller contexts. The readiness check does a REAL
# 1-token generation, so it also catches the "serves /v1/models but can't
# generate" case. On fresh/clean memory it gets the full 4096.
# Deploy to: ~/.local/bin/openclaw-mlc-run.sh  (chmod +x)
set -u
IMAGE="dustynv/mlc:0.20.0-r36.4.0"
# Non-reasoning Instruct-2507 (no <think> blocks), matches beast. q4f16_2 quant.
MODEL="HF://FutureProofHomes/Qwen3-4B-Instruct-2507-q4f16_2-MLC"
# q4f16_2 Instruct-2507 params are ~2.7GB. A 4096 KV cache needs ~3.73GB resident
# (params 2691 + KV 665 + prefill temp 377 @ chunk 256). That fits in 7.4GB ONLY if
# the CUDA allocator has enough *genuinely free* RAM — the Tegra allocator won't
# reclaim page cache. So we drop caches right before each launch (see openclaw-drop-
# caches, installed by mlc/enable-4k-jetson.sh) and probe 4096 first, falling back to
# 2048/1024 if it still won't allocate or can't generate. Note: max_total_seq_length
# must be set explicitly or MLC's mode heuristic silently caps the KV cache at 2048.
# No 1024 tier: it's worse than useless (below the old 2048 baseline) and, when a
# 4096 load fails on a restart race, a cold 1024 recompile stalls startup for minutes.
CTXS="${OPENCLAW_CTXS:-4096 2048}"
# Root-owned helper (passwordless via /etc/sudoers.d/openclaw-drop-caches). Absent on
# a box that never ran enable-4k-jetson.sh -> the `sudo -n` below just no-ops.
DROP_CACHES="sudo -n /usr/local/bin/openclaw-drop-caches"
GEN='{"model":"openclaw","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"stream":false}'
for CTX in $CTXS; do
  # Start every attempt from settled, clean memory: a leftover/prior container's GPU
  # memory releases lazily, so without this a 4096 load can OOM on a restart and
  # cascade down the fallback chain even though it fits from a clean state.
  docker rm -f openclaw-mlc >/dev/null 2>&1
  sleep 3                            # let the driver reclaim the prior container's VRAM
  $DROP_CACHES 2>/dev/null || true   # then free page cache so the KV pool can allocate
  echo "[openclaw-mlc] starting at context_window_size=$CTX"
  docker run --rm --name openclaw-mlc --runtime nvidia --ipc=host \
    -p 11434:8000 \
    -v /home/ajmalrasi/.cache/huggingface:/root/.cache/huggingface \
    -v /home/ajmalrasi/.cache/mlc_llm:/root/.cache/mlc_llm \
    "$IMAGE" \
    mlc_llm serve "$MODEL" --mode interactive \
      --overrides "context_window_size=${CTX};max_total_seq_length=${CTX};prefill_chunk_size=256" \
      --host 0.0.0.0 --port 8000 &
  DPID=$!
  up=false
  for i in $(seq 1 45); do
    curl -sf -m 3 http://localhost:11434/v1/models >/dev/null 2>&1 && { up=true; break; }
    kill -0 "$DPID" 2>/dev/null || { echo "[openclaw-mlc] died during load at ctx $CTX"; break; }
    sleep 6
  done
  if $up; then
    gcode=$(curl -s -o /dev/null -w "%{http_code}" -m 90 -H "Content-Type: application/json" \
      -d "$GEN" http://localhost:11434/v1/chat/completions 2>/dev/null)
    if [ "$gcode" = "200" ]; then
      echo "[openclaw-mlc] SERVING+GENERATING at context $CTX"
      wait "$DPID"; exit $?
    fi
    echo "[openclaw-mlc] serves but cannot generate at ctx $CTX (gen http=$gcode) -> lower"
  fi
  docker rm -f openclaw-mlc >/dev/null 2>&1
done
echo "[openclaw-mlc] FAILED at all contexts"; exit 1
