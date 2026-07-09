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
# q4f16_2 Instruct-2507 params are heavier (~2.7GB), so 4096 KV won't fit with
# generation headroom on 7.4GB — 2048 is the reliable ceiling. MLC recompiles the
# lib per context, so we don't waste cycles probing 4096/3072.
CTXS="${OPENCLAW_CTXS:-2048 1024}"
GEN='{"model":"openclaw","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"stream":false}'
docker rm -f openclaw-mlc >/dev/null 2>&1
for CTX in $CTXS; do
  echo "[openclaw-mlc] starting at context_window_size=$CTX"
  docker run --rm --name openclaw-mlc --runtime nvidia --ipc=host \
    -p 11434:8000 \
    -v /home/ajmalrasi/.cache/huggingface:/root/.cache/huggingface \
    -v /home/ajmalrasi/.cache/mlc_llm:/root/.cache/mlc_llm \
    "$IMAGE" \
    mlc_llm serve "$MODEL" --mode interactive \
      --overrides "context_window_size=${CTX};prefill_chunk_size=512" \
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
