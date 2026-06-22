#!/usr/bin/env bash
# Preload the shared model into RAM as soon as Ollama is up, before the rest of
# the system fragments memory, and pin it resident (keep_alive -1). Keeping ONE
# model hot is the whole point: clients get ~2-3s replies instead of paying a
# cold model-load on every call.
set -u
for _ in $(seq 1 60); do
  curl -sf http://localhost:11434/api/version >/dev/null 2>&1 && break
  sleep 1
done
curl -s http://localhost:11434/api/generate \
  -d '{"model":"openclaw","prompt":"","keep_alive":-1}' >/dev/null 2>&1
exit 0
