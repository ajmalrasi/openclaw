#!/usr/bin/env bash
# Provision the OpenClaw LLM backed by vLLM on this host (e.g. `beast`, an
# 8 GB Ampere laptop GPU). Serves the OpenClaw contract — OpenAI API on :11434,
# model name "openclaw" — as a drop-in replacement for the Ollama backend.
#
# This is the vLLM counterpart to ./install.sh (which is Ollama, used on the
# Jetson). Only one openclaw backend can own :11434 at a time, so this script
# retires the Ollama one first.
#
#   ./install-vllm.sh            # install + start the vLLM user service
#   ./install-vllm.sh --no-start # install the unit only, don't start it
#
# Requirements: Docker with the NVIDIA runtime, the invoking user in the
# `docker` group, and the vLLM image (pulled automatically if missing). The
# model checkpoint is downloaded to ~/.cache/huggingface on first start.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT="openclaw-vllm.service"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
NO_START="${1:-}"

echo ">> Checking Docker + NVIDIA runtime ..."
command -v docker >/dev/null || { echo "!! docker not found"; exit 1; }
id -nG | tr ' ' '\n' | grep -qx docker \
  || { echo "!! $USER is not in the 'docker' group. Add it and re-login."; exit 1; }

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo ">> Pulling $IMAGE (large, one-time) ..."
  docker pull "$IMAGE"
fi

echo ">> Retiring any Ollama openclaw backend on :11434 ..."
# The repo's own user service (if it was ever used here).
systemctl --user disable --now openclaw.service 2>/dev/null || true
# A system-wide Ollama install (the common case on beast) needs root. Try
# without a password; if that fails, tell the operator to do it by hand.
if curl -sf http://localhost:11434/api/version >/dev/null 2>&1; then
  if sudo -n systemctl disable --now ollama.service 2>/dev/null; then
    echo ">> Stopped system ollama.service."
  else
    echo "!! Something (Ollama) still holds :11434 and stopping it needs root."
    echo "!! Run this, then re-run this script:"
    echo "     sudo systemctl disable --now ollama.service"
    exit 1
  fi
fi

echo ">> Installing user service $UNIT ..."
mkdir -p "$HOME/.config/systemd/user"
install -m 0644 "$HERE/vllm/$UNIT" "$HOME/.config/systemd/user/$UNIT"
loginctl enable-linger "$USER" 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable "$UNIT"

if [[ "$NO_START" == "--no-start" ]]; then
  echo ">> Installed but not started (--no-start). Start with:"
  echo "     systemctl --user start $UNIT"
  exit 0
fi

echo ">> Starting $UNIT ..."
systemctl --user restart "$UNIT"

echo ">> Waiting for vLLM to answer on :11434 (first start loads the model) ..."
for _ in $(seq 1 120); do
  curl -sf http://localhost:11434/health >/dev/null 2>&1 && break
  sleep 2
done
if curl -sf http://localhost:11434/health >/dev/null 2>&1; then
  echo ">> OK. openclaw (vLLM) is live."
  echo ">> Apps should use: base_url=http://<host>:11434  model=openclaw"
else
  echo "!! vLLM did not come up. Check: journalctl --user -u $UNIT -e"
  exit 1
fi
