#!/usr/bin/env bash
# Install the validated Qwen3.5-4B W4A16 vLLM service on Jetson Orin Nano 8 GB.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT="openclaw-vllm-jetson.service"
IMAGE="${VLLM_IMAGE:-ghcr.io/nvidia-ai-iot/vllm@sha256:817f0f940d2d9c9067d861d2118d7bf58c40873598f0c35e19c8516269ebc4bd}"
MODEL="RedHatAI/Qwen3.5-4B-quantized.w4a16"
TOKENIZER_DIR="$HOME/.cache/vllm/qwen35-tokenizer-v4"
TOKENIZER_BASE="https://huggingface.co/$MODEL/resolve/main"
NO_START="${1:-}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "!! This installer is only for the aarch64 Jetson host."
  exit 1
fi

echo ">> Checking Docker + NVIDIA runtime ..."
command -v docker >/dev/null || { echo "!! docker not found"; exit 1; }
docker info >/dev/null 2>&1 || {
  echo "!! Docker is not usable by $USER."
  exit 1
}
docker info --format '{{json .Runtimes}}' | grep -q 'nvidia' || {
  echo "!! Docker's NVIDIA runtime is not configured."
  exit 1
}

echo ">> Pulling the tested NVIDIA Jetson-Orin image ..."
docker pull "$IMAGE"

echo ">> Preparing the checkpoint-compatible tokenizer ..."
mkdir -p "$TOKENIZER_DIR"
for file in tokenizer.json tokenizer_config.json chat_template.jinja; do
  if [[ ! -s "$TOKENIZER_DIR/$file" ]]; then
    curl -fL --retry 3 -o "$TOKENIZER_DIR/$file.tmp" "$TOKENIZER_BASE/$file"
    mv "$TOKENIZER_DIR/$file.tmp" "$TOKENIZER_DIR/$file"
  fi
done
sed 's/"tokenizer_class"[[:space:]]*:[[:space:]]*"TokenizersBackend"/"tokenizer_class": "Qwen2TokenizerFast"/' \
  "$TOKENIZER_DIR/tokenizer_config.json" > "$TOKENIZER_DIR/tokenizer_config.json.tmp"
mv "$TOKENIZER_DIR/tokenizer_config.json.tmp" "$TOKENIZER_DIR/tokenizer_config.json"

echo ">> Installing $UNIT ..."
mkdir -p "$HOME/.config/systemd/user" "$HOME/.cache/huggingface"
install -m 0644 "$HERE/vllm/$UNIT" "$HOME/.config/systemd/user/$UNIT"
loginctl enable-linger "$USER" 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable "$UNIT"

if [[ "$NO_START" == "--no-start" ]]; then
  echo ">> Installed but not started."
  exit 0
fi

echo ">> Replacing the trial container with the persistent service ..."
docker rm -f openclaw-vllm-trial >/dev/null 2>&1 || true
systemctl --user restart "$UNIT"

echo ">> Waiting for startup, then testing a real non-thinking response ..."
SMOKE_FILE="$(mktemp)"
trap 'rm -f "$SMOKE_FILE"' EXIT
ready=false
for _ in $(seq 1 300); do
  if curl -sf -m 5 http://127.0.0.1:11434/health >/dev/null 2>&1; then
    code=$(curl -s -o "$SMOKE_FILE" -w '%{http_code}' -m 120 \
      -H 'Content-Type: application/json' \
      -d '{"model":"openclaw","messages":[{"role":"user","content":"Reply with exactly: LIFEOS_OK"}],"max_tokens":32,"stream":false}' \
      http://127.0.0.1:11434/v1/chat/completions || true)
    if [[ "$code" == "200" ]] && grep -q 'LIFEOS_OK' "$SMOKE_FILE"; then
      ready=true
      break
    fi
  fi
  sleep 4
done

if $ready; then
  echo ">> OK. openclaw is live through vLLM on :11434."
  exit 0
fi

echo "!! vLLM did not produce the expected response." >&2
echo "!! Inspect it with: journalctl --user -u $UNIT -e" >&2
exit 1
