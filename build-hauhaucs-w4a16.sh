#!/usr/bin/env bash
# Build a vLLM-native W4A16 checkpoint of the HauhauCS aggressive Qwen3.5-4B
# model. The upstream release is GGUF-only, so this uses its BF16 safetensors
# reconstruction and writes a compressed-tensors checkpoint for Marlin kernels.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
MODEL_DIR="$HOME/.cache/openclaw-models/hauhaucs-w4a16"

command -v docker >/dev/null || { echo "!! docker not found"; exit 1; }

if [[ -s "$MODEL_DIR/config.json" && -s "$MODEL_DIR/model.safetensors" ]]; then
  echo ">> HauhauCS W4A16 checkpoint already exists: $MODEL_DIR"
  exit 0
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo ">> Pulling $IMAGE (large, one-time) ..."
  docker pull "$IMAGE"
fi

mkdir -p "$HOME/.cache/huggingface" "$MODEL_DIR"

docker_env=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  docker_env+=(--env HF_TOKEN)
fi

echo ">> Building HauhauCS Qwen3.5-4B W4A16 checkpoint on CPU ..."
docker run --rm \
  --entrypoint /bin/bash \
  "${docker_env[@]}" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$MODEL_DIR:/output" \
  -v "$HERE/vllm/quantize-hauhaucs-w4a16.py:/work/quantize.py:ro" \
  "$IMAGE" \
  -lc 'python3 -m pip install --no-cache-dir "llmcompressor>=0.11.0" && python3 /work/quantize.py'

[[ -s "$MODEL_DIR/config.json" && -s "$MODEL_DIR/model.safetensors" ]] \
  || { echo "!! Quantization did not produce a complete checkpoint"; exit 1; }

echo ">> W4A16 checkpoint ready: $MODEL_DIR"
