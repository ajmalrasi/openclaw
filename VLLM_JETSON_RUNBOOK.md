# Jetson Orin Nano vLLM runbook

## Production result (2026-09-10)

`openclaw` serves successfully on the Jetson Orin Nano Super 8 GB after the
upgrade to JetPack 7.2.1 / L4T r39.2.1. It uses NVIDIA's dedicated Jetson-Orin
vLLM container and the Red Hat Qwen3.5-4B W4A16 checkpoint.

- Endpoint: `http://192.168.3.30:11434/v1`
- Model alias: `openclaw`
- Checkpoint: `RedHatAI/Qwen3.5-4B-quantized.w4a16`
- Context: 4096 tokens
- Concurrency: one active sequence
- KV cache: 192 MB explicit allocation
- Mode: language-only, non-thinking, eager execution
- Tool calling: Qwen3 parser enabled
- Swap: 16 GB `/swapfile-vllm`, persistent through `/etc/fstab`

The service pins the tested container digest instead of tracking a moving
`latest` tag.

## Install or update

Run this repository's installer on the Jetson:

```bash
./install-vllm-jetson.sh
```

It verifies Docker's NVIDIA runtime, pulls the tested container, prepares the
checkpoint-compatible tokenizer, installs the user service, waits for readiness,
and requires a real `LIFEOS_OK` chat completion before returning success.

The first run downloads about 5.2 GB of model data. The persistent cache path
inside NVIDIA's container is `/data/models/huggingface`; mounting only
`/root/.cache/huggingface` causes the weights to be downloaded again after the
container is removed.

## Why these limits are required

The model weights occupy about 4.44 GiB while loading. Full multimodal startup
then exceeds the board's shared 8 GB memory during vision profiling. Production
therefore uses `--language-model-only` and disables the multimodal processor
cache.

Automatic KV sizing also leaves too little headroom on this unified-memory
system. The stable configuration explicitly allocates 192 MB for KV cache,
limits batching to one sequence and 1024 batched tokens, and disables MTP
speculative decoding. These are capacity decisions, not model-format changes.

The checkpoint declares the Transformers 5 `TokenizersBackend` class, while the
tested container currently includes Transformers 4. The installer preserves the
checkpoint's exact `tokenizer.json` and chat template but exposes it through the
compatible `Qwen2TokenizerFast` class.

## Operate

```bash
systemctl --user status openclaw-vllm-jetson.service
systemctl --user restart openclaw-vllm-jetson.service
systemctl --user stop openclaw-vllm-jetson.service
journalctl --user -u openclaw-vllm-jetson.service -f
docker logs -f openclaw-vllm-jetson
```

Smoke test:

```bash
curl -s http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"openclaw","messages":[{"role":"user","content":"Reply exactly: OK"}],"max_tokens":16}'
```

LifeOS uses:

```text
OPENCLAW_BASE_URL=http://host.docker.internal:11434
OPENCLAW_MODEL=openclaw
```

## Troubleshooting history

The generic `vllm/vllm-openai:latest` container failed because its PyTorch/CUDA
build did not include Orin's `sm_87` target. The working image is
`ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin`, pinned by digest in the service.
Its CUDA 12.6 PyTorch build was verified with a real tensor operation on the GPU.

If startup fails after container experiments, stop other model containers and
restart this service. Do not run MLC and vLLM simultaneously on the 8 GB board.
