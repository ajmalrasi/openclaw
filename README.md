# OpenClaw — one shared local LLM for all my apps

A single local-LLM daemon + a single resident model per host, owned in one
place. Every app (life_os, etc.) is a **client** that points at it. No app ships
or runs its own LLM server, and no app names a raw model — they all ask for
**`openclaw`**, and this repo decides what that is. The serving engine is an
implementation detail (currently TensorRT Edge-LLM on the Jetson; vLLM on
`beast` and EC2); the API and model name are identical everywhere.

## Why this exists

On an 8 GB Jetson Orin Nano, each loaded model eats ~2–3 GB. If every app loads
its own (or apps disagree on which model), they thrash in and out of RAM and
every call pays a cold-load penalty (measured: ~49 s cold vs ~2.3 s warm). One
shared, pinned model = warm replies and headroom to spare.

## The contract (what apps depend on)

| Setting    | Value                              |
|------------|------------------------------------|
| Base URL   | `http://<host>:11434`              |
| API        | OpenAI-compatible `/v1/chat/completions` |
| Model name | `openclaw`                         |

> **Use the OpenAI `/v1/*` API only.** The current engines (TensorRT Edge-LLM,
> vLLM, and historical MLC-LLM) do
> **not** serve Ollama-native `/api/generate` / `/api/chat` — those return 404.
> Apps that used raw `/api/*` must switch to `/v1/chat/completions`.

From inside a container on the same host, reach it at
`http://host.docker.internal:11434`.

Example — life_os (`backend/.env`):

```
OPENCLAW_BASE_URL=http://host.docker.internal:11434
OPENCLAW_MODEL=openclaw
```

That's the entire integration. Apps never need this repo's files at build time,
so it is **not** a submodule — it's a standalone service plus a config contract.

## What `openclaw` currently is

A shared language-model alias. The *API and model name are identical on every
host*; the engine, generation, quant, and exact variant differ per hardware:

| Host | Backend | What / quant | Speed | Context |
|------|---------|--------------|-------|---------|
| `jetson-orin` | **TensorRT Edge-LLM** | `Qwen/Qwen3.5-4B` INT4 AWQ, text-only non-MTP batch-two engine | Endpoint smoke-tested; earlier-engine performance results are not current-engine benchmarks | 6144 input / 8192 total |
| `beast` (RTX 3070 Ti laptop) | **vLLM** | `QuantTrio/Qwen3.5-4B-AWQ` (INT4 AWQ, FP8 KV, language-only) | **47.31 tok/s** sequential; **46.57 tok/s** aggregate in the 8-way 7K stress test | 8192 |
| `aws-g6` (EC2 g6.xlarge, NVIDIA L4) | **vLLM** | `google/gemma-4-12B-it-qat-w4a16-ct` (QAT W4A16, FP8 KV cache, language-only) | Endpoint smoke-tested; no benchmark run | 16384 |

All hosts return direct responses with no `<think>` blocks. On `beast`,
`enable_thinking=false` is a server-wide chat-template default.

The EC2 vLLM port is bound only to `127.0.0.1`. It is available either through
an SSH tunnel or through its Caddy HTTPS frontend. vLLM requires the same bearer
key on both routes. For tunnel-only access:

```bash
ssh -i ~/.ssh/id_ed25519 -N \
  -L 11434:127.0.0.1:11434 ubuntu@<ec2-public-ip>
# Base URL from this machine: http://127.0.0.1:11434/v1
# Model: openclaw
```

> **Jetson deployment (2026-09-12):** the active deployment is the TensorRT
> Edge-LLM user service, `openclaw-tensorrt-edgellm.service`, serving
> `/home/ajmalrasi/Qwen3.5-4B/engines/llm-b2-input6144-kv8192-vanilla` on port
> 11434. It is a text-only INT4 AWQ engine with 6,144-token input, 8,192-token
> total sequence capacity and engine batch capacity two; the current server
> admits one active sequence. vLLM is retained only as a disabled rollback
> service. The model may be intentionally stopped for maintenance; check
> `/v1/models` or the service state rather than assuming it is running.
> See [TENSORRT_EDGE_LLM_EXPERIMENT_LOG.md](./TENSORRT_EDGE_LLM_EXPERIMENT_LOG.md).

`beast` moved off Ollama to vLLM for a 2.4x speedup. The Jetson moved from
Ollama to MLC-LLM, then to vLLM after the JetPack 7 upgrade, and now to
TensorRT Edge-LLM.
See [MLC_MIGRATION.md](./MLC_MIGRATION.md), [TRTLLM_MIGRATION.md](./TRTLLM_MIGRATION.md),
[BENCHMARKS.md](./BENCHMARKS.md).

To change the model for **every** app on a host at once, update that host's
serving service. On the Jetson, this is TensorRT Edge-LLM; nothing in any app
changes either way.

## Install / update

**TensorRT Edge-LLM host** (Jetson) — JetPack 7.2.1, TensorRT Edge-LLM 0.10.1,
and existing `llm-b2-input6144-kv8192-vanilla` batch-two engine. The service source is
`tensorrt-edgellm/openclaw-tensorrt-edgellm.service`; its experiment and
deployment record is [TENSORRT_EDGE_LLM_EXPERIMENT_LOG.md](./TENSORRT_EDGE_LLM_EXPERIMENT_LOG.md).
Do not start vLLM alongside it on the 8 GB board.

**vLLM host** (beast) — Docker with the NVIDIA runtime, user in the `docker`
group. Retires the Ollama backend on :11434 and installs the vLLM user service:

```bash
./install-vllm.sh       # pull image if needed, stop Ollama, start vLLM on :11434
```

**vLLM host** (AWS EC2 g6.xlarge) — Ubuntu, Docker with the NVIDIA runtime, an
NVIDIA L4, and the `openclaw-ec2-ecr-readonly` instance profile. The installer
pulls one private ECR artifact containing both vLLM and the pinned model; it does
not download weights from Hugging Face. It installs system services that start
after Docker at every boot:

```bash
./install-vllm-ec2.sh   # pinned vLLM image + Gemma 4 W4A16, starts automatically
sudo systemctl status openclaw-vllm-ec2.service
sudo systemctl status openclaw-caddy-ec2.service
sudo journalctl -u openclaw-vllm-ec2.service -f
```

> `install.sh` / `Modelfile` are the retired **Ollama** provisioner, kept for
> reference only — no host runs Ollama anymore.

## Files

- `tensorrt-edgellm/openclaw-tensorrt-edgellm.service` — active Jetson
  TensorRT Edge-LLM service source, serving the existing 6,144-input / 8,192-KV
  batch-two INT4 AWQ engine on :11434 when started.
- `vllm/openclaw-vllm-jetson.service`, `install-vllm-jetson.sh`, and
  `VLLM_JETSON_RUNBOOK.md` — preserved Jetson vLLM rollback service and
  historical runbook; not the current Jetson backend.
- `mlc/openclaw-mlc-run.sh` — retired wrapper that starts the MLC container, tries
  4096 context and falls back if memory is too tight to generate; retained only
  for historical rollback.
- `MLC_RUNBOOK.md` and `MLC_MIGRATION.md` — retired JetPack 6.2 backend history.
- `vllm/openclaw-vllm.service` — the vLLM user service (beast): OpenAI API on
  :11434, model name `openclaw`, language-only Qwen3.5-4B INT4 AWQ, 8K context,
  and up to eight active sequences.
- `install-vllm.sh` — idempotent vLLM provisioner (retires Ollama on :11434).
- `vllm/openclaw-vllm-ec2.service` — boot-persistent EC2 system service: pinned
  vLLM container, loopback-only API, Gemma 4 12B W4A16 on the NVIDIA L4.
- `vllm/openclaw-ec2-init.service` and `.sh` — generate a per-VM bearer key and
  refresh the regional ECR image and `sslip.io` hostname on every boot.
- `install-vllm-ec2.sh` — installs/enables that EC2 service and verifies its GPU.
- `aws/Dockerfile.ec2` — pinned vLLM runtime with the Gemma checkpoint embedded.
- `aws/push-ecr-image.sh` — build the amd64 artifact and push it to private ECR.
- `vllm/openclaw-caddy-ec2.service` and `vllm/Caddyfile.ec2` — automatic HTTPS
  frontend for the EC2 OpenAI API; only `/v1/*` is public and vLLM checks its
  bearer key.
- `EC2_RUNBOOK.md` — AWS SSO, SSH tunnel, service operations, storage, and
  stop/start instructions for the g6.xlarge host.
- `TRTLLM_MIGRATION.md` — why TensorRT-LLM was rejected; `BENCHMARKS.md` — numbers.
- `Modelfile`, `install.sh`, `systemd/openclaw.service`, `bin/openclaw-warmup.sh`
  — **retired** Ollama provisioner, kept for reference only.

## Migration notes (Jetson)

**2026-09-12:** replaced the Jetson's vLLM deployment with TensorRT Edge-LLM
using the existing batch-two Qwen3.5-4B INT4 AWQ engine. The endpoint remains
OpenAI-compatible on :11434 with model alias `openclaw`; vLLM is rollback-only.

**2026-09-10:** upgraded the Jetson to JetPack 7.2.1 and replaced MLC with the
dedicated NVIDIA Jetson-Orin vLLM container serving Qwen3.5-4B W4A16. That is
now a preserved rollback configuration, not the active backend.

**2026-07-09:** the Jetson moved **Ollama → MLC-LLM** (~16 → ~25 tok/s). That
historical JetPack 6.2 migration is documented in
[MLC_MIGRATION.md](./MLC_MIGRATION.md) and [MLC_RUNBOOK.md](./MLC_RUNBOOK.md).

_(Earlier, 2026-06-22: replaced the app-specific `lifeos-ollama.service` with the
shared Ollama `openclaw.service`, since also retired.)_
