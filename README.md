# OpenClaw — one shared local LLM for all my apps

A single local-LLM daemon + a single resident model per host, owned in one
place. Every app (life_os, etc.) is a **client** that points at it. No app ships
or runs its own LLM server, and no app names a raw model — they all ask for
**`openclaw`**, and this repo decides what that is. The serving engine is an
implementation detail (Ollama on the Jetson, vLLM on `beast`); the API and model
name are identical everywhere.

## Why this exists

On an 8 GB Jetson Orin Nano, each loaded model eats ~2–3 GB. If every app loads
its own (or apps disagree on which model), they thrash in and out of RAM and
every call pays a cold-load penalty (measured: ~49 s cold vs ~2.3 s warm). One
shared, pinned model = warm replies and headroom to spare.

## The contract (what apps depend on)

| Setting    | Value                              |
|------------|------------------------------------|
| Base URL   | `http://<host>:11434`              |
| API        | OpenAI-compatible `/v1/chat/completions` (also native `/api/*`) |
| Model name | `openclaw`                         |

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

The **instruct (non-reasoning)** Qwen3 4B — no `<think>` blocks, so callers need
no special flags. The *contract above is identical on every host*; only the
serving engine differs per hardware:

| Host | Backend | What / quant | Speed |
|------|---------|--------------|-------|
| `jetson-orin` | Ollama | `qwen3:4b-instruct-2507` Q4_K_M (see [`Modelfile`](./Modelfile)) | ~16 tok/s |
| `beast` (RTX 3070 Ti laptop) | **vLLM** | `Eslzzyl/Qwen3-4B-Instruct-2507-AWQ` (INT4 AWQ) | **~96 tok/s** |

`beast` moved off Ollama to vLLM for a 2.4x speedup (correct output, same API).
TensorRT-LLM was evaluated and rejected — its pytorch backend can't serve a
working quantized model on this consumer Ampere GPU. See
[TRTLLM_MIGRATION.md](./TRTLLM_MIGRATION.md) and [BENCHMARKS.md](./BENCHMARKS.md).

To change the model for **every** app on a host at once: on Ollama hosts edit
`FROM` in the `Modelfile` and re-run `./install.sh`; on vLLM hosts edit
`OPENCLAW_MODEL` in [`vllm/openclaw-vllm.service`](./vllm/openclaw-vllm.service)
and re-run `./install-vllm.sh`. Nothing in any app changes either way.

## Install / update

**Ollama host** (Jetson) — Ollama must already be installed (e.g.
`~/.local/ollama/bin/ollama`):

```bash
./install.sh            # create/refresh the openclaw alias + warmup script
./install.sh --service  # also install & start the systemd user service
```

**vLLM host** (beast) — Docker with the NVIDIA runtime, user in the `docker`
group. Retires the Ollama backend on :11434 and installs the vLLM user service:

```bash
./install-vllm.sh       # pull image if needed, stop Ollama, start vLLM on :11434
```

## Files

- `Modelfile` — defines the `openclaw` model alias (Ollama hosts).
- `systemd/openclaw.service` — the shared Ollama user service (LAN-bound on
  :11434, `OLLAMA_KEEP_ALIVE=-1`, preloads `openclaw` at boot).
- `bin/openclaw-warmup.sh` — pins the model resident right after start (Ollama).
- `install.sh` — idempotent Ollama provisioner.
- `vllm/openclaw-vllm.service` — the vLLM user service (beast): OpenAI API on
  :11434, model name `openclaw`, INT4-AWQ Qwen3-4B.
- `install-vllm.sh` — idempotent vLLM provisioner (retires Ollama on :11434).
- `TRTLLM_MIGRATION.md` — why TensorRT-LLM was rejected; `BENCHMARKS.md` — numbers.

## Migration note (Jetson, 2026-06-22)

Replaces the old app-specific `lifeos-ollama.service` (which preloaded
`qwen3.5:4b`). To cut over:

```bash
systemctl --user disable --now lifeos-ollama.service
./install.sh --service
```

Both bind the same port (11434), so run only one at a time.
