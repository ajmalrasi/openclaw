# OpenClaw — one shared local LLM for all my apps

A single local-LLM daemon + a single resident model per host, owned in one
place. Every app (life_os, etc.) is a **client** that points at it. No app ships
or runs its own LLM server, and no app names a raw model — they all ask for
**`openclaw`**, and this repo decides what that is. The serving engine is an
implementation detail (MLC-LLM on the Jetson, vLLM on `beast`); the API and model
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
| API        | OpenAI-compatible `/v1/chat/completions` |
| Model name | `openclaw`                         |

> **Use the OpenAI `/v1/*` API only.** The current engines (MLC-LLM, vLLM) do
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

Qwen3 4B. The *API and model name are identical on every host*; the engine,
quant, and exact variant differ per hardware:

| Host | Backend | What / quant | Speed | Context |
|------|---------|--------------|-------|---------|
| `jetson-orin` | **MLC-LLM** (TVM) | `FutureProofHomes/Qwen3-4B-Instruct-2507-q4f16_2-MLC` (non-reasoning) | **~22 tok/s** | 2048 |
| `beast` (RTX 3070 Ti laptop) | **vLLM** | `Eslzzyl/Qwen3-4B-Instruct-2507-AWQ` (INT4 AWQ) | **~96 tok/s** | — |

Both hosts run **Instruct-2507 (non-reasoning)** — no `<think>` blocks anywhere.

> **Jetson caveat:** the only *working* prebuilt MLC of Instruct-2507 is the
> heavier `q4f16_2` quant (~2.7 GB params), which caps context at **2048** and
> needs **clean memory to load** (loads fine at boot; a mid-session restart on a
> busy box may fail — clear cache in jtop or reboot). A lighter `q4f16_1` build
> (compiled from the FP16 base) would restore up to 4096 and more headroom — see
> [MLC_MIGRATION.md](./MLC_MIGRATION.md).

`beast` moved off Ollama to vLLM for a 2.4x speedup. **The Jetson moved off
Ollama to MLC-LLM** for ~1.5x (25 vs 16 tok/s) — vLLM was tried first and
**cannot run on the Orin Nano's unified-memory iGPU** (NVML + contiguous-KV
walls); MLC's TVM-compiled kernels sidestep both. TensorRT-LLM was also rejected.
See [MLC_MIGRATION.md](./MLC_MIGRATION.md), [TRTLLM_MIGRATION.md](./TRTLLM_MIGRATION.md),
[BENCHMARKS.md](./BENCHMARKS.md).

To change the model for **every** app on a host at once: on the MLC host edit
`MODEL`/`CTXS` in [`mlc/openclaw-mlc-run.sh`](./mlc/openclaw-mlc-run.sh) and
restart `openclaw-mlc.service`; on vLLM hosts edit `OPENCLAW_MODEL` in
[`vllm/openclaw-vllm.service`](./vllm/openclaw-vllm.service) and re-run
`./install-vllm.sh`. Nothing in any app changes either way.

## Install / update

**MLC host** (Jetson) — Docker with the NVIDIA runtime, JetPack 6.2 (L4T r36.4),
user-service linger enabled. Full details in [MLC_RUNBOOK.md](./MLC_RUNBOOK.md):

```bash
# copy the unit + wrapper into place, then:
cp mlc/openclaw-mlc-run.sh ~/.local/bin/ && chmod +x ~/.local/bin/openclaw-mlc-run.sh
cp mlc/openclaw-mlc.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now openclaw-mlc.service   # serves :11434, auto-starts at boot
```

**vLLM host** (beast) — Docker with the NVIDIA runtime, user in the `docker`
group. Retires the Ollama backend on :11434 and installs the vLLM user service:

```bash
./install-vllm.sh       # pull image if needed, stop Ollama, start vLLM on :11434
```

> `install.sh` / `Modelfile` are the retired **Ollama** provisioner, kept for
> reference only — no host runs Ollama anymore.

## Files

- `mlc/openclaw-mlc.service` — the **MLC-LLM user service** (Jetson): OpenAI API
  on :11434, model name `openclaw`, auto-starts at boot.
- `mlc/openclaw-mlc-run.sh` — wrapper it runs: starts the MLC container, tries
  4096 context and falls back if memory is too tight to generate.
- `MLC_RUNBOOK.md` — how to run/tune/manage MLC; `MLC_MIGRATION.md` — why MLC
  (and why vLLM can't run on the Jetson).
- `vllm/openclaw-vllm.service` — the vLLM user service (beast): OpenAI API on
  :11434, model name `openclaw`, INT4-AWQ Qwen3-4B.
- `install-vllm.sh` — idempotent vLLM provisioner (retires Ollama on :11434).
- `TRTLLM_MIGRATION.md` — why TensorRT-LLM was rejected; `BENCHMARKS.md` — numbers.
- `Modelfile`, `install.sh`, `systemd/openclaw.service`, `bin/openclaw-warmup.sh`
  — **retired** Ollama provisioner, kept for reference only.

## Migration note (Jetson, 2026-07-09)

The Jetson moved **Ollama → MLC-LLM** (~16 → ~25 tok/s). Ollama was fully removed
(binary, models, unit files, image). vLLM was evaluated first and cannot run on
the Orin Nano — see [MLC_MIGRATION.md](./MLC_MIGRATION.md) for the full story and
[MLC_RUNBOOK.md](./MLC_RUNBOOK.md) for operations.

_(Earlier, 2026-06-22: replaced the app-specific `lifeos-ollama.service` with the
shared Ollama `openclaw.service`, since also retired.)_
