# OpenClaw — one shared local LLM for all my apps

A single Ollama daemon + a single resident model, owned in one place. Every app
(life_os, etc.) is a **client** that points at it. No app ships or runs its own
Ollama, and no app names a raw model — they all ask for **`openclaw`**, and this
repo decides what that is.

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

`qwen3:4b-instruct-2507` (see [`Modelfile`](./Modelfile)) — the **instruct
(non-reasoning)** Qwen3 4B, Q4_K_M. No `<think>` blocks, so callers need no
special flags. Full GPU offload (all 36 layers, ~3.1 GiB VRAM) at ~15–18 tok/s
— which only fits because the **desktop GUI is disabled** (`multi-user.target`);
see the Modelfile for the full story.

To change the model for **every** app at once: edit `FROM` in the `Modelfile`,
then re-run `./install.sh` on the host. Nothing in any app changes.

## Install / update

Ollama must already be installed (e.g. `~/.local/ollama/bin/ollama`).

```bash
./install.sh            # create/refresh the openclaw alias + warmup script
./install.sh --service  # also install & start the systemd user service
```

## Files

- `Modelfile` — defines the `openclaw` model alias.
- `systemd/openclaw.service` — the shared Ollama user service (LAN-bound on
  :11434, `OLLAMA_KEEP_ALIVE=-1`, preloads `openclaw` at boot).
- `bin/openclaw-warmup.sh` — pins the model resident right after start.
- `install.sh` — idempotent provisioner.

## Migration note (Jetson, 2026-06-22)

Replaces the old app-specific `lifeos-ollama.service` (which preloaded
`qwen3.5:4b`). To cut over:

```bash
systemctl --user disable --now lifeos-ollama.service
./install.sh --service
```

Both bind the same port (11434), so run only one at a time.
