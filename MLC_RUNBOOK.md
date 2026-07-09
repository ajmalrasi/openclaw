# MLC-LLM runbook (Jetson `openclaw` endpoint)

How to run, tune, and manage the MLC-LLM serving of `openclaw` on the Jetson
Orin Nano. Background & rationale: [MLC_MIGRATION.md](MLC_MIGRATION.md).

- **Host:** `ajmalrasi@192.168.3.30` (JetPack 6.2 / L4T r36.4.7, sm87, 7.4 GB)
- **Endpoint:** `http://192.168.3.30:11434/v1` (OpenAI-compatible)
- **Image:** `dustynv/mlc:0.20.0-r36.4.0`
- **Model:** `HF://FutureProofHomes/Qwen3-4B-Instruct-2507-q4f16_2-MLC`
  (non-reasoning, JIT-compiled for sm87, cached). Context **2048** — this
  `q4f16_2` quant's params are ~2.7 GB, so 4096 won't fit with generation
  headroom, and it needs clean memory to load.

---

## Quick start (manual, one-off)

```bash
docker run -d --name openclaw-mlc \
  --runtime nvidia --ipc=host --restart no \
  -p 11434:8000 \
  -v /home/ajmalrasi/.cache/huggingface:/root/.cache/huggingface \
  -v /home/ajmalrasi/.cache/mlc_llm:/root/.cache/mlc_llm \
  dustynv/mlc:0.20.0-r36.4.0 \
  mlc_llm serve HF://FutureProofHomes/Qwen3-4B-Instruct-2507-q4f16_2-MLC \
    --mode interactive \
    --overrides "context_window_size=2048;prefill_chunk_size=512" \
    --host 0.0.0.0 --port 8000
```

First run downloads the model (~2.7 GB) and **JIT-compiles the sm87 kernel
library** (a few minutes); both are cached in the mounted volumes, so restarts
are fast. **Note:** MLC compiles the lib per `context_window_size`, so changing
context triggers a fresh ~3 min compile.

### Parameter reference

| Part | Why |
|---|---|
| `--runtime nvidia` | Required — gives the container the Tegra GPU. |
| `--ipc=host` | Shared-memory for the engine. |
| `-p 11434:8000` | MLC serves on 8000 inside; map to **11434** (the openclaw contract port). |
| `-v …/huggingface` | Persist downloaded model weights. |
| `-v …/mlc_llm` | Persist the **JIT-compiled** `.so` (avoids recompiling every start). |
| `mlc_llm serve HF://…` | Serve model straight from HF; MLC compiles for the local GPU. |
| `--mode interactive` | Single-user (batch size 1), modest KV. Use `local`/`server` for more batching (more memory). |
| `context_window_size=2048` | KV cache span. **~0.28 MB/token**. Kept at 2048 because the `q4f16_2` params (~2.7 GB) leave little room; lower it if it OOMs on fragmented memory. |
| `prefill_chunk_size=512` | Caps the temp buffer during prefill (smaller = less memory). |
| `--host 0.0.0.0` | Listen on all interfaces (LAN). |

**Do NOT** pass `--served-model-name` / worry about the model name — MLC ignores
the request's `model` field and serves the one loaded model, so clients can send
`model: "openclaw"` as-is.

---

## Production service (systemd, boot-start)

Installed as a **user** service (linger enabled, so it starts at boot without
login). Files (also mirrored in this repo under [`mlc/`](mlc/)):

- Unit:    `~/.config/systemd/user/openclaw-mlc.service`
- Wrapper: `~/.local/bin/openclaw-mlc-run.sh`

The wrapper starts at **2048** (the reliable ceiling for the heavier `q4f16_2`
model) and falls back to `1024` if needed. Its readiness check does a **real
1-token generation** (90 s timeout for cold starts), not just a `/v1/models`
ping — so it also catches a "serves but can't *generate*" case. Override the
context list with `OPENCLAW_CTXS` (e.g. a lighter model could use `4096 2048`).

### Manage

```bash
systemctl --user status  openclaw-mlc.service
systemctl --user restart openclaw-mlc.service
systemctl --user stop    openclaw-mlc.service
systemctl --user start   openclaw-mlc.service
docker logs -f openclaw-mlc          # live model/engine logs
docker logs openclaw-mlc 2>&1 | grep "KV cache token capacity is"   # which context it settled at
```

Enable / disable boot-start:
```bash
systemctl --user enable  openclaw-mlc.service
systemctl --user disable openclaw-mlc.service
```

### Force a specific context (skip fallback)
```bash
OPENCLAW_CTXS=2048 ~/.local/bin/openclaw-mlc-run.sh    # or edit the unit's env
```

---

## Using it

```bash
# OpenAI-compatible chat (the openclaw contract). model name is ignored — send "openclaw".
curl -s http://192.168.3.30:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openclaw","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'

# list models
curl -s http://192.168.3.30:11434/v1/models
```

**Note:** this is the **non-reasoning** Instruct-2507 model — replies have **no
`<think>` block**, just the answer.

**Ollama-native `/api/*` endpoints are NOT served** — only OpenAI `/v1/*`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA: out of memory` at the **KV cache** step (startup) | fragmented memory can't give the big contiguous KV block | **Clear memory in jtop** (frees cache — lighter than reboot) then `systemctl --user restart openclaw-mlc`; or **reboot**. The wrapper also auto-falls-back to a smaller context. |
| Serves `/v1/models` (200) but **generations hang / error** | 4096 KV allocated at startup, but per-request working memory OOMs (too little headroom on busy memory) | **Clear memory in jtop** + `systemctl --user restart openclaw-mlc` (the wrapper's real-generation check will settle on a context that can generate). |
| Endpoint down, service crash-looping | all fallback contexts failed (badly fragmented) | `systemctl --user stop openclaw-mlc`, **clear memory (jtop) or reboot**, then start. |
| `NvRmGpuLibOpen failed` / `/dev/nvhost-gpu` missing | a bad `cma=` boot arg broke the GPU | restore `/boot/extlinux/extlinux.conf.bak` + reboot. **Never set `cma=` on Tegra.** |
| Slow first start | JIT compiling the sm87 lib | one-time; cached in `~/.cache/mlc_llm`. |
| `NVML_SUCCESS == r` assert | that's **vLLM**, not MLC | don't use vLLM here — see MLC_MIGRATION.md. |

**Golden rule:** if MLC won't allocate the KV cache *or* can't generate, the
problem is almost always **memory pressure / fragmentation** — clearing cache in
**jtop** (or a reboot) frees it, and restarting the service lets it grab a clean
context. Do not reach for `cma=`.

**Memory reality:** the `q4f16_2` Instruct-2507 uses ~5 GB at 2048 context
(params 2.7 + KV 0.6 + CUDA context + temp), leaving ~2 GB headroom on this 7.4 GB
box. The 2.7 GB params need **clean, low-fragmentation memory to load** — at boot
it's fine, but a mid-session restart on a busy/fragmented box can fail the param
load (clear cache in jtop or reboot). A lighter `q4f16_1` build (compiled from the
FP16 base) would load in ~2.1 GB and allow higher context — the more robust
long-term option.
