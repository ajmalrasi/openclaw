# For agents: use OpenClaw, don't run your own LLM

This machine/LAN already has ONE shared local LLM service. Every app is a
**client** of it. Do NOT install, bundle, run, or containerize your own LLM
server, and do NOT hardcode a raw model name (e.g. `qwen3.5:4b`, `llama3.2`). On
the 8 GB Jetson there is only room for ONE resident model — extra instances or a
different model cause RAM thrash and timeouts for everyone.

## The contract (all you depend on)

- **Base URL:** `http://192.168.3.30:11434` (LAN) · `http://host.docker.internal:11434` (from a container on the host)
- **API:** OpenAI-compatible `POST /v1/chat/completions` **only** — Ollama-native
  `/api/generate` and `/api/chat` are **gone** (return 404). Use `/v1/*`.
- **Model name:** `openclaw` ← always use this literal string
- **Auth:** none by default (no API key)

## How to integrate

Put the base URL and model in config/env, never inline. Mirror life_os:

```
OPENCLAW_BASE_URL=http://host.docker.internal:11434   # or the LAN URL
OPENCLAW_MODEL=openclaw
```

Call it like any OpenAI chat endpoint:

```bash
curl -s http://192.168.3.30:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"openclaw","stream":false,
       "messages":[{"role":"system","content":"..."},
                   {"role":"user","content":"..."}]}'
# response: data.choices[0].message.content
```

## What `openclaw` is (and isn't)

- It's **Qwen3 4B**. The name `openclaw` and the API are the same everywhere; the
  serving engine depends on the host (Jetson → **MLC-LLM**; `beast` laptop →
  vLLM). You never need to care which.
- **⚠️ `<think>` blocks:** the Jetson (`192.168.3.30`, the LAN default) runs the
  **thinking** variant, so replies begin with a `<think>…</think>` block before
  the real answer. **Strip it** before using the output (take everything after
  the last `</think>`). `beast` runs the non-thinking Instruct model and does
  not. Write parsing that tolerates both.
- Speed depends on the host: **~23–25 tok/s** on the Jetson (MLC), ~96 tok/s on
  `beast` (vLLM + INT4-AWQ). Either way it's **not** GPT-4 class — design accordingly.
- Design for it: keep prompts tight and explicit; if you need strict JSON, say
  *"return ONLY a JSON array, no prose, no code fences"* and parse tolerantly.
- It can be slow/unreachable under memory pressure. Treat every call as
  **best-effort**: wrap it, set a timeout, and fall back to deterministic
  behavior. Never let an LLM failure crash a request or block a job.

## Changing the model (one place, affects all apps)

This repo owns the service. To swap the model: on the Jetson edit `MODEL` in
[`mlc/openclaw-mlc-run.sh`](./mlc/openclaw-mlc-run.sh) and restart
`openclaw-mlc.service`; on `beast` edit the vLLM service. See
[`MLC_RUNBOOK.md`](./MLC_RUNBOOK.md).

Do **not** solve a model-quality problem by spinning up your own model in your
app — raise it against this repo so the change is shared.

---

**Rule of thumb:** your app provides prompts + parsing + graceful fallback.
OpenClaw provides the model. One model, one daemon, shared by all. See
[`README.md`](./README.md) for the operator/setup details.
