# For agents: use OpenClaw, don't run your own LLM

This machine/LAN already has ONE shared local LLM service. Every app is a
**client** of it. Do NOT install, bundle, run, or containerize your own Ollama,
and do NOT hardcode a raw model name (e.g. `qwen3.5:4b`, `llama3.2`). On the
8 GB Jetson there is only room for ONE resident model — extra instances or a
different model cause RAM thrash and timeouts for everyone.

## The contract (all you depend on)

- **Base URL:** `http://192.168.3.30:11434` (LAN) · `http://host.docker.internal:11434` (from a container on the host)
- **API:** OpenAI-compatible `POST /v1/chat/completions` (native `/api/*` also works)
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

- It's an Ollama model **alias**, currently → `qwen3:4b-instruct-2507`, the
  **instruct (non-reasoning)** Qwen3 4B. No `<think>` blocks, no per-request
  flags needed — just send messages and read the reply.
- Runs at ~8–9 tok/s on the Jetson (GPU layers capped via `num_gpu` to dodge a
  ~1.9 GiB single-alloc limit; rest on CPU — see the repo Modelfile). Usable,
  but **not** GPT-4 class and not instant.
- Design for it: keep prompts tight and explicit; if you need strict JSON, say
  *"return ONLY a JSON array, no prose, no code fences"* and parse tolerantly.
- It can be slow/unreachable under memory pressure. Treat every call as
  **best-effort**: wrap it, set a timeout, and fall back to deterministic
  behavior. Never let an LLM failure crash a request or block a job.

## Changing the model (one place, affects all apps)

This repo owns the service. To swap the model: edit the `FROM` line in the
[`Modelfile`](./Modelfile), push, `git pull` on the host, re-run `./install.sh`.

Do **not** solve a model-quality problem by spinning up your own model in your
app — raise it against this repo so the change is shared.

---

**Rule of thumb:** your app provides prompts + parsing + graceful fallback.
OpenClaw provides the model. One model, one daemon, shared by all. See
[`README.md`](./README.md) for the operator/setup details.
