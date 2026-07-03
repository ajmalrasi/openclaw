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
- **Always send `"think": false`** — see below, this is required now, not optional.

## How to integrate

Put the base URL and model in config/env, never inline. Mirror life_os:

```
OPENCLAW_BASE_URL=http://host.docker.internal:11434   # or the LAN URL
OPENCLAW_MODEL=openclaw
```

Call it like any OpenAI chat endpoint — **and always send `"think": false`**,
see below for why:

```bash
curl -s http://192.168.3.30:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"openclaw","stream":false,"think":false,
       "messages":[{"role":"system","content":"..."},
                   {"role":"user","content":"..."}]}'
# response: data.choices[0].message.content
```

Or via native `/api/chat`, same field:

```bash
curl -s http://192.168.3.30:11434/api/chat \
  -d '{"model":"openclaw","stream":false,"think":false,
       "messages":[{"role":"user","content":"..."}]}'
```

## What `openclaw` is (and isn't)

- It's an Ollama model **alias**, currently → `qwen3:4b`, a 4B **hybrid
  reasoning** model. It defaults to emitting a `<think>...</think>` block
  before every answer, which is why **every caller must pass `"think":
  false`** — Ollama has no Modelfile-level way to disable thinking by default
  yet (open upstream: ollama/ollama#14617, #14809), so this can't be baked
  into the alias itself. Forget the flag and a reply can take 130s+ and blow
  past your timeout.
- Fast (~2–3 s warm) with thinking off, but **not** GPT-4 class.
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
