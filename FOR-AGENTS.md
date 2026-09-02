# For agents: use OpenClaw, don't run your own LLM

This machine/LAN already has ONE shared local LLM service. Every app is a
**client** of it. Do NOT install, bundle, run, or containerize your own LLM
server, and do NOT hardcode a raw model name (e.g. `qwen3.5:4b`, `llama3.2`). On
the 8 GB Jetson there is only room for ONE resident model — extra instances or a
different model cause RAM thrash and timeouts for everyone.

## The contract (all you depend on)

- **Base URL:** `http://192.168.3.30:11434` (LAN) · `http://host.docker.internal:11434` (from a container on the host) · `http://127.0.0.1:11434` (through the EC2 SSH tunnel) · `https://<public-ip-with-dashes>.sslip.io` (public EC2)
- **API:** OpenAI-compatible `POST /v1/chat/completions` **only** — Ollama-native
  `/api/generate` and `/api/chat` are **gone** (return 404). Use `/v1/*`.
- **Model name:** `openclaw` ← always use this literal string
- **Auth:** none on the Jetson/`beast` LAN endpoints; both the EC2 SSH-tunnel
  and public HTTPS endpoints require `Authorization: Bearer <key>` (see
  `EC2_RUNBOOK.md`)

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

- It is a direct-response language model — **no `<think>` blocks** and no
  per-request flags needed; just send messages and read the reply. The Jetson
  runs Qwen3-4B Instruct-2507, `beast` runs language-only Qwen3.5-4B, and the
  EC2 L4 host runs language-only Gemma 4 12B QAT W4A16 with FP8 KV cache.
  The name `openclaw` and API are the same everywhere; the serving engine
  depends on the host (Jetson → **MLC-LLM**; `beast` and EC2 → vLLM).
- Speed depends on the host: **~22 tok/s** on the Jetson (MLC); Qwen3.5 on
  `beast` decodes at **~51 tok/s** for one active request and delivers ~156
  aggregate tok/s with four-way continuous batching. The current EC2 Gemma
  model has only received a short correctness smoke test; no benchmark was run.
  These are not apples-to-apples model comparisons. None is GPT-4 class —
  design accordingly.
- Keep prompts within **~4096 tokens** total on the Jetson (its current context
  window). `beast` is configured for 32,768 tokens and EC2 for 16,384 tokens. A
  client should still set a timeout and fallback appropriate to the host and
  workload.
- Design for it: keep prompts tight and explicit; if you need strict JSON, say
  *"return ONLY a JSON array, no prose, no code fences"* and parse tolerantly.
- It can be slow/unreachable under memory pressure. Treat every call as
  **best-effort**: wrap it, set a timeout, and fall back to deterministic
  behavior. Never let an LLM failure crash a request or block a job.

## Changing the model (one place, affects all apps)

This repo owns the service. To swap the model: on the Jetson edit `MODEL` in
[`mlc/openclaw-mlc-run.sh`](./mlc/openclaw-mlc-run.sh) and restart
`openclaw-mlc.service`; on `beast` edit the vLLM user service; on EC2 edit
[`vllm/openclaw-vllm-ec2.service`](./vllm/openclaw-vllm-ec2.service) and rerun
`./install-vllm-ec2.sh`. See [`MLC_RUNBOOK.md`](./MLC_RUNBOOK.md) and
[`EC2_RUNBOOK.md`](./EC2_RUNBOOK.md).

Do **not** solve a model-quality problem by spinning up your own model in your
app — raise it against this repo so the change is shared.

---

**Rule of thumb:** your app provides prompts + parsing + graceful fallback.
OpenClaw provides the model. One model, one daemon, shared by all. See
[`README.md`](./README.md) for the operator/setup details.
