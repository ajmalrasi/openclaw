# Beast vLLM runbook

`beast` is the OpenClaw vLLM host: an RTX 3070 Ti Laptop GPU with 8 GB VRAM.
It serves `QuantTrio/Qwen3.5-4B-AWQ` through the OpenAI-compatible endpoint on
port 11434, using the public model alias `openclaw`.

This document is specific to Beast and vLLM. It is not a Jetson or MLC record.

## Validated serving configuration

The user-systemd service is `openclaw-vllm.service`. Its source is
[`vllm/openclaw-vllm.service`](vllm/openclaw-vllm.service).

| Setting | Value | Purpose |
| --- | ---: | --- |
| Model | `QuantTrio/Qwen3.5-4B-AWQ` | INT4 AWQ Qwen3.5 checkpoint |
| Model alias | `openclaw` | Stable shared API contract |
| Total sequence limit | 6,144 tokens | Input plus generated output per request |
| Active sequences | 4 | Four requests execute simultaneously; later requests queue |
| Prefill chunk limit | 1,024 tokens | Bounds GDN/FLA prefill workspace, not prompt length |
| FP8 KV cache reservation | 1.2 GB | Leaves workspace headroom on the 8 GB GPU |
| GPU-memory utilization guard | 0.95 | Avoids startup rejection seen at 0.97 |
| Execution | eager | Required validated runtime mode |

A 4K-token input is allowed. With the 6,144-token total sequence limit, it
leaves approximately 2,144 tokens for output.

## Why the previous eight-way configuration crashed

The previous service admitted eight sequences and reserved 1.8 GB for FP8 KV
cache. Its reported cache capacity was 60,854 tokens, which can make eight-way
serving appear safe. That metric is cache capacity only; it does not reserve
temporary GDN/Flash Linear Attention workspace.

During the failing workload, vLLM had four approximately 3,045–3,061-token
requests running, with only 20.4% KV-cache use. The Qwen3.5 GDN/FLA path then
failed to allocate 16 MiB: only 3 MiB VRAM was free and the process already
held 7.53 GiB. The EngineCore exited, causing every in-flight request to return
HTTP 500 with `EngineDeadError`.

Reducing only `--max-model-len` would not have fixed this: the old KV cache was
manually fixed at 1.8 GB. The current configuration both reduces the context
limit and releases about 560 MB of cache reservation for transient workspace.

## Verification

On 2026-09-13, after applying the four-way configuration:

- Startup reported 7.43 GiB free before allocation and a 1.12 GiB KV cache.
- vLLM reported 35,328 cache tokens and theoretical 5.75x concurrency at the
  6,144-token sequence limit.
- Four simultaneous requests, each with 3,016 prompt tokens and a small output
  budget, all returned HTTP 200 (`finish_reason: stop`).
- The service remained active, with no `CUDA out of memory`, `EngineDeadError`,
  or fatal engine event in the post-test logs.

This validates the representative four-way workload that previously crashed.
It is not a guarantee that arbitrarily long outputs, tool schemas, or a
different vLLM/model build have the same memory profile.

## Operations

```bash
systemctl --user status openclaw-vllm.service
systemctl --user restart openclaw-vllm.service
journalctl --user -u openclaw-vllm.service -e --no-pager
```

The endpoint is `http://beast:11434/v1/chat/completions` (or Beast's LAN IP),
with model name `openclaw` and no authentication on the LAN.

If an OOM recurs, do not increase the KV cache or active-sequence cap. Capture
the EngineCore traceback and scheduler state first, then lower the active cap,
the prefill chunk limit, or the fixed KV reservation as indicated by the
observed allocation pressure.
