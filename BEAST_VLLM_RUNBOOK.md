# Beast vLLM runbook

`beast` is the OpenClaw vLLM host: an RTX 3070 Ti Laptop GPU with 8 GB VRAM.
Its current model is HauhauCS Qwen3.5-4B-VL W4A16. A cached
`QuantTrio/Qwen3.5-4B-AWQ` checkpoint is available as a temporary alternative.
Both use the OpenAI-compatible endpoint on port 11434 and public model alias
`openclaw`.

This document is specific to Beast and vLLM. It is not a Jetson or MLC record.

## Current HauhauCS serving configuration

The user-systemd service is `openclaw-vllm.service`. Its source is
[`vllm/openclaw-vllm.service`](vllm/openclaw-vllm.service).

The live HauhauCS service source sets an 8,192-token total sequence limit and
an eight-sequence scheduler cap, plus its multimodal memory settings. The
previous QuantTrio service used a separately validated, more conservative
6,144-token/four-sequence configuration. The HauhauCS settings and evidence
are in [`vllm/openclaw-vllm.service`](vllm/openclaw-vllm.service) and
[`UNCENSORED_VL_W4A16_EXPERIMENT_LOG.md`](UNCENSORED_VL_W4A16_EXPERIMENT_LOG.md).

## Temporary model switch

Install the selector on Beast once, then use it whenever you want to change
models:

```bash
mkdir -p ~/.local/bin
install -m 0755 vllm/switch-beast-model.sh ~/.local/bin/openclaw-model
```

If the repo is not checked out on Beast, copy `vllm/switch-beast-model.sh`
there first. Then run:

```bash
openclaw-model use quanttrio  # switch to QuantTrio Qwen3.5-4B-AWQ
openclaw-model use hauhaucs   # restore HauhauCS Qwen3.5-4B-VL W4A16
openclaw-model status         # show selection and endpoint status
```

The selector writes its own systemd drop-in, so the checked-in service file
and unrelated overrides are left intact. Switching restarts the shared vLLM
service; clients briefly lose the endpoint while the selected model loads.
QuantTrio uses its previously validated 6,144-token, four-sequence language-
only configuration and cached revision `32c292e3a73afe1138518180b1b6d2868c980ee2`.
Choosing `hauhaucs` removes only the selector's drop-in and returns to the
base service configuration. Both checkpoints are already cached on Beast.

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
