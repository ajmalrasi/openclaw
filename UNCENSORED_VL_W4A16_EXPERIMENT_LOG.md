# Uncensored Qwen3.5-4B-VL on Beast: W4A16 + video input experiment log

Goal: serve the uncensored HauhauCS Qwen3.5-4B checkpoint through vLLM on
`beast` (RTX 3070 Ti Laptop, 8 GB VRAM), with vision intact, fast enough to
process video frames, and cheap enough to run locally instead of on EC2.

Source model: `DreamFast/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark`
(bf16 safetensors reconstruction of the HauhauCS Aggressive GGUF).

## What didn't work

**AWQ via `llmcompressor` (self-quantized).** `AWQModifier` auto-detected the
hybrid full/linear-attention architecture and built mappings for it, but
crashed during the smoothing step:

```
TypeError: Qwen3_5GatedDeltaNet.forward() missing 1 required positional argument: 'hidden_states'
```

`llmcompressor`'s AWQ smoothing replays a cached forward call using only
recorded kwargs; the linear-attention (Gated DeltaNet) module expects
`hidden_states` positionally, so the replay fails. This is a library gap in
AWQ support for Qwen3.5's hybrid attention, not a resource problem — it
happened consistently regardless of GPU memory available.

**Full bf16 on 8 GB VRAM.** The checkpoint is ~8.5 GB in bf16 alone, larger
than the card's usable ~7.7 GiB, before KV cache or vision activations. vLLM's
`--cpu-offload-gb` makes it *possible* (streaming weights from system RAM),
and it does run at full precision, but it is slow: PCIe-bound, ~3-4x slower
than a fully GPU-resident quantized model in testing.

## What worked: RTN W4A16, no calibration

A previously-quantized checkpoint already existed on beast at
`~/.cache/openclaw-models/hauhaucs-w4a16/`, built with plain
`QuantizationModifier` (round-to-nearest, **not** AWQ):

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head", "re:.*visual.*", "re:.*linear_attn.*"],
)
oneshot(model=model, recipe=recipe)  # requires_calibration_data: false
```

This sidesteps the AWQ smoothing bug entirely (no smoothing step exists for
plain RTN quantization) and needs **no calibration dataset** — directly
avoiding any small-calibration-set quality concern. Only the full-attention
and MLP `Linear` layers are quantized to 4-bit; the vision tower, `lm_head`,
and the Gated DeltaNet linear-attention projections stay bf16. Quantization
itself ran on CPU (`device_map="cpu"`), avoiding GPU OOM during the build.

Result: ~4.92 GB checkpoint, `compressed-tensors` format, loads with vLLM's
Marlin W4A16 kernel automatically.

## The remaining blocker: vision encoder profiling OOM

Even with 4-bit weights, vLLM's one-time multimodal profiling step (sizing
memory for the worst-case image) tried to process an image at the default
`longest_edge: 16777216` px cap. That single profiling pass could OOM the
8 GB card by itself, independent of `max_num_seqs` or `max_model_len` —
reducing concurrency or context did not help, because the profiling cost is
tied to per-image resolution, not batch size.

Fix: cap per-frame resolution via `--mm-processor-kwargs`:

```
--mm-processor-kwargs '{"max_pixels": 200704}'
```

This dropped peak vision activation from ~1.7 GiB to ~0.36 GiB, which was
enough to fit the entire quantized model **fully GPU-resident** (zero
`--cpu-offload-gb`) alongside a usable KV cache.

## Validated serving configuration

Container: `openclaw-vllm-w4a16` (Docker, `vllm/vllm-openai:latest`,
`--restart unless-stopped`). Not yet wired into `openclaw-vllm.service`.

| Setting | Value | Purpose |
| --- | ---: | --- |
| Model path | `~/.cache/openclaw-models/hauhaucs-w4a16` | Local W4A16 checkpoint |
| Model alias | `openclaw` | Served name |
| `max-model-len` | 8192 | Context per request |
| `max-num-seqs` | 3 | Concurrent requests (4th+ queues) |
| `gpu-memory-utilization` | 0.90 | |
| `mm-processor-kwargs` | `{"max_pixels": 200704}` | Caps per-frame resolution, ~0.44 MP |
| `limit-mm-per-prompt` | `{"video":1,"image":0}` | One video per request |
| CPU offload | **0 GB** | Fully GPU-resident |
| KV cache | 28,672 tokens | 3.5x concurrency headroom at 8192 tokens/request |

Endpoint: `http://192.168.3.226:11434/v1/chat/completions` (LAN).

### Measured latency (single request, `Say OK in one word`, 100 tokens)

| Config | Time |
| --- | ---: |
| bf16, `--cpu-offload-gb 6` | >120s (timed out) |
| W4A16, `--cpu-offload-gb 2` | ~28-33s |
| W4A16, fully GPU-resident (after `max_pixels` fix) | ~3-4s |

## Native video input works

Qwen3.5-VL's own video processor (`fps: 2`, `min_frames: 4`, `max_frames: 768`
from `processor_config.json`) handles frame sampling when sent a `video_url`
content block directly — no need to manually split frames into an image
array. Confirmed with a synthetic 6s red→green→blue clip.

**Timestamps are approximate, not frame-exact.** Ground truth was 0-2s /
2-4s / 4-6s; the model returned 0-3s / 3-4s / 4-6s — correct order and rough
proportions, soft boundaries. This model was fine-tuned to remove refusals,
not for video temporal grounding, so downstream consumers of timestamps
should treat them as approximate.

### Getting clean, complete JSON

Two things were required together:

1. **Disable thinking** per-request (`chat_template_kwargs: {"enable_thinking": false}`)
   — otherwise visible chain-of-thought reasoning burns most of `max_tokens`
   before the actual answer, risking truncation.
2. **Guided/structured output** via `response_format: {"type": "json_schema", ...}`
   — guarantees complete, schema-valid JSON instead of free text that merely
   looks like JSON.

Applying both took the same request from 16.6s (verbose reasoning, truncated
JSON) to 3.5s (clean, complete JSON).

## Open items

- Not wired into systemd — a beast reboot needs the container relaunched
  manually (`docker run ...` command above is idempotent to rerun; Ollama's
  `openclaw-ollama.service` is enabled and will grab port 11434 first on
  boot, so stop it before relaunching vLLM).
- Real concurrent batch throughput (multiple videos in flight) not yet
  benchmarked — only sequential single-request latency has been measured.
- FP8 not tried as a quality/size tradeoff between W4A16 and bf16; on this
  Ampere (SM86) GPU it would only save memory, not add speed (no native FP8
  tensor cores), so it's a quality lever, not a speed one.
