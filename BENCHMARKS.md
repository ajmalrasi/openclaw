# Benchmarks

Same `openclaw` model (`qwen3:4b-instruct-2507-q4_K_M`, Q4_K_M) served via Ollama,
measured with the `/api/chat` `eval_count`/`eval_duration` fields (`stream: false`).
3 runs each, prompt: "Explain how photosynthesis works in detail, covering light
and dark reactions." (~1750-2000 output tokens per run).

## 2026-07-09

| Host | GPU | GPU offload | Generation | Prompt processing | Cold load |
|------|-----|-------------|------------|--------------------|-----------|
| `beast` (laptop) | RTX 3070 Ti Laptop (8 GB dedicated VRAM) | 92% (near-full) | **~40.5 tok/s** (40.5 / 40.7 / 40.5) | ~590-820 tok/s | ~1.6 s |
| `jetson-orin` | Orin iGPU (7.4 GB unified memory) | 55% reported / 36-37 layers actually offloaded | **~16.0 tok/s** (16.0 / 16.0 / 16.0) | ~116-216 tok/s | ~0.6 s |

## 2026-07-09 — backend swap on `beast`: Ollama → vLLM (INT4-AWQ)

After ruling out TensorRT-LLM (see [TRTLLM_MIGRATION.md](./TRTLLM_MIGRATION.md)),
`beast` moved to **vLLM** serving `Eslzzyl/Qwen3-4B-Instruct-2507-AWQ` (INT4 AWQ,
Marlin kernels). Same photosynthesis prompt; generation tok/s measured from a
streaming client (tokens ÷ time-between-first-and-last token, so TTFT excluded).

| Host | Backend | Model / quant | Generation | Output | vs Ollama |
|------|---------|---------------|------------|--------|-----------|
| `beast` | Ollama | qwen3:4b-instruct-2507 Q4_K_M | ~40.5 tok/s | ✅ correct | 1.0x |
| `beast` | TensorRT-LLM (pytorch) | INT4 W4A16-AWQ | ~100 tok/s | ❌ garbage (kernel bug) | — |
| `beast` | TensorRT-LLM (pytorch) | INT8 W8A8-SQ | — | ❌ won't load | — |
| **`beast`** | **vLLM** | **INT4 AWQ (Marlin)** | **~96 tok/s** (100.7 / 95.6 / 92.4) | ✅ correct | **2.4x** |

- vLLM TTFT is ~0.03 s (instant) after the model is resident.
- 8 GB serving flags that matter: `--max-num-seqs 4` (validated below) and
  `--enforce-eager` (skip CUDA graphs to save VRAM).
- The Jetson is still on Ollama at ~16 tok/s **for now**; migrating it to vLLM is
  the next step (pending an aarch64/JetPack vLLM build + engine benchmarking).

## 2026-07-24 — vLLM PagedAttention / continuous batching on `beast`

Compared the same resident vLLM model with `--max-num-seqs 1` and `4`. Both
runs used 10 warmups followed by 500 random prompts, each with 512 requested
input tokens and 128 output tokens, through the OpenAI-compatible chat endpoint.
The single-sequence run used a 1 RPS Poisson arrival rate and concurrency 1. The
batched run used an unlimited offered rate and concurrency 4 to saturate
continuous batching.

| Metric | Sequential (`max-num-seqs=1`) | Batched (`max-num-seqs=4`) | Change |
|--------|-------------------------------:|---------------------------:|-------:|
| Successful requests | 500 / 500 | 500 / 500 | no failures |
| Benchmark duration | 786.74 s | 248.66 s | **3.16x faster** |
| Request throughput | 0.64 req/s | 2.01 req/s | **3.14x** |
| Output throughput | 81.35 tok/s | 257.38 tok/s | **3.16x** |
| Peak output throughput | 105 tok/s | 376 tok/s | **3.58x** |
| Total token throughput | 411.83 tok/s | 1,302.97 tok/s | **3.16x** |
| Mean TTFT | 153.47 ms | 473.67 ms | 3.09x higher |
| Median TTFT | 153.29 ms | 557.41 ms | 3.64x higher |
| P99 TTFT | 166.17 ms | 634.76 ms | 3.82x higher |
| Mean TPOT | 11.17 ms | 11.93 ms | 6.8% higher |
| P99 TPOT | 12.05 ms | 15.35 ms | 27.4% higher |
| Mean ITL | 11.08 ms | 11.84 ms | 6.9% higher |
| P99 ITL | 12.31 ms | 12.28 ms | unchanged |

The server reserved essentially the same VRAM in both configurations: about
7,300 MiB according to `nvidia-smi`. vLLM reported 4.27 GiB available for the
paged KV cache, a capacity of 31,104 tokens, and theoretical concurrency of
7.59 requests at the configured 4,096-token context. During this workload, four
requests ran concurrently and used about 7-8% of the KV cache.

Conclusion: `max-num-seqs=4` triples aggregate throughput without increasing the
reserved VRAM ceiling. Under full saturation, the tradeoff is mean TTFT rising
from 153 ms to 474 ms; streaming cadence changes little. The `beast` service is
therefore configured for four sequences.

## 2026-07-24 — `beast` model switch to Qwen3.5-4B

After the batching benchmark above, `beast` moved to
`QuantTrio/Qwen3.5-4B-AWQ`, loaded with `--language-model-only` and the
server-wide chat-template default `{"enable_thinking":false}`. A live API check
returned `READY` exactly, with no `<think>` block and a null reasoning field.

- Resident GPU memory: 7,086 MiB
- Paged KV-cache budget: 1.62 GiB
- KV-cache capacity: 37,236 tokens
- Reported maximum concurrency at the configured 4,096-token context: 9.09x
- Serving concurrency remains capped at the validated `--max-num-seqs 4`

The performance figures above describe the previous Qwen3-4B Instruct-2507 AWQ
checkpoint and remain as the pre-switch baseline. Qwen3.5 performance
benchmarking is still pending.

Notes:
- The RTX 3070 Ti has dedicated VRAM independent of system RAM; the Jetson's
  GPU and CPU share one 7.4 GB pool, so other resident services (openclaw
  gateway, postgres, dockerd, etc.) compete with the model for the same
  memory budget.
- Forcing a clean reload on the Jetson with ~6 GB free (stopped
  `openclaw.service`, freed memory, restarted) still offloaded 36/37 layers
  to GPU — same as the normal 55%-reported run — and generation speed was
  unchanged (~15.3 tok/s). The `ollama ps` CPU/GPU % on Jetson does not
  reliably reflect actual layer placement; check `ollama serve` logs
  (`load_tensors: offloaded N/M layers to GPU`) for ground truth.
- Bottleneck on Jetson is raw GPU compute (Orin iGPU has far fewer CUDA cores
  and lower memory bandwidth than a discrete laptop GPU), not memory
  pressure or offload configuration — the RTX 3070 Ti is ~2.5x faster on
  generation.
