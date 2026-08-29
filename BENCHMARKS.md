# Benchmarks

## 2026-08-05 — AWS EC2 g6.xlarge / Qwen3.5-9B

The EC2 host serves `Qwen/Qwen3.5-9B` with vLLM 0.18.1 on an NVIDIA L4. The
weights are BF16, the KV cache is FP8, the configured context is 16,384 tokens,
and the service caps continuous batching at four sequences. Thinking is
disabled server-wide and the public model name remains `openclaw`.

A short calibration used two warmups followed by five random requests, each
with 512 requested input tokens and 128 output tokens, concurrency one, an
unlimited offered request rate, and EOS ignored.

| Metric | EC2 L4 / Qwen3.5-9B |
|--------|---------------------:|
| Successful requests | 5 / 5 |
| Benchmark duration | 41.27 s |
| Request throughput | 0.12 req/s |
| Output throughput | **15.51 tok/s** |
| Peak output throughput | 16 tok/s |
| Total token throughput | 77.55 tok/s |
| Mean / median / P99 TTFT | 214.43 / 216.05 / 217.38 ms |
| Mean / P99 TPOT | 63.30 / 63.32 ms |
| Mean / P99 ITL | 62.80 / 63.55 ms |

Runtime footprint:

- Model loading used 16.8 GiB of VRAM.
- vLLM allocated 3.69 GiB to the paged KV cache (60,192-token capacity).
- The server reserved about 22,272 MiB according to `nvidia-smi`.
- vLLM reported theoretical 12x concurrency at the 16,384-token context; the
  service remains deliberately capped at four sequences.

The planned 10-warmup, 500-request sequential and four-way tests were stopped
before completion to avoid paying for a long EC2 benchmark, so no partial score
is reported. The 15.51 tok/s calibration is not an apples-to-apples model
comparison with `beast` or `jetson-orin`: EC2 is serving a 9B BF16 model, while
those hosts serve quantized 4B models. It describes this endpoint's observed
single-request speed, not relative GPU capability.

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

The same 10-warmup, 500-request benchmark was repeated after the switch. The
requested workload remained 512 input and 128 output tokens; Qwen3.5's tokenizer
produced 262,203 total input tokens across the 500 requests.

| Metric | Sequential (`max-concurrency=1`) | Batched (`max-concurrency=4`) | Batching gain |
|--------|----------------------------------:|------------------------------:|--------------:|
| Successful requests | 500 / 500 | 500 / 500 | no failures |
| Benchmark duration | 1,352.67 s | 410.90 s | **3.29x faster** |
| Request throughput | 0.37 req/s | 1.22 req/s | **3.30x** |
| Output throughput | 47.31 tok/s | 155.76 tok/s | **3.29x** |
| Peak output throughput | 57 tok/s | 208 tok/s | **3.65x** |
| Total token throughput | 241.16 tok/s | 793.88 tok/s | **3.29x** |
| Mean TTFT | 191.77 ms | 619.75 ms | 3.23x higher |
| Median TTFT | 191.42 ms | 682.14 ms | 3.56x higher |
| P99 TTFT | 207.46 ms | 750.19 ms | 3.62x higher |
| Mean TPOT | 19.78 ms | 21.00 ms | 6.2% higher |
| P99 TPOT | 20.49 ms | 23.10 ms | 12.7% higher |
| Mean ITL | 19.62 ms | 20.84 ms | 6.2% higher |
| P99 ITL | 21.05 ms | 21.96 ms | 4.3% higher |

Compared with the previous Qwen3-4B Instruct-2507 AWQ baseline, Qwen3.5 is about
42% slower in sequential output throughput (47.31 vs 81.35 tok/s) and 39% slower
when continuously batched four ways (155.76 vs 257.38 tok/s). Its active
single-request decode cadence is about 50.6 tok/s (`1000 / mean TPOT`).

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

## 2026-08-30 — HauhauCS aggressive Qwen3.5-4B on `beast`

The requested HauhauCS release contains only GGUF files. vLLM 0.24.0's
experimental GGUF path failed before weight loading, so the service stayed on
native vLLM formats: the BF16 safetensors reconstruction was compressed locally
to symmetric group-128 W4A16 using LLM Compressor. The visual encoder,
`lm_head`, and Gated DeltaNet linear-attention projections remain in BF16,
matching the supported Qwen3.5 compressed-tensors layout. vLLM serves the
result with its Marlin WNA16 kernel and `--language-model-only`.

- Checkpoint size: 4.92 GiB
- Model-loading GPU memory: 4.48 GiB
- API model alias: `openclaw`
- Direct-response smoke test: returned `READY` exactly, reasoning field null
- Single-request smoke test: 110 completion tokens in 2.129 s (**51.7 tok/s**)
- Four-request smoke test: 456 completion tokens in 3.569 s
  (**127.8 aggregate tok/s**)

These are short smoke tests, not directly comparable to the earlier 500-request
benchmark. The original AWQ checkpoint remains cached and the pre-change user
service is backed up on `beast` for rollback.
