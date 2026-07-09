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
- 8 GB serving flags that matter: `--max-num-seqs 1` (single-user; caps the
  huge float32 logits buffer over Qwen's 152k vocab that OOMs vLLM's memory
  profiling) and `--enforce-eager` (skip CUDA graphs to save VRAM).
- The Jetson is still on Ollama at ~16 tok/s **for now**; migrating it to vLLM is
  the next step (pending an aarch64/JetPack vLLM build + engine benchmarking).

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
