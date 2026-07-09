# Jetson → MLC-LLM migration (findings & rationale)

**Outcome (2026-07-09):** On the Jetson Orin Nano, `openclaw` now serves via
**MLC-LLM (TVM)** instead of Ollama — **~25 tok/s vs Ollama's ~16 (~1.5×)**,
OpenAI-compatible API on `:11434`, model name `openclaw`. vLLM was tried
extensively first and **cannot run on this hardware**; the reasons are the
valuable part of this document.

Hardware: Jetson Orin Nano Super Dev Kit, **JetPack 6.2 (L4T r36.4.7)**,
CUDA 12.6, GPU compute **sm 8.7**, **7.4 GB unified memory** (CPU+GPU shared).

---

## TL;DR decision table

| Engine | NVML assert | Big-contiguous KV wall | Jetson support | Result |
|---|---|---|---|---|
| **MLC-LLM (TVM)** | avoids (own runtime) | avoids / tunable | first-class container | **shipped, ~25 tok/s** |
| llama.cpp / Ollama | avoids | avoids (chunked KV) | proven | the baseline we replaced |
| vLLM | **hits** | **hits** | container exists | **cannot start** |
| LMDeploy/TurboMind | avoids | likely hits | build-from-source | not worth it |

---

## Why vLLM cannot run on the Orin Nano

vLLM assumes a **discrete datacenter GPU**. Every one of those assumptions
breaks on Tegra's unified-memory iGPU. Two hard walls, proven empirically over
6 launch attempts:

### Wall 1 — NVML is a stub on Tegra
vLLM (via PyTorch's `CUDACachingAllocator`) calls `nvmlDeviceGetMemoryInfo()`
to track free VRAM. On Jetson NVML returns an error, so PyTorch aborts with:

```
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at
"/opt/pytorch/c10/cuda/CUDACachingAllocator.cpp":1016
```

This is a **platform-level PyTorch↔Tegra incompatibility**, not a vLLM bug
(pytorch#122068, dusty-nv/jetson-containers#1568). Ollama/llama.cpp never hit
it because they query the CUDA driver directly (`cudaMemGetInfo`), which Tegra
does support. Workarounds tried and their results:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → **worse** (uses CUDA VMM
  APIs unsupported on Tegra; triggers the assert directly).
- `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` → **worse** (OOMs
  immediately at weight allocation).
- `PYTORCH_NO_CUDA_MEMORY_CACHING=1` → got *past* the assert (best of the three)
  but then hit Wall 2.

### Wall 2 — big contiguous allocation vs NvMap/CMA
vLLM's PagedAttention pre-allocates a **single large contiguous** KV block
(~1 GB). Tegra's **NvMap** allocator forwards contiguous requests to the kernel,
which fails when it can't find a contiguous run — even with GBs *free but
fragmented*:

```
NvMapMemAllocInternalTagged: 1075072515 error 12   # 1.07 GB single alloc, ENOMEM
```

An NVIDIA engineer confirmed the root cause (forum thread on the NVML assert):
*"NvMap forwards the request to the kernel, and the allocation fails because the
kernel cannot provide contiguous memory."* The default CMA pool on this box is
only **256 MB** (`CmaTotal: 262144 kB`). Ollama never hits this because
llama.cpp grows its KV cache in **small chunks**, never asking for one big slab.

### Also ruled out
- **JetPack 7 upgrade** — would **not** fix it (same NvMap/CMA/unified-memory
  model), and it's a full reflash. JetPack 7.0 was Thor-only; 7.2 added Orin
  Nano (Q2 2026) but the vLLM containers target JP6, so you'd build from source,
  unvalidated. Wrong lever.
- **LMDeploy/TurboMind** — its C++ engine dodges the NVML assert, but has no
  prebuilt aarch64/sm87 wheels (build from source), no dusty-nv container, and
  still pre-allocates a big contiguous KV → likely hits Wall 2 anyway.
- **`cma=2048M` boot arg** — attempted to enlarge the CMA pool. **It broke the
  GPU** (see below). Do not do this.

---

## The CMA boot-arg trap (things that went wrong)

To get past Wall 2 we tried enlarging the CMA pool via
`/boot/extlinux/extlinux.conf` → `APPEND ... cma=2048M` + reboot.

**Result: the kernel could not place a 2 GB CMA region and fell back to
`CmaTotal: 0`, which broke the GPU entirely** — `/dev/nvhost-gpu` disappeared and
every GPU container failed with `NvRmGpuLibOpen failed` / `nvml error`. The
Tegra GPU driver *needs* a CMA pool to initialize; requesting an unplaceable
size zeroed it out.

**Fix:** restore the backup and reboot:
```bash
sudo cp /boot/extlinux/extlinux.conf.bak /boot/extlinux/extlinux.conf
sudo reboot
```

**Lesson:** on Tegra, `cma=` is **not** a free tunable — the GPU depends on it.
Don't raise it blindly.

---

## The real insight: fragmentation, not CMA size

After reverting CMA, MLC ran the **full 4096-token KV (~1.15 GB contiguous)**
successfully — with the *original* 256 MB CMA — **on a fresh boot**. The earlier
failures at 4096/2048 all happened *late in a session*, after container churn had
**fragmented** memory.

So the contiguous KV comes from **general memory**, and the enemy is
**fragmentation**, not CMA pool size. This is why the production setup:
1. Runs MLC as a **boot-time service** (fresh, unfragmented memory → full 4096).
2. Uses a **wrapper that falls back** to smaller contexts (3072→2048→1024) if the
   big KV can't be allocated, so the endpoint always comes up.

Per-token KV cost for Qwen3-4B q4f16 ≈ **0.28 MB/token**, so:
`4096 → ~1.15 GB`, `2048 → ~576 MB`, `1024 → ~288 MB`.

---

## Benchmark

| Host | Engine | Model / quant | tok/s | Context |
|---|---|---|---|---|
| jetson-orin | Ollama (llama.cpp) | qwen3:4b-instruct-2507 Q4_K_M | ~16 | 4096 |
| **jetson-orin** | **MLC-LLM (TVM)** | Qwen3-4B q4f16_1 | **~25** | up to 4096 |
| beast (RTX 3070 Ti) | vLLM | Qwen3-4B-Instruct-2507 AWQ | ~96 | — |

Measured end-to-end (`completion_tokens / wall_time`), 200–300 token gens,
`--mode interactive`. Stable across runs (25.2 / 25.3 / 25.4 / 24.6).

**Caveat:** the MLC model is the *original* Qwen3-4B (**hybrid-thinking**, emits
`<think>`), so 25 tok/s is decode speed *including* thinking tokens. Raw engine
speed is a genuine ~1.5× over Ollama; time-to-final-answer depends on thinking
overhead. Ollama ran the non-thinking Instruct-2507.

---

## Model note

- Serving **`mlc-ai/Qwen3-4B-q4f16_1-MLC`** (official MLC repo, JIT-compiled for
  sm87 on first run; lib cached in `~/.cache/mlc_llm`).
- This is the **original** Qwen3-4B, not `Instruct-2507` (what beast/Ollama use).
  All community MLC builds of `Qwen3-4B-Instruct-2507` are broken (missing
  `ndarray-cache.json`, packaged by a newer MLC). To match beast exactly you'd
  compile 2507 from the FP16 base (`mlc_llm convert_weight` + `gen_config` +
  `compile`) — deferred by choice.

## API compatibility

- **OpenAI `/v1/*`** (the documented openclaw contract): works. MLC ignores the
  request's `model` field and serves the single loaded model, so apps can send
  `model: "openclaw"` unchanged.
- **Ollama-native `/api/chat`, `/api/generate`, `/api/tags`:** **not implemented
  by MLC.** Apps using those raw endpoints will break and need a shim.

See [MLC_RUNBOOK.md](MLC_RUNBOOK.md) for how to run, params, and management.
