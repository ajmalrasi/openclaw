# TensorRT-LLM migration (WIP, not yet cut over)

Goal: replace Ollama with TensorRT-LLM as the `openclaw` backend on `beast`
(RTX 3070 Ti Laptop, 8GB VRAM, Ampere/compute 8.6, driver 580.126.20, CUDA 13.0).
Reachable at `192.168.3.226` for SSH continuation.

Ollama baseline to beat (see [BENCHMARKS.md](./BENCHMARKS.md)): **~40.5 tok/s**
generation on `qwen3:4b-instruct-2507-q4_K_M`.

## State as of 2026-07-09 ~03:20 IST

### Done
- `nvidia-container-toolkit` installed, Docker `nvidia` runtime configured and
  verified (`docker run --gpus all nvidia/cuda:... nvidia-smi` works).
- Pulled `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` (37.3GB image, already
  cached locally — `docker images` will show it).
- HF cache at `~/.cache/huggingface` (shared with host, mounted into
  containers via `-v ~/.cache/huggingface:/root/.cache/huggingface`).

### What failed, and why (important — don't repeat these)

1. **Full BF16 `Qwen/Qwen3-4B-Instruct-2507` via `trtllm-serve --backend
   pytorch`**: OOM. Weights alone need ~7.15GB, leaving ~0 for KV
   cache/activations on a 7.66GB-usable GPU (some VRAM reserved by driver).
   `torch.OutOfMemoryError` during `Executor creation`.

2. **`cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit`** (INT4, but quantized via
   `llm-compressor`, `quant_method: "compressed-tensors"` in config.json):
   **identical OOM, identical 7.15GB allocation.** TensorRT-LLM's `pytorch`
   backend did not recognize/apply this quant format — it silently loaded
   the weights as if unquantized.

3. **`Eslzzyl/Qwen3-4B-Instruct-2507-AWQ`** (genuine AutoAWQ format,
   `quant_method: "awq"`, GEMM kernel, verified via config.json before
   trying): **same identical 7.15GB OOM again.**

**Root cause**: TensorRT-LLM's `pytorch` backend (the modern `trtllm-serve`
quick-start path) only reliably auto-detects and applies quantization from
**NVIDIA ModelOpt-produced checkpoints** (`hf_quant_config.json` /
ModelOpt's own config shape). Community AutoAWQ and compressed-tensors
checkpoints get silently treated as full-precision. This is not documented
clearly anywhere obvious — discovered empirically.

Also ruled out without trying (hardware incompatible):
- **FP8** (`Qwen/Qwen3-4B-Instruct-2507-FP8`): needs Hopper/Ada (compute
  ≥8.9) for native FP8 tensor cores. This GPU is Ampere (8.6).
- **NVFP4** (`nvidia/Qwen3-8B-NVFP4`, `OPENZEKA/Qwen3-4B-Instruct-2507-NVFP4`):
  Blackwell-only (compute 10.0).

### Current plan: quantize it ourselves with ModelOpt

`nvidia-modelopt` is already installed inside the TensorRT-LLM release
container (visible in every log as a UserWarning about transformers version
mismatch). Plan:

1. Load `Qwen/Qwen3-4B-Instruct-2507` in BF16 with `transformers`.
2. Calibrate with `modelopt.torch.quantization` (`mtq.quantize(model,
   mtq.INT4_AWQ_CFG, forward_loop)`) using a small calibration text set
   (a few hundred short prompts is typical).
3. Export via `modelopt.torch.export.export_hf_checkpoint(...)` — this
   produces a checkpoint with the ModelOpt-native `hf_quant_config.json`
   that `trtllm-serve --backend pytorch` *should* actually respect (since
   it's the same library TRT-LLM uses internally to detect quant configs).
4. Point `trtllm-serve` at the exported local checkpoint dir and benchmark.

**Known risk**: step 1-2 (calibration) needs the BF16 model loaded on GPU
too — same 7.15GB weight footprint that caused the original OOM. Whether
this fits depends on calibration batch size/sequence length being small
enough to leave room for activations in the ~500MB headroom. If it doesn't
fit, fallback options: calibrate on CPU (slow, ~10-30min for a small model),
or use `device_map="auto"` to split calibration across CPU+GPU, or do the
calibration on a different machine with more VRAM and copy the resulting
checkpoint here for serving (serving needs far less memory than calibration
since inference doesn't need full-precision weights resident).

### Containers/images left on this machine
- `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` — kept, reused for next attempt.
- No `trtllm-qwen3` container currently running (removed after each failed
  attempt — always `docker rm` the old one before starting a new attempt to
  avoid port 8001 conflicts).
- HF cache has partial/full downloads of: `Qwen/Qwen3-4B-Instruct-2507`,
  `cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit`, `Eslzzyl/Qwen3-4B-Instruct-2507-AWQ`
  under `~/.cache/huggingface/hub/`. These are dead ends per above — don't
  need to be re-downloaded, but can be deleted to save space if needed
  (~15GB combined).

### Next steps (pick up here)
1. Draft and review the ModelOpt calibration script before running it
   (see plan above).
2. Test whether BF16 calibration fits in 7.66GB alongside model weights on
   this GPU; fall back to CPU calibration if not.
3. Export quantized checkpoint, point `trtllm-serve` at it locally
   (`-v /path/to/checkpoint:/model` + `trtllm-serve serve /model ...`).
4. Benchmark against Ollama baseline (40.5 tok/s) using the same
   photosynthesis prompt methodology as BENCHMARKS.md.
5. If TensorRT-LLM wins meaningfully, update `install.sh`/`Modelfile`/README
   to make `trtllm-serve` the `openclaw` backend and retire the Ollama
   systemd service on this machine (Jetson stays on Ollama regardless —
   TensorRT-LLM doesn't target Jetson's iGPU the same way).

### Useful commands
```bash
# check what's currently running
docker ps -a | grep trtllm

# clean slate before a new attempt
docker rm -f trtllm-qwen3 2>/dev/null

# check GPU headroom
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```
