# TensorRT Edge-LLM Deployment on Jetson

This document records the TensorRT Edge-LLM export and deployment investigation on Beast and the Jetson target. Failed attempts are preserved so that later work does not repeat them.

> **Superseding deployment status (2026-09-12):** the Jetson deployment is now
> TensorRT Edge-LLM, served by `openclaw-tensorrt-edgellm.service` on port
> 11434 with model alias `openclaw`. It uses the existing batch-two Qwen3.5-4B
> INT4 AWQ non-MTP engine. The vLLM service is rollback-only. The model may be
> deliberately stopped for maintenance, so confirm the live backend through
> `/v1/models` and the user-service state before operating on it.

## Current state

- Beast repository: `/home/ajmalrasi/TensorRT-Edge-LLM`
- User fork: `https://github.com/ajmalrasi/TensorRT-Edge-LLM`
- At the time checked, the fork `main` and NVIDIA upstream `main` resolved to the same commit: `e8b29522938901f6df19ebeedd4b69bc8edbcd97`.
- Export/quantization is being attempted inside NVIDIA container `edgellm-export`.
- Container image: `nvcr.io/nvidia/pytorch:26.05-py3`
- Workspace: `/workspace/TensorRT-Edge-LLM`
- Container virtual environment: `/workspace/venv`
- Current model: `Qwen/Qwen3.5-4B`
- Intended quantization: `int4_awq`
- No valid quantized output has been confirmed yet.

## Timeline and findings

### 1. Initial repository discovery on Beast

The repository was found at `/home/ajmalrasi/TensorRT-Edge-LLM`. A case-insensitive search also found its Python package directory at `/home/ajmalrasi/TensorRT-Edge-LLM/tensorrt_edgellm`.

### 2. Host virtual-environment installation failed

Command:

```bash
source ~/venv/bin/activate
cd /home/ajmalrasi/TensorRT-Edge-LLM
pip3 install .
```

Result:

```text
PermissionError: [Errno 13] Permission denied:
build/cp312-cp312-linux_x86_64/.cmake/api/v1/query/codemodel-v2
```

Cause:

- The existing repository `build/` directory was owned by `root:root`.
- The installation was run as `ajmalrasi`.
- CMake/scikit-build-core could not write its build metadata.

Resolution:

- The repository directory was removed and recreated/recloned by the user.
- No ownership repair or destructive cleanup was performed by this investigation.

### 3. Host venv installed incompatible package versions

After installation, the host venv contained:

```text
torch         2.5.1+cu121
transformers  5.5.0
modelopt      0.21.1
tensorrt      10.3.0
```

The checkout requested newer pinned versions, including:

```text
torch             2.13.0
transformers      5.14.1
nvidia-modelopt   0.45.0
numpy             2.2.6
```

Verification failures:

- `tensorrt-edgellm-export --help` failed in PyTorch custom-op schema inference with `types.UnionType` lacking `__origin__`.
- `tensorrt-edgellm-quantize llm --help` failed because `modelopt 0.21.1` lacked `NVFP4_DEFAULT_CFG`.
- ModelOpt also warned that the installed Transformers version did not expose the expected `Conv1D` integration.

Cause:

- The package entry points were installed, but the environment did not match the repository's dependency pins.

Decision:

- Move export and quantization into the NVIDIA PyTorch container instead of continuing to repair the host venv.

### 4. NVIDIA container and venv isolation issue

The running container was confirmed as:

```text
edgellm-export   nvcr.io/nvidia/pytorch:26.05-py3
```

The first container venv was created under the repository and did not see the container's base packages. The CLI entry points existed, but failed with:

```text
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'modelopt'
```

Cause:

- A normal venv isolates itself from packages installed in the container's base Python.
- The location of the venv was not the root cause; the isolation mode and incomplete dependency installation were the important factors.

The venv was subsequently moved/created at:

```text
/workspace/venv
```

### 5. Missing Transformers dependency

With `/workspace/venv` activated, both CLI commands initially failed with:

```text
ModuleNotFoundError: No module named 'transformers'
```

Cause:

- The Edge-LLM entry points were installed, but the complete base dependency sequence had not been completed in that environment.
- For the container workflow, installing the package alone is insufficient if dependencies were deliberately skipped or the requirements step was omitted.

The user then installed the required tools/dependencies. Subsequent checks showed:

```text
torch         2.13.0+cu130
transformers  5.14.1
modelopt      0.45.0
```

### 6. GPU quantization ran out of memory

Command:

```bash
tensorrt-edgellm-quantize llm \
    --model_dir Qwen/Qwen3.5-4B \
    --output_dir Qwen3.5-4B/quantized \
    --quantization int4_awq
```

Beast GPU:

```text
NVIDIA GeForce RTX 3070 Ti Laptop GPU
8,192 MiB VRAM
```

Result:

```text
torch.OutOfMemoryError: CUDA out of memory
Tried to allocate 46.00 MiB
GPU total: 7.66 GiB
GPU free: 1.50 MiB
```

Cause:

- INT4 is the output precision, but quantization first loads the model in a higher-precision representation.
- The 8 GB GPU could not hold the model and quantization working memory.

Conclusion:

- Reinstalling the same libraries would not solve this specific failure.
- A larger-VRAM GPU, CPU/offloaded quantization, or a smaller model is required.

### 7. CPU quantization avoided GPU OOM but exposed a Transformer Engine failure

Command reproduced:

```bash
tensorrt-edgellm-quantize llm \
    --model_dir Qwen/Qwen3.5-4B \
    --output_dir Qwen3.5-4B/quantized-cpu \
    --quantization int4_awq \
    --device cpu
```

Progress:

- The model downloaded successfully.
- All 723 weight files loaded successfully.
- CPU mode avoided the original CUDA out-of-memory failure.
- The process progressed to MTP draft quantization and calibration setup.

Failure:

```text
OSError: .../transformer_engine/libtransformer_engine.so:
undefined symbol: cublasLtGroupedMatrixLayoutInit_internal,
version libcublasLt.so.13
```

Failure path:

```text
ModelOpt
  -> Hugging Face plugin
  -> PEFT
  -> Transformer Engine
  -> incompatible transformer_engine shared library
```

Cause:

- The container's Transformer Engine binary expects a cuBLASLt symbol that is not available in the loaded CUDA 13 runtime library.
- The error is a binary compatibility mismatch, separate from the earlier GPU-memory problem.

Related warnings seen before the fatal error:

- ModelOpt warned that Transformers 5.14.1 is not tested with the current ModelOpt integration.
- ModelOpt reported Transformer Engine plugin import failures.
- The fast path was unavailable because optional `flash-linear-attention` and `causal-conv1d` packages were not installed. This was a warning, not the fatal error.
- Hugging Face access was unauthenticated, affecting rate limits rather than causing the crash.

## Recommended next investigation

1. Keep the working NVIDIA container and current `/workspace/venv`; do not return to the incompatible host venv.
2. Resolve the Transformer Engine mismatch inside the container, or prevent the PEFT/ModelOpt path from importing Transformer Engine when it is not needed for this model.
3. Re-run CPU quantization with a fresh output directory after that fix.
4. Treat any output produced by a failed run as incomplete until the command exits successfully and the expected quantized checkpoint files are verified.
5. After quantization succeeds, export the quantized checkpoint to ONNX on Beast, then transfer ONNX to the Jetson for hardware-specific TensorRT engine building.

## Important distinctions

- Host venv dependency mismatch: fixed by moving to the NVIDIA container and installing matching dependencies.
- Missing `transformers`: incomplete/isolated container environment; fixed after the dependency installation was completed.
- GPU OOM: hardware capacity limitation on the 8 GB RTX 3070 Ti Laptop GPU.
- CPU failure: Transformer Engine/cuBLASLt binary mismatch during MTP/PEFT integration.
