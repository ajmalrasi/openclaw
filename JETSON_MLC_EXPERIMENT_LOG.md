# Jetson Qwen3.5-4B / vLLM → MLC experiment journal

Created: 2026-09-11. Time zone: Asia/Kolkata (IST, UTC+05:30).
Latest detailed inspection: 2026-09-11, approximately 13:16–13:19 IST.
Owner: OpenClaw project. This is the continuing experiment record requested by the user.

## How to use and maintain this record

This file records the investigation from its initial benchmark and memory questions through the current MLC/TVM rebuild. It also summarizes the older Jetson history needed to understand the comparisons. Update it during every subsequent build, repair, benchmark, deployment change, and requested status check in this task.

For each substantive experiment, record: date/time, question, exact source/image/model/configuration, action, evidence/log location, result, failure, diagnosis and confidence, fix, verification, next issue, and whether rollback remains available. Give new experiments sequential IDs. Update the current-state section after each change. Preserve failed attempts and superseded conclusions; append a correction rather than silently rewriting history.

Routine checks should be one short timestamped entry. Do not continuously poll simply to update this document. Long jobs must remain detached. Record timestamps before replacing log files; prefer a new per-attempt log from now on.

Evidence labels:

- **Verified artifact:** inspected source, Docker metadata, saved log, or existing repository document.
- **Conversation record:** recovered prior responses/actions from this task, not independently remeasured now.
- **Interpretation:** explanation supported by observations, but not necessarily isolated experimentally.
- **Pending:** proposed or authorized work that has not passed validation.

Not every early raw measurement or overwritten retry log is still available. Exact numbers below are preserved where the conversation retained them. Unknown commands, durations, and timestamps are explicitly left unknown. This is a detailed reconstruction, not a claim that every original shell transcript survived.

## Current state

- Hardware: Jetson Orin Nano Super, nominal 8 GB unified CPU/GPU memory; Linux reports roughly 7.3 GiB usable.
- SSH: `ajmalrasi@192.168.3.30`.
- Host software: JetPack 7.2.1, L4T r39.2.1, CUDA 13.2; target Orin SM87.
- The original Qwen3.5 MLC model compiled and served successfully at concurrency one.
- Its measured end-to-end output throughput was **10.52 tok/s**.
- Its concurrency-two test failed with a recurrent-state shape error, without observed OOM.
- The corrected newer MLC image has been created, but its full TVM compiler repair is still building.
- Latest captured compiler progress: **55% at 3,462 build-seconds**. A process check shortly afterward showed the Docker build active at about 60 minutes elapsed.
- Latest captured memory: 2.7 GiB used, 4.7 GiB available, 205 MiB swap occupied.
- Build process: Docker PID `198243`; PID `198242` is its launcher shell, not the Docker process itself.
- Current build log: `/home/ajmalrasi/mlc-main-sm87-runtime-repair.log`.
- Target repair image: `openclaw-mlc:qwen35-main-sm87-runtime`; not yet produced at the last image check.
- vLLM and the MLC inference server are stopped to leave memory for compilation.
- Stopped containers `mlc-qwen35-compile` and `openclaw-mlc-jetson` both have exit code 0 and `OOMKilled=false`.
- The unrelated API and PostgreSQL containers remain running. Port 8000 belongs to the API. MLC tests used port 8001.
- No successful benchmark exists yet for the newly rebuilt upstream MLC/TVM combination.
- Concurrency two, FlashInfer execution, direct-response behavior, and tool calling remain open validation items.

### Critical correction to earlier performance claims

The **actual current-model vLLM benchmark was 10.50 decode tok/s and 10.46 end-to-end output tok/s**, not 20 tok/s. It was Qwen3.5-4B W4A16.

The later MLC Qwen3.5-4B result was **10.52 end-to-end output tok/s**. These short tests used different prompt/output lengths, so they do not establish a meaningful speed advantage for either engine.

The user's approximately 20 tok/s reference concerned an earlier Qwen3-4B deployment. Repository history also records approximately 22–25 tok/s on older Qwen3 MLC builds. Several later replies incorrectly treated that older reference as the measured current Qwen3.5 vLLM baseline. Statements that current MLC was “about twice as slow as current vLLM” were therefore unsupported.

Likewise, attributing most of the MLC slowdown specifically to FlashInfer was too strong. Fallback was verified, but its contribution has not been measured in a controlled comparison.

## User requirements and operating decisions

1. Benchmark the model deployed on the Jetson, not the conversational assistant.
2. Initial request: two minutes; narrowed to the smallest useful approximately one-minute benchmark.
3. Investigate concurrency four and larger KV cache; then prioritize concurrency two.
4. Explain memory ownership and cache/build-image options before changing anything during the initial diagnosis.
5. Keep the same Qwen3.5-4B base model when evaluating MLC; explain that quantized checkpoint bytes differ.
6. Explicit permission was given to stop vLLM because this is a development machine.
7. Long work must run detached on the Jetson and survive a Mac/SSH disconnect.
8. A user-reported power cut/restart occurred; detached jobs do not survive loss of power merely because they survive SSH loss.
9. The user authorized necessary compatibility fixes and a newer upstream rebuild.
10. Concurrency-one/two benchmark work must take at most five minutes, with hard request timeouts.
11. No frequent polling or long unattended token-consuming conversation.
12. Maintain this journal as the work proceeds.
13. Preserve the existing unrelated API/database services and the known-working model artifacts.
14. Never treat a successful compiler build as successful inference or a successful HTTP response as validated answer/tool quality.

## Earlier Jetson history (background, not this build campaign)

Sources: `MLC_MIGRATION.md`, `MLC_RUNBOOK.md`, `BENCHMARKS.md`, and `VLLM_JETSON_RUNBOOK.md`.

### H01 — Ollama baseline on JetPack 6.2

The older Orin system used JetPack 6.2 / L4T r36.4.7 and CUDA 12.6. Ollama served Qwen3-4B-Instruct-2507 Q4_K_M at roughly 16 tok/s. A clean reload still offloaded 36/37 layers and gave roughly 15.3 tok/s; the reported CPU/GPU percentage was not a reliable indication of actual layer placement.

### H02 — Earlier generic vLLM startup failures

Historical attempts encountered a PyTorch NVML assertion on Tegra. Tried allocator changes included expandable segments, cudaMallocAsync, and disabling caching. Disabling caching got beyond one assertion but exposed a large-allocation failure.

An attempted `cma=2048M` boot argument caused CMA to fall to zero and the GPU device to disappear. The saved boot configuration was restored and the machine rebooted. No such boot change is part of the current experiment.

Later, a fresh boot allowed a large MLC KV allocation with the original CMA setting. The old diagnosis evolved from “CMA pool too small” to “allocation/fragmentation matters.” Those historical notes contain broad claims about vLLM/JetPack that were superseded when a dedicated NVIDIA Orin vLLM image worked on JetPack 7.2.1.

### H03 — Earlier Qwen3 MLC deployment

- Original thinking model: `mlc-ai/Qwen3-4B-q4f16_1-MLC`, approximately 2.1 GB weights, up to 4K context, approximately 25 tok/s.
- Direct-response model: `FutureProofHomes/Qwen3-4B-Instruct-2507-q4f16_2-MLC`, approximately 2.7 GB weights, 2048 context, approximately 22 tok/s.
- Several community weight packages lacked `ndarray-cache.json` expected by the older runtime.
- The heavier direct-response checkpoint reduced context/headroom.
- Boot-time loading and a smaller-context fallback were used for allocation reliability.
- The public contract was OpenAI-compatible `/v1/chat/completions`, model alias `openclaw`, port 11434.
- Historical documentation saying Qwen3.5 had no MLC build is now obsolete.
- Historical Qwen3 KV-per-token estimates must not be reused for Qwen3.5 hybrid attention.

### H04 — Working Qwen3.5 vLLM on JetPack 7.2.1

Generic vLLM/PyTorch images lacked suitable SM87 support. The dedicated NVIDIA Orin image worked, pinned as:

```text
ghcr.io/nvidia-ai-iot/vllm@sha256:817f0f940d2d9c9067d861d2118d7bf58c40873598f0c35e19c8516269ebc4bd
```

Checkpoint: `RedHatAI/Qwen3.5-4B-quantized.w4a16`.

Problems and fixes documented in the runbook:

- Full vision profiling exceeded shared memory → language-only startup and no multimodal processor cache.
- Automatic KV allocation left too little headroom → explicit `192M` budget, one sequence, 1024 batched tokens, eager execution.
- Checkpoint tokenizer declared Transformers 5 `TokenizersBackend`, while the image used Transformers 4 → expose the same tokenizer JSON/template through `Qwen2TokenizerFast`.
- Wrong model-cache mount could trigger repeated downloads → mount host Hugging Face cache at container `/data/models/huggingface`.
- Persistent 16 GB swap file: `/swapfile-vllm`, recorded in fstab.
- Real `LIFEOS_OK` completion used for startup validation.
- Service configured the Qwen reasoning parser and `qwen3_coder` tool parser, with thinking disabled.
- These flags show configuration; they do not establish MLC tool compatibility.

## Current campaign: experiment and issue ledger

### E01 — Small vLLM baseline benchmark

**Evidence:** recovered original benchmark response from this task.

Configuration: Qwen3.5-4B Red Hat W4A16, vLLM, context 4096, concurrency one.

Procedure: 8-token warmup, 64-token calibration, one 555-token measured streaming request.

| Metric | Result |
|---|---:|
| Full benchmark duration | 60.44 s |
| Measured request duration | 53.06 s |
| Input tokens | 35 |
| Output tokens | 555 |
| Total tokens | 590 |
| Decode speed | 10.50 tok/s |
| End-to-end output throughput | 10.46 tok/s |
| Time to first token | 0.307 s |
| Time per generated token | 95.22 ms |
| Approximate prompt rate | 114 input tok/s |
| Completion reason | Requested length reached |

The tiny prompt cannot isolate prefill compute from fixed API/network latency. Approximate prompt rate is not a validated sustained prefill benchmark. The original raw client result path has not been recovered.

### E02 — Concurrency/KV capacity assessment

**Action:** inspected the running system; no concurrency/cache change was made during this diagnostic phase.

Snapshot: 7.31 GiB physical RAM, about 6.9 GiB used, 376 MiB available, approximately 3.1 GiB swap used, vLLM container approximately 6.20 GiB.

vLLM reported 192 MiB cache and approximately 1,056 live cached tokens. Earlier replies extrapolated approximately 1,400 at 256 MiB, 1.5 GiB for two full 4K contexts, and 3 GiB for four. These were rough capacity projections, not successful configurations; hybrid state and allocation granularity require actual validation.

Decision evolved from discussing four short requests to prioritizing two short requests at the existing cache size. Two full 4K contexts were not established as feasible on vLLM. Increasing the advertised sequence limit alone does not guarantee enough live-cache capacity or working memory.

Two prior startup kernel OOM kills were reported in the diagnostic record.

### E03 — Memory ownership and Linux cache investigation

**Evidence:** prior live measurements retained in the conversation.

| Consumer/measure | Historical snapshot |
|---|---:|
| vLLM container | 6.19 GiB |
| Engine process RSS, including allocations | 5.51 GiB |
| Loaded model weights, included in engine | 4.44 GiB |
| API/frontend RSS | 594 MiB |
| Explicit KV cache, included above | 192 MiB |
| Other application containers | 176 MiB |
| Linux page cache | 434 MiB |
| Reclaimable slab | 86 MiB |
| Available RAM | 373 MiB |
| Swap occupied | 3.16 GiB |
| Swap attributed to vLLM container | approximately 2.7 GiB |

These accounting categories overlap; do not add them as independent allocations. Largest free physical block was reported as 2 MiB, indicating fragmentation at that observation.

Conclusion: weights were only part of total memory. CUDA state, activations, recurrent state, KV, runtime, tokenizer, and services share the same physical pool. Dropping caches could temporarily change free-memory reporting but would not release live allocations or create durable capacity.

Suggested but not implemented here: FP8 KV with correctness checks, allocator tuning, carefully increasing KV to 256 MiB, or stopping unrelated applications with appropriate scope. More swap and CPU offload were not substitutes for GPU-resident shared RAM.

### E04 — Docker image size versus inference memory

NVIDIA image history contained `COPY multi:…`, evidence of multi-stage construction. Snapshot: approximately 20.6 GiB image/root filesystem, 134 MiB writable layer, only approximately 108 MiB container file-backed resident pages.

Docker infrastructure: dockerd approximately 60 MiB, containerd 39 MiB, shim 8 MiB.

Conclusion: build-artifact cleanup primarily saves disk. It does not remove gigabytes of live weights/workspace. Candidate disk cleanup included toolchains, headers, examples, OpenCV/Vulkan development components and unused backends, but a slimmed vLLM image was not built in this phase.

The later MLC build image is also large and is a development/compiler image. No final minimal multi-stage MLC inference image has been implemented.

### E05 — The 594 MiB frontend investigation

The earlier “coordinator” label was corrected: this process was the vLLM API/frontend, handling HTTP, templates, tokenization and streaming.

Observed: approximately 594 MiB RSS, 877 MiB swapped, approximately 596 MiB anonymous/native allocations, approximately 8 MiB mapped files, 34 threads, 248,044-entry vocabulary from a roughly 20 MiB tokenizer file. Small differences between categories reflect accounting/rounding, not separately additive memory.

Installed async serving rejected `VLLM_ENABLE_V1_MULTIPROCESSING=0` with `NotImplementedError`; merging the processes was not a supported configuration.

Potential allocator tests `MALLOC_ARENA_MAX=2` and `MALLOC_TRIM_THRESHOLD_` were discussed but not run. Estimated 100–250 MiB savings were speculative. Removing parsers, skipping tokenizer initialization, or building a custom synchronous frontend would change functionality/contract and were not implemented.

### E06 — Research: overhead and Qwen3.5 MLC support

Searches found vLLM's separate frontend/engine architecture, historical paging overhead papers, MLC's native compilation and low-concurrency modes, upstream Qwen3.5 support, and official converted weights.

Decision: try `mlc-ai/Qwen3.5-4B-q4f16_1-MLC`, approximately 2.39 GB published weight files. Same base model, different quantization/checkpoint bytes from Red Hat W4A16; identical quality was never guaranteed.

Research justified an experiment, not a prediction that MLC must outperform vLLM on Orin. Older datacenter benchmarks and older Qwen3 performance were not direct evidence for this combination.

### E07 — Stop vLLM and start the first container build

The user approved downtime. vLLM was stopped and the Jetson container recipe selected.

The broad recipe installed container CUDA development files and user-space libraries even though the host already had CUDA/cuDNN. The host driver is shared through the NVIDIA runtime, but development files are not automatically inherited into a new image.

The selected chain included more dependencies than text MLC strictly needed: CUDA stack, Python, PyTorch, torchvision, Transformers, Rust, Triton, LLVM, TVM, FlashInfer and MLC. This contributed to download, disk and compatibility costs.

Initial chain: 24 stages. Early check: stage 4, cuDNN about 67% downloaded after approximately 44 minutes, 6.4 GiB available RAM.

### E08 — Attached build corrected to detached execution

**Failure in execution setup:** the first build was running through a persistent SSH session, not truly detached. The assistant acknowledged this when asked.

**Fix:** stop/restart with `nohup`, redirected input/output and a remote log. Detached PID recorded as 158739. First three layers reused; interrupted cuDNN download restarted.

**Verification:** launcher adopted by init and output persisted in `/home/ajmalrasi/mlc-build-detached.log`.

SSH independence does not imply reboot persistence. The user later reported a power cut and restart. The conversation records that vLLM restarted and was stopped again; exact timing and lost/reused build work are not fully recovered.

### E09 — Python bootstrap package-index failure

**Failure:** stage 5 of 24 could not find ordinary `pip` on the Jetson-only wheel index.

**Fix 1:** modify `packages/build/python/install.sh` to use explicit public PyPI for bootstrap tools. Account for `UV_DEFAULT_INDEX`, which overrode the legacy command-line index setting.

**Fix 2:** broaden subsequent-stage fallback in `packages/build/python/Dockerfile`:

```text
PIP_EXTRA_INDEX_URL=https://pypi.org/simple
UV_DEFAULT_INDEX=https://pypi.org/simple
UV_EXTRA_INDEX_URL=${PIP_INDEX_URL}
```

The Jetson index remained available for specialized wheels. Python and NumPy subsequently passed; CMake stage 7 began. CUDA/cuDNN layers were reused.

### E10 — LLVM 20/22 conflict

**Failure:** stage 18/24 attempted LLVM 20 on top of LLVM 22.

**Fix:** pin the MLC recipe dependency to LLVM 22 in `packages/llm/mlc/config.py`.

**Result:** chain became 23 stages, first 17 layers reused, restart PID 205426.

### E11 — First full MLC image and post-build test failure

Base recipe pinned MLC `d1ea69a`; completed image:

```text
openclaw-mlc:qwen35-jp7-r39.2.tegra-aarch64-cu132-24.04
```

Final MLC stage spent over an hour in CUDA/CUTLASS. Memory pressure varied from roughly 174 MiB available plus 3.4 GiB swap to over 6 GiB available as components finished.

**Result:** image exported, but the outer build command returned failure during generic PyTorch tests:

```text
CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
```

Conversation diagnosis: PyTorch expected cuDNN 9.20, while exposed components included 9.21. Optional TensorRT-import/upload errors also appeared in logs.

**Decision:** preserve the built image and directly validate MLC; an outer test failure does not mean image export failed. Conversely, image export does not imply a usable inference runtime.

**Next issue:** TVM Python package/compiler missing.

### E12 — Missing TVM and mismatched TVM FFI

The image had MLC but omitted the TVM Python package, and the installed `apache-tvm-ffi` wheel did not match the source used to build MLC. The minimal/dummy TVM library also lacked compiler registrations needed by the CLI.

**Fix:** create `mlc/Dockerfile.qwen35-runtime`, reinstall TVM FFI from MLC's exact vendored TVM submodule, expose its Python package, and build a full compiler from that source. The old TVM target was `tvm`.

Full repair used CUDA/SM87, LLVM 22, cuBLAS/CUTLASS/Thrust, cuDNN disabled, and two compiler jobs. Detached PID 252404 and `mlc-runtime-repair.log` recorded.

At one check: 50% after 1h11, approximately 5.1 GiB available and 484 MiB swap.

### E13 — Optional backtrace header conflict

**Failure:** repair reached roughly 96% before hitting incompatible GCC/LLVM backtrace headers.

**Fix sequence:** an initial retry used an ineffective/wrong setting; it was stopped early. The exact vendored FFI option was then applied:

```text
TVM_FFI_USE_LIBBACKTRACE=OFF
```

**Verification:** corrected detached build proceeded; PID 295292 recorded. Compiler cache retained.

Saved final repair log reports `Built target tvm` at 2704.8 seconds, followed by roughly 53 seconds of export. The frequently repeated “46-minute build” refers to this successful TVM repair stage, not the entire multi-stage campaign.

### E14 — Missing pytest

**Failure after compiler success:** TVM imported testing helpers during CLI startup, but `pytest` was absent.

**Fix:** add `uv pip install pytest` to the repair layer.

**Verification:** TVM `0.24.dev0`, MLC import, CUDA/Orin detection and compile/chat/serve command availability passed. This was an import/CLI smoke test; actual model compilation still exposed another problem.

### E15 — Weight download and LLVM symbol collision

Official weights downloaded in detached container `mlc-qwen35-download`, approximately 2.3 GB on disk.

Path:

```text
/home/ajmalrasi/models/Qwen3.5-4B-q4f16_1-MLC
```

**Next failure:** model compiler hit an LLVM symbol collision/segfault.

**Attempt:** rebuild with TVM symbol isolation (`HIDE_PRIVATE_SYMBOLS=ON`), logged in `mlc-symbol-repair.log`.

**Further diagnosis/fix:** Triton loaded another bundled LLVM copy. Remove Triton from this MLC CUDA image with `uv pip uninstall triton`.

**Verification:** actual Qwen compilation then ran successfully. Earlier import success was insufficient to detect this compiler execution conflict.

### E16 — Original Qwen3.5 model compilation

Recovered container command:

```sh
mlc_llm compile /models/Qwen3.5-4B-q4f16_1-MLC \
  --device cuda \
  --overrides 'context_window_size=4096;prefill_chunk_size=1024;max_batch_size=2' \
  --output /models/Qwen3.5-4B-q4f16_1-MLC/Qwen3.5-4B-q4f16_1-cuda.so
```

Container: `mlc-qwen35-compile`, image `openclaw-mlc:qwen35-runtime`.

Docker timestamps: 2026-09-11 03:58:47–04:03:19 UTC, or 09:28:47–09:33:19 IST. Duration approximately 4m32s. Exit 0; no OOM kill. Output library approximately 31 MiB.

**Nonfatal but important failure:**

```text
Error caught when creating FlashInfer PagedKVCache:
Cannot open /root/.cache/flashinfer/0.6.11/87/cached_ops/
batch_prefill_tvm_dtype_q_float16_dtype_kv_float16_dtype_o_float16_
qk_head_dim_256_v_head_dim_256_enable_inline_rope_False/
batch_prefill_paged_kernel_mask_0.cuda.o

The model will fallback to TIR-based KV cache.
```

Compilation log already had `flashinfer=1`; enabling the flag alone was not sufficient. It also showed `cublas_gemm=0;faster_transformer=0;cudagraph=1;cutlass=1;ipc_allreduce_strategy=NONE`.

**Result:** working compiled library with generic TIR KV implementation. FlashInfer acceleration remained unresolved.

### E17 — First MLC endpoint smoke test

Port 8000 was occupied by the unrelated backend, so MLC used 8001.

HTTP 200, 32 generated tokens in 3.52s, roughly 9.1 end-to-end tok/s. Idle container approximately 2.78 GiB, available system RAM approximately 3.1 GiB.

**Quality limitation:** output spent its allowance on visible reasoning and did not reach the requested final phrase. This demonstrated generation, not final-answer or non-thinking contract correctness.

Compiled capacity was batch two; runtime initially allowed one request and 4096 total KV tokens. Detached server had restart enabled at that point.

### E18 — Bounded concurrency-one and concurrency-two benchmark

Total reported test duration: under three minutes, within user's five-minute cap.

| Metric | Concurrency one | Concurrency two |
|---|---:|---|
| Prompt tokens | 40 | Not preserved in recovered summary |
| Completion tokens | 128 | No response bytes returned |
| Wall time | 12.16865 s | Both clients hit 70 s timeout |
| Output throughput | 10.52 tok/s end-to-end | No successful throughput result |
| HTTP outcome | 200 | Timeout |
| OOM observed | No | No |

Concurrency-two evidence:

```text
ValueError: Mismatched output.shape[0]
rnn_state_get_1
expected to match seq_slot_ids.shape[0]
```

Container memory approximately 3.703 GiB; system available approximately 2.9 GiB; estimated engine requirement approximately 4.81 GB.

**Diagnosis:** batched recurrent-state shape handling failed. This is stronger evidence for a software defect than memory exhaustion for this short workload. It does not prove two arbitrary full 4K contexts fit.

**Recovery:** restore concurrency one. Restored command used interactive mode and:

```text
max_num_sequence=1;max_total_seq_length=4096;prefill_chunk_size=1024
```

Recovered restored-server log estimates 4676.286 MB total: 2257.462 MB parameters, 216.271 MB KV, 2202.553 MB temporary buffer. This is an estimator, not resident memory. An earlier approximately 3.85 GB compilation estimate described a different estimate/stage; neither should replace measured memory.

### E19 — Performance and concurrency diagnosis

Prior observations recorded MAXN_SUPER mode, roughly 53°C GPU temperature, no detected thermal throttling, warmed test and sufficient memory. These are historical observations, not current telemetry.

Research and logs support separate questions:

- Standard attention: FlashInfer kernel failed and TIR fallback was used.
- Hybrid recurrent layers: concurrency two failed in RNN state handling.
- Kernel scheduling: compilation does not guarantee well-tuned SM87 decode/GEMV.
- Benchmark comparability: Qwen3 and Qwen3.5, quantization, prompt size, and decode/end-to-end metrics differ.

Qwen3.5's approximately 75% GatedDeltaNet / 25% standard attention mix makes a claim that FlashInfer alone explains all performance particularly uncertain.

Decision: try current upstream MLC with its pinned TVM before experimental Orin fork changes. Recompiling unchanged source was unlikely to resolve the state error.

### E20 — New upstream rebuild and memory interference

Target MLC revision `9fa644f5`, installed version `0.26.dev6`. Vendored TVM subsequently verified as:

```text
837cb9de1127b48ce48e4cefe09e83215b9d4ba7
```

The initial new build mistakenly left the old MLC server running. A check found almost all RAM and approximately 8.7 GB swap used.

**Fix:** stop the temporary MLC server. Available RAM improved from approximately 200 MiB to 795 MiB, and swap headroom improved. Restored-server Docker metadata shows it stopped at 2026-09-11 04:39:06 UTC / 10:09:06 IST.

This was an avoidable deviation from the user's earlier request to free model memory before building.

**Estimate correction:** an initial 20–50 minute total estimate assumed extensive cache reuse. Changed source invalidated substantial compilation work; this estimate proved too optimistic.

### E21 — Wrong GPU architecture selected

The recipe inherited `87;110;120;121` and took the last architecture, selecting SM121 as its primary target.

**Attempt 1:** stop the build and restart an Orin-specific variant.

**New issue:** an active child was compiling SM87, but its parent command still scheduled SM110/120/121 later. Inspecting only the current child was insufficient.

**Attempt 2 / fix:** force all relevant architecture values inside `packages/llm/mlc/build.sh`:

```sh
export CUDAARCHS=87
export CUDA_ARCHITECTURES=87
export TORCH_CUDA_ARCH_LIST=8.7
export FLASHINFER_CUDA_ARCH_LIST=8.7
archs=87
```

**Verification:** parent CUDA commands contained only compute_87/sm_87, and active compiler showed `__CUDA_ARCH__=870`. Old processes stopped.

Distinct logs retained:

- `mlc-main-build.log`
- `mlc-main-orin-build.log`
- `mlc-main-sm87-build.log`

This was two stopped incorrect attempts followed by the corrected SM87 build, not one uninterrupted successful build.

### E22 — Upstream SM87 image completion

Long CUTLASS/GEMV translation units continued after other targets printed 100%. CPU-active compiler processes and changing logs showed work rather than a stall. Per-target percentages were not overall task completion.

Wheel `mlc_llm 0.26.dev6` built and installed. Upload to `jetson-ai-lab.io` failed due to DNS, but recipe treated publishing as nonfatal. Local wheel installation remained successful.

Final base image:

```text
openclaw-mlc:qwen35-main-sm87-r39.2.tegra-aarch64-cu132-24.04
sha256:3e8b24c9d284225e200788224b3bce97d586e4a482080e51771dfe15b0d8d4ca
```

Created 2026-09-11 approximately 11:13:31 IST. Inspect reported 16,324,664,129 bytes; Docker image-list presentation approximately 48.6 GB. These are different storage representations, not inference memory or quantities to add.

**Next failures:** generic PyTorch cuDNN sublibrary mismatch recurred, and direct MLC import found `ModuleNotFoundError: No module named 'tvm'`.

The cuDNN issue is outside the chosen TVM build configured with cuDNN OFF; it remains unresolved in the broad image and is not grounds to declare every package healthy.

### E23 — New-revision runtime repair: renamed target

Applying the existing repair recipe to new TVM failed quickly:

```text
gmake: No rule to make target 'tvm'
```

**Cause:** new TVM split compiler/runtime; appropriate compiler target is `tvm_compiler`, with libraries under `/opt/tvm-full-build/lib`.

**Fix attempt:** detect `tvm_compiler` by piping CMake target help into grep; otherwise fall back to `tvm`. Update library lookup paths for the new lib directory.

**New issue:** detection returned false and ran the obsolete target. Configuration succeeded but build failed in approximately 15 seconds. The precise reason for the detection mismatch was not isolated; do not present a speculative pipe/regex explanation as established fact.

**Final fix:** directly build the confirmed target:

```sh
cmake --build /opt/tvm-full-build --target tvm_compiler -j2
```

A first local patch failed to match the exact file text, so no change was applied. Read the real lines, patched successfully, copied the Dockerfile to the Jetson, and restarted detached.

**Verification:** configuration succeeded and actual C++ compilation started at 1%; later passed 54–55%.

### E24 — Current repair, inspection, and documentation

Exact launched build (run from remote `/home/ajmalrasi/openclaw`):

```sh
nohup docker build --network host \
  --build-arg BASE_IMAGE=openclaw-mlc:qwen35-main-sm87-r39.2.tegra-aarch64-cu132-24.04 \
  --tag openclaw-mlc:qwen35-main-sm87-runtime \
  --file mlc/Dockerfile.qwen35-runtime . \
  > /home/ajmalrasi/mlc-main-sm87-runtime-repair.log 2>&1 < /dev/null &
```

Current Dockerfile settings include matching vendored FFI, LLVM 22, Release mode, CUDA/cuBLAS/CUTLASS/Thrust ON, cuDNN/NCCL OFF, FFI libbacktrace OFF, hidden private symbols, SM87 and two compile jobs; pytest installed and Triton removed.

Library paths include both `/opt/tvm-full-build/lib` and `/opt/tvm-full-build`. New CMake warns that `BUILD_DUMMY_LIBTVM` was unused; the explicit compiler target is the relevant operation.

**Reproducibility caveat:** the Dockerfile's default BASE_IMAGE still names the older image while its target now assumes newer TVM. The active command passes the correct newer base explicitly. Do not reuse the current file with its default and assume it reproduces the older working runtime. This mismatch is documented, not changed during this documentation-only work.

At latest inspection, build active, 55%, approximately one hour elapsed, adequate memory. No model recompilation or new benchmark started during this documentation turn.

## Artifact inventory and recovery references

All remote paths below are on the Jetson.

| Artifact | Location / identity | State |
|---|---|---|
| Current journal | `openclaw/JETSON_MLC_EXPERIMENT_LOG.md` | Continuing record |
| Local repair recipe | `/Users/ajmalrasi/openclaw/mlc/Dockerfile.qwen35-runtime` | New TVM target |
| Remote repair recipe | `/home/ajmalrasi/openclaw/mlc/Dockerfile.qwen35-runtime` | Used by active build |
| Container recipe clone | `/home/ajmalrasi/src/jetson-containers` | Contains local fixes |
| Official MLC weights | `/home/ajmalrasi/models/Qwen3.5-4B-q4f16_1-MLC` | Downloaded |
| Original model library | Same directory, `Qwen3.5-4B-q4f16_1-cuda.so` | 31 MiB, old runtime |
| Original working runtime | `openclaw-mlc:qwen35-runtime`, ID prefix `5518bed77c40` | Preserved |
| Original MLC base | `openclaw-mlc:qwen35-jp7-r39.2.tegra-aarch64-cu132-24.04`, ID `98a1eb2ffd2f` | Preserved |
| New MLC base | `openclaw-mlc:qwen35-main-sm87-r39.2.tegra-aarch64-cu132-24.04`, ID `3e8b24c9d284` | Created |
| New repaired runtime | `openclaw-mlc:qwen35-main-sm87-runtime` | Build pending |
| Download container | `mlc-qwen35-download` | Exited 0 |
| Compile container | `mlc-qwen35-compile` | Exited 0; compilation log available |
| Last server container | `openclaw-mlc-jetson` | Exited 0 |
| vLLM service recipe | `vllm/openclaw-vllm-jetson.service` | Rollback reference |
| Older Qwen3 service wrapper | `mlc/openclaw-mlc-run.sh` | Historical; not current Qwen3.5 config |

Remote build logs retained:

- `/home/ajmalrasi/mlc-build-detached.log`: initial dependency/full image campaign; final modified Sep 10 16:52 IST.
- `/home/ajmalrasi/mlc-build-logs/`: initial per-stage build/test evidence.
- `/home/ajmalrasi/mlc-runtime-repair.log`: original full TVM repair; final modified Sep 10 23:54 IST.
- `/home/ajmalrasi/mlc-symbol-repair.log`: symbol/pytest repair; final modified Sep 11 00:45 IST.
- `/home/ajmalrasi/mlc-main-build.log` and `mlc-main-build-logs/`: first newer-source attempt.
- `/home/ajmalrasi/mlc-main-orin-build.log` and `mlc-main-orin-build-logs/`: partially corrected architecture attempt.
- `/home/ajmalrasi/mlc-main-sm87-build.log` and `mlc-main-sm87-build-logs/`: corrected SM87 build and generic test failure.
- `/home/ajmalrasi/mlc-main-sm87-runtime-repair.log`: current repair. Earlier short failures were overwritten by restarts; conversation history preserves their errors.

Remote recipe changes verified with git diff:

1. `packages/build/python/install.sh`: bootstrap PyPI and uv index handling.
2. `packages/build/python/Dockerfile`: subsequent-stage index fallback.
3. `packages/llm/mlc/config.py`: LLVM 22 dependency.
4. `packages/llm/mlc/build.sh`: four explicit SM87 architecture settings.

These edits live in a separate repository and are not automatically captured by committing OpenClaw documentation. Preserve/export them before cleaning that checkout.

## What has and has not been established

| Question | Evidence-backed answer |
|---|---|
| Can Qwen3.5-4B generate through MLC on this Jetson? | Yes, original repaired runtime and model library generated successfully. |
| Is MLC faster than current Qwen3.5 vLLM? | Not established; both short tests were approximately 10.5 output tok/s. |
| Does MLC have a smaller measured idle footprint here? | Yes in recorded snapshots, but compare identical runtime states before quantifying savings. |
| Does concurrency two work? | No on the original MLC build; new build not tested. |
| Was the concurrency failure an OOM? | Observed error was an RNN state shape mismatch; no OOM observed in that test. |
| Does two-way 4K-per-request serving fit? | Not validated. A 4096 total KV budget is not two full 4096 contexts. |
| Is FlashInfer acceleration active? | Original compilation fell back to TIR; new path unverified. |
| Was FlashInfer disabled by missing flag? | No; original compile explicitly tried flashinfer=1. |
| Has Qwen3.5 answer quality been validated? | Only basic generation; first short response used visible reasoning. |
| Does MLC tool calling match vLLM? | Pending; prior config noted use_function_calling=false. |
| Was a slim inference image created? | No. Current image includes development dependencies. |
| Does the current detached build survive SSH loss? | Yes, launched detached with remote logging. |
| Does detached mean power-loss recovery? | No. Must inspect/restart after a Jetson reboot. |

## Outstanding work in execution order

1. Check current TVM repair once on the next requested check; record completion, error, or progress.
2. After success, verify matching TVM/FFI/MLC imports, library discovery, compiler CLI and real CUDA access.
3. Compile a separate new Qwen model library against the new runtime, retaining the old library. Suggested new name: `Qwen3.5-4B-q4f16_1-main-sm87.so`.
4. Explicitly target SM87 and verify the actual FlashInfer path. A configured switch or successful import is insufficient; inspect compilation and runtime evidence.
5. Investigate if needed: recipe lists generated FlashInfer head dimensions 128 while the observed missing Qwen kernel requires head dimension 256. This is a lead from inspected artifacts, not a proven root cause because JIT generation may supply additional dimensions.
6. Start concurrency one on port 8001 and verify the actual final answer, reasoning behavior, and served model/API contract.
7. Run a bounded, comparable short benchmark, recording prompt/output counts, warmup, wall time, TTFT/decode if streaming, memory and exact artifact versions.
8. Test concurrency two with an overall limit of five minutes for the benchmark session. Record per-request results and aggregate throughput; fail quickly on state errors.
9. If batching still fails, use the exact new traceback/source to isolate a fix. Do not assume FlashInfer or a flag resolves recurrent state.
10. Separately validate tool-call structure, tool-result round trip, stop/finish behavior, and non-thinking requirements.
11. Restore the best validated configuration. Update the production runbook/public endpoint only when migration validation passes.
12. Consider smaller inference-image packaging only after the required runtime libraries are identified and validated.
13. Consider applicable SM87/GDN optimizations from the experimental Orin work only after a clean upstream baseline. No wholesale experimental fork has been adopted.

## Lessons, corrections, and unresolved uncertainty

- A successful source build, exported image, import test, compiled model, HTTP response, correct final answer, successful batching and usable tool calling are separate milestones.
- Preserve exact revision/quantization/context and metric definitions; “4B” is not enough to compare speed or memory.
- Image disk size is not resident RAM. Do not sum shared Docker layer sizes as unique disk consumption.
- A full compiler is needed for model compilation even if MLC has a minimal runtime library.
- Reuse vendored FFI/TVM revisions together. Copying old binary components into a new source stack is not a verified shortcut.
- Removing Triton solved the observed LLVM conflict in this chosen path; it is not a claim Triton is universally unnecessary.
- Fixing an optional generic test dependency is not required to claim only the independently verified MLC path works. But unresolved tests must remain recorded.
- Keeping inference running during a memory-heavy rebuild increased swap and contradicted the user's requested workflow.
- Inspect parent compiler flags, not just one active GPU target.
- Build progress percentages and “100%” refer to targets, not necessarily the whole image.
- Prior remaining-time estimates were repeatedly optimistic. Use measured phase timings and identify what is still unbuilt.
- The approximately 46-minute successful old TVM repair, approximately 4m32s model compile, and currently hour-long new TVM build are different operations. Do not describe each as rebuilding “everything.”
- Prior projections of 12–16 or 20+ tok/s were hypotheses, not results.
- The Orin FlashInfer issue's reported 25% improvement is anecdotal; issue #3492's large gains concern other model sizes/architectures, including MoE. Router optimizations do not automatically apply to dense 4B.
- Available RAM in one short test does not establish maximum context/concurrency capacity.
- Some relative container ages were nonsensical (“56 years”) during inspection; use recorded absolute timestamps/logs and note clock anomalies.
- Raw earlier client benchmark files, exact power-cut timing, and overwritten short-retry logs remain evidence gaps. Do not invent them.

## Research references consulted during the original investigation

These links are provenance for earlier research, not a fresh revalidation of moving upstream pages during this documentation turn.

- [Official Qwen3.5 MLC weights](https://huggingface.co/mlc-ai/Qwen3.5-4B-q4f16_1-MLC)
- [MLC Qwen3.5 support request #3448](https://github.com/mlc-ai/mlc-llm/issues/3448)
- [MLC Qwen3.5 model implementation](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/model/qwen35/qwen35_model.py)
- [FlashInfer SM87 issue #2579](https://github.com/flashinfer-ai/flashinfer/issues/2579)
- [MLC Orin optimization experiment #3492](https://github.com/mlc-ai/mlc-llm/issues/3492)
- [FlashInfer installation documentation](https://docs.flashinfer.ai/installation.html)
- [vLLM process architecture](https://github.com/vllm-project/vllm/blob/main/docs/design/arch_overview.md)
- [PagedAttention/vLLM paper](https://arxiv.org/abs/2309.06180)
- [vAttention paper](https://arxiv.org/abs/2405.04437)
- [MLC introduction](https://llm.mlc.ai/docs/get_started/introduction)
- [MLC engine modes](https://github.com/mlc-ai/mlc-llm/blob/main/docs/deploy/python_engine.rst)
- [Historical MLC performance benchmarks](https://github.com/mlc-ai/llm-perf-bench)
- [Docker multi-stage builds](https://docs.docker.com/get-started/docker-concepts/building-images/multi-stage-builds/)
- [Linux cache-dropping documentation](https://kernel.org/doc/html/latest/admin-guide/sysctl/vm.html#drop-caches)
- [vLLM quantized KV cache](https://docs.vllm.ai/en/v0.18.0/features/quantization/quantized_kvcache/)
- [NVIDIA tegrastats reference](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)

## Append-only update log

- 2026-09-11 approximately 13:16–13:19 IST: reconstructed the campaign from task history, runbooks, surviving Docker metadata, compiler logs and remote recipe diffs. Current build active at 55%. Corrected current-model vLLM/MLC comparison, separated original and new TVM builds, recorded failed target detection, and preserved pending validation. No inference experiment launched in this documentation turn.
- 2026-09-11 14:27 IST: one requested status check. The detached repair had completed and exported `openclaw-mlc:qwen35-main-sm87-runtime` at 14:06:32 IST. The full new TVM build reached 100%, linked `lib/libtvm_compiler.so`, installed pytest, removed Triton, and exported successfully. Post-build host availability was 6.4 GiB RAM with 199 MiB swap in use. No runtime import, model recompilation, server start, or benchmark was run in this check.
- 2026-09-11 approximately 14:29 IST: runtime validation against `openclaw-mlc:qwen35-main-sm87-runtime` passed. `tvm` reported `0.26.dev0`; `mlc_llm` imported; `tvm.cuda().exist` was `True`; CUDA device `cuda:0` was visible; MLC exposed `compile` and `serve`; and `libtvm_compiler.so`, `libtvm_runtime.so`, and `libtvm_ffi.so` were present under `/opt/tvm-full-build/lib`. This validates import, compiler registration, library discovery and CUDA visibility only. It does not validate model compilation, serving, quality, FlashInfer use, or batching. Per user request, this result is appended; earlier summary sections are not rewritten.
- 2026-09-11 approximately 14:30 IST: launched the next authorized step as a detached Jetson Docker job, PID `224953`, with log `/home/ajmalrasi/mlc-qwen35-main-sm87-compile.log`. It uses `openclaw-mlc:qwen35-main-sm87-runtime`, only Orin SM87 environment targets (`CUDAARCHS=87`, `CUDA_ARCHITECTURES=87`, `TORCH_CUDA_ARCH_LIST=8.7`, `FLASHINFER_CUDA_ARCH_LIST=8.7`), 4K context, 1024-token prefill chunks and compiled capacity two. It writes a separate `Qwen3.5-4B-q4f16_1-main-sm87.so`, preserving the original `Qwen3.5-4B-q4f16_1-cuda.so`. This step remains pending; no output or FlashInfer claim has been made yet. Per user request, this is an append-only entry.
- 2026-09-11 14:42–14:43 IST: one requested compile status check. The separate compilation had already exited, with no `Qwen3.5-4B-q4f16_1-main-sm87.so` produced. Before exit, it verified CUDA `sm_87`, host `aarch64` / `cortex-a78ae`, Qwen3.5 model discovery, FlashInfer enabled in the optimization configuration, and all requested 4K/1024/batch-two overrides. It then stopped during “Exporting the model to TVM compiler” after roughly seven seconds. The only emitted diagnostics were `note: run with TVM_BACKTRACE=1` and `BlockBuilder destroyed with remaining blocks`; no exception, exit code, or stack trace survived because the detached container used `--rm`. This proves neither FlashInfer execution nor a model binary. A single short diagnostic retry with `TVM_BACKTRACE=1` is the next authorized action; it will not overwrite the preserved original model library.
- 2026-09-11 approximately 14:44–14:45 IST: the first diagnostic command was rejected locally because of shell quoting, so it never ran on the Jetson and changed no artifact. The corrected short `TVM_BACKTRACE=1` retry then ran and exited `1`; its log is `/home/ajmalrasi/mlc-qwen35-main-sm87-diagnose.log` (133 lines). It identifies a deterministic upstream source incompatibility during Qwen3.5 recurrent-state export, before CUDA kernel compilation: `tvm.error.InternalError: TIR is ill-formed: buffer seq_id is used ... without a prior DeclBuffer or other declaration.` The traceback reaches `mlc_llm/model/qwen35/qwen35_model.py:775`, then `mlc_llm/nn/rnn_state.py:205`, inside `RNNState.create_get_func`. TVM's parser rejects the generated function as not well formed. This is a compiler/model-source compatibility issue, not an Orin-memory, CUDA-target, FlashInfer-cache, or model-weight failure. No new `.so` was produced; the old original-runtime library remains preserved. Next action is to identify the compatible upstream MLC/TVM revision or a minimal validated source fix for this recurrent-state TIR generation. Per user request, this is an append-only entry.
- 2026-09-11 approximately 14:46–14:48 IST: searched upstream MLC/TVM sources and issues for the exact verifier failure. No published issue or merged fix for `buffer seq_id ... without a prior DeclBuffer` was found. The built source is exactly current upstream MLC main, `9fa644f54b04983adea4d0168f49fc6af4a893ba`; `git ls-remote` returned the same hash for upstream `main`, so another rebuild of the same revision cannot resolve this error. The failing code is current `python/mlc_llm/nn/rnn_state.py` high-dimensional getter: it stores a manually constructed `T.BufferLoad(storage, [seq_id, history_id, *vs])` through `T.buffer_store`. The inline source comment says the semantically equivalent direct indexed expression was avoided only for Python versions before 3.11; this runtime uses Python 3.12. Proposed minimal repair: replace that low-level load/store with direct buffer indexing so the TIR parser can see the declared buffer relationship, then build only a thin derived image and run one compile validation. Do not disable `check_well_formed`: that would silence the verifier rather than repair the malformed IR. This remains a proposed patch, not a validated fix. Relevant upstream evidence: [current Qwen3.5 source](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/model/qwen35/qwen35_model.py), [Qwen3.5 addition](https://github.com/mlc-ai/mlc-llm/issues/3448), and [Orin Qwen experiment](https://github.com/mlc-ai/mlc-llm/issues/3492). Per user request, this is an append-only entry.
- 2026-09-11 approximately 14:50 IST: began the proposed minimal repair in detached mode, build PID `229162`; log `/home/ajmalrasi/mlc-qwen35-rnnstate-fix-build.log`; target image `openclaw-mlc:qwen35-main-sm87-rnnstate-fix`. New source patch: `mlc/patches/qwen35-rnnstate-tir.diff`. It replaces only the high-dimensional recurrent-state getter's manual `T.buffer_store(T.BufferLoad(...))` with semantically equivalent direct indexed assignment. Derived Dockerfile: `mlc/Dockerfile.qwen35-rnnstate-fix`; it starts from the completed runtime image, applies the patch to the vendored source, and copies the patched file into the installed MLC Python package. The base TVM libraries, model weights, original runtime image and original model `.so` are untouched. Build completion and the patch's compiler validity remain pending. Per user request, this is an append-only entry.
- 2026-09-11 18:50 IST: one requested status check found the first thin-image build had exited before changing the image. `git apply --check` rejected `qwen35-rnnstate-tir.diff` as corrupt at line 18 because its hunk header stated 14 old / 7 new lines while the patch contained 11 old / 4 new lines. This was a patch-format error, not a source-context conflict. Corrected only the header to `@@ -199,11 +199,4 @@`; started a new detached retry, PID `272192`, with a distinct log `/home/ajmalrasi/mlc-qwen35-rnnstate-fix-build-retry.log`. No source was applied in the failed attempt, and no existing artifact changed. Per user request, this is an append-only entry.
- 2026-09-11 18:52 IST: user clarified that **everything** in this work—every request, action, rationale, command-level outcome, failure, correction, validation, estimate, status check, decision, and next step—must be appended to this Markdown journal itself. Remote files such as Docker/compiler logs are supporting evidence only; referring to a log does not replace recording the event and its meaning here. Keep this file append-only. Do not silently revise prior entries; append corrections. Apply this rule to all remaining work in the Qwen3.5/vLLM/MLC migration.
- 2026-09-11 18:51 IST: one requested status check found detached patch-image retry PID `272192` had exited before image creation. It used the cached completed runtime base and copied the corrected-format patch, but `git apply --check` rejected the patch context at `python/mlc_llm/nn/rnn_state.py:199` with `patch does not apply`. Thus the header-count repair was insufficient: the hand-authored hunk's starting line/context does not exactly match the source inside the image. The image `openclaw-mlc:qwen35-main-sm87-rnnstate-fix` was not created; no patch was applied, no model library was altered, and post-check host memory remained 6.4 GiB available with 195 MiB swap used. Next step: inspect numbered source lines from the exact base image and generate a context-accurate patch before another derived-image attempt. Per the standing instruction, this entire result is recorded here rather than only in the remote build log.
- 2026-09-11 approximately 18:54 IST: user asked for the way forward. Plan: (1) extract the exact `rnn_state.py` from the completed base image rather than hand-count a diff; (2) create and validate a context-accurate one-file patch that changes only the direct high-dimensional buffer assignment; (3) build a thin derived image detached, which should take seconds to minutes because it reuses the completed runtime; (4) run one bounded compile validation that outputs a separate new model `.so`; (5) if it compiles, verify whether FlashInfer is active or again falls back, start a concurrency-one server, and run the previously authorized bounded concurrency-one/two tests; (6) if the one-line source repair does not compile or run correctly, stop using the new mainline stack, preserve all artifacts and report the exact upstream defect with the minimal reproducer. Do not rebuild TVM/MLC again unless this evidence proves a different revision is necessary. The original MLC image/model library remain rollback artifacts throughout.
- 2026-09-11 18:53–18:54 IST: user explicitly confirmed the six-step execution plan: extract the exact runtime source, apply the direct-indexing semantic fix, build a thin detached derived image, compile a separate SM87 model library, then validate FlashInfer plus bounded concurrency one/two; if the patch fails at compile or runtime, stop this current-mainline path and preserve the older working MLC setup for an exact minimal upstream report. Inspected completed image `openclaw-mlc:qwen35-main-sm87-runtime`, immutable image ID `sha256:f358fa9dcd9c4d967abad218438e85b2dba56c0f24943974cc93e0d7a845ded0`, created 14:06:32 IST. The vendored and installed `rnn_state.py` copies are byte-identical, SHA-256 `26857c7f4bb7c46af48eddb06ebf4f3d923b7102ee3c20e9ea4e250e33e2ccb7`. Exact numbered inspection shows the high-dimensional hunk starts at source line 223, not line 199 as the second hand-authored patch claimed; line 199 belongs to the already-direct-indexed one-dimensional getter. Corrected only the patch hunk start from 199 to 223, retaining the single semantic change at lines 226–233. This explains the prior context failure precisely and does not alter either preserved image or model library. Next validation is `git apply --check` against the exact image source before launching the thin build.
- 2026-09-11 approximately 18:54–18:56 IST: the line-223/minimal-context patch still failed `git apply --check` against the completed runtime, despite a byte comparison showing its eleven claimed old lines equal source lines 223–233. No image build was launched from that failed check. Extracted the full file from a temporary container with `docker cp`, created the one-line edited copy, and generated the unified diff mechanically instead of repairing the hand-authored patch again. The exact generated hunk is `@@ -223,14 +223,7 @@` and includes the three real trailing context lines after the replacement; it removes the manual `T.buffer_store(... T.BufferLoad(...))` block and adds only `output[vi, *vs] = storage[seq_id, history_id, *vs]`. Removed the stale, unverified blob-index metadata from the patch. After copying the generated patch and unchanged derived Dockerfile to `/home/ajmalrasi/openclaw`, an isolated container check against image `openclaw-mlc:qwen35-main-sm87-runtime` printed `PATCH_CHECK_OK`. This is the first successful context/format validation; it does not yet prove the derived image builds or the model compiles.
- 2026-09-11 18:56 IST: launched the context-validated thin image build from `/home/ajmalrasi/openclaw` with output redirected to `/home/ajmalrasi/mlc-qwen35-rnnstate-fix-build-exact.log`. The reported launcher PID was `274323`; because of shell/background grouping it was a transient wrapper rather than a durable Docker PID, so the image and log—not that PID—are the completion evidence. Precheck found the target tag absent, only the unrelated healthy API and PostgreSQL containers running, 6.4 GiB RAM available, and 195 MiB swap occupied. The build reused immutable base `sha256:f358fa...` from cache, applied the verified patch, copied the patched file to the installed Python package, and exported in under one second. New image: `openclaw-mlc:qwen35-main-sm87-rnnstate-fix`, ID `sha256:a6e7da8967d19609308e58bba9132314a575dc100d3d22eadf1ea88f7621733e`, created 18:56:09 IST. The original runtime image and both existing model artifacts remain unchanged. Next step is a named, detached, SM87-only compile producing the separate `Qwen3.5-4B-q4f16_1-main-sm87.so`; retain its container so an exit code and complete logs survive failure.
- 2026-09-11 approximately 18:57 IST: verified both patched `rnn_state.py` copies inside the derived image are byte-identical with SHA-256 `b4179b9267205b136cd3670b032778fb5120ae11c27d2a43dd0e6ba4d11c8bd3`, and directly inspected the installed high-dimensional getter to confirm it contains the one direct-index assignment and no low-level load/store block. Confirmed the separate target `.so` did not already exist. Launched retained detached container `mlc-qwen35-main-sm87-patched-compile`, ID `0dbcbfac5a0757d3a6b0bb60ad918dee7c2cce261b5b97d224cc5b7c6857d15c`, using the patched image, NVIDIA runtime, the existing weight directory, `TVM_BACKTRACE=1`, and only SM87 architecture environment values (`87` / `8.7`). Exact compile configuration remains context 4096, prefill chunk 1024, max batch two, with output `/home/ajmalrasi/models/Qwen3.5-4B-q4f16_1-MLC/Qwen3.5-4B-q4f16_1-main-sm87.so`. The old `Qwen3.5-4B-q4f16_1-cuda.so` is not the output target. The named container is the detached job and preserves its exit status and logs. Compilation result, FlashInfer behavior and output existence remain pending.
- 2026-09-11 approximately 18:57–18:58 IST: one immediate launch verification found the named compile container running, not OOM-killed, with CUDA device `cuda:0`, target architecture `sm_87`, host `aarch64-unknown-linux-gnu` / `cortex-a78ae`, Qwen3.5 model type, FlashInfer requested in optimizer configuration, and the exact 4096/1024/batch-two overrides applied. It reached `Exporting the model to TVM compiler`, the stage where the unpatched image failed after roughly seven seconds. At this observation the prior malformed-TIR traceback had not reappeared, but the process was still active; reaching the stage is not compilation success and `flashinfer=1` is not evidence that its kernels build or execute. The job is being left detached without continuous polling.
- 2026-09-11 approximately 18:58–18:59 IST: the retained patched compile container exited `1` after about 27 seconds, with `OOMKilled=false`, and produced no `Qwen3.5-4B-q4f16_1-main-sm87.so`. The direct-indexing change did **not** repair the compiler failure. TVM emitted the same deterministic verifier error during model export: `TIR is ill-formed: buffer seq_id is used at <root>.body.block.body.body.body.body.body.block.reads[2].region[0].min.buffer without a prior DeclBuffer or other declaration`, followed by `Program is not well-formed`; patched file traceback locations are `rnn_state.py:205` at the decorator and line 230 returning `_func_high_dim`. Therefore the earlier diagnosis that direct indexing would expose the buffer relationship was disproved. The compile failed before CUDA kernel generation, so FlashInfer execution, concurrency-one serving, the bounded benchmark and concurrency two were not attempted. This satisfies the user's explicit stop condition: no further current-mainline MLC repair will be pursued in this path. Full retained log: `/home/ajmalrasi/mlc-qwen35-main-sm87-patched-compile.log`, 132 lines / 20,365 bytes. The failed named container remains retained. The original model library still exists with SHA-256 `917ee583d732167dd1f5af59e0805514fc6372c5e12251f09f2512bd70310743`; preserved images remain old working runtime `sha256:5518bed...`, unpatched new runtime `sha256:f358fa...`, and patched diagnostic image `sha256:a6e7da...`. Only the unrelated API and PostgreSQL containers are running; post-failure availability is 6.4 GiB RAM and 195 MiB swap used.
- 2026-09-11 approximately 18:59 IST: collected exact report metadata. MLC source is `9fa644f54b04983adea4d0168f49fc6af4a893ba`; vendored TVM is `837cb9de1127b48ce48e4cefe09e83215b9d4ba7`; Python is 3.12.14 built with Clang 22.1.3; reported packages are TVM `0.26.dev0` and MLC `0.1.dev0`; CUDA visibility remains true under the NVIDIA runtime. A first metadata command omitted the NVIDIA runtime and consequently failed only while importing MLC with `OSError: libcuda.so.1` missing; rerunning the same read-only check with the runtime succeeded. This incidental import failure did not alter any artifact and is unrelated to the TIR bug.
- 2026-09-11 approximately 18:59 IST: reduced the failure to a model-weight-independent call: `RNNState.create_get_func((32, 128, 128), "float32", 2, 2, 0)` inside the patched image. It reproduces the identical `buffer seq_id ... without a prior DeclBuffer` verifier failure directly in `_func_high_dim`. The `(32,128,128)` dimensions come from Qwen3.5's first recurrent state (`n_vh=32`, key dimension 128, value dimension 128). This establishes a concise upstream reproducer without compiling weights or reaching CUDA kernels. Created local report draft `mlc/QWEN35_RNNSTATE_UPSTREAM_BUG.md` containing exact revisions, reproducer, expected/actual behavior, both source forms, command, artifact identities and evidence paths. It explicitly labels the typed-local/read-region explanation as an investigation lead rather than a proven root cause. The current-mainline experiment is now stopped; the older working MLC image and library are preserved for rollback.
- 2026-09-11 approximately 19:00 IST: final preservation check. One locally malformed quoting attempt was rejected by the Mac shell with `unmatched \"` before reaching SSH; it changed nothing. The corrected read-only check confirmed `openclaw-mlc-jetson` and the original `mlc-qwen35-compile` container are both preserved as exited-zero containers based on `openclaw-mlc:qwen35-runtime`; the failed new compile is separately retained as exited one; the older working 31 MiB model library remains at its original path; and the vLLM system service remains inactive as it was during this build campaign. No service was restarted, no image/container/model was deleted, and the unrelated API/database remained untouched. This is the terminal state for the authorized current-mainline attempt. Any later work should begin from the report and journal rather than continue speculative patches in this stopped path.
- 2026-09-11 after 19:00 IST: user explicitly requested restoration of the known-working Qwen3.5 vLLM service. Authorized action is to start the existing `openclaw-vllm-jetson.service` without changing its model, image, configuration, API contract or unrelated services; then verify service/container state, model startup and a real OpenAI-compatible completion using model alias `openclaw`. This is a production restoration, not a continuation of the stopped current-mainline MLC experiment.
- 2026-09-11 after 19:00 IST: initial system-level lookup found no unit because the service is installed in the user's systemd instance, not the system instance. Read-only `systemctl --user` inspection found the enabled unit at `/home/ajmalrasi/.config/systemd/user/openclaw-vllm-jetson.service`, retaining the validated pinned NVIDIA image, Red Hat Qwen3.5-4B W4A16 checkpoint, tokenizer compatibility mount, language-only mode, 4096 context, explicit 192 MiB KV cache, one sequence, eager execution, disabled thinking, and Qwen parsers. Its stale state was `failed` with exit 137 from the prior experiment stop; historical logs show it had loaded 4.44 GiB of weights before that stop. This check made no configuration change. Next action is `systemctl --user reset-failed` followed by starting this exact existing unit.
- 2026-09-11 19:05–19:06 IST: cleared the stale failed state and started the unchanged user service. Systemd reported active/running and Docker created `openclaw-vllm-jetson` from the pinned image. The first bounded 60-second HTTP readiness check ended before `/v1/models` became available, but the service remained active and startup logs showed forward progress rather than failure: Qwen3.5 architecture resolved, text-only mode applied, 4096 maximum length and 1024-token chunked prefill applied, CUDA/NCCL initialized, Triton/FLA GDN prefill selected, Marlin W4A16 selected, FlashAttention 2 selected, both checkpoint shards loaded, and model loading completed at 4.44 GiB in 11.08 seconds. A tensor-shape warning also appeared during warm-up, matching the previously observed warning and not yet an endpoint failure. Readiness and real response verification remain pending; continue with one additional bounded window rather than restarting the progressing service.
- 2026-09-11 approximately 19:07–19:08 IST: a second bounded 60-second readiness window also ended before the HTTP listener became ready. The unit and container remained active with no new traceback or exit. Process/memory inspection showed the `VLLM::EngineCore` alive and CPU-active at approximately 77%, container CPU approximately 90%, container memory 6.209 GiB, system available RAM approximately 386 MiB, and swap use increased to 1.3 GiB; substantial block I/O indicated active memory/loading work rather than an idle dead process. A first `tegrastats` attempt used unsupported `--count` syntax on this installed version and returned only its usage error; it changed nothing. No OOM evidence was observed. Decision: do not restart or change configuration while the engine is active; allow another bounded readiness interval.
- 2026-09-11 approximately 19:08–19:09 IST: a third bounded readiness window ended without an HTTP listener. At roughly four minutes service elapsed, the engine remained in running state and CPU-active at about 77%, with RSS approximately 6.08 GiB and no new log error after the previously recorded warm-up shape warning. This remains slower than desired but is evidence of an active first-start/warm-up process rather than a dead or exited unit. No restart or configuration mutation was performed; continue waiting in bounded intervals.
- 2026-09-11 approximately 19:09–19:11 IST: a fourth bounded readiness check still found no HTTP listener at about 5 minutes 26 seconds service elapsed. Both processes remained alive; the API process became runnable with RSS about 763 MiB, while the engine remained alive with about 5.57 GiB RSS and approximately 70% average CPU. This memory redistribution and continuing CPU activity indicate ongoing initialization. No crash, OOM, restart, or new configuration change occurred. Continue the authorized restoration rather than declaring readiness prematurely.
- 2026-09-11 approximately 19:11–19:12 IST: restoration completed successfully. `/v1/models` returned HTTP 200 and advertised only model alias `openclaw`, rooted at `RedHatAI/Qwen3.5-4B-quantized.w4a16`, with maximum length 4096. Server logs report application startup complete at 19:11:06 IST, approximately 5 minutes 53 seconds after systemd start; this was a slow cold start under unified-memory/swap pressure, not an endpoint benchmark. A real LAN `POST /v1/chat/completions` with a 70-second client cap, non-streaming mode, temperature zero and 16-token cap returned HTTP 200 in 24.70 seconds with exactly `LIFEOS_OK`, finish reason `stop`, no visible reasoning, no tool call, 22 prompt tokens and 5 completion tokens. This validates the configured direct-response API contract, not sustained performance or concurrency. Final state check: user service active, container `openclaw-vllm-jetson` up, container memory 6.152 GiB, system available RAM approximately 419 MiB, swap use 3.0 GiB, and GPU KV cache idle after the request. The tensor-shape warning repeated for the real 22-token request even though the output was correct; preserve it as an unresolved runtime warning rather than silently dismissing it. No MLC artifact or unrelated service was changed during restoration. Qwen3.5 vLLM is now running at `http://192.168.3.30:11434/v1/chat/completions` using model name `openclaw`.
- 2026-09-11 after successful restoration: user requested commit, push and synchronization. Fetched `origin`; local `main` and `origin/main` were exactly even before the new commit (zero commits ahead and zero behind). Commit scope is limited to this campaign's journal plus `mlc/Dockerfile.qwen35-runtime`, `mlc/Dockerfile.qwen35-rnnstate-fix`, `mlc/patches/qwen35-rnnstate-tir.diff`, and `mlc/QWEN35_RNNSTATE_UPSTREAM_BUG.md`. Existing `.gitignore` changes and the untracked `examples/` tree are unrelated user work and will remain unstaged. Next steps are whitespace validation, one scoped commit, push to `origin/main`, and an upstream synchronization check.
