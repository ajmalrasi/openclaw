# P1: native chunk and physical-slot feasibility

Date: 2026-09-14, Asia/Kolkata. Status: **bounded feasibility gate passed**.
Continuous serving is **not implemented or enabled yet**. Next phase: persistent
sequence ownership and sampling-free native execution boundaries (P2).

## What ran

One native runtime on the Orin Nano Super 8 GB, using the existing batch-two,
6144-input/8192-sequence Qwen3.5-4B INT4 AWQ engine, FP16 attention KV, FP32 recurrent
state, no speculation or context reuse. Normal serving and its watchdog were
stopped only during each build/test and automatically restored afterward. The
API/database containers were left running and verified healthy.

- Native source/archive: `e8b29522938901f6df19ebeedd4b69bc8edbcd97`, release 0.10.1.
- Implementation checkout: `/Users/ajmalrasi/openclaw-tensorrt-concurrency-review`,
  branch `codex/continuous-batching-p1`; changes are uncommitted.
- Probe: `examples/llm/continuousBatchingProbe.cpp`, optional CMake target
  `continuous_batching_probe`. Runtime header change: one friend declaration.
- Jetson probe/evidence directory: `/home/ajmalrasi/continuous-batching-p1-20260914`.
- Engine: `/home/ajmalrasi/Qwen3.5-4B/engines/llm-b2-input6144-kv8192-vanilla/llm.engine`.
- Engine SHA-256: `8a968e59bca3dbad7e193a7431d4fb3acf7e3e0f6a7f34858e519d3d9e1bd4d3`.
- Config SHA-256: `7dbe1219ab15152d781a94001f3e4c964a4ec70c198082868cad0773db200d1d`.
- Tested probe SHA-256: `e6d150f07240b31455da62c958796031e2af14611dd488997a6ab6b24067ca35`
  (identical locally and on the Jetson).

The probe was compiled with warnings treated as errors and linked against the
existing pinned archive and CUDA device-link object. Production source, libraries,
Python bindings and model artifacts were not overwritten. The optional CMake
target is provided but a fresh full CMake build was not exercised.

## Attempts and retained evidence

| Attempt | Result | Evidence |
| --- | --- | --- |
| 1 | Compile error: vector-typed `OptionalInputTensors` was initialized with `nullopt`; corrected to `{}`. No inference. Service restored. | [attempt1.log](attempt1.log) |
| 2 | Idle-health precondition failed; exited before maintenance. No inference. | [attempt2.log](attempt2.log) |
| 3 | Build succeeded; reproduced incorrect one-token resumed-prefill output. Isolation passed. Service restored. | [attempt3.log](attempt3.log) |
| 4 | Corrected execution contract; 24 output comparisons and five exact isolation checks passed. Exit 0; service restored. | [attempt4.log](attempt4.log) |

The two model-executing attempts together consumed under 40 seconds including
initialization, within the total five-minute validation cap. Attempt 3 had a
295-second external deadline; attempt 4 was reduced to 250 seconds to retain a
cumulative bound. These were functional checks, not throughput benchmarks.

## Actual finding and correction

The attention plugin's `deduceModeVanilla()` chooses decoding when start indices
are nonempty and the runtime sequence length is one. Its decode path interprets
`context_lengths` as absolute KV endpoints. The general prefill helper supplied
the chunk length (one), which was incompatible with that mode: full versus
64+1 and 64+64+1 prompts produced maximum absolute logit errors of 11.3906 and
16.0312, including wrong greedy tokens.

The proof adapter now sends a logical one-token resumed prompt chunk through
decode-shaped execution with `context_lengths = committed + 1`. It discards the
decoder's sampled token and advances only prompt accounting. Multi-token chunks
use prefill execution; a cold one-token prompt remains initial prefill.

This is a binding/metadata contract correction using the **existing serialized
engine**, not a new engine or a plugin binary replacement. P2/P3 must expose a
sampling-free forward operation with the same contract; the proof's discarded
sample is not the intended production implementation.

## Passing evidence

- Lengths 1, 3, 4, 63, 64, 65, 127, 128 and 129: full slot-0 prefill versus
  slot-1 chunks of at most 64 tokens, followed by teacher-forced decode.
- Identical singleton shapes produced exactly equal logits across physical slots.
  Multi-token 64+63 and 64+64 continuation was also exactly equal to full prefill.
- One-token-tail paths preserved greedy IDs. Across passing comparisons,
  maximum absolute logit error was 0.0761719 and maximum relative L2 was 0.00426784.
- A decoded while B prefilled with exactly matching A reference logits.
- Five exact inactive-state checks covered 54,002,176–56,164,864 bytes per check:
  every recurrent/convolution byte, live attention-KV prefix and page-table row.
  Unwritten KV capacity was intentionally excluded.
- Both rows of a two-sequence decode matched independent full-prefill baselines
  exactly. Prefill/decode profile switching, slot 1 alone and reuse were exercised.
- Committed endpoints matched the native cache-manager result after every step.

Numeric screens were declared before testing: finite logits, matching greedy
IDs, max absolute error ≤0.1 and relative L2 ≤0.005; equal-shape singleton checks
required exact equality. No tolerance was relaxed after failure. These FP16
screens establish limited feasibility, **not** broad model-quality acceptance.

## Memory and restoration

Attempt 4 reported CUDA free memory of 283,189,248 bytes after initialization and
117,321,728 bytes at completion. The latter includes approximately 54 MiB of
host-only state snapshots plus other test/lazy allocations on unified memory.
These are not exclusive GPU-use measurements, a leak test, or a production
memory budget. Production state isolation uses selected bindings, not the
diagnostic snapshot copies.

At 15:44:19 IST automatic restoration reported `healthy=true`, exit 0. Subsequent
checks confirmed both the model service and watchdog timer active, `/v1/models`
identifying TensorRT `openclaw`, and the original two Python engine-directory
support edits as the only dirty production files. `/health` still honestly
reports `max_num_seqs=1`. See [restored health](restored-health.json).

## Remaining work

P2–P8 remain pending. In particular: stable ownership/generation handles,
sampling-free steps, production chunk handling, automatic admission and slot
reuse, independent sampling/limits/cancellation, HTTP/SSE integration, CUDA graphs,
memory/performance qualification and deployment. Unequal-length two-row decode,
larger/diverse prompts, full active-state numerical comparisons and long-term
stability were not established by this short probe.
