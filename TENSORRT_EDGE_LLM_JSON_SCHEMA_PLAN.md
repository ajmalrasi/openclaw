# JSON Schema structured output for Jetson TensorRT Edge-LLM

High-level implementation plan, 2026-09-16. This document deliberately does not contain a detailed design or authorize implementation.

## Objective

Make the existing OpenAI-compatible Jetson endpoint enforce caller-provided JSON Schema during generation, rather than relying on prompting and post-generation parsing. The public model alias, endpoint, one-resident-model constraint, and current batch-two continuous scheduler must remain intact.

The current server explicitly rejects `response_format` because structured decoding is absent. Tool-call parsing is not a substitute: it interprets model output after generation and cannot ensure arbitrary JSON follows a schema.

## Chosen direction

Implement guided decoding in TensorRT Edge-LLM. Prefer integrating a maintained grammar backend such as XGrammar, subject to aarch64/Jetson compatibility verification. Do not create a new JSON-Schema compiler or migrate the service to full TensorRT-LLM as part of this work.

## Major workstreams

1. Establish the grammar dependency
   - Verify licensing, source/build compatibility, tokenizer requirements, CUDA 13.2/aarch64 support, and memory footprint on the Jetson.
   - Define the supported JSON-Schema subset and clear errors for unsupported schemas.

2. Define the OpenAI contract
   - Support the project-required `response_format` JSON-Schema request shape.
   - Validate schemas before scheduler admission; return actionable 4xx errors without affecting other requests.
   - Define interaction rules for tools, stop strings, logprobs, streaming, thinking, and unsupported decoding modes.

3. Carry constraints into native inference
   - Extend the Python protocol/runtime, pybind bridge, and native generation request with per-request guide configuration.
   - Compile a grammar state for each admitted request and retain it independently for the lifetime of its physical scheduler slot.

4. Constrain sampling on the GPU
   - Before every first-token and decode-token sample, derive allowed tokens from the request's current grammar state and mask disallowed logits.
   - Advance the grammar state only with the token actually emitted.
   - Reuse the existing per-slot logit-bias/mask path where appropriate; do not copy full logits to the CPU for each token.

5. Preserve batch-two scheduling correctness
   - Keep grammar state separate for each live sequence despite batched model forwards.
   - Safely release/reset guide state on completion, cancellation, failures, disconnects, and slot reuse.
   - Begin with the deployed vanilla decoder. Treat future speculative decoding as a separate compatibility gate.

6. Integrate API responses
   - Support both non-streaming and SSE responses without releasing partial invalid JSON as a completed result.
   - Preserve correct token usage, terminal events, error handling, and OpenAI response shape.

7. Qualify before deployment
   - Add unit, native, and endpoint tests for valid schemas, nested objects/arrays, escaping/Unicode, required fields, enums, numeric limits, invalid/unsupported schemas, cancellation, slot reuse, and two simultaneous constrained requests.
   - Measure added latency, decode throughput, and memory on the 8 GB Orin. Retain the existing rollback service until acceptance gates pass.

## Primary code areas

| Layer | Location | Role |
| --- | --- | --- |
| API schema and validation | `experimental/server/api/protocol.py`, `experimental/server/api/serving_chat.py` | Accept and validate `response_format` instead of rejecting it. |
| Python request construction | `experimental/server/runtime/engine.py` | Carry guide settings into native generation requests. |
| Python/C++ bridge | `experimental/pybind/edgellm_pybind.cpp` | Expose guide fields and native status safely. |
| Native request model | `cpp/runtime/llmRuntimeUtils.h` | Store per-request guide configuration/state ownership. |
| Native sampling | `cpp/runtime/llmRankRuntime.cpp`, `cpp/runtime/decoding/vanillaDecoder.cpp`, `cpp/runtime/decoding/logitBias.*` | Apply an allowed-token mask before each sample. |
| Scheduler lifecycle | current continuous batching runtime | Keep per-slot grammar state correct through admission, cancellation, completion, and reuse. |
| Build and tests | CMake/dependency setup and server/native tests | Build the grammar backend and verify correctness/performance. |

## Phased delivery

| Phase | Outcome | Gate |
| --- | --- | --- |
| 0. Feasibility | Grammar backend builds and initializes with the Qwen tokenizer on Jetson. | Dependency, license, memory, and tokenizer checks pass. |
| 1. Single-request native path | One JSON Schema constrains vanilla generation. | Generated output validates against representative schemas. |
| 2. API contract | `response_format` is validated and mapped end-to-end. | Correct 2xx/4xx behavior for streaming and non-streaming requests. |
| 3. Batch-two lifecycle | Independent guides work through concurrent admission and slot reuse. | Concurrent/cancelled requests cannot leak or corrupt guide state. |
| 4. Qualification | Performance, memory, reliability, and rollback validation. | Meets agreed Jetson acceptance targets with retained evidence. |

Phase 0 completed on 2026-09-22. XGrammar 0.2.7 built on the Jetson, all 73
upstream C++ tests passed, and the exact Qwen3.5 tokenizer/schema matcher gate
passed. The dependency is feasible with a material approximately 104 MiB
standalone peak-RSS caution. See
[`TENSORRT_EDGE_LLM_JSON_SCHEMA_PHASE0_RESULTS.md`](./TENSORRT_EDGE_LLM_JSON_SCHEMA_PHASE0_RESULTS.md).

Phases 1–3 completed on 2026-09-23 in signed-off local commit `2cb8bb8` on
the isolated `feature/json-schema-guided-decoding` branch. The candidate masks the logits
in the scheduler's **existing CPU sampling row**; it adds no new full-logits
device-to-host transfer, but it does not yet implement the GPU mask described
in workstream 4. GPU-side masking, full-build qualification, overhead/memory
measurement, production-sized engine fit, rollback verification, and any
deployment remain Phase 4 work. Passing functional Phases 1–3 must not be
read as production acceptance. See
[`TENSORRT_EDGE_LLM_JSON_SCHEMA_PHASE1_3_RESULTS.md`](./TENSORRT_EDGE_LLM_JSON_SCHEMA_PHASE1_3_RESULTS.md).

## Scale and constraints

- A proof of concept is roughly 1–2 engineer-weeks.
- A deployable batch-two/streaming vanilla implementation is roughly 4–6 engineer-weeks.
- Broad, upstream-quality support is roughly 6–10+ weeks.

These are planning estimates, not commitments. The first technical gate is whether the selected grammar backend works acceptably on Jetson aarch64/CUDA 13.2 and does not break the current memory budget.

## Explicit non-goals for this plan

- No implementation, dependency installation, service restart, engine rebuild, or benchmark is authorized by this document.
- No promise of all JSON-Schema keywords before a supported subset is specified and tested.
- No replacement of the current TensorRT Edge-LLM deployment with full TensorRT-LLM.
