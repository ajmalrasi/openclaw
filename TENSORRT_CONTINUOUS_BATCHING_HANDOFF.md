# TensorRT Edge-LLM continuous batching implementation handoff

This is the reusable handoff for implementing any phase of the TensorRT Edge-LLM continuous-batching project. Give an agent this file and ask it to implement a named phase, for example: “Use TENSORRT_CONTINUOUS_BATCHING_HANDOFF.md and implement P3.”

## Mission

Implement proper native continuous batching for the Jetson TensorRT Edge-LLM server, including chunked prefill. The target behavior is genuine staggered admission: request A can be decoding, request B can arrive and prefill in bounded chunks, and a newly completed request can release its physical slot to request C while B remains active. Python request coalescing alone is not an acceptable final solution.

The implementation must preserve correctness for the hybrid Qwen3.5 model: attention KV cache, recurrent/GDN state and causal-convolution state must remain owned by the correct logical request. Sampling, limits, cancellation, errors, streaming and request cleanup must be independent.

## Required reading

1. [Agent instructions](AGENTS.md)
2. [Shared API and Jetson operating contract](FOR-AGENTS.md)
3. [Eight-phase implementation plan](TENSORRT_CONTINUOUS_BATCHING_PLAN.md)
4. [TensorRT experiment journal](TENSORRT_EDGE_LLM_EXPERIMENT_LOG.md)
5. [Concurrency source analysis](TENSORRT_CONCURRENCY_ANALYSIS.md)
6. [P1 feasibility results](tensorrt-edgellm/evidence/continuous-batching-p1-20260914/RESULTS.md)
7. [P2 persistent-state results](tensorrt-edgellm/evidence/continuous-batching-p2-20260914/RESULTS.md)
8. [P1 native harness notes](https://github.com/ajmalrasi/TensorRT-Edge-LLM/blob/codex/continuous-batching-p1/examples/llm/continuousBatchingProbe.md)
9. [P2 step-runtime notes](https://github.com/ajmalrasi/TensorRT-Edge-LLM/blob/codex/continuous-batching-p1/examples/llm/sequenceStepRuntime.md)
10. [Jetson deployment and rollback procedure](TRT_EDGE_LLM_JETSON_DEPLOYMENT.md)

No other Markdown file is required for implementing these eight phases.

## Source and branches

- Local workspace: `/Users/ajmalrasi/openclaw`
- TensorRT source checkout: `/Users/ajmalrasi/openclaw-tensorrt-concurrency-review`
- Source repository: [ajmalrasi/TensorRT-Edge-LLM](https://github.com/ajmalrasi/TensorRT-Edge-LLM)
- Implementation branch: `codex/continuous-batching-p1`
- Pushed source commit: `cf57e1c`
- Existing base: `e8b29522938901f6df19ebeedd4b69bc8edbcd97` (TensorRT Edge-LLM 0.10.1)

Do not work directly on the live deployment checkout. Preserve unrelated dirty files. Use a `codex/` branch. Commit C++ changes with `git commit -s`; do not add AI co-authors.

## Current verified deployment

- Jetson: Orin Nano Super, 8 GB unified memory, JetPack 7.2.1 / L4T r39.2.1, CUDA 13.2, SM87, TensorRT 10.16.2.
- Host: `ajmalrasi@192.168.3.30`.
- Service: `openclaw-tensorrt-edgellm.service`.
- API: `http://192.168.3.30:11434/v1/chat/completions`.
- Model alias: `openclaw`.
- Engine: `/home/ajmalrasi/Qwen3.5-4B/engines/llm-b2-input6144-kv8192-vanilla`.
- Engine: vanilla Qwen3.5-4B INT4 AWQ, batch capacity 2, max input 6144, max sequence 8192, FP16 KV, FP32 recurrent state, no MTP, no context reuse.
- Current health honestly reports `max_batch_size=2` and `max_num_seqs=1`. Do not claim continuous batching until later phase gates pass.
- Preserve one resident model and unrelated API/database containers.
- Watchdog: `openclaw-tensorrt-watchdog.timer`; pause it during controlled model maintenance and restore its prior state.
- Total benchmark-session cap: five minutes, including warmup. Keep long jobs detached.

## Existing implementation

P1 proved selected physical slots, resumed multi-token chunks, one-token tail handling, profile switching and two-row decode on the existing engine. Important contract: a resumed logical one-token prompt tail uses decode-shaped execution with absolute context lengths; the proof adapter discards the decoder sample and advances prompt accounting only.

P2 added, but has not been integrated into production serving:

- `cpp/runtime/state/sequenceSlots.{h,cpp}`: generation-tagged handles, request-local state, independent options, prompt/output accounting, bounded admission and reuse protection.
- `cpp/runtime/sequenceStepRuntime.{h,cpp}`: exclusive runtime lease, selected physical views `{0}`, `{1}`, `{0,1}`, sampling-free forward steps, explicit completion, finish/release and failure poisoning.
- Native tests and diagnostic probe under `examples/llm/` and `unittests/cpp/runtime/state/`.

P2 is scoped to single-rank, two-slot, text-only vanilla Qwen3.5. It does not provide a scheduler, automatic chunking, sampling policy, HTTP/SSE integration, CUDA graphs or production deployment.

## Phase ranking and gates

| Phase | Scope | Difficulty |
| --- | --- | ---: |
| P1 | Engine feasibility and native proof | 8/10 — complete |
| P2 | Persistent ownership and native steps | 8/10 — complete, not production-integrated |
| P3 | Correct chunked prefill and numerical qualification | 10/10 — next |
| P4 | Native continuous scheduler and slot reuse | 9/10 |
| P5 | Independent sampling, limits, cancellation and failures | 9/10 |
| P6 | Pybind, HTTP and independent SSE streaming | 7/10 |
| P7 | CUDA graphs, tuning, memory and performance qualification | 10/10 |
| P8 | Deployment, watchdog and rollback verification | 5/10 |

P3 must resolve or rigorously qualify the retained full-versus-chunked discrepancy. Test boundaries 1, 3, 4, 63, 64, 65, 127, 128 and 129, one-token tails, longer prompts, all three state types, active-state comparisons and teacher-forced continuation. Do not loosen thresholds.

P4 must add one native worker, bounded queue, decode-first scheduling, guaranteed prefill progress, chunk limits, immediate slot reuse and no busy waiting. Prove A starts, B arrives, A finishes, C reuses A while B continues.

P5 must make temperature, top-p/top-k, RNG seed/counter, output limits, EOS/thinking, stop strings, logit bias, logprobs, cancellation, deadlines and failures independent. Corrupting CUDA failure must mark the runtime unready.

P6 must replace the HTTP single-generation lease with bounded native submission and independent result/SSE ownership. Test mixed clients, parameters, disconnect isolation, usage, overload and shutdown.

P7 comes only after correctness: qualify finite graph views, chunk size, profile switching, allocations, memory, singleton regression, TTFT, inter-token gaps and aggregate throughput. Do not promise a 2x speedup.

P8 deploys only a qualified candidate and verifies health/models, streaming/non-streaming, staggered requests, restart, watchdog and rollback.

## Non-negotiable rules

- One native execution owner; never invoke one rank runtime concurrently.
- No state-sized copies, per-token heap allocation or per-request CUDA graph capture.
- Keep physical slots stable; use selected-row views and generation tags.
- PR #199 coalescing is not the final continuous-batching solution.
- Preserve the legacy whole-request API until P6 migration is complete.
- Use eager execution until P7 graph qualification.
- Update the experiment journal after every experiment, build, service change, benchmark or requested status check.
- Never report an untested phase as complete. Distinguish implementation, mechanism verification, numerical qualification, performance qualification and deployment.

## Completion report

When finishing a phase, report:

1. Exact phase and scope.
2. Files changed and commit hash.
3. Tests and exact evidence paths.
4. Failures retained, diagnosis and unresolved issues.
5. Jetson/service/watchdog state before and after.
6. Whether the next phase is safe to start.
