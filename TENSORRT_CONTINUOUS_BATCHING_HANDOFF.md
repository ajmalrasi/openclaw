# TensorRT Edge-LLM continuous batching implementation handoff

This is the reusable handoff for implementing any phase of the TensorRT Edge-LLM continuous-batching project. Give an agent this file and ask it to implement a named phase, for example: “Use TENSORRT_CONTINUOUS_BATCHING_HANDOFF.md and implement P6.”

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
10. [P3 chunked-prefill results](tensorrt-edgellm/evidence/continuous-batching-p3-20260915/RESULTS.md)
11. [P3 chunk API notes](../openclaw-tensorrt-concurrency-review/examples/llm/chunkedPrefill.md)
12. [P4 scheduler results](tensorrt-edgellm/evidence/continuous-batching-p4-20260915/RESULTS.md)
13. [P4 native API notes](../openclaw-tensorrt-concurrency-review/examples/llm/continuousScheduler.md)
14. [Jetson deployment and rollback procedure](TRT_EDGE_LLM_JETSON_DEPLOYMENT.md)

Phase 5 completion is recorded only in this handoff, as explicitly requested by the user. The plan, experiment journal and older native notes retain their prior phase status; use the P5 section below for current status and integration requirements. No other Markdown file is required for implementing these eight phases.

## Source and branches

- Local workspace: `/Users/ajmalrasi/openclaw`
- TensorRT source checkout: `/Users/ajmalrasi/openclaw-tensorrt-concurrency-review`
- Source repository: [ajmalrasi/TensorRT-Edge-LLM](https://github.com/ajmalrasi/TensorRT-Edge-LLM)
- Implementation branch: `codex/continuous-batching-p1`
- Latest source commit: `1496d34` (P6/P7), pushed to `origin/codex/continuous-batching-p1`.
- Previously pushed source commit: `cf57e1c` (P2).
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

P3 adds `beginPrefillChunk(handle)` with a fixed 128-token cap. Resumed final chunks contain 64–128 true prompt tokens; smaller remainders are absorbed by shortening the preceding chunk. Cold one-token prompts remain supported. Arbitrary low-level partitions, including raw 64+64+1, are not qualified. The first singleton-only workaround also failed for remainders 2–4; both failures remain recorded. The final policy passed 487 exact logit and 11,648 exact active-state comparisons across passing runs, plus 20 exact inactive snapshots in both slot orders. P4 must use this helper from the first prompt step. No scheduler or HTTP deployment has occurred.

P4 adds `ContinuousScheduler` and `GreedySchedulerBackend`: one worker, bounded FIFO admission, decode-first rounds and round-robin P3 chunks, safe-boundary slot reuse, tickets, basic cancellation, idle sleep and failure poisoning. Twenty host tests pass normally, with ASan/UBSan and ThreadSanitizer. Two Jetson runs proved automatic A/B/C staggering and exact serial greedy output equality. The API intentionally accepts only prepared tokens and output length; EOS and independent sampling policy are P5 work. Queued cancellation settles at admission/shutdown. No HTTP integration or production replacement occurred.

## Phase ranking and gates

| Phase | Scope | Difficulty |
| --- | --- | ---: |
| P1 | Engine feasibility and native proof | 8/10 — complete |
| P2 | Persistent ownership and native steps | 8/10 — complete, not production-integrated |
| P3 | Correct chunked prefill and numerical qualification | 10/10 — complete for the fixed policy, not deployed |
| P4 | Native continuous scheduler and slot reuse | 9/10 — complete, not deployed |
| P5 | Independent sampling, limits, cancellation and failures | 9/10 — complete, native-only |
| P6 | Pybind, HTTP and independent SSE streaming | 7/10 — complete, candidate-only |
| P7 | CUDA graphs, tuning, memory and performance qualification | 10/10 — complete, candidate-only |
| P8 | Deployment, watchdog and rollback verification | 5/10 |

P3 qualified the fixed policy through 6144 tokens, all remainder boundaries, four formatted chat fixtures, all three state types, teacher-forced continuation and both slot orders. Raw short-tail numerical drift remains an unsupported diagnostic path; never loosen thresholds or substitute arbitrary manual spans in P4.

P4 passed: A starts decoding, B arrives, A finishes, C reuses A’s slot while B continues, then B/C decode together. Preserve its single-owner and qualified chunking contract in P5.

P5 passed independent temperature, top-p/top-k, seed/counter, output limits, EOS/thinking, stop strings, bias, logprobs, cancellation, deadlines and bounded output checks. P6 completed native pybind/HTTP ticket ownership and independent SSE. P7 qualified three startup-only decode graph views, graph replays, bounded allocations, singleton latency, staggered TTFT, aggregate throughput, near-capacity requests and reuse. The candidate remains opt-in and the production endpoint still reports one active sequence.

P8 is safe to start: deploy only the qualified candidate and verify health/models, streaming/non-streaming, staggered requests, restart, watchdog and rollback.

## Non-negotiable rules

- One native execution owner; never invoke one rank runtime concurrently.
- No state-sized copies, per-token heap allocation or per-request CUDA graph capture.
- Keep physical slots stable; use selected-row views and generation tags.
- PR #199 coalescing is not the final continuous-batching solution.
- Preserve the legacy whole-request API until P8 migration is complete.
- Capture only the three verified startup decode views; never capture per request.
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


## P5 completion and P6 integration contract — 2026-09-15

**P5 complete within the native single-rank, two-slot Qwen3.5 scope. P6 is safe to start.**
Source commit **`711270f`**, DCO signed, on `codex/continuous-batching-p1`, not pushed.
Parent is P4 `e37d897`. No production HTTP/SSE integration or deployment occurred.

### Changed native files

- `cpp/runtime/continuousScheduler.{h,cpp}`: options-based submission, per-request
  terminal data, immediate queued cancellation/deadline scanning at safe boundaries,
  active deadlines, bounded stream publication and slow-consumer isolation.
- `cpp/runtime/greedySchedulerBackend.{h,cpp}`: `SamplingSchedulerBackend`, with
  `GreedySchedulerBackend` retained as an alias; shared forwards followed by each
  row's independent sampler and stopping state.
- `cpp/runtime/state/sequenceSlots.h`: request-local EOS/thinking metadata and
  explicit ignore-EOS diagnostic control.
- `cpp/runtime/state/sequencePolicy.{h,cpp}`: deterministic sampling, logprobs,
  UTF-8-safe decoded text, cross-token stops and thinking/EOS state.
- `cpp/runtime/state/sequenceChannel.{h,cpp}`: preallocated record/byte ring with
  separate closure notification and consumer-side reads.
- `unittests/cpp/runtime/continuousSchedulerTest.cpp` and
  `unittests/cpp/runtime/state/sequencePolicyTest.cpp`: policy/lifecycle tests.
- `examples/llm/continuousBatchingProbe.cpp`: `--policies` native validation.

Operator wrapper: `tensorrt-edgellm/run-continuous-batching-p5.sh` in OpenClaw.
Only this handoff receives Phase 5 Markdown updates per the user's instruction.

### Verified results and exact evidence

All artifacts are under `tensorrt-edgellm/evidence/continuous-batching-p5-20260915/`.

| Check | Result | Evidence |
| --- | --- | --- |
| Final host suite | 33 passed: scheduler, existing slot/chunk, policy and channel tests | [host-tests.xml](tensorrt-edgellm/evidence/continuous-batching-p5-20260915/host-tests.xml) |
| ASan/UBSan | 33 passed; no findings | [host-sanitized.log](tensorrt-edgellm/evidence/continuous-batching-p5-20260915/host-sanitized.log) |
| ThreadSanitizer | 33 passed; no race reports | [host-thread.log](tensorrt-edgellm/evidence/continuous-batching-p5-20260915/host-thread.log) |
| Initial native matrix | Mixed sampling/limits, serial equality, stop/EOS/streams, slow-peer isolation and injected failure passed | [attempt2.log](tensorrt-edgellm/evidence/continuous-batching-p5-20260915/attempt2.log) |
| Final native matrix | All four P5 gates passed, including native cancellation, reuse and deadlines | [attempt3.log](tensorrt-edgellm/evidence/continuous-batching-p5-20260915/attempt3.log) |
| Preservation | Original engine/deployed patch unchanged; model/watchdog and unrelated containers healthy | [final-verification.txt](tensorrt-edgellm/evidence/continuous-batching-p5-20260915/final-verification.txt) |
| Source identity | 15 tested C++/header hashes match local committed source | [source-sha256.txt](tensorrt-edgellm/evidence/continuous-batching-p5-20260915/source-sha256.txt) |

Native A/B/C use prompt lengths 65/513/129 and output limits 6/12/12.
Temperatures are 0.7/1.2/0, top-k 20/40/1, and RNG draw counts 6/12/0.
The final run uses logprob counts 5/0/2. All concurrent token vectors and decoded
texts exactly equal their serial references; A's logprob history also matches
exactly. Ready rows with different settings execute together.

A forced repeated token tests a two-token stop match without leaking the stop
prefix; a forced primary EOS finishes at one token. A one-record slow stream
terminates only that request and its partner still exactly matches baseline.
Native queued/prefill cancellation, stale-ticket reuse and active deadlines pass,
with unaffected peer/later outputs. Host tests additionally exercise decode
cancellation, two-prefill fairness, queue/byte limits, 100 submissions from four
producers, shutdown/startup/worker failure settlement, Unicode boundaries and
channel wraparound/fullness. No numerical threshold was loosened.

### Operational record and retained attempts

- Attempt 1 exited at its idle-health precondition while the endpoint was busy.
  It stopped no service and performed no build/inference. Retained:
  [attempt1.log](tensorrt-edgellm/evidence/continuous-batching-p5-20260915/attempt1.log).
- Attempt 2 compiled successfully; build completion to healthy restoration was
  **09:45:22–09:45:58 IST**, at most 36 seconds including loading/restoration.
- Final attempt 3 was **09:47:37–09:48:16 IST**, at most 39 seconds on the same
  measure. Combined conservative model-validation bound: **75 seconds**, below
  five minutes. Each run also had a 180-second external inference deadline;
  compilation time is separate. Both jobs ran detached.
- Each maintenance run saved the deployed patch, paused the watchdog, stopped
  the model to free memory, built isolated candidate sources beside the original
  pinned archive, and restored the original model/prior active timer state.
- Final independent verification at **15:21:00 IST** found model/watchdog active,
  TensorRT `openclaw` healthy/idle, `max_num_seqs=1`, original deployed source base
  and two-file Python patch unchanged, unrelated API/database healthy. Available
  RAM 279 MiB / swap 982 MiB is only a snapshot. Original engine SHA-256 remains
  `8a968e59bca3dbad7e193a7431d4fb3acf7e3e0f6a7f34858e519d3d9e1bd4d3`.

There were no native compilation or inference failures. The post-forward fault
is an intentional simulated reported error after real GPU work, not an actual
illegal CUDA access. It fails the ticket, closes admission, reports unhealthy,
and prevents a new step lease on the same parent after scheduler destruction.
No in-place recovery from corrupted CUDA state is claimed.

### API and semantics P6 must preserve

1. Use `submit(preparedTokens, SchedulerRequestOptions)` and map HTTP settings
   explicitly. The integer-limit overload retains P4 greedy/ignore-EOS diagnostic
   behavior and is not the full serving API. Construct `SamplingSchedulerBackend`
   with the loaded tokenizer so EOS/thinking metadata and raw token pieces are
   available. Format/tokenize once with the request's thinking option before
   submission. Continue using the P3 chunk helper from the first prompt step.
2. `SequenceOptions` owns temperature, top-k/top-p, seed, output limit,
   numLogprobs (0–50), finite bias in [-100,100], EOS IDs and thinking state.
   Bias precedes sampling and logprobs; logprobs use the full biased **unscaled**
   distribution. Temperature <=0.001/top-k=1 is greedy; the legacy default tuple
   near temperature=1/top-k<=1/top-p=1 also remains greedy. Otherwise scale,
   apply top-k then nucleus filtering and draw independently. Equal logits break
   ties by token ID. Seed/counter use SplitMix64; they are deterministic across
   scheduling/reuse but do **not** promise legacy CUDA/Philox seed equivalence.
3. Primary EOS always stops unless explicitly ignored; secondary EOS stops when
   thinking is disabled/done. Qwen thinking-start/end and first-token handling
   follow the legacy path independently. Stop matching spans token/UTF-8
   boundaries, withholds possible stop prefixes and removes matched stop text.
   Accepted-token usage includes EOS and stop-completing tokens; committed cache
   length still excludes the pending sampled input. A partner never clamps a
   request's output limit. Existing nonzero frequency/presence penalties remain
   unsupported, as in the current endpoint.
4. `streamRecords=0` selects final-result-only. Otherwise record and byte caps
   bound the stream separately. `ticket.read(timeout)` returns an optional
   sample/text update plus closed state; a final text-only flush has token=-1.
   One consumer drains each ticket. The consumer may allocate; the worker's
   sampling, text and ring path uses preallocated storage. A full ring finishes
   only that request with `kSlowConsumer`, never waits on the reader, and retains
   a separate terminal result. Channel closure precedes terminal-future readiness;
   queued updates remain drainable after closure. Retained completed futures are
   caller-owned memory.
5. `queueDeadline` applies before admission; `deadline` spans queue and active
   work. Both use absolute steady-clock time and default to no deadline. All
   queue flags/deadlines are scanned at every safe boundary even with both slots
   occupied. Active cancellation/deadlines also act at boundaries, without kernel
   preemption. Natural completion already committed by a forward wins a late
   cancellation/deadline. Cancellation may retain an accepted token in terminal
   accounting before that token was streamed. Stale tickets affect only their
   original submission. Shutdown settles outstanding work and joins one worker.
6. Native validation bounds prompt/sequence at 6144/8192, bias at 1024 entries,
   EOS lists at 256 IDs, stop lists at 64 entries / 4096 bytes each / 16 KiB total
   metadata, stream records at 8192 and stream bytes at 1 MiB. These also must fit
   the scheduler queue byte cap (default 256 KiB, count eight). Admission accounts
   for prompt, dynamic policy metadata and ring capacity. Invalid settings fail
   synchronously without changing active requests; no silent truncation.
7. Worker exceptions conservatively fail outstanding requests and poison the
   parent lease, including errors after forward completion. Recreate the parent
   runtime to recover. Runtime/stream must outlive scheduler destruction; close
   joins the worker but the lease remains until destruction. No HTTP thread or
   per-request consumer may call close on the shared worker.

### Remaining qualification

The CPU sampler performs full-vocabulary sorting and logit D2H with preallocated
scratch. Efficient GPU sampling, performance, peak/long-term memory and graphs
are P7 work. No throughput result is claimed. A full clean CMake build was not
run; new sources compiled/linked on Jetson against the preserved pinned archive.
P3's unsupported raw short-tail drift remains unchanged. HTTP/SSE, Python/GIL,
tool/reasoning parsers and endpoint concurrency are **P6**, and production
rollout remains **P8**. Current production still admits one active sequence.
