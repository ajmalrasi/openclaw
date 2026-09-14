# Phase 2 results — 2026-09-14

**Persistent ownership and native step mechanism implemented and verified.**
**Full-versus-chunked numerical qualification is still failing on one new fixture.**
The production endpoint is unchanged and still admits one active sequence.

## Implemented

Source checkout: `/Users/ajmalrasi/openclaw-tensorrt-concurrency-review`, branch
`codex/continuous-batching-p1` (P1 and P2 changes remain uncommitted).

- `cpp/runtime/state/sequenceSlots.{h,cpp}`: stable request identity, per-pool and
  per-allocation handles, prompt/output ownership, per-request controls, bounded
  admission, independent headroom, sampled/committed counts and reuse protection.
- `cpp/runtime/sequenceStepRuntime.{h,cpp}`: exclusive parent-runtime lease,
  preallocated physical-slot views, forward-only prefill/decode, explicit GPU
  completion before endpoint publication, finish/release and failure poisoning.
- Native forward execution never invokes sampling. A one-token prompt tail
  remains prompt work even when its physical execution uses the decode profile.
- Existing engine, attention/recurrent/conv allocations and native archive are
  reused. Production source, bindings, libraries and service configuration were
  not overwritten. Existing Python engine-directory edits remain intact.

Scope is deliberately single-rank, text-only, two-slot vanilla Qwen3.5. The
final single-rank exclusion guard was added after the successful inference run
and checked separately with the Jetson compiler; it does not change execution
on the tested single-rank deployment.

## Passing verification

- Nine CPU tests passed normally: [host-tests.xml](host-tests.xml).
- The same nine passed with ASan+UBSan: [host-sanitized.xml](host-sanitized.xml).
- Seven native logit comparisons matched independent execution **exactly**.
  For chunked tails, the independent reference used the identical P1 chunk
  recipe; separate full-prompt numerical differences remain recorded below.
- Four byte-exact isolation checks passed, including slot C replacing A while
  B retained 130 committed tokens and its independent settings.
- Two-row decoding with unequal endpoints (65 versus 129 before the step;
  66 versus 130 afterward) matched independent singleton decoding exactly.
- Six invalid operations were rejected: second session, in-flight release,
  overlapping forward, stale release, duplicate slot and unordered slots.
- The legacy call was rejected during the lease and worked again after
  session destruction, returning the exact two expected greedy tokens.
- Sampler output scratch remained byte-for-byte unchanged across all native
  steps. No generated token appeared during prefill; sampled tokens did not
  enter KV before the next decode forward.

Successful evidence: [attempt3.log](attempt3.log), ending with
`LEGACY_RESTORED passed=1`, `P2_STATE_GATE passed=1` and `PROBE_RESULT passed=1`.
This result is explicitly scoped to P2 mechanism preservation, not P3 quality.

## Failures retained and not waived

1. [Attempt 1](attempt1.log): post-tail decode on a new synthetic fixture differed
   from full prefill by max absolute 0.113281 and relative L2 0.00932514. Greedy
   output matched, but the unchanged 0.1/0.005 screen failed. The legacy test
   also omitted a mandatory message field; fixed the test input.
2. [Attempt 2](attempt2.log): the same fixture through the original P1 path
   reproduced exactly the same numerical difference. An experimental padded
   two-position prefill tail improved errors to 0.0820312/0.00737054, but still
   failed. It was not adopted; its source is preserved in
   [attempt2-sequenceStepRuntime.cpp](attempt2-sequenceStepRuntime.cpp).
   The legacy test also needed the coordinator-prepared `formattedRequests`
   entry; that test fixture was corrected without weakening runtime validation.
3. Attempt 3 retains `P1_TAIL_BASELINE quality_passed=0`. It verifies exact
   new-versus-existing execution independently. No numerical threshold was
   relaxed and the failed quality gate was not converted into a pass.

Next required work is P3 chunk scheduling and numerical qualification: broader
boundary/active-state comparisons, isolation of kernel/tactic rounding effects
and a justified solution or acceptance policy. Matching greedy IDs on a short
fixture does not establish broad answer quality.

## Operations and limits

Each build/test ran detached under a dedicated user service, with model/watchdog
maintenance and automatic restoration. Model-executing portions of all three
attempts together were under one minute including initialization, below the
five-minute cap. Attempts 1/2 used 290-second process deadlines; attempt 3 was
reduced to 260 seconds for the remaining cumulative budget. A systemd property
update intended to further bound attempt 2 was rejected; it did not alter the
existing process deadline. This was not a performance benchmark.

The API/database containers were preserved. See [restored health](restored-health.json)
and the end of each log for service/watchdog recovery. No throughput, long-term
memory stability, graph execution, injected CUDA-fault recovery, complete sampling
policy or HTTP/SSE concurrency is claimed. A full clean CMake build was not run;
new native sources were compiled and linked against the pinned existing archive.
