# Phase 4 results — 2026-09-15

**P4 complete: automatic native continuous scheduling and slot reuse.**
**P5 is safe to start. Production HTTP still admits one active sequence.**

Source commit: **`e37d897`**, DCO signed, local branch
`codex/continuous-batching-p1`, not pushed. Parent: `f416525` (P3).

## Implementation

- `cpp/runtime/continuousScheduler.{h,cpp}`: one worker, bounded FIFO queue
  (default eight requests / 256 KiB token payload), up to two resident sequences,
  native tickets/futures, decode-first rounds, one round-robin prefill chunk per
  round, immediate safe-boundary slot reuse, condition-variable idle sleep,
  cancellation and fail-closed shutdown/error plumbing.
- `cpp/runtime/greedySchedulerBackend.{h,cpp}`: exclusive step-runtime adapter,
  qualified P3 chunks from the first step, stable physical views, greedy
  length-limited output using a preallocated pinned logit buffer.
- `cpp/runtime/sequenceStepRuntime.{h,cpp}`: external poison hook so errors after
  forward completion also retain the parent lease and require runtime recreation.
- `cpp/CMakeLists.txt`: explicit Threads dependency; runtime sources are included
  by the existing runtime source glob.
- `unittests/cpp/runtime/continuousSchedulerTest.cpp`: nine scheduler tests.
- `examples/llm/continuousBatchingProbe.cpp`: `--scheduler` native proof.
- `examples/llm/continuousScheduler.md`: lifetime, scheduling and restricted API.

No HTTP interface, Python integration, engine, deployed runtime or model was
replaced. No state-sized serving copies, per-token heap allocation or graphs
were added. Diagnostic tracing and terminal-result copies are outside the
per-token execution path. Result vectors and admission state are bounded by
request limits; consumers retain their own completed futures.

## Evidence

| Check | Result | Artifact |
| --- | --- | --- |
| Final host tests | 20 passed: 9 scheduler + 11 slot/chunk | [host-tests.xml](host-tests.xml) |
| ASan/UBSan | 20 passed, no findings | [host-sanitized.log](host-sanitized.log) |
| ThreadSanitizer | 20 passed, no race reports | [host-thread.log](host-thread.log) |
| First Jetson run | All five native gates passed | [attempt1.log](attempt1.log) |
| Final-source Jetson run | All five native gates passed | [attempt2.log](attempt2.log) |
| Native step/admission/release trace | Request, slot, generation, partner, cursor, monotonic timestamp | [events.json](events.json) |
| Preservation and source identities | Tested C++/headers match local files; engine and deployed patch unchanged | [final-verification.txt](final-verification.txt), [source-sha256.txt](source-sha256.txt) |

Host coverage includes FIFO slot reuse; two-prefill fairness; queue count/byte
caps; invalid requests; queued/active and stale-ticket cancellation; worker
failure settling active/queued requests; shutdown; idle sleep/wake; cancelled
queue-head skipping; four concurrent producers and 100 successful submissions.
Source formatting, whitespace and operator-shell syntax checks also passed.

## Actual staggered execution

The native test first generates serial references using the same scheduler,
then submits A with 65 prompt / 6 output tokens. After A's first decode the
observer submits B (1025/12) and C (129/12) through native tickets. The observer
does not select slots or schedule forwards; the worker does all admission,
chunking, batching, completion and reuse automatically.

Final-run monotonic times in microseconds:

| Event | Time | Physical ownership |
| --- | ---: | --- |
| A admitted | 210583528536 | slot 0, generation 4 |
| B admitted after A decode | 210583714050 | slot 1, generation 1 |
| A released | 210584689869 | slot 0 becomes free |
| C admitted | 210584690175 | slot 0, generation 5 |
| C released | 210586360199 | B still owns slot 1 |
| B released | 210586531074 | slot 1 becomes free |

C was admitted 306 microseconds after A's release event in this diagnostic
trace; this is not a latency benchmark. B continued prefill after C admission.
Seven two-row B/C decode forwards were observed (14 row events). All three
concurrent output vectors exactly equal their serial greedy references.
`P4_SCHEDULER_GATE passed=1 exact_outputs=1 reused=1 b_continued=1 paired_decode=1`.
No logit threshold was relaxed. This is scheduler/greedy-output mechanism
qualification; P3's broader logit and state comparisons remain separate evidence.

## Retained failure and corrections

[host-attempt1.xml](host-attempt1.xml) records six passing tests and one failed
paired-decode assertion. C originally had only four output tokens and finished
before B's long prefill completed. Extending C to twelve created the intended
paired-decode coverage; no scheduler or numerical threshold was weakened.

Review after the first native pass found a cancelled queue head could postpone
another live admission by one boundary. A bounded skip loop and targeted test
fix this; final native attempt 2 tests that final implementation. Additional
checks guard ticket-ID overflow and harmless default-ticket cancellation.
There were no native compile/inference failures in either run.

## Operations and limits

Both jobs ran detached with an external 180-second loading/inference deadline.
Before each run the wrapper required idle healthy TensorRT `openclaw`, saved
the deployed two-file Python patch, paused the watchdog and stopped the model.
It compiled candidate sources beside the original pinned static archive and
reused its device-link object. Full clean CMake configuration/build was not run.
The previously exported/built engine was used unchanged; no export/build of
model artifacts was necessary for this runtime-only phase.

Attempt 1 build completion → healthy restoration: **09:24:32–09:25:04 IST**.
Attempt 2: **09:26:44–09:27:16 IST**. Combined conservative model-validation
bound is **64 seconds**, including loading and both restorations, below five
minutes. Compilation time is separate. No throughput benchmark ran.

Before: original model/watchdog active, endpoint healthy/idle.
After: both active, endpoint healthy/idle at 09:27:42, `max_num_seqs=1`, original
engine SHA-256 `8a968e59bca3dbad7e193a7431d4fb3acf7e3e0f6a7f34858e519d3d9e1bd4d3`,
source base `e8b29522938901f6df19ebeedd4b69bc8edbcd97`, deployed patch unchanged,
API/database healthy. Final RAM available 176 MiB / swap 724 MiB is only a
snapshot, not long-term stability qualification. Rollback artifacts preserved.

## Remaining scope

P4 exposes only prepared token IDs and output length, with greedy sampling and
length termination. EOS, independent sampling settings/RNG, stop strings,
thinking, deadlines, logprobs and streaming are **not implemented here**.
Queued cancellation settles at admission or shutdown; immediate removal while
both slots are occupied is P5 work. Any worker error conservatively poisons the
whole runtime. Full injected CUDA-fault coverage remains P5. No HTTP/SSE,
performance, graph, long-term memory or production concurrency claim is made.
P3's unsupported raw short-tail drift remains unchanged.

**Next: P5 independent request policies and complete cancellation/fault behavior.**
