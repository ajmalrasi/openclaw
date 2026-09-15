# Phase 3 results — 2026-09-15

**P3 complete for the fixed, qualified chunk policy on the existing Jetson engine.**
**P4 can begin using `beginPrefillChunk`; production still serves one active sequence.**

## Implementation and scope

TensorRT source commit: **`f416525`**, signed off, on local branch
`codex/continuous-batching-p1`. This commit has not been pushed.

- `cpp/runtime/state/prefillChunk.h`: fixed cap of 128 true prompt tokens.
- `cpp/runtime/sequenceStepRuntime.{h,cpp}`: `beginPrefillChunk(handle)`, with
  rejection of incompatible manually partitioned history before GPU execution.
- `unittests/cpp/runtime/state/sequenceSlotsTest.cpp`: exhaustive partition,
  token-conservation and first-output accounting tests.
- `examples/llm/continuousBatchingProbe.cpp`: active-state numerical comparison,
  broad prompt/remainder matrix, teacher continuation and manual interleaving.
- `examples/llm/chunkedPrefill.md` and `sequenceStepRuntime.md`: API contract,
  numerical scope and unsupported raw partitions.

The fixed policy keeps intermediate endpoints aligned to 64 tokens. With
129–191 tokens remaining it consumes 64, leaving a final 65–127-token chunk;
otherwise it consumes up to 128. Every resumed final chunk contains 64–128 true
prompt tokens. A complete cold prompt can be shorter, including one token.
Examples: 129 becomes **64+65**; 130 becomes **64+66**; 257 becomes **128+64+65**.
There is no padding, replay, state-sized serving copy, additional model instance,
or per-chunk sampling. The admission-owned prompt is formatted/tokenized once;
only final completion permits accepting the first output token.

Scope remains eager, single-rank, text-only vanilla Qwen3.5-4B INT4 AWQ, two
physical slots, FP16 KV, FP32 recurrent state, no context reuse/MTP. This phase
adds a prompt continuation primitive, not the P4 scheduler or P6 HTTP integration.

## Passing evidence

| Verification | Result | Evidence |
| --- | --- | --- |
| CPU state and policy tests | 11 passed; all lengths 1–6144 conserve tokens, respect cap/tail restrictions and reject sampling before final prefill | [host-tests.xml](host-tests.xml) |
| Same final tests with ASan/UBSan | 11 passed | [host-sanitized.xml](host-sanitized.xml) |
| Initial boundary/long matrix | 16 prompts, 80 logit comparisons, 2048 active-state tensor/plane comparisons; all exact | [attempt2.log](attempt2.log) |
| Final expanded matrix | 75 prompts, 407 logit comparisons including interleaving, 9600 active-state tensor/plane comparisons; all exact | [attempt4.log](attempt4.log) |
| Manual interleaving | Five B chunks alternated with A decode in each physical-slot order; 20 byte-exact inactive snapshots, 16 exact logit comparisons | [attempt4.log](attempt4.log) |
| Mixed manual-policy history | Rejected before execution; runtime remained healthy | [attempt4.log](attempt4.log) |
| Deployment preservation | Model/watchdog active, original engine and Python patch hashes unchanged, API/database healthy | [final-verification.txt](final-verification.txt) |

Across the two passing numerical runs: **487 exact logit comparisons and 11,648
exact active-state comparisons**, excluding the explicitly failing raw-tail
negative control. There are 91 prompt-case executions, including repeated
boundary controls across runs, not 91 unique prompt lengths.

Coverage includes 1,3,4,63,64,65,127,128,129; every length 129–193; multiple-chunk
tails; 255/256/257/258/259/260/319/320/321/513/1025/2049/6144; four final formatted
chat fixtures at 155/229/244/2468 tokens (explanation, Python, Unicode/JSON and
varied numbered records). Synthetic cases use four teacher-forced steps; chat
cases use eight. Active state is compared after prefill and after continuation:
all 24 recurrent and 24 convolution tensors plus K/V live prefixes in eight
attention layers. Unwritten KV capacity is excluded. Inactive checks also cover
the physical page-table row. Sampled-token scratch is unchanged during the
interleaved forward-only calls, and pending-output versus committed-cache
accounting is checked explicitly.

The exact matches greatly exceed the unchanged acceptance screens: finite
logits, equal greedy IDs, max absolute ≤0.1 and relative L2 ≤0.005. Active-state
screens were declared before testing at finite/0.1/0.005 per tensor/plane;
inactive state requires byte equality. No threshold was loosened.

## Failures retained and diagnosis

1. [Attempt 1](attempt1.log) failed compilation on a signed/unsigned comparison
   in the new diagnostic's host capacity check. An explicit cast fixed it. No
   inference ran and automatic restoration succeeded.
2. [Attempt 3](attempt3.log) disproved the first policy, which avoided only
   singleton tails. Remainders 2–4 caused four logit-screen failures and 304
   active-state-screen failures, excluding the raw singleton control. The largest
   failing logit example was length 132, teacher step 3: **0.488281 max absolute,
   0.0165032 relative L2**. Interleaving itself passed, although its aggregate
   printed false because that variable inherited earlier matrix failures.
3. The raw **64+64+1** control in attempts 3 and 4 still reproduces the original
   post-tail **0.113281 / 0.00932514** failure. It is explicitly outside the
   qualified policy. Its prefill logit error is smaller (0.0234375 / 0.00132449),
   illustrating why first-token comparisons alone were insufficient.
4. The first three attempt-3 chat controls were below the cap (47/61/64 tokens).
   Final attempt 4 extended them past the cap before the one-time formatting and
   tokenization, so they genuinely exercised continuation.

**Resolution:** select supported execution shapes by moving true tokens across
chunk boundaries. The raw small-shape kernels are not repaired or claimed safe.
SM87 GDN has distinct S=1 and S>1 kernels, but its prefill is a sequential
recurrence. Alignment here is an empirical policy, not a proven GDN block-size
requirement. Attribution among GEMM tactics, convolution, GDN and attention
rounding is still unisolated. A future smaller-cap policy must undergo its own
qualification; P4 must use the qualified helper rather than manual spans.

## Operations, identity and limits

All four attempts ran detached in dedicated user services. They saved the
original Python deployment patch, stopped the model/watchdog for maintenance,
compiled separate source/header overlays against the pinned existing archive,
and automatically restored the original service and prior active timer state.
The live checkout, native archive, plugin, Python bindings and engine were not
replaced. The C++ changes were compiled/linked on Jetson with warnings treated
as errors; a full clean CMake build was not run. No export or engine build was
needed because this is runtime-only execution over the previously validated
serialized engine.

Source base: `e8b29522938901f6df19ebeedd4b69bc8edbcd97`; P2 parent: `cf57e1c`.
Engine SHA-256: `8a968e59bca3dbad7e193a7431d4fb3acf7e3e0f6a7f34858e519d3d9e1bd4d3`.
Final probe and runtime/header hashes match the local committed files exactly;
see [final verification](final-verification.txt). Earlier source snapshots are
not separately archived; their logged hashes, retained failure logs and
candidate-policy descriptions document those attempts.

Conservative combined model-validation time is **at most 255 seconds**: 72 + 58
+ 125 seconds measured from build completion through restoration for attempts
2/3/4, including initialization and all three restorations. Actual model time
is smaller. Attempt 1 performed no model validation. Each attempt also had an
external process deadline. This is functional/numerical validation, not a
throughput benchmark; build time is separate.

Before: model/watchdog active, endpoint healthy/idle, API/database healthy.
After final restoration at **09:02:20 IST**: model/watchdog active, endpoint
healthy/idle with `max_num_seqs=1`. Independent final verification at **09:04:02
IST** found the healthy service handling one active request, no queued requests,
and the unrelated containers healthy. No request was interrupted by that read.
Only the two original deployed Python support files remain dirty, with an exact
match to the saved patch. Rollback artifacts are preserved.

Memory snapshots are diagnostic, not a stability result: final probe CUDA free
bytes 91,340,800 include host snapshots/staging effects on unified memory;
post-restoration available RAM was 93 MiB, increasing to 183 MiB at final check.
No performance, long-term leak, CUDA graph, broad answer-quality or HTTP/SSE
concurrency qualification is claimed. Those remain later-phase work.

**Next:** P4 may implement one native scheduler using `beginPrefillChunk`, stable
slots and the existing exclusive execution lease. Production deployment remains
P8 work.
