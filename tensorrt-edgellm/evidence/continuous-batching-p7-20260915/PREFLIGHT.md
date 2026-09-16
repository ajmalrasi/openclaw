# P7 prerequisite findings — 2026-09-15

Source: `a90574f894d07d6e1fbc7eda6dffa43a0dcbb37d`, clean checkout on `codex/continuous-batching-p1`.

P6 is not qualified. Its prior task and authoritative journal explicitly leave C++/pybind compilation and Jetson HTTP validation pending. Existing 118 passing Python regressions do not establish continuous HTTP behavior.

## Concrete source findings

- `experimental/server/api/serving_chat.py:create_chat_completion` and `api/routes.py:chat_completions` both prepare an engine request before calling generation/streaming. `runtime/engine_client.py:prepare_request` unconditionally takes the legacy semaphore. Both new continuous branches require the prepared request to be absent, so actual prepared HTTP requests bypass those branches and reach the legacy runtime while the native scheduler owns its exclusive lease. This affects streaming and non-streaming.
- Health/readiness routes return literal healthy/ready without consulting `continuous_healthy`.
- HTTP defaults allocate a stream record for every allowed output token. A record embeds 50 logprob entries even when none were requested; 2048 records alone exceed the native 256 KiB queue budget. Normal default streaming requests cannot fit that admission budget.
- Native text-only channel flushes use token -1, but the bridge includes that sentinel in emitted token IDs, which would inflate usage.
- The non-streaming ticket is hidden inside a blocking helper, preventing the HTTP task from cancelling it on disconnect. Native overload/deadline statuses do not have explicit HTTP mappings.

These are source findings, not observed GPU/HTTP test results. No benchmark, build, service stop or deployment occurred during this inspection.

## P7 work after the prerequisite gate

1. Capture only the three stable decode binding views `{0}`, `{1}`, `{0,1}` at session initialization. Capture warmup executes a forward and changes hybrid state, so capture cannot occur on live requests; reset scratch state before admission. Qualify eager/graph equality and inactive-state isolation after profile switching and slot reuse.
2. Remove allocation from graph lookup: existing `snapshotBindings()` constructs a vector on each matching decode execution. Verify graph identity including profile, addresses and shapes; bound captures to three per session and expose capture/replay counts.
3. Measure prefill/decode preparation, forward duration, profile switches and CPU sampling. Retain P3's 128-token policy unless alternate partitions pass the numerical/state matrix.
4. One external 300-second benchmark deadline must cover warmup and sequential preserved/candidate runs on the same engine, with only one resident model. Record singleton latency/throughput (≤10% regression), staggered waiting, median/p95 gaps, aggregate output rate, near-capacity cases, repeated slot reuse, allocations and memory/thermal snapshots.
5. Report correctness, latency, memory and throughput gates separately. Long-term stability remains unqualified by a short run. P8 cannot start until the required gates pass.

## Live preflight

Model and watchdog active; TensorRT openclaw healthy/idle; batch capacity two, max_num_seqs one; available RAM 304 MiB, swap 994 MiB; unrelated API/database healthy. No remote changes.
