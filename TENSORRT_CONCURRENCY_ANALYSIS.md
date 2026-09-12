# TensorRT Edge-LLM serving concurrency investigation

Read-only investigation: 2026-09-12, approximately 19:55–20:02 IST.

Follow-up: the user subsequently selected proper continuous batching plus chunked prefill. The implementation design in [TENSORRT_CONTINUOUS_BATCHING_PLAN.md](./TENSORRT_CONTINUOUS_BATCHING_PLAN.md) supersedes this report's recommendation to start with request coalescing. The diagnostic findings below remain preserved.

## Conclusion

The HTTP server deliberately serializes generation. The loaded engine supports two sequences in one native batch, but each HTTP request is constructed as a batch of one. Increasing queue capacity, removing the inference lock, or starting more HTTP workers does not implement batching.

Two fixes have different scope: Python request coalescing can batch requests waiting together; continuous admission of requests arriving during generation requires native scheduler work. The former is a practical first implementation, but does not fulfill the latter behavior.

## Evidence and current deployment

- Cloned the user fork into `/Users/ajmalrasi/openclaw-tensorrt-concurrency-review`; HEAD is `e8b29522938901f6df19ebeedd4b69bc8edbcd97` (0.10.1). The Jetson checkout has the same HEAD.
- Read-only SSH inspection confirmed the Jetson has only the previously recorded existing-engine-loading edits: eight added lines in `experimental/server/config.py`, thirteen in `experimental/server/runtime/engine.py`. These edits introduce `--engine-dir` and bypass checkpoint building; they do not change scheduling.
- `openclaw-tensorrt-edgellm.service` is active/running, PID `358655`. `/v1/models` identifies `openclaw`, owner `tensorrt-edgellm`, total sequence limit 8192.
- `/health` returns healthy, `max_batch_size=2`, `max_num_seqs=1`, `max_input_len=6144`, `max_model_len=8192`, text-only, no speculative decoding or context reuse. It was idle at the snapshot.
- The existing `llm-b2-input6144-kv8192-vanilla/config.json` confirms batch 2, input 6144, KV capacity 8192, 128 KV pages, FP16 KV and 24 recurrent / 8 attention layers.
- Service `MemoryCurrent` was 6,643,195,904 bytes (about 6.19 GiB; cgroup memory, not a GPU-only measurement). A later system snapshot showed 332 MiB available RAM and 1085 MiB swap used. This is limited headroom; two-slot persistent generation has not been verified in this investigation.
- Prior AIPerf results are taken from the continuing journal, not rerun or independently re-read from raw exports: increasing offered concurrency from one to two left aggregate throughput near 20 output tokens/s and raised mean TTFT from 3.226 to 23.552 seconds. That is consistent with the verified serial path.

## Where concurrency is blocked

Source references below are relative to the clean local clone and the pinned fork revision above. Deployed `engine.py` line numbers differ because of its thirteen added lines.

| Layer | Exact source | Effect |
| --- | --- | --- |
| HTTP admission | `experimental/server/runtime/engine_client.py:51`, `:55`, `:387` | `_AdmissionController` uses `asyncio.Semaphore(1)`; preparation must acquire it before generation. The lease is held until completion or stream cleanup. |
| Published capability | `experimental/server/runtime/engine_client.py:193` | `max_num_seqs=1` is hardcoded; it is not an adjustable concurrency knob. |
| Request construction | `experimental/server/runtime/engine.py:1072` | `request.requests = [req]` builds exactly one native slot per HTTP request. |
| Python runtime entry | `experimental/server/runtime/engine.py:1226` | `_handle_request()` holds `_infer_lock` through the entire native call. Direct high-level callers also have a one-slot admission semaphore. |
| Native runtime entry | `cpp/runtime/llmRankRuntime.cpp:856` | An atomic guard rejects overlapping `handleRequest()` calls on the same runtime. |
| Native lifecycle | `cpp/runtime/llmRankRuntime.cpp:908`, `:1495`, `:3069` | Batch membership is initialized once; the decode loop can evict finished slots, but contains no admission step for new arrivals. |

The Python binding releases the GIL around `handle_request` (`experimental/pybind/edgellm_pybind.cpp:935`), so the Python GIL is not the explanation. Even `LLM.generate([prompt_a, prompt_b])` loops over prompts and executes them sequentially (`experimental/server/runtime/engine.py:1292`). Existing tests explicitly assert serialization (`tests/python-unittests/test_server_runtime.py:63`); these were inspected, not executed.

The [server concurrency documentation](https://github.com/ajmalrasi/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/docs/source/user_guide/examples/experimental-server.md#runtime-concurrency) describes the same single-generation-state restriction.

## Fix 1: coalesce waiting HTTP requests into native batches

Implement a scheduler at the `EngineClient` / `LLM` boundary, sharing the existing resident runtime and engine. The native request and per-slot streaming bindings already exist, so this design should not require an engine rebuild or C++ binding change. Actual memory and correctness still require on-device validation.

1. Replace per-request exclusive admission with a bounded pending queue owned by one dispatch worker. Keep native execution exclusive. Adapt shutdown and lease accounting together; changing the semaphore value alone is insufficient.
2. Select up to two compatible text requests already waiting. When idle, optionally allow a configurable 5–10 ms coalescing window; this is an initial tuning proposal, not a measured optimum. Never wait indefinitely to fill a batch. Bound unfairness when looking past incompatible requests.
3. Build one `LLMGenerationRequest` containing both native `Request` slots. Preserve template behavior and every batch-wide setting. Leave formatting to the existing coordinator, or consistently supply all formatted slots; do not mix partial prepared state.
4. Attach a separate `StreamChannel` to each streaming slot and call native `handle_request()` once under the existing inference lock. Consume channels independently and route content, usage, finish reasons, logprobs, reasoning and tool parsing back to the corresponding HTTP request. Non-streaming slots can use response arrays; those final arrays are available after the whole native batch returns.
5. Make the scheduler own the batch worker's lifetime. A single client's disconnect cancels only its channel, and a completed client's stream must not wait unnecessarily for its batch partner. The current one-request iterator's unconditional `worker.join()` cannot simply be copied into each batch consumer. Release the runtime only after its native batch has exited; drain/cancel all work before shutdown.
6. Report two-slot capability only when this path is enabled, and expose real active-slot / queued-request counts rather than changing health metadata alone.

Compatibility is more than matching temperature. `LLMGenerationRequest` shares temperature, top-p, top-k, maximum output length, LoRA selection, template flags, thinking/speculation, logprob count and cache policies across the batch (`cpp/runtime/llmRuntimeUtils.h:95`). Stop strings and logit bias are already per slot. Initially serialize unsupported media/audio/cache flows and incompatible requests without silently changing their parameters.

One extra correctness trap: `clampMaxGenerateLengthForKVCapacity()` takes the minimum output headroom across every slot (`cpp/runtime/llmRuntimeUtils.cpp:723`). A long prompt can shorten its short-prompt partner's output even when both request the same maximum. Only co-batch when both preserve their individual effective output budgets, or add native per-slot output limits. Token checks must account for native reserved KV headroom.

This fix batches both prefill and decode through the current native path. It does not implement the earlier desired policy of prefilling arrivals individually and then joining their decodes. Unequal prompt lengths and large prefills need measurement.

## Fix 2: continuous batching for staggered arrivals

Example: request A is generating; B arrives two seconds later. Fix 1 leaves B waiting until A's native call finishes. Likewise, a finished slot cannot be refilled while its partner continues. A short coalescing window cannot solve this.

For B to join A, refactor the C++ request lifecycle into persistent per-sequence state with submission, prefill, decode-step and completion operations. At iteration boundaries, admit pending requests into free slots, prefill them while preserving active sequences, then batch decode up to two. Maintain stable request-to-slot identity through eviction and compaction, independent cancellation, per-sequence output budgets and sampling semantics.

Qwen3.5 makes state preservation substantial: active attention KV pages plus recurrent and convolution state must survive another request's prefill. Current `setUpForPrefillExecution()` resets page mappings, recurrent state for new rows, and cache lengths (`cpp/runtime/llmRankRuntime.cpp:2800`). Re-entering it naively can overwrite A. Managed cache and eviction helpers may be reusable, but cache reuse is not an existing continuous scheduler. The public Python binding currently exposes whole-request execution, so step/admission support also needs an API boundary.

The existing engine may remain usable by scheduling prefill and decode as separate executions; that requires validation of state mappings, profiles and buffer ownership. Interleaving a large prefill can pause the current stream. Lowering that pause with chunked prefill is additional work, not a property supplied by a Python queue.

## Recommendation and acceptance checks

Start with Fix 1 to expose the batch-two capability for compatible requests arriving together or queued under load. Treat Fix 2 as the required next scope if the requirement is for arbitrary arrivals to join generation already in progress. Do not promise continuous behavior from coalescing.

Validate scheduler grouping and per-client result routing, incompatible fields, unequal prompt/output lengths, one-slot EOS/cancellation, timeout, overload and shutdown. Then run a single bounded Jetson session of at most five minutes: one-request reference, simultaneous pair, staggered pair and cancellation. Capture token timestamps and memory; simultaneous streams must overlap, while the staggered case explicitly distinguishes coalescing from continuous batching. Judge speedup from the current persistent endpoint, not an older synthetic engine table.

No implementation, service restart, model inference, benchmark, build, remote write, commit or push was performed in this investigation. Only the local source clone, this analysis and the continuing journal were created/updated.
