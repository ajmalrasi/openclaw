# TensorRT Edge-LLM JSON Schema — Phases 1–3 results

Date: 2026-09-23 (IST)

## Verdict

**Functional Phases 1–3 pass in an isolated candidate.** The Qwen3.5-4B
TensorRT Edge-LLM candidate constrains vanilla generation to a caller-supplied
JSON Schema, accepts the OpenAI-compatible `response_format` envelope, and
keeps independent grammar state for two concurrently scheduled requests.
Streaming, invalid-schema rejection, incomplete-output failure, cancellation,
and physical-slot reuse passed the endpoint gate.

This is **not** a production deployment or Phase 4 qualification. The deployed
Jetson checkout and service configuration were not changed. The model service
was already inactive and remains inactive; the watchdog timer remains active.

## Implementation

- Source branch: `feature/json-schema-guided-decoding`, signed-off local commit
  `2cb8bb8`, in
  `/Users/ajmalrasi/openclaw-tensorrt-concurrency-review`, based on qualified
  continuous-batching commit `1496d34`.
- XGrammar 0.2.7 C++ core is an opt-in CMake dependency (`ENABLE_XGRAMMAR=ON`)
  and uses the exact Qwen tokenizer pieces and EOS metadata.
- Request admission validates a bounded JSON-Schema subset and compiles the
  grammar before taking a scheduler slot. Unsupported/invalid features return
  HTTP 400. Tools, thinking and stop strings cannot be combined with this
  mode; text-only vanilla decoding is required.
- A matcher belongs to each physical slot, advances only on its emitted token,
  and is discarded on completion/cancellation/release. Guide failures are
  request-local; incomplete output at the token cap is an error rather than
  successful partial JSON.
- The mask is applied to the **existing CPU logits row used by this scheduler's
  sampler**. No additional full-logits device-to-host copy was introduced.
  GPU-side masking remains unimplemented and is part of Phase 4 optimization.

## Verification

The isolated Jetson source/build/evidence directory is
`/home/ajmalrasi/json-schema-p1-p3-20260922/`; retained local logs and gate
scripts are in `artifacts/json-schema-p1-p3-20260922/`.

- Native candidate binding built and imported; 23/23 native scheduler/policy
  tests passed (`scheduler-tests.xml`).
- Host Python suite passed 116/116, including async tests. Jetson Python suite
  passed 112 tests, with four async tests skipped because that venv lacks
  `pytest-asyncio` (`python-final.log`, `python-tests-final.xml`).
- Opt-in CMake target `edgellmXGrammar` configured and built on Jetson CUDA
  13.2/SM87 (`CMAKE_XGRAMMAR_GATE=passed` in `cmake-gate.log`). This was the
  dependency target, not a full release rebuild.
- Candidate endpoint ran only on loopback port 11435 using the existing
  `llm-b2-input4096-kv4608-vanilla` engine. It passed schema-valid object and
  SSE output, nested Unicode constant `東京😊`, HTTP 400 for unsupported and
  unresolved/external references and tool conflict, HTTP 500 for an incomplete
  one-token result, and two concurrent different object/array schemas.
- A streaming client disconnected from a deliberately long constrained
  request; `/health.active_requests` returned to zero before the subsequent
  constrained request completed. Final candidate health was healthy and idle,
  with `max_batch_size=2`, `max_num_seqs=2`, and no queued requests.
- Final endpoint marker: `PHASE1_3_ENDPOINT_GATE passed=1`, 11.087 seconds
  (`endpoint-gate.log`, `candidate-server.log`). The candidate was stopped and
  port 11435 was unbound afterward. Port 11434 remained unbound, production
  model unit inactive, watchdog timer active.

## Failed attempts and corrections

The build and endpoint logs retain failed attempts. Native build wrappers
initially referenced a nonexistent source, lost an executable bit during sync,
and missed Python tests in a partial source copy; each was corrected. The
production-sized 6144-input/8192-KV engine could not load in the isolated
candidate under available unified memory. The smaller existing batch-two
engine loaded successfully. An endpoint harness URL incompatibility and a
stale copied header causing a candidate ABI mismatch were corrected. The
first cancellation test's follow-up request did not prove slot release;
the final gate explicitly required idle health after stream disconnect.
The final Jetson Python run required disabling strict marker validation to
skip async tests in the venv lacking the plugin; those tests passed on host.
Four CMake attempts exposed architecture, compiler-path, incomplete source
copy, and wrong default CUDA-version settings; the final CUDA 13.2 config
passed. No failed attempt was silently treated as a product pass.

## Phase 4 remains

Before production cutover, perform a full release build and measure guided
decoding latency, throughput and memory against the existing service under
the five-minute benchmark cap. Resolve fit of the production-sized engine and
evaluate GPU-side masking, then validate reliability, rollback and deployment
acceptance. No performance improvement or production memory fit is claimed by
these functional gates.
