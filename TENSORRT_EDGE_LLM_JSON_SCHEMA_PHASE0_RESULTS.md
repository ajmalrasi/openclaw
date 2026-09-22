# TensorRT Edge-LLM JSON Schema — Phase 0 results

Date: 2026-09-22 (IST)

## Verdict

**Phase 0 passes.** XGrammar 0.2.7 builds from source on the Jetson Orin
Nano Super (`aarch64`, GCC 13.3, JetPack 7.2.1/L4T r39.2.1) and initializes
against the exact deployed Qwen3.5 tokenizer. It compiles representative JSON
Schemas, produces the expected token mask, accepts valid Unicode JSON, and
rejects an invalid token sequence.

This is a dependency/tokenizer feasibility result. It does not implement
guided decoding in TensorRT Edge-LLM, exercise the GPU logit-mask path, generate
model output, or change the deployed service.

## Selected dependency

- XGrammar release: `v0.2.7`
- Source commit: `82505d0d987c36a4209fb3d8571cf6b0f28b5acd`
- License: Apache-2.0
- Source: <https://github.com/mlc-ai/xgrammar>
- Release: <https://github.com/mlc-ai/xgrammar/releases/tag/v0.2.7>
- Security baseline: 0.2.7 is newer than the 0.1.32 fix for
  CVE-2026-25048/GHSA-7rgv-gqhr-fxg3.

PyPI supplies Python 3.12 aarch64 wheels for XGrammar 0.2.7 and
`apache-tvm-ffi` 0.1.14.post0. The Python distribution declares PyTorch as a
dependency, while the deployed Edge-LLM environment intentionally has no
PyTorch. The recommended integration is therefore XGrammar's C++17 core,
linked into the native runtime, not the Python package.

XGrammar's C++ core contains no CUDA compilation unit. CUDA 13.2 is therefore
not a dependency-build blocker. Phase 1 must connect its CPU-produced DLPack
bitmask to Edge-LLM's existing GPU logit mask/bias path without copying full
logits to the CPU.

## Jetson evidence

Evidence directory on the Jetson:
`/home/ajmalrasi/xgrammar-phase0-20260922/`

Retained local evidence:
`artifacts/xgrammar-phase0-20260922/`

- Clean source build produced a 24 MiB static library.
- All 73 upstream C++ tests passed.
- Qwen tokenizer: 248,044 base tokens, 248,077 tokens including additions,
  248,320 model-logit vocabulary, EOS 248046, byte-level vocabulary.
- Valid fixture `{"name":"Ada","age":37,"tags":["math","é"]}` was
  accepted and completed.
- Invalid fixture was rejected at token index 3.
- A 248,320-token mask occupies 7,760 32-bit words, or 31,040 bytes per slot.
- Representative nested/Unicode, union/constant, string-pattern, and bounded
  numeric schemas compiled in 0–2 ms in the warm standalone process.
- The five-schema compiler cache occupied 982,000 bytes.
- Standalone process high-water RSS was 106,624 KiB (about 104.1 MiB), including
  vocabulary JSON parsing, tokenizer preprocessing, compiler, matchers, cache,
  and test fixtures.
- Final gate: `PHASE0_GATE passed=1`.

The measured peak is feasible but material on an 8 GB unified-memory device.
Production integration should construct tokenizer data without the JSON fixture
copy, cap the grammar cache, bound schema size/complexity, and re-measure peak
and steady-state memory with the model resident before deployment.

## Failures and corrections retained

1. The first build configuration inherited XGrammar's checked-in default and
   re-enabled Python bindings, then stopped because `tvm_ffi` was intentionally
   absent. A build-directory `config.cmake` explicitly disabled Python bindings
   and enabled C++ tests.
2. The corrected source build and all tests passed. Its wrapper then stopped
   because GNU `/usr/bin/time` is not installed. No package was installed;
   process memory was measured in the harness instead.
3. The first tokenizer harness used `FromVocabAndMetadata` with detector-only
   metadata. That API expects fully serialized tokenizer metadata. The harness
   was corrected to use the public direct constructor with detected byte-level
   settings, the exact 248,320 logit vocabulary, and EOS 248046.

No failed attempt changed Edge-LLM source, engines, packages, service settings,
or model artifacts.

## Initial supported-schema contract for Phase 1

The API must validate before scheduler admission and reject unsupported or
ambiguous semantics with an actionable HTTP 400. It must not rely on XGrammar
warnings because XGrammar deliberately weakens some unsupported constraints.

Recommended initial supported subset:

- `type`: object, array, string, integer, number, boolean, null, and type arrays
- `properties`, `required`, and `additionalProperties: false`
- `items`, `prefixItems`, `minItems`, and `maxItems`
- `enum`, `const`, and `anyOf`
- integer/number `minimum`, `maximum`, `exclusiveMinimum`, and
  `exclusiveMaximum`
- string `pattern`, `minLength`, and `maxLength`
- `minProperties` and `maxProperties`
- local `#/$defs/...` references only after explicit depth, size, and cycle
  limits are enforced

Reject or defer initially:

- external or remote `$ref`
- `not`, `if`/`then`/`else`, `dependentRequired`, and `dependentSchemas`
- `uniqueItems`, `contains`, `minContains`, and `maxContains`
- `multipleOf` for numbers and any integer `multipleOf` case the backend may
  ignore
- overlapping or otherwise unprovable `oneOf`
- unknown `format` values and any schema feature that produces a backend
  warning or relaxed fallback

The external-reference smoke case demonstrated the reason for this boundary:
XGrammar warned about `https://example.com/schema.json` but compiled it as an
unconstrained grammar instead of rejecting it.

## State and next gate

The TensorRT model service was already inactive when Phase 0 began and port
11434 was unbound. The correct watchdog timer remains active, but it does
nothing while the model unit is inactive. Phase 0 did not stop or restart
either unit.

Phase 1 may now implement a single-request native path in an isolated build.
Its gate remains actual constrained model generation whose decoded output
validates against representative schemas. No production service cutover is
authorized by this result.
