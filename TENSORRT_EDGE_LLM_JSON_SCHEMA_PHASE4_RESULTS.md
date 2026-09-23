# TensorRT Edge-LLM JSON Schema — Phase 4 qualification status

2026-09-23. Functional baseline commit: `2cb8bb8`; GPU-mask release:
`b921d36` (`feature/json-schema-guided-decoding`). The GPU-mask release is now
the persistent production runtime. The actual engine's memory headroom remains tight.

| Gate | Result | Evidence |
| --- | --- | --- |
| 4A, full CUDA/Jetson core build | Passed; `libedgellmCore.a` 79,352,636 bytes | `artifacts/json-schema-phase4a-20260923/` |
| 4B, full Python binding and import | Passed; `_edgellm_runtime` 100,609,584 bytes, `json_schema` exposed | `artifacts/json-schema-phase4b-20260923/` |
| 4C, full-binding endpoint on 4096/4608 engine | Passed object, SSE, Unicode, invalid/incomplete response, batch-two independence, cancel/reuse | `artifacts/json-schema-phase4c-20260923/` |
| 4D, short performance/memory calibration on 4096/4608 | Functional pass; 20.097 s workload | `artifacts/json-schema-phase4d-20260923/` |
| 4E, currently configured 6144/8192 production-engine fit | Passed isolated startup and full endpoint gate; previous partial-binding OOM preserved in journal | `artifacts/json-schema-phase4e-20260923/` |
| 4F, short production-engine performance/memory calibration | Functional pass; low headroom later found in baseline too | `artifacts/json-schema-phase4f-20260923/` |
| 4G, rollback preservation | Existing deployed binding, unit and both engine bundles intact; production unit not switched | `TENSORRT_EDGE_LLM_EXPERIMENT_LOG.md` |
| 4H, existing-binding memory baseline | Passed; the same engine is memory-constrained without JSON Schema | `artifacts/json-schema-phase4h-20260923/` |
| 4I, reversible live cutover | Passed public alias/port health, full JSON-Schema endpoint suite, and manual watchdog real-generation check | `artifacts/json-schema-phase4i-20260923/` |
| 4J, persistent production promotion | Passed stable release copy/hash, service restart, full live suite, watchdog, enabled units and linger | `artifacts/json-schema-production-20260923/` |
| Post-reboot verification | User rebooted Jetson; automatic service/timer startup, full live suite and watchdog passed | `TENSORRT_EDGE_LLM_EXPERIMENT_LOG.md` |
| GPU mask full binding rebuild | Passed on isolated Jetson source with two compile workers; final host-mask correction rebuilt and imported | `artifacts/json-schema-gpu-mask-20260923/`, `artifacts/json-schema-gpu-mask-final-build-20260923/` |
| GPU mask matched engine A/B | Passed full endpoint gate and three paired guided/plain performance samples under 300 s | `artifacts/json-schema-gpu-mask-ab-20260923/` |
| GPU mask bounded concurrent reliability | Passed 252 schema-valid requests in 126 simultaneous pairs over 165.463 s | `artifacts/json-schema-gpu-mask-reliability-20260923/` |
| GPU mask focused CUDA unit | Passed separate batch rows and partial mask word (1/1); model and timer restored | `artifacts/json-schema-gpu-mask-unit-20260923/` |
| GPU mask persistent production promotion | Passed versioned copy/hash, full live suite, non-greedy/SSE smoke, watchdog, health and boot-unit checks | `artifacts/json-schema-gpu-mask-release-20260923/` |

Both benchmark sessions were detached and capped at 300 seconds including
startup. The host has six CPU cores. The full binding build used three compile
workers at 6.6 GiB available; the core build used one because memory was
tighter at its launch. No inference engine was rebuilt.

## Short performance observation

The 4F actual-engine HTTP workload alternated three plain and three guided
streaming requests after warmup, then sent two guided requests together. All
guided outputs were schema-valid. Median plain TTFT was 0.161 s and guided
TTFT 0.163 s. Median measured decode rates were 22.376 versus 21.707
completion tokens/s, respectively. **This is not a controlled grammar-overhead
estimate:** plain output included Markdown fences and used 45 tokens, while
guided output used 39 tokens. The guided pair completed successfully in
approximately 2.47–2.52 seconds per request.

The later matched GPU-mask A/B used the same production-sized engine,
prompt, schema, graph setting, and binding environment. Both versions
produced identical 39-token guided JSON. Three paired guided samples had
median TTFT 0.1642 s on the CPU-mask baseline and 0.1618 s on the GPU-mask
candidate; median decode rates were 21.538 and 21.782 tokens/s, respectively.
Plain medians were 22.264 and 22.180 tokens/s. These small samples show no
material regression; the approximately 1.1% guided difference is exploratory,
not a general throughput claim. Both sessions were detached and limited to
five minutes including startup.

## Memory gate: baseline attribution and bounded reliability

The production-sized engine loaded and served requests, but after warmup the
candidate process used 6,873,112 KiB RSS and 150,404 KiB swap. System
`MemAvailable` fell to **78,568 KiB**. At the final sample it had 6,835,732 KiB
RSS, 185,188 KiB process swap and 103,280 KiB available. This is too little
headroom to claim stable serving on an 8 GB unified-memory Jetson, particularly
given the earlier OOM when this engine was tried under different conditions.
The short pass does not supersede a long-running reliability gate.

A subsequent isolated run of the **existing deployed binding without JSON
Schema** on the same engine also showed severe pressure: 61,960 KiB available
at readiness, and 113,232 KiB available after eight plain requests, with
172,648 KiB process swap. Candidate final RSS/swap exceeded the baseline final
by about 15/12 MiB, respectively. The runs are sequential and workloads not
identical, so this is not a precise incremental-cost measurement. It does
show that the large memory shortfall is **not demonstrated to be caused by the
JSON-Schema feature**. No speculative candidate-code memory change was made.

The smaller 4096/4608 engine left approximately 388,080 KiB available at the
end of its short workload, but switching engine capacity would change the
current serving contract and has not been done.

The GPU-mask candidate subsequently served 252 schema-valid requests in 126
concurrent pairs without request errors or OOM. Final health was healthy/idle;
sampled high-water RSS was 6,853,972 KiB, final RSS 6,738,776 KiB, process
swap 279,980 KiB, and system available memory 150,676 KiB. RSS did not trend
upward over this 165-second window. This is a bounded reliability pass, not a
claim of indefinite operation or adequate spare memory for unrelated jobs.

## Production deployment state

The existing `openclaw-tensorrt-edgellm.service` is active on port 11434 with
public alias `openclaw` and the unchanged 6144-input/8192-KV engine. Its
permanent JSON-Schema override now points at versioned source and full binding
in `/home/ajmalrasi/tensorrt-json-schema-release-b921d36/`; the binding SHA-256
is `00346d199043dd8ce967154ff2bb041c1aa9cf43d63bdb9bdc0fcb1a0c3221e5`.
The previous release and its drop-in copy remain for recovery. The service and
watchdog timer are enabled and active with user-systemd linger on. A
user-performed Jetson reboot directly verified automatic startup of the
previous production release; the new release has passed a real service
restart, full live schema suite (10.789 s), eight non-greedy object requests,
one non-greedy SSE request, and the watchdog real-generation check. **The new
release has not itself been reboot-tested.** CUDA graphs remain disabled
because that is the qualified configuration. Final production process RSS was
6,848,076 KiB, process swap 188,832 KiB, and system available memory 122 MiB.

## Architecture and residual risks

The new path derives XGrammar's packed allowed-token mask on the host, uploads
at most two 31 KiB mask rows, and applies the token mask to logits in CUDA
before sampling. It preserves independent masks for the two scheduler rows.
The existing scheduler still copies full logits to its CPU sampler every
token. GPU **masking** is implemented and measured; GPU **sampling** and removal
of that pre-existing logits transfer are not. This is a deviation from the
original plan's workstream 4, not a claim that the entire sampling path moved
to the device. There is no new full-logits transfer caused by this feature.

The current production engine has very little spare unified memory even with
the previous binding. The bounded concurrent pass and user-performed reboot
do not establish long-term stability or capacity for other memory-heavy jobs.
The prior release and service configuration remain recoverable.

The TensorRT experiment journal retains failed attempts, launch records,
terminal markers and exact remote evidence paths. The separate MLC journal is
not part of this experiment.
