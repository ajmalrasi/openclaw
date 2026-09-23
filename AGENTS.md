# Jetson experiment continuity

When working on the Jetson Qwen3.5 vLLM-to-MLC migration, read
`JETSON_MLC_EXPERIMENT_LOG.md` before starting experiments or interpreting old
benchmark results. The user explicitly requested a detailed, continuing record.

TensorRT Edge-LLM work, including JSON-Schema guided decoding, is a separate
track. Do not read or update the MLC journal for that work; use
`TENSORRT_EDGE_LLM_EXPERIMENT_LOG.md` instead.

Update that journal after each experiment, repair, benchmark, service change,
or requested status check. Record the configuration, evidence/log path, result,
failure, fix, verification, and next unresolved issue. Preserve failed attempts
and label corrections, estimates, and missing evidence. Update current state.

Keep long Jetson jobs detached. Do not poll continuously merely to maintain
the journal. Honor the user's five-minute benchmark cap and free model memory
before memory-intensive builds. Preserve working artifacts and unrelated services.
For future Jetson builds, use safe multicore parallelism when memory allows
(typically two compile workers with at least 4 GiB available on the 8 GB
Orin); use one worker only when memory pressure or a failure justifies it.
Do not disrupt an already-running build merely to change its worker count.

Older MLC runbooks describe Qwen3 on JetPack 6; they are not the current
Qwen3.5 experiment configuration. See `FOR-AGENTS.md` for the shared API contract.
