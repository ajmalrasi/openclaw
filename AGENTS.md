# Jetson experiment continuity

When working on the Jetson Qwen3.5/vLLM/MLC migration, read
`JETSON_MLC_EXPERIMENT_LOG.md` before starting experiments or interpreting old
benchmark results. The user explicitly requested a detailed, continuing record.

Update that journal after each experiment, repair, benchmark, service change,
or requested status check. Record the configuration, evidence/log path, result,
failure, fix, verification, and next unresolved issue. Preserve failed attempts
and label corrections, estimates, and missing evidence. Update current state.

Keep long Jetson jobs detached. Do not poll continuously merely to maintain
the journal. Honor the user's five-minute benchmark cap and free model memory
before memory-intensive builds. Preserve working artifacts and unrelated services.

Older MLC runbooks describe Qwen3 on JetPack 6; they are not the current
Qwen3.5 experiment configuration. See `FOR-AGENTS.md` for the shared API contract.
