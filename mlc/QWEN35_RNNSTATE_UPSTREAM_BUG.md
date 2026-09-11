# Qwen3.5 compile failure: TIRx rejects the high-dimensional RNN-state getter

## Summary

Current MLC main cannot export the Qwen3.5 recurrent-state getter with its pinned TVM. TVM reports that `seq_id` is a buffer used without declaration. Replacing the manual `T.buffer_store(T.BufferLoad(...))` with the semantically equivalent direct indexed assignment does not change the failure.

## Exact versions

- MLC commit: `9fa644f54b04983adea4d0168f49fc6af4a893ba`
- Pinned TVM commit: `837cb9de1127b48ce48e4cefe09e83215b9d4ba7`
- Python: `3.12.14`, Clang `22.1.3`
- TVM package: `0.26.dev0`
- MLC package: `0.1.dev0` (`mlc_llm 0.26.dev6` wheel build reported by the image recipe)
- Target used by the full compile: CUDA `sm_87`, AArch64 `cortex-a78ae`

## Minimal reproducer

Run in an environment containing the versions above:

```python
from mlc_llm.nn.rnn_state import RNNState

RNNState.create_get_func((32, 128, 128), "float32", 2, 2, 0)
```

The shape is the first Qwen3.5 recurrent state for this configuration: `linear_num_value_heads=32`, `linear_key_head_dim=128`, and `linear_value_head_dim=128`. The reproducer fails during the `@T.prim_func(s_tir=True)` parse and does not require model weights or CUDA kernel compilation.

## Actual result

```text
tvm.error.InternalError: TIR is ill-formed: buffer seq_id is used at
<root>.body.block.body.body.body.body.body.block.reads[2].region[0].min.buffer
without a prior DeclBuffer or other declaration.
```

The wrapped diagnostic is:

```text
tvm.error.DiagnosticError: error: Program is not well-formed.
```

## Expected result

`RNNState.create_get_func` should return a well-formed `tirx.PrimFunc`, allowing Qwen3.5 export to proceed.

## Source and attempted minimal fix

The failure is in the high-dimensional getter in `python/mlc_llm/nn/rnn_state.py`. Current main uses:

```python
T.buffer_store(
    output,
    T.BufferLoad(storage, [seq_id, history_id, *vs]),
    [vi, *vs],
)
```

On Python 3.12, this was changed to the source comment's stated equivalent:

```python
output[vi, *vs] = storage[seq_id, history_id, *vs]
```

Both forms fail with the same verifier error naming `seq_id`. This indicates the issue is not resolved by changing only the buffer load/store syntax; the typed local IDs or generated block read region likely need investigation. That last sentence is a diagnosis lead, not an established root cause.

Do not work around this by disabling `check_well_formed`; that would suppress validation rather than establish correct IR.

## Full compile reproduction

```sh
mlc_llm compile /models/Qwen3.5-4B-q4f16_1-MLC \
  --device cuda \
  --overrides 'context_window_size=4096;prefill_chunk_size=1024;max_batch_size=2' \
  --output /models/Qwen3.5-4B-q4f16_1-MLC/Qwen3.5-4B-q4f16_1-main-sm87.so
```

The retained compile container exited `1`, was not OOM-killed, and produced no output library. Full log on the test host: `/home/ajmalrasi/mlc-qwen35-main-sm87-patched-compile.log`.

## Preserved evidence

- Unpatched runtime image: `sha256:f358fa9dcd9c4d967abad218438e85b2dba56c0f24943974cc93e0d7a845ded0`
- One-line patched image: `sha256:a6e7da8967d19609308e58bba9132314a575dc100d3d22eadf1ea88f7621733e`
- Patched `rnn_state.py` SHA-256: `b4179b9267205b136cd3670b032778fb5120ae11c27d2a43dd0e6ba4d11c8bd3`
- Retained failed container: `mlc-qwen35-main-sm87-patched-compile`
- Preserved older working model library SHA-256: `917ee583d732167dd1f5af59e0805514fc6372c5e12251f09f2512bd70310743`
