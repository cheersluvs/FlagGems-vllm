# Standalone H800 verification, 2026-10-14

Source: user-provided runner output and failure `result.json` / traceback
excerpts in the conversation. This file transcribes those observations; it is
not an imported copy of the report archive. Successful jobs were reported by
their exit codes; their individual JSON/PTX files have not been retrieved.

- Reproducer commit: `763904a758e01696933d35ca15651e5bf1d070c8`
- Card report directory: `/tmp/flagtree-cp-async.wz6flwzl`
- Python: `/tmp/qsa-vllm031-h800.PQnVpV/venv/bin/python`
- GPU: NVIDIA H800
- Imported Triton: `3.6.0`, `/opt/flagtree/triton/__init__.py`
- Replay assembler: `/opt/flagtree/triton/backends/nvidia/bin/ptxas`

| Job | Exit code | Observed result |
| --- | --- | --- |
| replay-original | 1 | Illegal instruction at synchronize; captured CUBIN hash matches |
| replay-no-hint | 0 | Success (script exits zero only after numerical validation) |
| replay-fraction-half | 1 | Illegal instruction at synchronize; two PTX edits |
| source-original | 1 | Illegal instruction at synchronize; four async cache-hint sites |
| source-no-policy | 0 | Success (script exits zero only after numerical validation) |
| source-stage1 | 0 | Success (script exits zero only after numerical validation) |

All three failing jobs report the same exception:

```text
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
```

The reduced source reaches `minimal.py:70`, `torch.cuda.synchronize()`.
Both failing replays reach `replay.py:212`, `torch.cuda.synchronize()`.
These line numbers refer to the tested commit above. Compilation/assembly and
launch return before the error is observed; no actual trapping PC is available.

Reported CUBIN SHA256 values:

```text
replay-original (matches_captured_cubin=true)
dd097a244194ef3a4e13a7e6bf0a271be4421fd3ecd1d2a9188fd4c2ff5aa9b7
replay-fraction-half (matches_captured_cubin=false)
7f53e5c6ffac59d483a4e17a38f681d81ad7db15cc2f784714cd825e9ab59972
```

The source reduction is now a confirmed reproducer for this FlagTree runtime
on H800. Its stock-Triton comparison and SASS inspection remain unperformed;
the earlier stock comparison and uninitialized-descriptor disassembly evidence
apply to the original native scoring kernel. This run does not establish the
faulting instruction or a compiler/toolchain version boundary.
