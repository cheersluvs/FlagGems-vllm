# H800: fractional cache policy / cp.async reproduction

This directory needs only PyTorch, the selected Triton/FlagTree runtime, and
its bundled `ptxas`. It does not import vLLM or FlagGems and does not run TopK.

## What has been verified

The original native QSA scoring kernel and PTX interventions were verified on
H800. The original captured PTX fails with an illegal instruction; removing
only the cache-hint qualifier and policy operand from its four `cp.async`
instructions passes. Changing the fractional policy from 1.0 to 0.5 still fails.

`captured.ptx` is byte-for-byte the verified failing H=1, D=32 artifact.
`captured.json` records its ABI, launch metadata, hashes, and provenance.
`replay.py` packages it for independent execution and was confirmed on H800
on 2026-10-14. It uses Triton's internal binary loader
APIs observed in Triton 3.6.0 / 3.7.1; other versions may require adaptation.
The ABI stub is never compiled; only the captured PTX is assembled and loaded.

`minimal.py` is a **confirmed source reproduction on H800**. It retains a
staged dot loop and `evict_first` load policy, removing paging, causal masking,
head reduction, and integration dependencies. The default case emits four async
cache hints and fails at synchronization with an illegal instruction. Removing
the policy or using STAGES=1 passes. See [validation.md](validation.md) for all
six results. A successful default run on another environment means that the
failure was not reproduced there; inspect its PTX and `async_cache_hints`.

## Run on the existing H800 container

Activate the previously prepared vLLM 0.31 environment (used here only for its
PyTorch/runtime dependencies):

```bash
source /tmp/qsa-vllm031-h800.PQnVpV/activate_qsa.sh
PYTHONPATH=/opt/flagtree "$PYTHON" -c \
  'import torch,triton; print(torch.__version__,triton.__version__,triton.__file__); print(torch.cuda.get_device_name())'
PYTHONPATH=/opt/flagtree "$PYTHON" repros/qsa_cp_async_cache_hint/run.py
```

Use a fresh checkout containing this directory. No installation or replacement
of the existing compiler is required. `/opt/flagtree` must be the directory
containing the intended `triton` package. The activation script supplies the
existing CUDA compatibility-library configuration. On another machine, activate
its own compatible environment rather than copying these machine-local paths.

The runner prints a report directory and executes six jobs sequentially, each
in a separate process with a private Triton cache. It continues after failures,
keeps stdout/stderr in `JOB/run.log`, and records all exit codes in
`summary.json`. Its overall exit code is nonzero if any job fails, including an
expected original-kernel failure. Do not interpret that exit code alone as a
regression; inspect `JOB/result.json` for the failing phase and actual error.

Run one case directly:

```bash
PYTHONPATH=/opt/flagtree "$PYTHON" repros/qsa_cp_async_cache_hint/replay.py --variant original
PYTHONPATH=/opt/flagtree "$PYTHON" repros/qsa_cp_async_cache_hint/replay.py --variant no-hint
PYTHONPATH=/opt/flagtree "$PYTHON" repros/qsa_cp_async_cache_hint/replay.py --variant fraction-half
PYTHONPATH=/opt/flagtree "$PYTHON" repros/qsa_cp_async_cache_hint/minimal.py
```

Optional: add `--verify-inputs /tmp/qsa-prefill-h800.kwkjy154/h1d32-input.pt`
to a replay command to check generated input values and strides against the
original saved card inputs. By default the script reconstructs that recipe
without depending on the old report directory.

For the stock compiler control, run the same commands with the stock Triton
environment and without `/opt/flagtree` on `PYTHONPATH`. Captured-PTX replay
deliberately bypasses source compilation: using stock Triton does **not** repair
the already captured FlagTree PTX.

To return the complete report, archive the directory printed by the runner:

```bash
tar -czf qsa-cp-async-report.tar.gz -C /tmp flagtree-cp-async.REPLACE_WITH_PRINTED_SUFFIX
```

## Issue draft

[issue.md](issue.md) is an English draft for FlagTree. It has not been posted.
It includes the H800 validation of both standalone paths. The reduced source
has not yet been tested under stock Triton, and its SASS has not been inspected.
