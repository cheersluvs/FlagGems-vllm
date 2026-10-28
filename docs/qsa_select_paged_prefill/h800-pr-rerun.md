# Revalidate the NV PR on H800

This validation-only branch is based on PR preparation commit
`a4f8ab958880a3dc820867c8305e4a1bd6ec45b5`. All 27 pinned PR and framework
files are unchanged. It reuses the previously exercised stage timer and
FlagTree/CUDA-13.4 bootstrap from research commit `16bc82f`.
Historical candidate helpers are timer imports only; the selected candidate
must be production. These harness files do not belong in the operator PR.

## Frozen run

| Item | Setting |
| --- | --- |
| Correctness | Entire retained tests/test_qsa_select_paged_prefill.py, all 24 functions and their parametrizations; no --quick |
| Allowed skip | Only the second-SM90 device test on a single visible GPU |
| Performance matrix | All ten FP8 cases, unchanged PR benchmark workloads |
| Baseline | SHA256-pinned native vLLM source and native CUDA TopK |
| Candidate | Official package export and latest-upstream persistent_topk from the PR |
| Compiler | Same verified FlagTree directory for both score paths; original PTXAS 13.4/CUDA compat wrapper |
| Timer | --mode kernel, 2 alternating rounds, 20ms warmup/20ms measurement |
| GPU guard | Exclusive idle H800, 80C initial limit, 120s cooldown timeout |
| Reports | Score, TopK, directly measured pipeline, per-case timings/ratios and arithmetic means |
| Acceptance metric | Score-only kernel speedup, excluding TopK |
| Scope | Targeted QSA production suite; not every test in the repository |

Each benchmark case also performs its existing before/after score, selection
and common-logits checks. Accuracy and performance share the checked compiler,
GPU and native-extension identity. A changed/missing PR file fails before GPU
work. New empty/partial reports cannot count as successful reruns.

The historical 30-case overall-acceptance.json is a legacy timer output and
is not this rerun's verdict. Use pr-validation-summary.json for completeness,
arithmetic means and the explicit ten-case scope; original per-case data is
preserved in stage-summary.json and TSV outputs.

The full retained correctness file takes longer than the old 49-case iteration
profile. Two-round timing stays short. Tests and timings are not substituted
with historical results.

## Run and upload

Use a fresh worktree so an existing dirty GPU checkout remains untouched. The
wrapper saves and uploads reports for pass, correctness failure, compiler
failure and GPU-admission rejection. Upload failure prints a retry command
and retains the report/result worktree; no rerun is needed for upload retry.

Use the existing Python and FlagTree environment, without installing or
modifying compilers. The isolated CUDA compatibility wrapper may reuse or
download verified user-space packages under /tmp.

```bash
QSA_STOCK_PYTHON=/tmp/qsa-vllm031-h800.PQnVpV/venv/bin/python bash tools/run_qsa_prefill_nv_pr_h800.sh --flagtree /opt/flagtree
```

Run as child bash, never source the script into an interactive container shell.
After return, send the printed result branch and commit back to this chat.
