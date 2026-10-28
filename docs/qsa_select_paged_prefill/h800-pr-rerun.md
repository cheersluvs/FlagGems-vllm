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

## Import dependency fix after report 993b035

That run passed the PR byte/GPU/toolchain checks but failed before correctness:
latest-upstream mhc_pre imports flag_gems while the selected vLLM virtualenv
does not contain it. No accuracy or timing results were generated.

The wrapper now downloads the SHA256-pinned pure-Python FlagGems 5.0.2 wheel
and extracts only flag_gems and its dist-info into the report's python-deps
folder. It installs no packages into the virtualenv, invokes no pip resolver
and does not change Torch, FlagTree or PTXAS. Every compiler-profile check
verifies dependency bytes/import location and records the dependency identity.
The existing compiler identity checks remain active after package imports.

No PR production/test/config byte changed. There are no completed GPU results
to resume from report 993b035; rerun preflight, correctness and all ten timings.

Report 15de94b hit the fresh-directory guard before preflight: the dependency
bootstrap's wheel, JSON and python-deps directory were not yet allowlisted.
The guard now permits precisely those additional bootstrap names. Completed
accuracy/benchmark outputs and every other unexpected entry still fail closed.
A CPU regression drives the actual CLI parser through this guard (GPU parent
mocked), checking all ten cases are scheduled and existing stage-summary.json
is rejected. No production or timing code was modified by this correction.

Report 9677817 passed GPU admission and reached imports, then aborted in the
native stable-extension schema registration API. There were no QSA results.
Static inspection of the pinned FlagGems 5.0.2 wheel shows its patch_util
imports legacy vLLM extension modules and defines fallback schemas when they
appear absent. Loading FlagGems first can therefore conflict with newer native
stable-extension registration; this is a hypothesis pending the next H800 run.

The runtime now loads the pinned native vLLM reference before importing
FlagGems-vllm. No schema, kernel, package byte or compiler is patched. Import
phase markers and Python faulthandler are enabled to locate any remaining
native abort. Existing post-import compiler/module and public-entry identity
checks remain mandatory. PR production hashes remain unchanged.
