# QSA prefill: output overlap validation trial

This is the first production optimization trial after the baseline and host
profile. H800 report [e760e9e](https://github.com/cheersluvs/FlagGems-vllm/blob/e760e9e65185bb774473661bd4a623a74bdc29e2/results/qsa_select_paged_prefill/h800/stages-20261015-031813/stage-summary.json)
records 104 independent passes, one expected second-device skip, and eight
successful balanced benchmark cases. Core_1 unprofiled host validation is
20.925 us, hot outer allocation 6.104 us, and device guard 1.703 us. cProfile
identifies five overlap checks and ten repeated span sums per validation;
instrumented times are not production latency and cannot be added to wall.

The following choices are frozen before implementing the trial.

| Project field | Choice |
| --- | --- |
| Editable | Only output/input overlap helper and its validation call in production; alias tests; benchmark-only frozen old validator and paired A/B support; documentation |
| Read-only | Score/fill/Top-K kernels, public entry body, all other validation rules, workspace ownership, tuning keys/candidates, references and tolerances |
| Hypothesis | Computing the contiguous output's byte range once, then checking input byte ranges, removes repeated metadata work without dropping alias checks |
| Build / tests | Python-only; CPU differential tests against exact old helpers, real Torch alias/stride tests on H800 and full independent QSA correctness gate |
| Benchmark | Schema 2 paired full-entry old/new validation in one worker, same inputs and compiled score/Top-K callables. Primary metric wall us; retain per-round ratios and raw latencies |
| Active set | Existing 22 cases, including BF16/FP8, H/D, ragged padded strides, fast path and one-row chunks; focused eight-case run also supported |
| Timeout / profiler | Existing 900 s per isolated worker; optional unprofiled validation microbench and cProfile after all formal timings |
| Autotune | No new kernel: original six candidates, original libtuner keys and policy; both A/B entries share the same kernel objects and configuration cache |
| Acceptance | Candidate only until card correctness and stable A/B show benefit without important regressions; otherwise revert the production change. No native-vLLM speedup claim |
| Docs / missing files | workflow.md read; optimization.md/deep_opt.md absent in this checkout. This trial is a host metadata change, not a new lowering/kernel experiment |

| Contract field | Preserved behavior |
| --- | --- |
| Interface / dtype | Same ten arguments, BF16/FP8 Q/cache, int32 output, H/D inferred from Q; current NVIDIA SM90 gate |
| Alias / layout | Output must be contiguous and must not overlap any input's conservative byte hull, including padding holes; adjacent/disjoint slices sharing storage remain valid |
| Empty / metadata | Empty tensor byte ranges never overlap; existing stride/rank/type/device/error checks and exception messages remain unchanged |
| Semantics / numerics | Same FP32 scoring, per-head ReLU then head sum, unordered Top-K/ties and -1 padding; finite Q/K, inference-only |
| Workspace / concurrency | Same per-call allocations; no global cache, stale metadata or new stream ownership assumption |
| Torch use | Production reads metadata only; no compute, copy, cast or reference fallback |

| Path | Change / verification |
| --- | --- |
| Output span | Already validated contiguous: numel * element_size, computed once per call |
| Input span | Contiguous inputs use numel * element_size; strided inputs retain the original conservative span formula |
| Disjoint addresses | Input start >= output end cannot overlap, so skip its end calculation |
| Other production paths | Public dispatch, empty/select-all, score and Top-K unchanged |
| Frozen A/B baseline | Exact old three validator helpers copied from commit 416083e, verified by AST against its source snapshot. A copied public function uses only that old validator, with original shared kernel globals; it never swaps production module globals |

The full-entry A/B builder rejects changed frozen helper sources or public-entry
source (hashes are independent of Python AST version). Thus this
baseline measures this validation-only trial and cannot silently become a
baseline for a later workspace/dispatch optimization. Graph timings validate
unchanged device work; CPU validation runs during capture, not each replay.
Full graph stage times and standalone CPU microbench times are diagnostic;
the production full wall A/B remains the optimization metric.

The largest e760e9e case still drifted: temperature snapshots rose from 69 to
87 degrees Celsius, while boundary samples showed 1590 MHz SM clocks and no
active slowdown reason. Those samples do not establish the in-kernel cause.
Its full wall/event/graph medians are 1364.040/1365.411/1331.157 us; per-round
latency still grows substantially. Use paired round samples and do not adopt
a long-case speedup/regression claim without stable conditions.

Local checks cover 20,000 randomized byte-hull comparisons, empty/strided
metadata and exception equivalence, pinned helper/source identity, shared GPU
launcher construction without module-global mutation, and automatic upload
after successful/failed test runs. H800 adds 20 actual-storage alias/adjacency
cases and one conservative padding-hole rejection case; the full independent
correctness gate remains mandatory before timing.

Run from any directory, in a fresh checkout. The runner performs the full
independent correctness suite, defaults to the 22-case A/B, then uploads all
top-level report JSON/TXT/TSV through a new results worktree. Failed tests with
an available summary are also uploaded, with their exit status preserved.

```bash
if bash <<'QSA_AB'
set -euo pipefail
source /tmp/qsa-vllm031-h800.PQnVpV/activate_qsa.sh
qsa_repo=$(mktemp -d /tmp/flaggems-qsa-stage.XXXXXX)
git clone --depth 1 --single-branch \
  --branch codex/qsa-select-paged-prefill-h800 \
  https://github.com/cheersluvs/FlagGems-vllm.git "$qsa_repo"
CUDA_VISIBLE_DEVICES=0 QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
  bash "$qsa_repo/tools/run_qsa_prefill_validation_ab_h800.sh"
QSA_AB
then
  echo "运行和报告回传均完成。"
else
  echo "运行或回传失败，当前会话已保留，请检查上方输出。" >&2
fi
```

Optional runner arguments `--cases core_0 core_1 core_2 core_3
upstream_bfloat16_s0 upstream_float8_e4m3fn_s0 mixed_padded zero_visible`
select the prior focused active set. The 22-case default also covers H/D and
budget=0 regressions. Graph support remains optional; `--no-graph` still
compares both full-entry wall timings.

New report `validation-ab.tsv` contains both full wall medians and paired-round
ratios. `full` is the candidate; `full_validation_baseline` is the frozen old
validator running with the same current public bytecode, scoring and Top-K.
Ratios above one favor the candidate. CPU `host-summary.tsv` also includes
`validation_baseline`; profiles run outside formal timing. The reported source
snapshot and validation_ab identity fields expose which functions were shared.

Trial status: **keep** after H800 report `27f984c2` from code `7739abad`:
125 correctness passes, one expected second-device skip, and all 22 A/B cases
passed. Short scored cases reduced full wall medians by 6.47–9.31%; zero-visible
by 25.71%. Core_2, long core_3 and one-row chunks have no resolved benefit.
See the [full result and measurement limitations](qsa_prefill_host_validation_results.md).
This is a host-validation gain, not a GPU kernel or native-vLLM speedup.
