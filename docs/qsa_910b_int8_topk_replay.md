# Focused replay of the budget_exact duplicate-index failure

Report b0f2bb4dc03bbb0ac687c9bd93f30e60fc380ec8 passed nine logical workloads.
Only int8_budget_exact failed, at candidate index uniqueness in the first
128-row chunk. Neither this chunk's native output nor its scores had yet
been checked when the structural assertion stopped execution. The failure
is a genuine index-contract violation; score tolerance is not relaxed.

## Frozen project truth

| Field | Choice before implementation |
| --- | --- |
| Task | Replay exactly the first 128 rows of budget_exact, holding INT8 inputs and FP32 logits fixed |
| Editable | New replay script/runner and this document |
| Read-only | Workload generator/seed, scorer, all production and TopK files, preceding validators |
| Build/validation | Existing FlagTree; local syntax/lint and missing-dependency flow; NPU checks pending |
| Active set | Full original input allocation and RNG order, first128 rows, width8192, visible4096..4128, H4 D128 page192 K512 |
| Benchmark/profiler | None; correctness diagnosis only |
| Timeout | 600 seconds for one isolated replay |
| Autotune | Exempt diagnostic; frozen scorer/TopK parameters; original/sentinel auto128, auto32 and radix128 are explicit experiments |
| Fallback | Existing library behavior only; no output repair, deduplication or replacement TopK |

## Contract

| Field | Frozen requirement |
| --- | --- |
| Input | Same make_inputs(plan) from prior run; do not shrink generator shapes before producing cache values |
| Reference | Every one of 128 rows: CPU INT32 dots and FP32 scoring; CPU TopK selected scores |
| Output | Unique INT32 K512 indices in range; same native V1 contract; no padding in these visible>512 rows |
| Accuracy | Exact integer dots, score and selected-score rtol/atol1e-3 |
| Native | Direct unchanged V1 API, independent structural/CPU check even if candidate duplicates |
| Torch use | Inputs, allocations, validation/CPU reference only in this standalone probe |
| Unsupported | Other dtype/layout/workload/gradient paths remain outside this focused replay |

## Experimental paths

| Path | Invocation | Source | Check | Timing |
| --- | --- | --- | --- | --- |
| auto128_original | One128-row TopK call, empty output allocation | Existing TopK, auto route | All128 independent CPU TopK checks and duplicate details | None |
| auto128_sentinel | One128-row TopK call, output initialized -2 | Same TopK, auto route | Separate output-initialization control | None |
| auto32 | Four32-row TopK calls | Same TopK, auto route | Same frozen scores/rows | None |
| radix128 | One128-row TopK call | Same TopK, explicit existing radix route | Same frozen scores/rows | None |
| native | One128-row V1 call | Same INT8/scales/weights as candidate | Independent CPU validation | None |

Each TopK experiment uses a fresh module instance so its capability/fallback
state does not leak between experiments. The original auto128 experiment
runs first with the old empty-output allocation; the other output buffers
are initialized to -2 to expose unwritten entries. The native call follows
the TopK experiments so it cannot change device state before the first
reproduction. Source files are unmodified. Extra diagnostics and allocation
order may still affect a state-dependent failure; a non-reproduction alone
does not establish a fix.
Every result includes raw indices, duplicate row/index/count details, loader
identity and library state. The environment route override is restored.

The diagnostic returns zero only if scores, native and all four TopK
experiments pass; rejected experiments produce exit1 while remaining
experiments still run. Report upload occurs in either case. A smaller batch
or radix success is diagnostic evidence, not permission to silently change
the production TopK route or declare ten-shape acceptance.
