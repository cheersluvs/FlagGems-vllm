# QSA kernel benchmark without standalone pytest

User explicitly requests benchmark only after62 iteration nodes passed in report
22d3c2ae. Add --skip-accuracy to the verified FlagTree runner's existing forwarded
arguments. This skips only standalone pytest,not preflight or per-stage numeric
validation/guards. No prior report is automatically relabeled/reused as this run's
accuracy evidence.

| Project truth | Decision |
| --- | --- |
| Editable | Benchmark CLI/parent orchestration,CPU tests/source archival,this document |
| Frozen | All kernels/TopK/configs/30 fixtures,timers,guard thresholds,numerical checks,acceptance requirements |
| Validation | CPU parser/parent skip-default/preflight/worker/guard/report checks;changed-file pre-commit |
| Benchmark | Existing30 kernel cases,2 rounds,score/TopK/sequential pipeline;normal automatic upload |
| Accuracy | Record skipped_by_request with zero counts,not passed;no formal acceptance or full suite claim |
| Autotune/fallback | Existing tuning untouched;no production code or fallback changes |

| Contract | Decision |
| --- | --- |
| API/math | Existing QSA/native source contract and dtype/layout/outputs unchanged |
| Reference | Existing pinned native source and verified same-compiler runtime |
| Scope | Forward-only;test orchestration only |

| Path | Behavior | Check |
| --- | --- | --- |
| Default | Existing iteration/full pytest gate | Defaults unchanged |
| --skip-accuracy | Write explicit skipped record;continue preflight/guarded workers | No pytest spawned;all ordinary GPU checks retained |
| GPU occupied | Existing rejection,exit2 | Never bypass admission or timing guards |

Standalone acceptance stays invalid/incomplete until independently reviewed
correctness evidence is associated;stage timings and grouped statistics remain
available. Report target_met is not automatically asserted from benchmark-only
runs. Use the ordinary retained joint-stage runner with --skip-accuracy;do not
use the rejected global-unroll candidate. Fresh source/runtime/workload identity
records remain mandatory. H800 execution stays pending;local checks do not
constitute GPU numerical or timing validation.

Usage:

```bash
bash tools/run_qsa_prefill_stage_joint_h800.sh --flagtree /opt/flagtree --skip-accuracy
```

Local validation:99 related CPU regression checks passed. The actual parent
orchestration test covers default pytest execution versus explicit skipping,
continued preflight/workers,continued idle admission and truthful archived status.
Parser tests reject use outside the verified FlagTree parent. Existing acceptance
rejects skipped accuracy rather than marking it passed;no acceptance rule was
relaxed. AST comparison confirms only parent/main scheduling changed;worker,
per-stage numerical checks,measurement functions,runtime and guards are unchanged.
All kernels,TopK,tuning YAML and revised30 fixtures are untouched. Changed-file
pre-commit/diff checks pass. H800 execution is pending. This saves the standalone
pytest time only,not per-case JIT/autotuning,warmup or numeric validation time.
