# Raw empty-block finish paired validation

## Project truth

| Field | Contract |
| --- | --- |
| Goal | Retest only passing logical1024 versus empty1024 over four complete workloads |
| Frozen | Production/TopK/Cube/QW/K/KS/source arithmetic, inputs, workspace, compiler correction |
| Editable | Optional paired-suite flag, optional capture guard, CPU tests, runner/docs |
| Cases | first_chunk, budget_exact, tail113, long100k; all requests/chunks |
| Timing | Two reversed rounds, three active sweeps, both four-stage raw paths charged |
| Accounting | Strict original CPU-window containment, expected host/device counts; no task discarded or timestamp tolerance |
| Guard | 5ms CPU sleep before launch and after synchronize inside each CPU window; no device work; duration sums exclude sleeps/host gaps |
| Reason | Prior captures include tasks crossing CPU record windows; padding creates capture margin without reassigning tasks |
| Evidence | Same accepted capture supplies stage means; generated source, compiled IR and rejected traces retained |
| Validation | Selected boundary fixtures and all-row pre/post checks; exact candidate/baseline and existing INT8 tolerance |
| Autotune | Fixed diagnostic exception; no production tuning or dispatch changes |
| Acceptance | Exploratory score comparison only, not CANN V1 or official harness mean |

## Contract

| Field | Contract |
| --- | --- |
| Selected arms | logical1024 and empty1024 only; no slow logical-tail candidates launched |
| Output | Identical FP32 results and -inf masks; output poisoning outside timed sweeps |
| Sample policy | Retain every active sample, reject any incomplete capture, never adjust boundaries after export |
| Clock limit | Guard does not prove clock alignment; strict capture checks may still fail, reported without valid speedup |
| Environment | Same isolated FlagTree task-mode correction and pinned CANN/OPP |

## Paths

| Mode | Behavior |
| --- | --- |
| --finish-empty-pair | Two selected arms, same four complete workloads, 5ms capture guard |
| Existing --finish-routing-probe | Unchanged original exploratory five-arm mode |
| Other prefix timers | Default guard0 preserves existing capture behavior |

No previous failed capture is promoted to an accepted measurement. In the
20261116-034905 report all four prechecks passed, while all three timing
attempts failed strict containment. Selected standalone finish task aggregates
suggest empty-fill gains, but complete-score acceptance and postchecks are
still required. The new run performs both in one invocation and uploads reports
on failure too. CPU sleeps are explicit capture scaffolding, not device latency.

Local checks passed: 71 CPU tests (34 source/bounds probes, 24 raw-address,
5 compiler-mode, 4 full-input planner, 4 pair/guard tests), CLI guards,
Python/Bash syntax and focused pre-commit. Device validation is pending.
