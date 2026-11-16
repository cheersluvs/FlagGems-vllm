# Ten-input empty-finish score comparison

## Project truth

| Field | Contract |
| --- | --- |
| Goal | Validate useful empty-block raw finish against current INT8 score on all ten frozen inputs |
| Frozen | All kernels/arithmetic/native bitcode/compiler/TopK/production and workload tensors |
| Editable | Diagnostic suite selection/aggregation, CLI flag/runner, CPU tests |
| Inputs | Existing ten INT8 fixtures, all requests/global chunks; raw split cost charged |
| Timing | Three arms, two reversed rounds, three active; existing 5ms capture guard and strict task counts |
| Validation | Existing boundary and all-row pre/post exact raw pair, current tolerance, independent CPU oracles, K/KS exact |
| Evidence | Same trace for complete score and stage sums; source/IR retained |
| Autotune | Fixed diagnostic; no new tuning or production route |
| Acceptance | Score-only diagnostic, no CANN V1/TopK/formal harness acceptance |

## Contract

| Field | Contract |
| --- | --- |
| Arms | current, logical1024, empty1024 |
| Numerical behavior | Same passing candidate; invisible blocks write -inf, partial/full paths unchanged |
| Scratch | Existing shared case scratch; GPU packing charged each piece; no cached packed K across chunks |
| Sample handling | No slow-sample removal; reject incomplete task capture; no average of surviving cases |
| Reporting | Per-case current/raw, current/optimized, raw/optimized; ten-case arithmetic means only if all pass |

## Paths

| Mode | Behavior |
| --- | --- |
| --finish-empty-ten | All ten workloads, current plus both raw arms, guarded capture |
| --finish-empty-pair | Previous four-workload raw pair unchanged |
| Other modes | Preserve prior selection and timer defaults |

The prior four-input result 5b1ff860ff8bc32c4eb32cd770289cd562efc8c0 passed
pre/post validation and strict capture checks. Its raw self-comparison mean
2.094914x is not a comparison against current INT8 or CANN. This run supplies
the current INT8 score comparison in the same environment/capture, without
changing the production operator or claiming overall selection performance.

Local checks: 72 CPU tests passed (34 source/bounds probes, 24 raw address,
5 compiler-mode, 5 workload planner, 4 pair/guard tests), CLI guards and
Python/Bash syntax. Focused pre-commit passed after import formatting.
NPU compilation, correctness and timing for the expanded set remain pending.
