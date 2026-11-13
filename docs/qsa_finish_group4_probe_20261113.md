# Four-query raw score output reuse probe

## Frozen project scope

| Field | Scope |
| --- | --- |
| Evidence | 6e5b88ee0a81883f59a55a25f4a42f77e8292c6d: finish21.50us/57.4% in compact and591.83us/54.3% in long100k after QW vectorization |
| Editable | New isolated finish helper, mode/runner and CPU tests |
| Read-only | Raw C++/Cube, Q/W and K packing bodies, current INT8/production, harness and installed toolchain |
| Frozen baseline | QW two-dimensional original group grid, K two-dimensional capped, single-query page64-capped output |
| Active set | compact128/W1088 and long100k112/W29696; optional existing budget_exact |
| Validation | Exact whole-output equality to previous raw output, current score/mask tolerance1e-3, timed/counter postchecks |
| Timing | Original single-query finish vs group4-grid vs group4-capped; stage and complete four-task raw pipeline, rounds2/warmup3/active6 |
| Profiler | Same-run compiler artifacts and PipeUtilization/Memory, two iterations per metric |
| Autotune | Exempt diagnostic, no production config/registration change |
| Timeout | Existing worker1800 seconds and automatic upload |

## Contract

| Field | Scope |
| --- | --- |
| Raw | Existing [ceil(rows/4),key_tiles,16,64] FP32; only its first four output rows correspond to queries |
| Group | Four queries must share the page map, already proven by the selected-chunk guard |
| Shared data | One page ID and one64-element KS load per group/key tile; row visibility loaded as four integers |
| Load bounds | Group maximum visibility limits KS and rectangular Raw loads; invalid query rows remain masked |
| Output | Each row retains its own visibility mask and -inf padding; no arithmetic change to FP32 Raw*float(KS) |
| Layout | Raw4x64 contiguous submatrix, output4x64 with known row strideWIDTH |
| Partial groups | CPU address/mask checks and once-only device diagnostic with rows5,width257,unequal/zero visibility, shuffled pages, and Raw capacity shorter than dense output |
| Resources |256 FP32 Raw values,64 scales, four visibility values plus indices/masks; no enlarged Cube or TopK state |

## Paths

| Arm | Interpretation |
| --- | --- |
| original | Previous one-query page64-capped finish with fixed fast QW/K |
| group4_grid |4x64 output unit, independent grid(group_count,key_tiles_out) |
| group4_capped | Same4x64 body, at most20 programs and grid-stride loop |
| complete | Fixed QW + fixed K + unchanged raw Cube + selected finish; all four device tasks counted |
| Result | Keep as an unregistered candidate after evidence; no formal harness mean, CANN V1 or TopK claim |

Loading Raw up to a group's maximum visibility may read values invisible to
some of its rows. Those values exist in the current full-tile Cube output and
are masked to -inf independently at the final store. This does not expand
attention visibility. Entirely invisible key tiles issue no Raw/KS/page read.

Mac verification: twenty-four raw/vector/packing/finish CPU tests and five
compiler-copy tests, generated source/Python/Bash syntax and repository hooks.
Tests evaluate actual mask/base expressions against the old per-row formula,
check shared reads past individual visibility still produce masked output,
and prove full dense store coverage and capped-loop tail coverage. NPU
compilation, exact outputs and performance remain pending the card run.
