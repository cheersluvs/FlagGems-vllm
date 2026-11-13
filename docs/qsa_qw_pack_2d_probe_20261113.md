# Two-dimensional query and weight packing probe

## Frozen project scope

| Field | Scope |
| --- | --- |
| Evidence | 1fa0864219a9c43f39134989628baf9e51332685: compact QW94.49us is73.6% of128.40us raw pipeline; long QW94.53us is8.0% |
| Editable | Isolated Q/W probe, flags/runner, CPU contract tests and this note |
| Read-only | Raw C++/Cube, K packing and page64 finish bodies, current INT8 scorer, harness, original installed toolchain |
| Frozen baseline | Previously verified raw_k_2d_capped and raw_finish_page64_capped for every complete arm |
| Active set | compact first128 rows/W1088 and long100k first112/W29696; optional budget_exact |
| Validation | Entire Q NZ and W.T buffers exactly equal old packer; all candidate scores/masks equal old raw result, current at rtol=atol=1e-3 |
| Timing | QW stage and four-task complete raw pipeline; two interleaved rounds, warmup3/active6, strict task/name counts |
| Profiler | Compile and IR checks plus PipeUtilization/Memory, two iterations each, in the same invocation |
| Autotune | Exempt targeted diagnostic, no production configuration or registration change |
| Timeout | Existing worker1800 seconds and automatic upload |

## Contract

| Field | Scope |
| --- | --- |
| Q | INT8[rows,4,128]; flatten four queries into16 heads; output[ceil(rows/4),2048] NZ |
| W | W.T for the native transposed load: [head16,output_lane16]; four nonzero output lanes, per-head half(1024*QS*weight) |
| Query load | Four static16x32 source slices, row stride128, contiguous32 columns; same2048 values per group |
| Coefficients | Load QS/weight once as16-element vectors, same FP32 multiplication order and FP16 rounding, broadcast only after loading |
| Tail | Missing heads become zero; every Q/W output element written; no alias or Torch compute fallback |
| Numerical | No change to INT8 dot, half middle, FP32 accumulation or final scale/visibility |
| Resources | Small16x32 data tile and16x16 weight tile, including offsets/masks; no enlarged Cube or TopK state |

## Paths

| Arm | Interpretation |
| --- | --- |
| original | Unchanged flat Q/W packer with one program per four-query group |
| two_d_grid | Same group grid; four statically unrolled16x32 loads and explicit two-dimensional W.T store |
| two_d_capped | Identical group body; at most20 programs, grid-stride loop across groups |
| complete | Selected QW + fixed fast K + unchanged raw Cube + fixed page64-capped finish; all preparation charged |
| current | Existing routed INT8 score stage, not CANN or TopK timing |
| Result | Candidate only; formal harness mean and full operator regression remain separate |

The old QW artifact contains an extracted-access loop for Q. Tests inspect the
actual generated addresses against the independent NZ/transpose contract and
exercise partial query groups in CPU checks. Card cases validate entire packed
buffers before timing and exact outputs again after counter collection. Additional
device packing checks for rows1 and5 cover partial groups with nonuniform and
signed coefficients; these small checks run once, outside all timed windows.

Mac verification: twenty-one raw/vector/packing CPU checks and five existing
compiler-copy tests; generated query-pointer and W.T-store expressions are
evaluated against independent layout contracts for full and partial groups.
Python/Bash syntax, formatter/linter and repository pre-commit hooks are checked.
There is no Mac NPU execution or performance result for this candidate yet.
