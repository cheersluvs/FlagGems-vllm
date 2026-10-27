# Measure source-level R32 paired issue across all10 FP8 fixtures

## Scope board

| Item | Decision |
| --- | --- |
| Request | Measure optimization effect on every existing FP8 benchmark fixture,not only the prior five enabled cases |
| Editable | New private adapter,prelaunch gate,tests,runner and this board |
| Read-only | Production,compiler,formal benchmark shapes,reference and TopK/tolerances |
| Applicability | All10 cases execute a source candidate;compact retains its compact grid and splits its original query tile into two halves |
| Interpretation | New coverage includes multi-request ABI,stage1/K4 and original P1->P2. Report combined tile/query-split/issue effect,not isolated scheduling improvement |

## Implementation board

| Item | Decision |
| --- | --- |
| Candidate | Same frozen paired Python source from the existing compile proof;TILE_R32,QUERY_PARTS2;retain each tile's N/K tiles/warps/stages |
| Target tiles | FP8 H4/D128,N64,warps4,TILE_R>=32;main K16/R64 chunks plus first chunk;other small tails stay original;compact preserves its original config/grid and gets a separate split-query candidate |
| New specializations | Compile ordinary original-source control and paired source with identical candidate kwargs before launch;compare dot result shapes/work and inspect emitted waits |
| Resource policy | Require complete resource queries,zero spills and native hardware limits;record resource increases rather than silently hiding applicable cases |
| Prior evidence | For already-validated single-request R32/K16/stage2 preserve exact canonical executable/resource gate;do not impose its ABI hash on new multi-request signatures |
| Freeze | Save source-control/candidate PTX,TTGIR,CUBIN and receipts before execution;reject any new specialization or binary change during timing |

## Validation board

| Item | Decision |
| --- | --- |
| Numeric | Existing FP32 score/selection/pipeline checks before and after timing,including request first/last rows;original offset/page/mask boundaries retained |
| Compile | Two FP8 dot results per iteration with same shapes as the same-config unchanged-source control;two WGMMA waits ending0;source movement is exactly the previously checked assignment |
| Overlap | Emit whether wait1/0 is retained or compiler serializes to0/0. Applied source without retained overlap is not described as a scheduling gain |
| Timing | Existing two alternating rounds,20ms warmup/rep;score,unchanged common TopK,actual pipeline,four A/A controls and exclusive GPU guards |
| Coverage | Require at least one actually compiled/executed candidate chunk in every case,including compact;never count an unchanged A/A control as optimization |
| Output | All10 per-case data and all10 mean,with retained-overlap count and compact variant explicitly identified. No production promotion |

New compiler specializations are independently recorded and checked before any
execution. Unlike the earlier restricted probe,their executable hashes cannot
be equated to a single-request/K16 capture. This is an explicit extension of the
source-JIT probe,not an IR override or a change to compiler/production files.


Compact extension preserves the complete original request-slot binary search,
query/cache reads and grid. Reshape/permute/split divides the query rows into two
16-row halves;each keeps full-D FP32 dot,per-head ReLU/head sum and the original
visible/live store predicates. A sequential split-source compile control differs
only in the placement of the second dot. This is a new compact query-split
variant,so its effect is explicitly reported as split-plus-paired versus the
original single-dot compact production path.

Local validation:31 CPU checks passed,including actual target coverage for all10,
compact prefix/grid/load AST preservation,query/head lane reconstruction,
independent compile/spill/wait gates and no execution after failed preparation.
An additional24 NumPy FP32 split/reduction/masked-write comparisons passed for
1/15/16/17/31/32 valid rows and0/1/63/64 visible columns. These do not replace
FP8 GPU correctness. CLI,shell and changed-file hooks pass;GPU results pending.
