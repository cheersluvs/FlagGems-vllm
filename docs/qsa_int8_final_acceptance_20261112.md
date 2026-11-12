# Step 5: frozen INT8 integrated entry acceptance data

Status: runner prepared; device correctness/performance results pending.
The registered public Ascend operator is still BF16. This acceptance run targets
the established, unregistered PreparedSharedMapInt8Selection INT8 integrated
entry, not a newly registered INT8 API.

The mixed double-buffer control had a local benefit over its same-algorithm
single-slot version but remained slower than this integrated entry. The query16
second-Cube candidate also regressed. Installed direct Fixpipe/copy wrappers
require Ascend910_95; the generic L1-buffer candidate crashed bishengir-compile
with SIGSEGV in report 5e56c515. These candidates are not promoted. This run
changes reporting/coverage only, not scoring, TopK or compiler implementations.

| Item | Frozen choice |
| --- | --- |
| Cases | All ten existing migrated INT8 cases; original complete query rows and scan widths |
| Candidate | Existing shared-map, page64/query16/direct-page dispatch; automatic unchanged persistent_topk |
| Baseline | torch_npu.npu_quant_lightning_indexer CANN V1, existing adapter and arguments |
| Compiler | Existing isolated FlagTree and CANN9.1 environment |
| Timer | Repository benchmark harness --mode kernel |
| Aggregation | Arithmetic mean of ten per-case CANN fused latency / integrated candidate latency ratios |
| Additional data | Candidate score_all and topk_all, each covering every chunk of the full input |
| Initial timing settings | warmup20, iter50; same settings for baseline/candidate/stages, no additional profiler or round loop |
| Correctness | Every-row index structure/range/count/uniqueness/padding; existing recorded sample rows against FP32 CPU selected-score reference, rtol/atol1e-3; existing all-chunk scorer control comparison when query16 is active |

TopK isolated timing reads materialized FP32 scores; score timing uses the
ordinary scratch/chunk dispatch. Their sum need not equal the combined call.
The native baseline remains fused: do not label any native score/TopK split.
Input generation, metadata proof, allocations and CPU checks remain outside the
prepared kernel timing scope. The existing harness scope is retained.

QSA_HARNESS_ACCEPTANCE=1 enables all-ten-case stage collection. The old
four-case QSA_HARNESS_STAGES diagnostic mode remains available and still has no
acceptance mean. Experimental mixed, handoff, alternate TopK policy and scratch
flags are rejected for acceptance. The runner uploads partial reports on a
failed worker; the summary must not compute a mean from a partial case set.

Each case saves acceptance-checks.json with geometry, sampled row IDs,
reference checks, route counts, TopK identity and scorer/workload hashes.
Summaries require all ten full-input records and all stage measurements.
The terminal prints the complete stage table and arithmetic mean. Successful
collection is not a claim that a speedup target has been reached; assess the
reported per-case values and mean after the card run.
