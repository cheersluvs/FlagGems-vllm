# Step 5: frozen INT8 integrated entry acceptance data

Status: all ten full-input correctness checks and harness kernel measurements completed on 2026-11-12. Performance remains below CANN V1.
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

## Final result

- Report: `codex/qsa-ascend910b-results-20261112-064923-2179099`, commit `7e0aed0e95d4377e84ad9e66348437f44fd0b65a`.
- Measured implementation: `c49ca4ef7e5d3b1657144638115fc3efa326eebc`; resume runner: `a53ad33abc4a7031560f4fdbdf9c55b46a262482`.
- Device: Ascend910B4-1; existing FlagTree/Triton3.5.1, torch2.10.0+cpu, torch_npu2.10.0, CANN9.1/verified OPP.
- All ten full inputs passed the recorded correctness contract. This includes all-row index structure and sampled-row CPU selected-score checks; it is not an exhaustive CPU numerical reference over every row.
- Arithmetic mean of ten per-case fused-call kernel speedups: **0.5548143802445699x**. This is not a ratio of summed latencies. Every case is slower than the native CANN V1 call.
- Seven cases were reused from `da80530ecca93a6d3d8515e00bcd2e0a18cfae5e`; short_tail, compact and long100k were completed in a subsequent session after a device startup timeout. Resume checked unchanged measured source/workload/TopK hashes. These are not independent repeated measurements of all ten cases.

Units: milliseconds. Stage timings are candidate-only; CANN remains a fused baseline.

| Case | Score | TopK | Combined | CANN V1 | CANN/candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| int8_first_chunk | 7.027456 | 7.653179 | 15.538965 | 8.041386 | 0.517498x |
| int8_budget_exact | 11.355305 | 11.715438 | 23.311007 | 14.283604 | 0.612741x |
| int8_dominant_long | 10.936410 | 14.906696 | 25.705682 | 14.727142 | 0.572914x |
| int8_tail112 | 10.556602 | 15.091632 | 25.983135 | 14.295453 | 0.550182x |
| int8_tail113 | 10.697865 | 14.904904 | 25.467578 | 14.301214 | 0.561546x |
| int8_two_balanced | 13.979743 | 14.978869 | 28.810378 | 15.570125 | 0.540435x |
| int8_two_skewed | 13.419348 | 15.419034 | 28.540584 | 14.532934 | 0.509202x |
| int8_short_tail | 1.705929 | 2.123668 | 3.846259 | 2.527643 | 0.657169x |
| int8_compact | 2.502948 | 4.021520 | 6.850653 | 3.166371 | 0.462200x |
| int8_long100k | 37.892897 | 50.650956 | 89.025406 | 50.233213 | 0.564257x |

## Disposition

The unchanged INT8 integrated entry is frozen as the current baseline. Mixed
single/double-slot and second-Cube candidates remain diagnostic controls; none
is promoted by this acceptance run. The generic TLE L1 expression failed to
compile in report `5e56c51513fe4d17fb2f109bb922f9b002066bc3`, and direct installed
Fixpipe/copy wrappers are gated to Ascend910_95. Under the operator-only constraint,
close that local-handoff route for this environment rather than modify compilers.

This completes the plan's data collection and candidate disposition. It does not
register a public INT8 API or claim performance parity with CANN. A future
optimization cycle must start from this frozen full-input harness baseline and
state a new operator-level hypothesis; chunk or same-algorithm double-buffer
speedups alone are insufficient for production promotion.
