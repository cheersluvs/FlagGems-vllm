# Query16 second-Cube overhead diagnosis

The previous ten-chunk experiment passed correctness but score speedup was
0.583–0.627 versus the current INT8 integrated trial. This experiment diagnoses
that regression; it does not change the registered BF16 operator or promote
an INT8 implementation.

| Contract | Frozen choice |
| --- | --- |
| Workloads | Last 32 rows of longest request: budget_exact and long100k |
| Inputs | Existing seeded INT8 fixture, H4 D128, page192, original scan width |
| Complete scorers | Original FP32 Vector reduction and second-Cube reduction |
| TopK | Existing persistent_topk, same code and tolerance as previous run |
| Correctness | CPU raw-QK sink or original FP32 score reference, rtol/atol 1e-3 |
| Timing | Existing harness kernel timer; two reverse-order rounds |
| Acceptance | Diagnostic chunks only; no full-input arithmetic mean |

| Arm | Change and interpretation |
| --- | --- |
| current_int8_integrated | Existing query16 route, page traversal, grid and launch options |
| query16_second_cube | Same route with FP16 operands and second Cube reduction |
| qk_sink | QK plus FP32 four-head sum into the same 16x64 output footprint; omits dequantization and weighting |
| half_vector | Same positive QK/1024 and weight*query_scale FP16 rounding as Cube candidate, then FP32 Vector reduction |

The QK sink makes the first matrix multiply observable, preventing wholesale
dead-code elimination. Compiler input and available outputs are saved to inspect
retained operations. The sink is a lower-cost diagnostic proxy, not a measurement
of isolated Cube latency. Code removal changes live tensors, layouts and scheduling;
prefix differences are not additive exclusive stage durations. Likewise,
half_vector includes half-to-float conversion and differs from Cube placement.

| Evidence | Collection |
| --- | --- |
| Correctness and compilation | Each generated variant is compiled and checked before timing in the same run |
| Compiler identity | Generated source/hash, launch args, compiler input IR and available output hashes/text |
| Paired time | All four score arms; current/Cube additionally retain TopK and combined timings |
| Profiling | Separate CPU/NPU trace and PipeUtilization session per arm, three iterations each |
| Limitations | Task traces/counters do not prove intra-kernel overlap or individually attribute synchronization/layout cost |

`QSA_QUERY16_REDUCTION_DIAG=1` selects these two cases. It requires the existing
`QSA_FUSED_TEN_CHUNKS=1`, `QSA_COMPARE_CURRENT_INT8=1`, and
`QSA_QUERY16_REDUCTION_CUBE=1` flags. The report summarizes incomplete profiler
collection as failure while preserving successful correctness/timing evidence.
Only consider another Cube rewrite if these measurements show a specific,
removable cost; otherwise close this candidate without production integration.
