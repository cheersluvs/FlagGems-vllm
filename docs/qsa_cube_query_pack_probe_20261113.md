# Raw Cube query packing probe

## Frozen project scope

| Field | Contract |
| --- | --- |
| Current operator | Diagnostic raw INT8 two-Cube score chain, not a registered production change |
| Evidence | Report60350a51: group4 finish regressed; freeze one-query page64-capped finish |
| Editable | New native query-pack variant/helper/tests, optional native build arguments, probe routing/runner |
| Read-only | Original native source, QW/K packing bodies, current INT8 implementation, production/TopK/harness |
| Build | Existing ccec AIC-only bitcode command with QSA_QUERY_PACK=4/8/16; isolated FlagTree task-mode copy |
| Validation | Packed Q/W full equality, raw useful rows and final masks/scores versus old raw/current/oracle; tail and signed weights |
| Timing | Current score, per-arm QW/Cube/finish and complete QW+K+Cube+finish; microseconds, smaller is better |
| Active set | compact128/W1088 and long100k112/W29696; diagnostic rows17/key-tail65 in both larger packings |
| Aggregation | Two reversed/interleaved rounds, six active samples per round; medians of per-round means; no formal harness mean |
| Profiler | Same-run compiler/LLVM artifacts and Cube PipeUtilization/Memory, two iterations per metric |
| Timeout | Native build180s per command; worker1800s; upload root artifacts even on failure |
| Autotune | Explicit4/8/16 diagnostic sweep; no libtuner/registration needed until production integration |
| Fallback | No production fallback; torch computation only in diagnostic preparation and independent oracles |

## Alignment contract

| Field | Contract |
| --- | --- |
| Reference | Existing integer QK and half-chain CPU oracle in qsa_raw_score_chunks_910b.py |
| Device | Ascend910B, CANN9.1, existing pinned OPP/environment |
| Inputs | Contiguous INT8 Q[rows,4,128], INT8 paged K[pages,192,1,128], positive finite scales, finite signed weights, shared physical page map |
| Native shapes | First Cube M=(queries*4),K128,N64; second Cube M16,K=(queries*4),N64 |
| Precision | First INT32 accumulation; Fixpipe /1024, FP16/ReLU to L1; FP16 coefficient qs*weight*1024; second FP32 accumulation |
| Output | Raw[ceil(rows/queries),key_tiles,16,64] FP32; first queries rows useful, dense FP32 scores with per-query -inf mask |
| Packing | Original QW vector method extends head dimension only; same NZ order and W.T convention; K pack unchanged |
| Output method | Original single-query64-key capped body, only row//queries and row%queries address mapping changes |
| Tail | Zero-pad missing query heads/coefficient rows and missing key columns; explicit base-pointer byte offsets |
| Comparison | Q/W/K exact; final mask exact, raw4 oracle tolerance1e-3 and current score tolerance1e-3; no requirement for bitwise identity after K dimension change |
| Unsupported | Non-shared maps, H!=4,D!=128, nonfinite/overflow, nonpositive scales, training/backward; reject, no fallback |
| Unknown | Larger native LoadData/transpose behavior and real performance; verify on device in this run |

## Paths and resources

| Arm | Method | Check | Risk |
| --- | --- | --- | --- |
| original4 | Frozen QW grid+K capped+original native+single-query finish | Existing raw4 result | Baseline |
| query8 | New first M32/second K32, output16 rows | Pack equality, full score comparison, boundary diagnostic | Transpose/load layout |
| query16 | New first M64/second K64, output16 rows | Same | Larger local storage and second Cube work |
| complete | All four tasks charged, each arm's own QW and finish | Host/device task-count gate | Preparation benefit cannot be omitted |
| external | Raw custom C++ through existing TLE bitcode boundary | Retain source/build flags/LLVM/final object | Not a pure Triton production implementation |

The second Cube always produces16 rows, so increasing query packing reduces
native calls, repeated K loads and padded Raw writes. It also expands the
sparse weight matrix and the reduction dimension; a speedup is not assumed.
The original16-query-head first tile becomes32/64 head rows. L0A allocation
covers both INT8 first operands and FP16 second operands. L0B covers max(N*K
INT8, M*N FP16). L0C covers max(M*N INT32,16*N FP32). At query16 these are
8KiB,8KiB,16KiB respectively; all lifetimes remain fenced as in the baseline.
No new overlap, double buffering, Fixpipe opcode proof or TopK claim is made.

Local validation:24 existing raw/vector checks, five compiler-copy checks and
eight query-pack checks pass. The new checks evaluate generated NZ and finish
address/mask expressions, cover rows1/5/17 and zero visibility, verify disjoint
byte-offset regions and local operand capacities, preprocess distinct native
ABI symbols, and check variant build flags/artifacts leave the baseline intact.
Generated Python syntax and invalid CLI combinations are checked. CANN native
compilation, numerical device output, counters and latency remain pending the
910B run; the Mac has no NPU. A failed8/16-query candidate is retained with its
traceback while the other candidate still runs; any rejection makes the final
probe status incomplete rather than silently accepting partial success.
