# Query16 merged output-store probe

## Frozen project truth

| Field | Contract |
| --- | --- |
| Evidence | Report25c171: long full finish596.17us, fill346.33us, copy64/1024=372.81/37.27us; output-loop granularity is the next target |
| Editable | New output-merge helper/tests, diagnostic routing and runner |
| Read-only | Query16 native C++, pack QW/K, original64-key finish, current INT8/production/TopK/harness and installed compiler |
| Build | Existing query16 bitcode/isolated FlagTree metadata correction, generated Triton output kernels |
| Validation | Exact complete outputs versus original64-key finish, existing raw/current tolerance, signed/zero/unequal/tail controls, postchecks |
| Benchmark | Original64, merge256/512/1024 output stages and each complete QW+K+Cube+finish pipeline; device us, smaller is better |
| Active set | compact128/W1088 and long100k112/W29696; synthetic rows17,width257,max-visible65 and signed weights |
| Aggregation | Two reversed/interleaved rounds, six active samples, per-round means then median; raw timing samples retained |
| Profiler | Same-run source/IR/assembly; pre-native input/sink audit, scalarization markers and store shape; PipeUtilization/Memory for all surviving arms on long100k, original/best on compact |
| Timeout | Existing1800s worker and automatic root-artifact upload |
| Autotune | Diagnostic width sweep only, exempt from production registration/libtuner |
| Fallback | No production fallback; torch computation only in CPU preparation/oracles |

## Alignment contract

| Field | Contract |
| --- | --- |
| Reference | Frozen single-query64-key finish over query16 Raw output |
| Inputs | Raw[group,key_tile,16,64] FP32, positive FP16 KS, shared int32 page map/page192, individual int32 visible |
| Computation | FP32 Raw*float(KS), individual -inf visibility mask, dense FP32 output; no precision/reduction change |
| Read granule | Every leaf loads a regular64-key Raw/KS segment, no enlarged flattened Raw gather |
| Store granule | Balanced order-preserving concatenation of4/8/16 leaves, one masked256/512/1024-key store per output unit |
| Scope | One query per unit, same raw16 grouping; no four-query output broadcast or extra prepared dense-score input |
| Safe address | Inactive leaf selects table page0 and Raw tile0 before forming pointers; masks also remain, including dense width beyond Raw capacity |
| Resource | Up to1024 FP32 result values plus concatenation intermediates/KS/masks; actual allocation and lowering recorded, no assumed UB occupancy or speedup |
| Unsupported | Existing shared-map H4/D128/finite-scale guards; reject unsupported paths, no fallback; forward-only |
| Unknown | Order-preserving concatenation lowering, allocation/copy overhead and actual store coalescing; inspect and measure in the same run |

## Paths

| Arm | Method | Functional gate | Timing |
| --- | --- | --- | --- |
| original | Frozen64-key finish | Existing raw/current/oracle gates | Output and full four-task pipeline |
| merge256 | Four leaves, balanced concat, one store | Whole-output exact and boundary | Same |
| merge512 | Eight leaves | Same | Same |
| merge1024 | Sixteen leaves | Same | Same |
| Unsupported candidate | Compile, IR or numeric rejection | Preserve traceback; other widths continue | Never time numerically invalid arms; final status incomplete on any rejection |

No shared-map preparation, full-logits layout conversion or dense copy control
is inserted into the complete pipeline. A shape change in the GM store must
not reintroduce per-element Raw gather, so vector IR markers and allocation/
copy/store evidence are exported. A marker is evidence for review, not an
automatic performance claim. Actual native instruction counts, exclusive
pipeline costs and overlap are not inferred from static IR. Formal acceptance
still uses the benchmark harness; this is score-only and TopK is unchanged.

Local validation: six new checks execute the generated leaf with bounded
memory, including inactive pointers, shuffled pages/page boundaries and zero/
unequal visibility; check balanced concatenation order; prove every output is
stored once for all widths and tails; and verify static output-view auditing.
The43 existing raw/query-pack/prefix/compiler-copy checks also remain in scope.
Generated Python/CLI/Bash syntax and repository hooks are checked. Native
device compilation, concatenation lowering, exact full outputs, counters and
performance are pending the card run. No production gain is claimed yet.
