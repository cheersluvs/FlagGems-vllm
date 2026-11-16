# Direct two-dimensional output probe

## Project truth

| Field | Contract |
| --- | --- |
| Evidence | Report20b81935: join/transpose/reshape regressed256/512 and overflowed UB1024; retire that expression |
| Editable | New generated blocked source/tests, shared diagnostic runner parameterization, routing, missing-trace retry |
| Read-only | Native query16 Cube, QW/K pack, original64-key output, production, TopK, harness, installed compiler |
| Build | Existing query16 native bitcode and isolated metadata correction; no new compiler patch |
| Validation | Whole-output exact, existing half/current score oracle and rows17/width257/visible65 boundary; postchecks |
| Active set | compact128/W1088 and long100k112/W29696 |
| Timing | Output and full four-task pipeline, two reversed rounds/six active, per-round mean then median us; no formal acceptance |
| Hardware | Original/best on compact, all valid arms on long, existing PipeUtilization/Memory two iterations |
| Evidence | All sources/IR, input/sink liveness, scalarization markers, output view/write sites and allocation sites |
| Autotune | Explicit256/512/1024 diagnostic sweep, no production config/registration change |
| Timeout/fallback | Existing1800s worker/upload; no production torch fallback; torch only diagnostic preparation/oracles |

## Contract

| Field | Contract |
| --- | --- |
| Input/output | Same query16 Raw FP32, positive FP16 KS, shared int32 page table/page192, individual visible; dense FP32 score/-inf |
| Arithmetic | FP32 Raw*float(KS), original visibility; no change to dot or numerical precision |
| Layout | Direct [4/8/16 key tiles,64 keys] Raw view with strides[1024,1]; output with strides[64,1] |
| Address expressions | Construct pointers directly from separate tile/column aranges; no reused precomputed2D offset tensor |
| Safe inactive addresses | Select tile0/page0 before masked reads; inactive tiles never form speculative out-of-range bases |
| Tail | Zero/unequal visible, rows17, dense width beyond Raw tiles, partial key/page and shuffled pages |
| Resource | Up to1024 values plus2D addresses/masks/scales; actual UB lowering unknown until compilation; no join/transpose/reshape chain |
| Unsupported | Existing raw H4/D128/shared-map/finite-scale guards; forward-only, no fallback |

## Paths

| Arm | Method | Gate |
| --- | --- | --- |
| original64 | Frozen single-query output | Existing reference |
| blocked256/512/1024 | One2D load per Raw/KS operand and contiguous2D output store | Whole-output exact and boundary; independent compile/rejection |
| complete | Same QW+K+native Cube+selected output | Four tasks charged |
| missing trace | Retry whole capture at most3 times; preserve missing-file reason | Never invent timing samples |

Reusing the width-sweep runner keeps timing/counters and comparisons identical.
Width labels remain merge256/512/1024 in the table for runner compatibility;
the report records implementation=blocked2d and distinct source artifacts.
Pre-native regular views do not prove efficient final instructions. The2D
mask/broadcast lowering itself may regress or fail, and is explicitly tested.
No production integration, TopK or formal harness speedup is claimed.

Local validation includes the49 existing checks and four new checks executing
the actual generated2D body on bounded scalar memory. They cover all output
positions, zero/unequal visibility, inactive pointers, row17, partial widths,
shuffled pages and page crossings, and verify no concat/transpose/reshape is
present. A mock profiler export test verifies exactly three missing-file
attempts before failure. Device lowering, UB usage and speed remain pending.

Report f76af6b2: all widths compiled but the IR gate found an extra Out read.
The two-dimensional masked store lowered to an old-Out copy plus select and
writeback; no candidate reached correctness validation or timing. Keep the
strict gate. The revised expression flattens the contiguous value tensor once
and uses a one-dimensional output pointer/mask. This is a single view reshape,
without join or transpose chains. Actual elimination of the Out read remains
a device-IR gate, followed by exact boundary and post-timing validation.
