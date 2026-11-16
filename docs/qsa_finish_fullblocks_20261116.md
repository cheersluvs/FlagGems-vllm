# Full-visible output block probe

## Project truth

| Field | Contract |
| --- | --- |
| Evidence | Report6e5df3ba: blocked1024 long finish321.79/full496.51us, compact regresses; three extracted tile loops remain |
| Editable | New full-block source/tests, width-runner variant mapping, blocked-mode entry and CPU gate |
| Read-only | Native query16, QW/K packing, original64 finish and blocked1024 source, production/TopK/harness/compiler |
| Active set | compact128/W1088 and long100k112/W29696; independent full/tail CPU and device boundary inputs |
| Build | Existing isolated metadata correction and query16 bitcode; no new compiler changes |
| Validation | Whole outputs exact versus original64; prior raw/current tolerance, boundary and postchecks |
| Timing | Original64 vs blocked1024 vs full1024, output and full four-task score; two reversed rounds/six active samples |
| Metric/aggregation | Device us, baseline/candidate ratios, median of round means; no formal acceptance |
| Hardware | Same existing PipeUtilization/Memory selection, IR/liveness/store audit, no timing inferred from static counts |
| Timeout/config | Existing1800s/upload; diagnostic sweep, no production libtuner/registration; no torch fallback |

## Contract

| Field | Contract |
| --- | --- |
| Inputs | Frozen Raw[group,tiles,16,64] FP32, FP16 KS, shared int32 map/page192, per-row visible; H4/D128 guards |
| Output | Same dense FP32 Raw*float(KS), -inf after each row's visible bound |
| Full condition | (block+1)*1024 <= width and <= visible[row]; implies every Raw/KS address exists |
| Full method | Unmasked regular-stride Raw read using logical tiles directly; page lookup and KS loads remain, one flat1024 store |
| Partial method | Exact prior blocked1024 safe-address/masked source, no approximate visibility or skipped output |
| Branch scope | Separate JIT void helpers own local values and stores; branch carries no tensor result between iterations |
| Boundaries | Full block, unequal/zero visible, exact1024 boundary, partial second block, row17, shuffled pages and width beyond Raw capacity |
| Unsupported/precision | Existing guard scope, forward only; no arithmetic precision change or torch compute fallback |

## Paths

| Label | Interpretation | Gate |
| --- | --- | --- |
| original | Original64 output | Existing reference |
| blocked1024 | Frozen previous2D masked output | Exact comparison |
| full1024 | Full-visible block helper plus unchanged masked tail helper | Same, including independent full/tail boundary |
| complete | Same QW+K+native query16+selected output | All four tasks charged |

The compact case has no fully visible1024-key block and is a useful tail-only
control. It is not routed to this candidate in production. The fast branch is
an operator-level hypothesis: inspect Raw's generated copy/loops, not assume
the branch eliminates all page/KS extraction or guarantees speedup. No TopK,
overlap, CANN-stage timing or formal harness acceptance is inferred here.

Local validation adds three tests to the existing53: execute both generated
sources with bounded memory on rows17/width2057 and mixed visible lengths,
verify exact outputs and both branch choices, prove the tail AST is unchanged,
and check the fast helper has no masks, conditional tile selection or loops.
The device script also validates an independent synthetic full/tail boundary
against a CPU Raw*KS oracle before timing. Device lowering, branch behavior,
resource usage and speed remain pending; production is unchanged.
