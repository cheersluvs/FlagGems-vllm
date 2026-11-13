# Query16 raw score finalization diagnostics

## Frozen project truth

| Field | Contract |
| --- | --- |
| Evidence | Report594e665: query16 full raw pipeline34.78/766.63us; finish22.56/590.41us in compact/long100k |
| Editable | New finish diagnostic helper/tests and mode routing only |
| Read-only | Query16 native C++, QW/K pack methods, full finish, production/current INT8/TopK/harness, installed compiler |
| Build | Existing raw runner, query16 bitcode only, existing isolated compiler metadata correction |
| Validation | Exact diagnostic outputs versus independent CPU controls; original query16 correctness/oracle/tail gates and postchecks remain |
| Timing | Two reversed rounds, six active samples, same synchronized trace and strict task counts; us, lower is better |
| Active set | compact128/W1088 and long100k112/W29696; short synthetic rows17/W257/max65 for all output controls |
| Aggregation | Per-round sample mean then median; raw signed differences retained, no additive percentage attribution |
| Hardware | Full output on compact; fill/raw/ks/full/copy1024 on long100k, PipeUtilization+Memory two iterations |
| Compiler evidence | Preserve every source/IR and check required input read roots and an output sink remain in pre-native IR |
| Timeout | Existing1800s worker, automatic report upload even after diagnostic failure |
| Autotune | Diagnostic controls, no production dispatch/config/registration change |
| Fallback | Torch only in preparation/oracles; no production compute fallback |

## Alignment contract

| Field | Contract |
| --- | --- |
| Reference | Frozen query16 output and existing raw/current integer/half CPU oracle; every diagnostic has its own expected output |
| Inputs | Raw[ceil(rows/16),tiles,16,64] FP32; positive FP16 KS, shared int32 paged map/page192, per-row visible, dense FP32 output |
| Full | Unchanged query16 single-row64-key capped finish; exact mask, raw*float(KS), -inf outside individual visibility |
| Controls | Change one access/arithmetic group where possible; write the same dense rows*width output to keep consumed data observable |
| Promotion | FP16 KS to FP32, FP32 add/multiply; no dot/precision change |
| Boundaries | Zero and unequal visible, row17, max-visible65, dense width257 beyond Raw capacity, shuffled physical map, signed weights |
| Dense source | Precomputed diagnostic full output outside timing; its preparation is NOT a production speedup or charged complete pipeline |
| Unsupported | Non-shared maps, nonfinite/overflow, nonpositive scales, backward; existing raw guard rejects |
| Unknown | Compiler changes and overlap make differences conditional; no inference of exact exclusive instruction costs |

## Diagnostic paths

| Arm | Work retained | Comparison meaning |
| --- | --- | --- |
| fill | Loop, visibility and zero/-inf store,64 keys per unit | Control/masking/output-write floor for this implementation |
| raw | fill+visible Raw read | Raw read path increment over fill |
| ks | fill+page/KS read and FP32 conversion | KS access path increment over fill |
| sum | Both input paths with FP32 addition and same masking/store | Makes both loads live without multiplication; includes the addition |
| full | Frozen Raw*KS finish | Actual reference output stage |
| no_visible | Raw*KS with global max-visible bound instead of per-query visibility | Diagnostic only; changes read coverage/output mask, not a valid optimized operator |
| copy64 | Precomputed dense score copy, original64-key unit/grid | Layout/control comparison, no dynamic visibility/page/KS work |
| copy1024 | Same precomputed dense score, contiguous1024-key units/max20 programs | Larger-copy throughput control; changes loop granularity as well |
| complete | Frozen QW+K+query16 Cube+original full finish | All four device tasks charged, no prefix/dense-preparation shortcuts |

Every prefix writes a full-size GM sink and must match its own CPU oracle.
IR roots are followed through memref views to copy/load/store operations to
reject a missing data read or output write. This is pre-native liveness
evidence, not an ISA or traffic proof. Only the full and complete arms implement
the score contract. In particular, full-minus-sum is multiply-vs-add plus any
lowering difference; full-minus-no_visible changes visibility and read coverage;
copy64-minus-copy1024 is not an isolated scalar-loop time. Differences can be
negative because execution overlaps and generated code changes. No differences
are clamped or added into a purported whole-kernel breakdown. TopK is unchanged
and is outside this diagnostic; formal acceptance still uses the harness.

Local validation: six new finish-prefix checks plus37 existing raw/query-pack/
compiler-copy checks. The new tests execute generated bodies with bounded
scalar memory, prove exact oracle values and single output coverage for every
arm, exercise zero visibility and dense tails, reject missing IR reads/sinks,
and preserve negative timing differences. The liveness parser also recognizes
all four input reads and the output write in the saved actual query16 finish
IR from report594e665. Device compilation, outputs, counters and timing remain
pending. No optimization or production speedup is claimed by these controls.
