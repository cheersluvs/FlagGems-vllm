# FP8 paired-issue score-width crossover probe

## Scope board

| Item | Decision |
| --- | --- |
| Evidence | Full10 report8de856: changed5 score1.0138x,full10 score1.0066x; long100k1.0393x is strongest; small positive changes can overlap A/A variation |
| One hypothesis | Benefit depends on scoring width; isolate width before introducing a dispatch threshold |
| Editable | New controlled-fixture adapter/tests/runner/document only; production,compiler,TopK and formal benchmark table unchanged |
| Matrix | Width8192/16384/24576/29696;single request,4519 query rows,H4,D128,page196,FP8;original R64 versus source paired R32 |

## Implementation board

| Item | Decision |
| --- | --- |
| Seed | Verified long100k fixture;take its first4519 Q rows;reuse the exact same152-page cache and1x152 table for all points |
| Metadata | Starts[0,4519],output[4519,512],visible[row]=width-floor((4518-row)/4),max_seq_len=width*4;both width and visible bound change together |
| Grid | One public chunk per point under512MiB budget;fixed N64,K16,P2,warps4,stages2;only rows per CTA differ between production and candidate |
| Provenance | Enforce seed fingerprint first;fresh diagnostic input fingerprints recorded before/after;Q/cache/table/starts fingerprints must be identical across all four points |
| Runtime | Reuse full10 source-JIT prelaunch executable/resource gates and unchanged two-layout boundary evidence;reuse boundary checks across widths with explicit receipt |
| Timing | Two alternating rounds,20ms warmup/rep;score,common unchanged TopK,actual pipeline and four A/A controls per point |

## Validation board

| Item | Decision |
| --- | --- |
| Numerical | Existing FP32 score and selection/pipeline checks before/after timing at every width;no tolerance changes |
| Artifacts | Same offline executable protocol/resources at all points;export per-width JSON/PTX/TTGIR in an archive;print per-point data |
| Guards | Existing exclusive GPU,foreign-PID and thermal guards remain;parent verifies historical seed report before running diagnostic worker |
| Limits | Synthetic rectangular causal ramp,not a new captured inference fixture. Four points only bracket a crossover;no threshold or production promotion inferred automatically |

The original full10 manifest identifies the seed and compiler only. Diagnostic
inputs are explicitly generated and fingerprinted on the card;they are not
represented as historical captured fingerprints. The unchanged cache allocation
stays fixed,while the amount of its prefix actually read necessarily varies.

Local validation:28 CPU checks passed,covering fixed-length/causal visible
shifts,cache capacity/one-chunk budget bounds,adapter dispatch isolation,
nonmisleading diagnostic summary and the existing source-JIT/full10 gates.
CLI,shell syntax and changed-file hooks pass. GPU measurements remain pending.
