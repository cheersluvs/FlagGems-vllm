# Full10 FP8 restricted R32 paired-issue regression

## Scope board

| Item | Decision |
| --- | --- |
| Evidence | f64f09f: two-case source-JIT correctness passed; R64/paired score1.0077x and1.0411x; tile-only R32 regresses |
| Editable | New full10 manifest/adapter/tests/runner and generic last-variant A/A indexing in existing runtime driver |
| Read-only | Production, compiler, TopK, compile proof, numerical tolerances and original10 workload definitions |
| Candidate domain | Single request,H4,D128,FP8,R64/P2,N64,K16,warps4,stages2,ctas1; only TILE_R32 and existing right-dot movement |
| Controls | Other configurations,small tails,compact route and two multi-request cases retain original binaries; multi-request ABI lacks matching offline proof |

## Implementation board

| Item | Decision |
| --- | --- |
| Manifest | All10 cases/inputs/configs from verified production report78c3ca4f,source314a9de2/SHA0d7e80e2; cross-check existing three-case profile manifest |
| Config mapping | One distinct recorded choice repeats across chunks; two distinct choices follow first-use main/tail chunk order. Assert count/geometry and cross-check profiled cases |
| Adapter | Reuse verified source-JIT runtime worker with two variants and one direct comparison; exact compile proof/prelaunch resource checks retained |
| Boundaries | Existing two single-request layouts and three visible/page/tile scenarios; candidate16+16+3 split across35 offset rows; unchanged original masks/canaries |
| Inactive routes | Compare PTX/CUBIN/configs between both sides; assert same TopK and buffers |

## Validation board

| Item | Decision |
| --- | --- |
| Matrix | 5 changed cases:budget_exact,dominant_long,tail112,tail113,long100k. 5 original controls:first_chunk,two_balanced,two_skewed,short_tail,compact |
| Local | Complete manifest/geometry/input/config validation,profiled-case agreement,policy/fallback tests,source-JIT regression tests and CLI/shell/hooks |
| Card | Before/after FP32 score and selection/pipeline checks on all10; runtime executable/resource equality; four score/pipeline A/A controls per case |
| Timing | Two rounds,20ms warmup/rep; score,unchanged common TopK and actually executed pipeline reported separately; original GPU guards retained |
| Decision | Full10 and changed5/control5 arithmetic means reported; no production promotion from a compile-only result or two-case success |

The multi-request cases remain in the benchmark and are tested, but are not
claims of validating the new paired schedule for a different ABI. This round
does not autotune new choices or extend the candidate to unrelated FP8 tiles.

Local validation:25 CPU checks passed, including full10 fixture/profile agreement,
changed5/control5 coverage, multi-request fallback, isolated adapter state and
actual returned compile-proof loading under two runtime variants. CLI lists all10;
shell syntax and changed-file hooks pass. No full10 GPU result is claimed yet.
