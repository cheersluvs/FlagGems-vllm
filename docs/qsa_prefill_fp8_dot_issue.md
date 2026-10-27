# FP8 source-level paired dot issue probe

## Scope board

| Item | Decision |
| --- | --- |
| Hypothesis | Issue the independent right-query full-D dot immediately after the left dot, before the left epilogue; inspect whether waits/layout transitions change |
| Editable | New diagnostic source generator, offline compile driver, tests and runner |
| Read-only | Production source SHA256 0d7e80e240966565a7c78151c8344f17e0627e73920f5229aadcc0f0fb42f10e, compiler, assembler, launch options, masks, TopK and tolerances |
| Excluded | Producer-first IR overrides, compiler patches, manual barriers, FP8 buffer edits and BF16 candidates |

## Implementation board

| Item | Decision |
| --- | --- |
| Source | Extract original FP8 kernel and store helper; move exactly one existing right-dot assignment. No added/reassociated arithmetic or casts |
| Target | Existing H4/D128, TILE_R64, BLOCK_N64, K_TILES16, QUERY_PARTS2 main specialization; no new autotuning |
| Compiler | Original installed FlagTree/PTXAS13.4, explicit SM90 ASTSource, CUDA_VISIBLE_DEVICES empty; existing scoped CUDA hint selection only |
| Input | Original TTIR ABI/options from successful FP8 compile report; pin original TTIR and metadata SHA256 |
| Controls | Compile production source first; canonical executable PTX must reproduce captured production; freeze source/compiler/assembler identities |

## Validation board

| Item | Decision |
| --- | --- |
| Local | Source movement/reversibility, complete captured ABI, negative gates, CLI and shell checks |
| Card | Two CPU-only source compilations; display registers/shared/spills/MMA/dot/wait/barrier counts and ordering |
| Gate | Reject spills or increased registers/shared; require same dot/MMA work and changed executable PTX; changed synchronization needs later lifetime review |
| Limits | Compile feasibility is not numerical correctness or speedup; no automatic GPU run or production integration |
| Timing | None now. Later short A/B must show score, TopK and measured combined pipeline separately |

The earlier IR producer-first candidate is abandoned for production: its gain
requires compiler-level scheduling. This probe makes a different, source-only
change. Source order alone does not guarantee hardware overlap. An unchanged
binary or resource regression ends this candidate without a GPU test.

Run the shell runner with `--source-report /tmp/qsa-fp8-producer-h800.qRmT0K
--flagtree /opt/flagtree`. It uploads reports even when compilation fails.
The two prior representative cases share the same captured main specialization;
compile it once per side, not once per case. Small tails and other FP8 shapes
remain outside this compile probe. No runtime correctness is claimed for them.

Local validation: 11 CPU checks passed, including source AST equivalence after
normalizing the single moved assignment and rejection of changed work/resources.
The actual archived TTIR/options passed preparation: 17 runtime arguments and
10 original constexpr values, including num_requests=1 and stride_table_page=1.
MLIR f8E4M3FN/f32 pointer types are mapped to the pinned ASTSource's
fp8e4nv/fp32 spelling. This Mac has not compiled or run either GPU kernel.


## TILE_R32 follow-up boards

| Scope | Decision |
| --- | --- |
| Evidence | Report64f812a: R64 paired issue works (wait1/0), but registers158->206, shared56KiB unchanged, zero spill; no runtime measurement |
| One hypothesis | Halve query rows per CTA to reduce simultaneously live accumulators while preserving paired issue |
| Editable/read-only | Extend this diagnostic only; production, compiler, TopK and workloads remain unchanged |

| Implementation | Decision |
| --- | --- |
| Three compilations | R64 original source control; R32 original order control; R32 paired issue candidate |
| Changed constant | Only TILE_R64->32 for the two R32 jobs; BLOCK_N64/K_TILES16/PARTS2/warps4/stages2 remain fixed |
| Comparison | Compare instruction work between the two R32 jobs; R64 and R32 have different work per CTA and must not be compared by raw MMA counts |
| Source | Same original helper and kernel; paired moves only the existing right dot assignment |

| Validation | Decision |
| --- | --- |
| Compile gates | Original R64 must reproduce prior executable/resources; R32 paired must preserve same-tile dot/MMA work, have wait1 before first epilogue and wait0 afterwards, zero spill, registers<=158 and shared<=57344 |
| Control-relative cost | Print R32 sequential versus paired resource delta; allow an increase versus R32 sequential only within the original R64 resource ceiling, since eliminating all overlap cost is not required |
| Limits | Smaller tile approximately doubles main CTA count/query reloads for aligned requests. Lower per-CTA resources do not establish speedup; changed pipeline needs lifetime review before runtime |
| CLI | Add --tile32 to run these three compilations; omit it to replay the historical two-job experiment |
