# FP8 source-only R32 dot issue runtime probe

## Scope board

| Item | Decision |
| --- | --- |
| Prior proof | 8ba03f061282d82e245ed95d4d60931de169c3db: R64 original158regs/56KiB, R32 sequential94regs/36KiB, R32 paired114regs/36KiB; all zero spills |
| Editable | New runtime adapter/tests/runner only; use the exact previously compiled Python sources |
| Read-only | Production source, compiler/PTXAS, TopK, numerical tolerances, workload manifest and historical probes |
| Scope | Two saved FP8 main scenarios; TR8/P1 tail uses exact production binary; no full10 acceptance or production integration |

## Implementation board

| Item | Decision |
| --- | --- |
| Variants | R64 production, R32 sequential, R32 paired. Same fixed configuration except TILE_R32 for the latter two; regenerate grid using its own tile |
| Source path | Ordinary Triton JIT of the exact saved kernel.py; no IR override, compiler edit or binary substitution |
| Prelaunch | Compile/warmup without running, require canonical executable PTX and registers/shared/spills equal the offline proof, then launch; freeze all captured chunk routes |
| Numerical | Reuse FP32 score reference, selection and pipeline checks; offset/masked/guarded FP8 boundary fixtures for both R32 source variants |
| Timing | Same fixture/buffers, direct measurements for all three pairs, score/actual pipeline separately; one common unchanged TopK callable measured A/A and reported for every pair |
| Duration | Two alternating rounds,20ms warmup/20ms rep; validate each variant before and after the entire timing block instead of repeating references for every pair |
| Guards | Existing exclusive idle H800,foreign PID and temperature80C/cooldown120s gates remain |

## Lifetime review and validation board

The pinned R32 paired TTGIR SHA256 is
ac22a8c4abce758120f233abd2191fddfbfd229ab57aadc6f2e8f67b8882143e.
Its sequential control SHA256 is
d9ec6aa88520e1fbf7dcf143fd0b5b10078b203ecf46f87f477cc99085767f17.

| Evidence | Interpretation |
| --- | --- |
| K allocation | Two slots of64x128 FP8 (8192bytes each); Q left/right are immutable128x64 FP8 allocations outside the loop |
| Prologue | K slot0 copy then async commit; loop-carried token initialized from this commit |
| Loop read | Consume committed current copy with async wait0 and CTA barrier, then both dots read the same indexed K view |
| Paired consumers | Commit left and right separately; wait1 materializes left, wait0 materializes right; right wait0 precedes next K copy |
| Ring progression | Read counter starts-1 and write counter0; each increments and wraps at2. Prologue/current reads0,1,0,...; next writes1,0,1,... |
| Reuse | PTX has a CTA barrier after final WGMMA wait0 and before next copy; current K copy occurs after both consumers; final async wait0 and barrier drain before exit |
| Resource aliasing | Shared epilogue scratch may be reused only with the emitted CTA barriers; no barrier or allocation is removed by this probe |
| Assumptions | One warp group (warps4), compiler-generated sync retained, pinned executable equivalence required; structural review is not a race-checker proof |
| Gate | Verify exact original compiled IR hashes and final PTX ordering; numerical boundary/reference checks still mandatory |

TopK is unchanged. Its standalone measurement uses the same function and common
logits; pipeline is timed as executed and is never computed by summing stages.
Direct seq32/paired32 timing isolates the scheduling effect; R64/R32 comparisons
include the tile/grid change. No speedup is inferred from resource reductions.

Local validation:17 CPU checks passed, including prelaunch rejection without
candidate execution, frozen-specialization rejection, unchanged small-tail
routing and grid recomputation with TILE_R32. The complete returned compile
archive/source snapshot passed the new proof loader and both lifetime checks.
CLI/help,shell syntax and changed-file hooks passed. GPU correctness/performance
has not been run on this Mac.


## Prelaunch lazy-resource repair

| Board | Decision |
| --- | --- |
| Evidence | Report b5b4771 stopped in first production boundary call: warmup returns a CompiledKernel without n_regs; no candidate timing was obtained |
| Scope | Runtime checker/tests/document only; preserve source, compilation proof, fixtures and all gates |
| Implementation | First verify executable PTX and metadata.shared; if register/spill fields are absent, call pinned CompiledKernel._init_handles to load the CUDA module and query resources without invoking a launcher |
| Validation | A lazy fake kernel must initialize once before launch; altered PTX/shared must reject before loading; unsupported/missing resource interfaces and spills reject before launch |
| Replay | Reuse the exact prior offline compile report. Runtime validation restarts because the first boundary was incomplete; no completed timings exist to resume |

Pinned FlagTree compiler.py creates n_regs/n_spills in _init_handles via
load_binary. Module loading and resource queries do not execute the GPU kernel.
The checker never accesses the run property or invokes the generated launcher.

Repair validation:20 CPU checks pass, including lazy module initialization,
no execution after a resource mismatch, the original prelaunch/fallback/grid
checks, and unchanged source/ABI checks. No GPU runtime result is claimed.
