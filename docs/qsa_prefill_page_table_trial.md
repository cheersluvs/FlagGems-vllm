# QSA scalar page-table fetch trial, 2026-10-21

Pre-coding plan. Baseline is current production source SHA256
f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c,
validated by full H800 report07993da. Packed-grid is rejected; its scheduler
is not part of this experiment.

## Project truth

| Field | Frozen choice |
| --- | --- |
| Editable | Private page-table fetch candidate/kernel, current-production A/B plumbing,8 candidate gates and profile nodes, CPU proofs, docs |
| Read-only | All production code, grid/host/chunks, ten rectangular and six compact config choices, math, dtype/layout contract, TopK, native source,30 fixtures, timer/tolerances and GPU guards |
| Hypothesis | Explicitly fetch at most two unique physical page IDs per key tile instead of one ID per key column, then broadcast them across the existing column lanes |
| Scope | Rectangular score only, both BF16 and FP8; unchanged compact path. Compile-time BLOCK_N<=PAGE_SIZE enables two-page fetch; otherwise retain general per-column fetch |
| Evidence | Actual production BLOCK_N64/128 and benchmark page196 imply at most two logical pages per tile; previous empty-grid reduction did not yield stable benefit. No profiler counter establishes page-table bottleneck yet |
| Validation | CPU execute actual load/mask branch over page boundaries, table strides/bounds/invalid IDs; exact unchanged AST outside lookup;8 H800 candidate gates plus40 production iteration nodes; full profile retained |
| Benchmark | Same30 scored-prefill cases, same FlagTree/ptxas13.4/CUDA13.4 compat; kernel mode, two alternating rounds, warmup200ms/rep100ms; score/TopK/pipeline versus native plus score/pipeline versus actual current production |
| Goal | Full-pipeline per-case >=1 or near1 (proposed0.98 floor), equal-weight arithmetic mean >1.1; experimental/iteration reports do not assert formal release validation |
| Fallback | No Torch compute fallback; no new allocation, GPU preprocessing or page-table conversion in candidate. Extra reference normalization remains test-only |

## Semantic contract

| Field | Unchanged behavior |
| --- | --- |
| Inputs | Current BF16/FP8, H/D and page sizes, padded/strided Q/cache/table, nonzero query base and chunk boundaries |
| Indexing | Identical logical_page/page_offset per column; physical IDs masked by page-table bounds and num_pages, including invalid IDs |
| Math | Exact key x query FP32 dot, ReLU before real-head reduction and existing visible-column stores |
| Output | Existing int32 selected indices/-1 padding, tie/tolerance rules, caller output and None return |
| Host/TopK | Exact current public/score launcher bytecode, chunk order/workspaces and persistent_topk identity |
| Compiler specificity | Existing Triton scalar load/where and constexpr branch only; no assembly, cache policy or NVIDIA instruction |

## Paths and gates

| Path | Change | Gate |
| --- | --- | --- |
| BLOCK_N<=PAGE_SIZE | Load first_page and optional first_page+1 with scalar address masks; tl.where broadcasts the proper ID to each column | CPU equality for every live lane and no out-of-bounds unmasked table read; H800 page64/196 with BLOCK_N64 and128 |
| BLOCK_N>PAGE_SIZE | Exact original vector fetch | H800 page16 and page64/BLOCK_N128; CPU arbitrary small page sizes |
| Invalid/tail columns | Preserve all original per-column live and physical-page masks | Short visible rows, table last page, negative/too-large physical IDs and page stride2 |
| Cache/autotune | Existing aligned rectangular entry class and original ten keys/configs; compact clone uses original six | Independent caches; source pin and actual returned per-chunk config/PTX/CUBIN snapshots |
| A/B | Isolated current-production output/workspace and real public traces | Same five-stage before/after correctness and native frozen common logits; actual pipeline timings, never sums |
| Keep/revert | Production stays current until measured | Reject unstable sub-percent effects or material active regressions; static source load counts are not memory-transaction/latency measurements |

## Implementation and delivery scope

The explicit private kernel copies the exact current rectangular signature,
cache/tuner decorators, request/query/key grid and all arithmetic/masks outside
the physical-page load. The host score launcher and public API are rebound
from the original bytecode, with independent rectangular/compact caches and
the same TopK object. The new kernel is not wired into the public export.

CPU proof executes the actual new scalar/general load branch with checked fake
memory across arbitrary page sizes, both frozen block widths, partial visible
ranges, padded table strides and invalid page IDs. Replacing just that branch
with the old load recovers exact original kernel AST. The page-table variant
uses the same30 experimental full-pipeline objective and isolated current
production A/B as packed-grid; no packed scheduler or its gate is included.

H800 candidate coverage comprises6 BF16/FP8 score tests at page16/64/196,
each checking BLOCK_N64 and128 across three request/chunk splits, plus2 real
full-entry BF16/FP8 native/FP32 comparisons. Full-entry keeps padded candidate
inputs and nonzero query base, with native's logically identical page table
normalized to contiguous columns outside timing. The default iteration profile
therefore runs48 exact nodes (40 production+8 new candidates); `full` remains
available for complete production validation before integration.

Timing remains two alternating rounds with the same stage timer, thermal/PID
guards and automatic result upload. Actual compiled configs/PTX/CUBIN/static
resources for candidate and current production are captured outside timing.
There is no claim that scalar source loads reduce actual memory transactions,
register use or latency until the H800 report is inspected.

Local validation completed:208 CPU contract/orchestration tests ran (207
passed,1 skipped). Changed-file pre-commit, Python syntax, frozen-file byte
comparisons and CPU-only CLI listing of exactly48 iteration nodes passed.
There is no local Torch/Triton/CUDA installation; H800 compilation, numerical
checks, speedup and keep/revert remain pending. Delivery uses the existing
safe child-bash runner and automatically uploads reports on success/failure.
