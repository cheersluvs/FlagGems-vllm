# QSA query subtiles trial, 2026-10-21

Pre-coding plan. Current production remains pinned to SHA256
f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c.
The BF16 page-table candidate remains separate pending full validation. This
experiment isolates query subtiles against actual current production, without
stacking that candidate. workflow.md applies; optimization.md/deep_opt.md are
absent. No GPU profiler measurements are available locally.

## Project truth

| Field | Frozen choice |
| --- | --- |
| Operator/reference | qsa_select_paged_prefill; existing pinned native vLLM, no single torch API |
| Editable | Private query-split kernel/factory, fixed-config comparison helper, benchmark selection/diagnostics, CPU/GPU gates and docs |
| Read-only | Production/exports/YAML, all previous kernels, TopK,30 scored-prefill fixtures, native/reference tolerances and timers/guards |
| Evidence/hypothesis | FP8 TILE_R64/K_TILES16 compiled at226 registers/thread; split FP32 dot results into two query halves with immediate per-half head reduction/store, aiming to reduce simultaneously live accumulators. Register pressure is not a proven bottleneck |
| Autotune | Existing ten rectangular/six compact choices, original keys/strategy/aligned entry class; no new performance parameter. Fixed-config stage inherits each actual baseline chunk's configuration, then independent original autotuning remains for main comparisons |
| Active set/metric | Same30, FlagTree/ptxas13.4/CUDA13.4 compatibility runtime, kernel mode/us/baseline divided by candidate, pipeline primary: equal-weight mean>1.1, per-case>=1 or near1 (proposed0.98 floor); score/TopK always reported |
| Validation/build | Local AST and executable CPU contracts, Python syntax and pre-commit; GPU JIT/numerical checks on H800.40 production iteration nodes plus candidate score/full-entry nodes; full profile before integration |
| Timing | Two alternating rounds, warmup200ms/rep100ms, initial80C, case1800s/accuracy7200s/cooldown900s interval10s; current five stages plus fixed-config score comparison for FP8 rectangular calls eligible to split |
| Profiler | Same collected register/spill/shared/static PTX diagnostics; no hardware-counter/occupancy/traffic claim |
| Fallback/integration | No Torch compute fallback or GPU preprocessing/extra workspace; private experiment, no production export change |

## Semantic contract

| Field | Unchanged behavior |
| --- | --- |
| Inputs | Existing BF16/FP8 e4m3fn, parameterized H/D, query/cache/table strides, pages, nonzero query base/chunk offsets and tail rows |
| Math | Each output retains one full-D FP32 dot, ReLU per real head before head sum; no splitting/reordering D accumulation or changing tolerances |
| Output | Existing int32 indices/-1 padding, caller output and None return; invalid-page/live/store masks and unspecified tie ordering |
| Special/unsupported | Existing validation/NaN/Inf/tie treatment, forward-only/no autograd promises, no new dtype or layout scope |
| Host | Original launcher/public bytecode, grids/chunks/workspaces and persistent_topk identity |
| Torch | Test/reference only; none in candidate compute |

## Paths and gates

| Path | Condition | Implementation/config | Coverage/risk |
| --- | --- | --- | --- |
| Split | IS_FP8 and TILE_R>=32 | Two half-query loads outside key loop, keys loaded once per key tile; two sequential full-D dot/ReLU/head-sum/store epilogues; same whole TILE_R grid and ten configs | Direct fixed configs TILE_R32/64, uneven request/chunk tails, padded H/D and stride2; compiler can reschedule, so register savings not guaranteed |
| Original | BF16 or TILE_R<32 | Exact original query load/dot/epilogue, original vector page lookup | BF16 controls and FP8 TILE_R8; original path AST proof |
| Compact/early return | Existing gate | Exact existing kernel/API | Existing production checks; no benefit attributed |
| Fixed-config score | FP8 rectangular baseline has eligible tile | Rebind identical original host bytecode with per-chunk fixed entry launchers, baseline original JIT and candidate split JIT, exact same configs, separate existing workspaces | Untimed resource capture and before/after existing FP32 score checks; compare to independently tuned main stages, never conflate |
| Keep/revert | Full validated same-runtime data | Preserve only stable measured gains without material regressions; full correctness before integration | Two live query operands, extra instructions/layout moves or compiler scheduling may erase gains |

## Implementation and local validation

The split path loads two contiguous half-query operands outside the key loop.
The existing visible vector is reshaped/transposed/split into matching halves,
without extra global visibility reads. Each key tile is fetched once and used
by both full-D dots. Each half immediately applies the original ReLU/head-sum
and visible/invalid-page stores. Source ordering is not a guarantee that the
compiler will keep accumulator lifetimes short; resource/latency data decide.
The original BF16/tiny-tile branches are AST-identical after constexpr path
selection. The compact kernel, original host/public bytecode, config set/keys,
TopK, workspaces, chunking, shape set and numeric tolerances are unchanged.
No Torch compute/copy/cast fallback was introduced (the kernel `.to(tl.int64)`
is the existing Triton address cast).

The supplemental `score_matched` stage uses a separate pair of modules with
the exact original host bytecode and per-chunk fixed launchers. Both sides
inherit actual baseline public-entry configurations and their existing
independent workspace buffers. The baseline uses original JIT and candidate
uses split JIT. Unknown chunks, omitted preparation, config mismatches or new
compilation keys after freezing fail the trial. Fixed kernels are warmed and
resource-snapshotted outside timing, FP32-checked before/after timing, and the
fixed comparison is measured first when applicable. Main five independently
autotuned stages remain available in the same run to avoid another remote
round trip; promotion requires auditing the fixed and tuned results together.
No fixed-config score improvement is asserted before that audit.

Iteration correctness uses46 exact nodes:40 existing production+4 direct score
nodes+2 full-entry nodes. Direct score launches cover existing TILE_R8/32/64
configs, BLOCK_N64/128, K_TILES4/16, BF16 H4/D128, FP8 H4/D128,H1/D32,H3/D96,
three chunks across uneven request boundaries, nonzero query base and padded
page table/cache/query. Full-entry forces existing TILE_R64/K_TILES16/w4 so
the new FP8 path is actually executed, and reuses native/FP32 whole-output
selection gates with logically identical normalized native page table.
`--accuracy-profile full` still selects the complete production test file and
all new candidate nodes; iteration is not formal production validation.

Local result:221 CPU contract/orchestration tests ran,220 passed,1 skipped.
Proofs cover unchanged original path AST and per-half full-D dot/epilogue,
actual query address/mask execution, visibility-half mapping, host/TopK/cache
identity, fixed-config exact-key replay/freezing, report coverage and CLI
profiles. Frozen production/benchmark/previous test and kernel sources were
compared byte-for-byte, new Python syntax and diff checks passed, changed-file
pre-commit passed. No local Torch/Triton/CUDA is available. H800 JIT/numerics,
register/spill/shared/static PTX observations and performance remain pending.
The safe child-shell command uses a fresh checkout and automatically uploads
the result on success/failure. Production integration remains pending measured
benefit and complete validation.
