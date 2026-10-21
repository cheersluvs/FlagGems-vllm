# QSA joint full-query/subtile tuning trial, 2026-10-21

Pre-coding plan. Report5045a0f shows useful fixed TILE_R64 and32/w8 gains but
some32/w4 regressions. The next hypothesis is choosing the algorithm jointly
with the existing tiles, preserving every original full-query choice. Current
production source stays SHA256
f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c.
No BF16 page-table candidate is stacked. workflow.md applies; optimization.md
and deep_opt.md remain absent in this checkout.

## Project truth

| Field | Frozen choice |
| --- | --- |
| Op/reference | qsa_select_paged_prefill; existing pinned native vLLM, no single torch API |
| Editable | New private query joint kernel/factory/policy, new YAML group only, benchmark selection/fixed-comparison/control diagnostics, CPU/GPU gates, docs |
| Read-only | Production source/exports, all existing YAML groups, old trial kernels/tests, TopK,30 scored-prefill cases, numeric tolerances/reference and timer/guard implementations |
| Hypothesis | QUERY_PARTS=1/2 jointly tuned per existing runtime key chooses full-query for slow regions without case-name or backend-specific register limits |
| Autotune | New qsa_select_paged_prefill_query_joint_score group: original10 with QUERY_PARTS1 plus7 TILE_R>=32 twins with PARTS2; BF16 early prune retains only original10 PARTS1; same original keys/strategy/aligned entry class; no empty pruned sets |
| Validation/build | Local AST/math equivalence, YAML/pruner/cache/host contracts, syntax and pre-commit; GPU direct PARTS1/2 score and forced full-entry tests plus40 production iteration nodes; full suite before integration |
| Active/metric | Same30, same FlagTree/ptxas13.4/CUDA13.4 compatibility runtime; kernel us, baseline/candidate, equal-weight native pipeline arithmetic mean>1.1 and every case>=1 or near1 (proposed0.98 floor); score/TopK/current-production A/B retained |
| Fixed comparison | FP8 chosen joint tile is replayed with PARTS1 versus2, exact same other parameters and separate existing workspaces; labeled distinctly from old production-config comparison |
| BF16 control | Existing short_batch3 receives an additional pipeline buffer-swap replay with the same timer/2 rounds; record actual configurations and buffer layouts, before/after FP32 output checks; diagnose timing/buffer sensitivity without assuming a cause |
| Runtime/timeouts | Two rounds/warmup200ms/rep100ms/initial80C; case1800s/accuracy7200s/cooldown900s interval10s; automatic report upload |
| Profiler | Actual compiled resources/static PTX only, no hardware counters/occupancy or proven bottleneck claim |
| Fallback/integration | No Torch compute/copy/cast fallback, new workspace, custom assembly or production export; private experiment only |

## Semantic contract

| Field | Unchanged behavior |
| --- | --- |
| Inputs | BF16/FP8 e4m3fn with existing H/D, pages, query/cache/table strides, nonzero query bases/chunk offsets, invalid-page/live masks |
| Math | PARTS1 recovers original path; PARTS2 recovers previous split source for FP8 eligible tiles. Full-D FP32 dot per output, ReLU before real-head sum, no source split of D accumulation or tolerance change |
| Output | Existing int32 selection/-1 padding, caller output and None return; tie order unspecified |
| Unsupported | Existing validation errors and forward-only/autograd/special-value limitations; no extra dtype/layout promise |
| Host | Original score/public bytecode, grid/chunks/workspaces and persistent_topk identity; controls use existing buffers |
| Torch | Only existing tests/reference use Torch compute |

## Paths and gates

| Path | Trigger | Implementation/config | Coverage and risk |
| --- | --- | --- | --- |
| Full | QUERY_PARTS1, or BF16/tiny tile | Original query/dot/epilogue; original10 geometries remain available | Forced GPU PARTS1, AST original proof, BF16 controls; more candidates do not guarantee tuner always finds faster result |
| Split | FP8,TILE_R>=32,QUERY_PARTS2 | Exact previous two-half queries and immediate epilogues; Q outside key loop, one K fetch | Both modes at fixed32/64 tiles, H1/D32,H3/D96,H4/D128, padded strides/request/chunk tails; existing compiler scheduling risk |
| Compact/early returns | Existing host condition | Exact unchanged compact/identity path | Existing production40; no split gain attributed |
| Fixed diagnostic | Selected FP8 rectangular tile eligible | Compare PARTS1/2 on chosen joint tile, normalize only QUERY_PARTS for exact tile matching | Separate supplemental score_matched label; no silent conflation with old fixed production-config metric |
| Control replay | Existing BF16 short_batch3 only | Swap score/TopK workspace/output ownership between canonical and candidate pipeline functions; same math and timer | Record config equality ignoring PARTS1, buffer shape/stride, paired rounds and FP32 validation; does not establish root cause automatically |
| Keep/revert | Audited H800 result | No production merge before stable gains/full correctness and control review | Extra cold compilation/tuning cost; no promised runtime or performance gain |

## Implementation and local delivery

QUERY_PARTS is a new constexpr performance parameter in the private kernel.
The added YAML group explicitly lists17 choices,10 full-query and7 eligible
split twins. The early pruner keeps all17 for FP8 and exactly the10 original
full-query geometries for BF16. Existing groups are byte-identical: removing
only the inserted experimental YAML block restores the complete original file.
The rectangular factory preserves the pruner when isolating its aligned entry;
compact keeps its original six choices/JIT/entry behavior. Runtime trace checks
reject ineligible selected modes. Mode search can still be noisy; the existence
of full-query choices is not a promise the tuner always selects a faster one.

Supplemental score_matched now compares QUERY_PARTS1 versus2 using each chosen
joint tile. Both sides use the same joint JIT and exact same other parameters;
small/compact tails stay original. It is explicitly labeled joint_tile_full_query
and query_parts_pair[1,2], distinct from the prior trial's original-production
configuration comparison. Main native and true current-production score/TopK/
pipeline comparisons retain their original meaning. Fixed launchers reject
unknown/unprepared keys, check every non-mode parameter, and freeze compiled
records outside timing. Old fixed-config behavior remains the default for old
trial selections.

The one BF16 control uses canonical code with candidate buffers versus joint
code with canonical buffers, including corresponding TopK scratch/output. Its
pipeline_buffer_swap row has before/after full FP32 selection checks using the
correct swapped output owners, plus the same paired timer/guard. Reports contain
workspace/output shapes, strides, alignment, actual per-chunk configs, equality
of all non-mode config fields and identical TopK binding. Metadata reads do not
hash uninitialized scratch. Config equality or a change on swapping does not
alone establish a causal explanation for the earlier0.9747x observation.
The supplemental control is required by the experimental report gate.

The46 iteration nodes are40 original production plus6 new candidate nodes.
Each FP8 direct node explicitly executes both modes at TILE_R8/32/64 across
request/chunk tails, query_base97, padded strides and H1/D32,H3/D96,H4/D128.
Both full-entry FP8 modes are forced through the existing64/16/w4 choice and
native/FP32 checks; BF16 checks mode1. The full profile keeps all production
tests before integration. No numerical tolerance, workload or TopK was changed.

Local validation:228 CPU contract/orchestration tests ran,227 passed,1 skipped.
AST proof recovers the exact prior kernel after only removing the new parameter,
mode predicate and tuner plumbing. YAML/config pruning, actual host/TopK/cache
identity, matched-mode records/freezing, control buffer ownership and selection
validation output mapping, acceptance gates and CLI profiles passed. Production
source SHA, all old YAML groups, frozen benchmark/previous GPU tests/kernels,
new Python syntax, diff whitespace and changed-file pre-commit passed.
No local Torch/Triton/CUDA is available. H800 JIT/numerics, selected modes,
performance, additional cold tuning time and the BF16 control remain pending.
Private candidate only; no production exports or TopK implementation changes.
The child-shell runner uses a fresh checkout,2 rounds and automatic report upload.
