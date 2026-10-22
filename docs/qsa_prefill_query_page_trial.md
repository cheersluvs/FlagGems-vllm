# Combined BF16 page-table and joint-query trial

## Project truth (before implementation)

| Field | Frozen choice |
| --- | --- |
| Op/reference | Existing qsa_select_paged_prefill and pinned native vLLM source; no single Torch API |
| Editable | New private combined kernel/factory, candidate GPU/CPU tests, runner, this document; candidate routing/accuracy/source archive in benchmark tools |
| Read-only | Canonical production source/exports, all existing private kernels/factories, all YAML groups,30 scored-prefill cases, original timer/guard/reference/tolerances and TopK |
| Build/validation | Local syntax/AST/math/cache/YAML/acceptance/profile tests and pre-commit; full canonical production correctness file plus combined candidate boundary/mode/all-config/full-entry checks on H800 |
| Active set | Frozen30:12 upstream BF16,8 observed BF16,10 FP8; two score-skipping cases remain excluded |
| Metric | Same-compiler --mode kernel us; native baseline/candidate; separately score,TopK,sequential pipeline; same-run current-production score/pipeline and fixed-query diagnostics retained |
| Aggregation | Equal-weight arithmetic mean of30 native pipeline ratios>1.1 and every case>=1/near1 (proposed floor0.98); experimental only, not formal production acceptance |
| Runtime |2 rounds,warmup200ms/rep100ms,initial80C; case1800s,full correctness7200s,cooldown900s/10s; automatic report upload |
| Autotune | Reuse unchanged qsa_select_paged_prefill_query_joint_score17 choices, original keys/strategy/entry/pruner; BF1610 full-query choices,FP8 all17; compact original6. No new config/YAML parameter |
| Profiler | Compiled resources/static instruction sites only; no hardware counters/occupancy proof |
| Fallback | No Torch compute fallback or new production path; no new workspace/asm/cache policy/TopK change |

## Semantic contract

| Field | Frozen behavior |
| --- | --- |
| Inputs | Existing BF16/FP8 e4m3fn,H/D,query/cache/table strides,integer metadata,nonzero query bases,chunks,invalid page/live masks |
| Math | Full-D FP32 dot per head,ReLU before real-head reduction,unchanged score scaling and tolerances; no split of reduction D |
| Outputs | Same int32 selection/-1 padding,caller output identity and None return;tied indices follow unchanged native normalization |
| Promotion/scope | Existing FP32 scores and integer outputs;forward only,unchanged exceptions/unsupported/autograd/special-value coverage |
| Host | Exact canonical score/public function bytecode;independent entries/caches;canonical public export remains canonical |
| Torch | Only existing test/benchmark references perform compute; production never imports this private trial |

## Paths

| Path | Trigger | Implementation/config | Validation and risks |
| --- | --- | --- | --- |
| BF16 page fetch | not IS_FP8 and BLOCK_N<=PAGE_SIZE | Exact previously measured two scalar page-ID loads and broadcast, with original masks | Actual page-branch AST emulation; GPU page16/64/196,nondivisible boundaries,strided tables,all10 choices |
| General page fetch | FP8 or BLOCK_N>PAGE_SIZE | Exact original vector fetch | AST specialization recovers query-joint for FP8; GPU fallback/page-boundary tests |
| Full query | QUERY_PARTS1 or tiny/BF16 | Exact prior joint full-query path | Forced modes and all eligible configs; BF16 recovers prior page-table candidate |
| Split query | FP8,TILE_R>=32,QUERY_PARTS2 | Exact prior two-half full-D query epilogues,one K fetch | Forced modes,H/D and tail coverage,all17 FP8 choices |
| Compact/early returns | Existing dispatch | Exact canonical compact/identity paths | Full canonical suite plus combined compact/full-entry checks |
| Report | Existing query-joint diagnostics | Main5 stages,FP8 fixed mode1/2 comparison,BF16 short_batch3 swap,combined page-gate metadata | Experimental acceptance validates source/config/page metadata; full correctness before integration |

The only kernel change from query-joint is replacing the vector physical-page
load with the exact BF16-only page-table branch. Neither optimization alters the
other dtype's active math. No fixture-name dispatch is introduced. The runner
uses --accuracy-profile full: canonical production tests keep their real public
export, while new candidate tests explicitly bind the private combined module.
This validates the unchanged production contract and the modified candidate
paths; it does not replace production exports or claim production integration.

## Implementation and validation

The private score kernel reuses the existing17-choice joint YAML group. Removing
only its BF16 page-load branch recovers the exact joint score AST. Disabling the
FP8-only split path, removing the mode parameter and restoring the decorator
plumbing recovers the exact previously measured BF16 page-table score AST.
The actual page/mask AST was executed with checked fake memory for both dtypes,
page sizes1/7/16/64/96/196/256,block widths64/128,table strides1/2/3,invalid page
IDs and live/dead columns. No production or YAML file is changed.

The factory preserves canonical host bytecode and TopK identity, with separate
rectangular/compact entries and caches. Candidate selection is explicitly
`query-joint-page`. Reports retain native score/TopK/pipeline and current-production
score/pipeline comparisons, selected-tile full/split diagnostics and the one BF16
buffer swap. The combined metadata records source pin, page gate, config snapshot
and unchanged TopK; acceptance checks reuse the existing joint gates and require
this metadata too. Private results cannot claim formal production integration.

The full GPU profile runs two files: the unchanged complete canonical production
suite (through the real public export), and40 combined candidate tests. The latter
include6 forced query-mode/head-dim/full-entry nodes,8 page-boundary/full-entry
nodes,8 rectangular every-config nodes,6 compact every-config nodes and12 public
candidate-entry nodes across both dtypes,512/1/0MiB workspace budgets and both
rectangular/compact envelopes. All17 FP8 and10 BF16 rectangular choices are
explicitly checked, alongside all6 compact choices, padded strides,int64 metadata,
nonzero query bases,request/chunk tails and untouched invalid-column sentinels.
The full candidate-entry checks also compare canonical output and FP32 reference
without rebinding the real production export. These40 nodes are required to pass;
missing,duplicate,skipped or incomplete candidate families fail the full gate.

This is full canonical contract validation plus targeted full coverage of the
modified candidate paths, not a claim that every canonical parameterized test
was rerun through the private module. A separate iteration profile remains54
nodes for later work (40 canonical +14 candidate), but the delivered runner
explicitly requests full,not iteration. The30 benchmark definitions and timing
parameters remain frozen. No focused control cycles are repeated in this run.

Local validation:98 related CPU tests passed, including7 new composition/gate
checks. New kernel syntax, source pins, unchanged YAML/production/TopK identities,
profile/CLI plumbing,shell syntax,diff checks and changed-file pre-commit passed.
No local Torch/Triton/CUDA is available: H800 JIT,numerics,selected configurations
and all30 performance results remain pending. Keep the candidate private until
those results have been audited; no production integration is part of this change.

Run `bash tools/run_qsa_prefill_query_page_h800.sh --flagtree /opt/flagtree` from
a clean checkout under the existing QSA vLLM0.31 activation and QSA_STOCK_PYTHON.
The verified FlagTree/ptxas13.4/CUDA13.4 compatibility profile is routed explicitly;
no compiler is installed into or replaced in the virtualenv. The existing runner
selects an idle H800 and uploads complete or partial JSON/TXT/TSV reports to a
result branch under a `flagtree134-query-page-*` report directory. Run the shell
entry in a child bash so test/upload failures preserve the interactive container.
