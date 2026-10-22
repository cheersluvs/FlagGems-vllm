# Combined QSA short-batch self-comparisons

## Project truth (before implementation)

| Field | Frozen choice |
| --- | --- |
| Op/reference | Existing qsa_select_paged_prefill, audited query-joint-page candidate and pinned native vLLM; no single Torch API |
| Editable | New diagnostic/CPU tests/child-shell entry/this document; optional CLI/worker/source archive hook and report label in existing runner |
| Read-only | All production and private kernels/factories/YAML, workload definitions, original timer/guard/reference/tolerances, TopK, previous diagnostics and accuracy profiles |
| Validation | Local syntax/CPU scheduling/owner/CLI/record tests and pre-commit; on H800 existing54 iteration correctness nodes plus before/after FP32/native checks for all diagnostic rows |
| Active set | Only observed_short_batch3_bfloat16; no30-case acceptance claim |
| Metric | Same verified FlagTree/ptxas13.4/CUDA13.4compat,--mode kernel us; original2-round AB/BA timer per comparison; all score/TopK/sequential pipeline data |
| Schedule/aggregation |4 cycles;4 buffer conditions x3 stages x3 comparisons =144 pairs total; per-row medians/ratios and raw samples; do not replace native/current-production main rows |
| Runtime | warmup200ms/rep100ms,initial80C; case1800s,accuracy7200s,cooldown900s/10s; original automatic report upload |
| Autotune | Public tuning unchanged; diagnostic fixed replay of existing TR8/BN64/KT4/W4/stages2/ctas1 choice; combined QUERY_PARTS1. Explicit diagnostic exemption,no new config |
| Profiler | Existing compiled-resource records/metadata only; no new hardware counter or cache-state claim |
| Fallback | No Torch compute fallback,new kernel,workspace,cache flush,policy or TopK implementation change |

## Semantic contract

| Field | Frozen behavior |
| --- | --- |
| Inputs | Exact BF16 short_batch3 fixture,request lengths,visible blocks,page196,padded/table strides and chunk geometry |
| Math | Canonical/combined existing full-D FP32 score,ReLU before head sum; combined BF16 scalar page gate unchanged |
| Outputs | Existing int32 selection/-1 padding,caller output buffers,None return; per-side outputs checked immediately before another side can overwrite |
| Promotion/scope | Existing FP32 scores/integer metadata and outputs;forward only;no new dtype/layout/autograd/special-value guarantee |
| References | Original score and selection helpers/tolerances only,outside timing;native common TopK input immutable |

## Execution paths

| Path | Trigger | Implementation | Checks/risks |
| --- | --- | --- | --- |
| Original run | No new flag | All existing benchmark and query controls unchanged | Regression tests |
| P/P | production_self | Same fixed production module on both sides | Shared A/A or B/B reuses exact callable; A/B or B/A same kernel,different owner |
| C/C | combined_self | Same fixed combined module on both sides | Same callable identity guarantee on shared buffers |
| P/C | production_vs_combined | Fixed original production versus audited combined kernel | Same non-mode tile constants,original TopK |
| Conditions | Each4-cycle block | Existing balanced A/B,B/A,A/A,B/B condition order;rotate stages and rotate/reverse pair order | Preserve order/timestamps;does not reproduce identical device cache state |
| Report | Success/failure | Nested combined_self_controls JSON and *.combined-controls.tsv plus existing native/current-production tables | Retain abnormal self ratios;no threshold forcing self ratios to1;partial failures uploaded |

P and C refer to production and combined code. A and B are existing canonical and
candidate score/scratch/output banks. Self-comparisons deliberately use the same
module and,for shared banks,the identical closure object;they must not create two
independently compiled copies masquerading as a self-control. Both real modules
are warmed and fixed before any timing. No GPU allocations are added by this
control setup. Checks cover source hashes,module/TopK identity,tile equality,
compiled records before/after and all per-side outputs. Numerical or source
changes fail;timing ratios away from1 are retained as diagnostic evidence.

This follow-up only investigates whether code/buffer/order comparisons reproduce.
It neither proves a cache/compiler/address root cause nor updates30-case means.
The previous normal and swapped results remain available without substitution.

## Delivery and interpretation

The dedicated entry is
`bash tools/run_qsa_prefill_combined_controls_h800.sh --flagtree /opt/flagtree`
under the existing QSA vLLM0.31 activation and QSA_STOCK_PYTHON. The runner selects
an idle H800 with the original guards and routes the verified FlagTree/ptxas13.4/
CUDA13.4compat libraries. No compiler installation or kernel change is needed.
It runs the unchanged54-node iteration correctness profile (40 canonical plus14
combined),then only short_batch3. The previously passed full261+1-skipped result
remains separately archived; it is not relabeled as part of this diagnostic.

The original native score/TopK/pipeline,current-production score/pipeline and
buffer-swap rows remain in triton-stage-ab.tsv. Supplemental self-controls are
in the per-case JSON under combined_self_controls and *.combined-controls.tsv,
including source SHA256/paths,actual fixed resources before/after,buffer pointers/
shapes/strides,comparison/module/callable identity,cycle/order/timestamps,validation
before/after,and every raw timing/paired ratio. The144 supplemental comparisons
produce576 timing samples; with the six original timed stage pairs,600 samples
are expected. Forty-eight supplemental rows use the identical callable on both
sides. Score and TopK checks reuse existing helpers. Pipeline checks launch one
side once,then inspect that exact result with the existing FP32 checker and no
further output-writing launches before inspection.

The existing runner uploads complete or partial reports under
flagtree134-combined-controls-*. Pending rows and measured samples are saved
before validation can fail; completed validated rows remain in TSV. Scripts run
in a child bash,so failures preserve the user's interactive container. The30-case
overall acceptance field remains ineligible because this is a single-case run;
a successful diagnostic is not30-case performance acceptance.

Interpret self ratios alongside cross-code ratios at the same condition/cycle.
A ratio far from1 for the exact same callable demonstrates measurement/context
sensitivity in that comparison;it does not prove a particular cache or driver
cause. A cross-code difference still needs repeatability across buffers/order.
Keep abnormal self ratios visible without rejection,normalization or substitution
into main metrics. Do not infer a cause from elapsed times alone. The scope is
short_batch3;short96 and the remaining29 benchmark cases are not measured here.

Local validation:105 related CPU checks passed,including7 new scheduling,
source-pin,CLI,exact-callable/owner,shared-output,recompilation and single-launch
pipeline-inspection checks. A deliberately abnormal0.5 self ratio is retained
by the orchestration test. Changed-file pre-commit,Python/shell syntax,diff checks
and all frozen production/candidate/YAML/TopK/timer/guard identities passed.
No local Torch/Triton/CUDA is available;actual H800 compilation,numerics and
performance remain pending. No optimization or performance result is claimed by
this script-only delivery.
