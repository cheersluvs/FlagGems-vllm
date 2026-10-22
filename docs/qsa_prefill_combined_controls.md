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

## Audited H800 result: 2026-10-22 02:32:40 UTC

Report commit `9e9f617d3e4562eeba7bff28f755ec6f762d5bdd` tested source
`048d0c0dba903129c69e657087aad18fda34ac00`. The diagnostic passed. Its
54 iteration correctness nodes passed in193.27s; all144 supplemental comparisons
passed their before/after numerical checks. This is the existing40 canonical
plus14 combined iteration profile,not a new full-suite run.

The archive audit verified18 files (17 unique Git blobs),100 source snapshots,
seven main-stage TSV rows (six timed,one inapplicable),144 supplemental rows,
600 raw baseline/candidate timing samples and450 guard observations. File Git
SHA1/size,source SHA256,per-row sample statistics,AB/BA order,identity flags,
source/buffer ownership,compiled records,validation results and acceptance
metadata were checked. All48 shared-bank self comparisons used the exact same
callable. Production remains SHA256
`f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c`;
combined remains SHA256
`1ef12d431b65971c5617f0a34f614360008c652a36125d5679da00206b212d7c`.
The combined kernel is identical to the previous30-case report.

Both sides used the same verified FlagTree3.6.0 module,ptxas13.4.59 and
CUDA13.4 compatibility libcuda (Driver API13040);runtime identities matched
between preflight,accuracy and worker. Sampled temperature was32–36C,SM clock
1980MHz,with no foreign-process observations or guard rejections. Benchmark
telemetry spans295.41s. Correctness plus this benchmark span totals488.68s
(8m8.68s),excluding setup and gaps;this is not the full runner wall time.

### Supplemental self and cross-code comparisons

Every cell below is an equal-weight arithmetic mean of16 per-comparison
speedups: four cycles times four buffer conditions. Each speedup is that row's
baseline two-round median divided by its candidate two-round median. These
are not pooled-sample ratios and are not speedups versus native vLLM.
P is current production; C is the combined candidate.

| Comparison | Score | TopK | Sequential score + TopK |
| --- | ---: | ---: | ---: |
| P / P | 1.000276x | 0.996978x | 1.000074x |
| C / C | 1.000038x | 0.999695x | 1.000075x |
| P / C | 1.029989x | 1.003572x | 1.015216x |

All16 P/C score comparisons and all16 P/C pipeline comparisons exceeded1;
all32 individual paired ratios exceeded1 for each of these two stages.
Score comparison ratios ranged1.019531–1.043194,pipeline1.008596–1.020908.
P/C pipeline raw samples ranged22.496–22.720us for P and22.144–22.368us
for C. The earlier27–29us slow pipeline samples did not recur.

TopK code was unchanged. Its P/C mean1.003572x is a measured difference,
not evidence of a TopK optimization. Individual same-callable C/C TopK
comparisons ranged0.980233–1.009390,illustrating small-percent timing
variation even without a code change.

The next table isolates buffer conditions. Each cell averages four cycles;
A and B retain the existing canonical and candidate buffer allocations.

| Comparison | Stage | A/B | B/A | A/A | B/B |
| --- | --- | ---: | ---: | ---: | ---: |
| P / P | score | 1.002287x | 0.998745x | 1.002853x | 0.997220x |
| P / P | topk | 0.995904x | 0.990824x | 1.000303x | 1.000882x |
| P / P | pipeline | 1.004615x | 0.996474x | 0.999384x | 0.999824x |
| C / C | score | 1.002661x | 0.998100x | 1.001673x | 0.997719x |
| C / C | topk | 0.999715x | 1.002502x | 0.996550x | 1.000012x |
| C / C | pipeline | 1.004681x | 0.995349x | 1.000269x | 1.000000x |
| P / C | score | 1.030085x | 1.036264x | 1.027400x | 1.026208x |
| P / C | topk | 1.002653x | 1.005903x | 1.002952x | 1.002781x |
| P / C | pipeline | 1.019790x | 1.010748x | 1.015217x | 1.015110x |

For pipeline,the exact-callable P/P shared-bank comparisons ranged
0.996476–1.001411 (mean0.999604); C/C ranged0.998927–1.001434
(mean1.000135). Different-bank self comparisons showed a small directional
association: A/B means about1.0046–1.0047,B/A about0.9953–0.9965. This
supports treating buffer condition as part of measurement context;it does
not identify address,cache,compiler or driver behavior as the cause.
P/C still improved on both shared banks,about1.0152x on A and1.0151x on B,
so the observed cross-code benefit is not confined to assigning P and C
different buffer banks.

### Original main-stage measurements

These rows remain separate from the144 supplemental comparisons. Latencies
are per-row medians in microseconds; pipeline is timed sequential execution,
not the sum of separately timed score and TopK medians.

| Stage | Baseline | Baseline us | Combined us | Speedup |
| --- | --- | ---: | ---: | ---: |
| Score | Native vLLM | 52.528 | 12.128 | 4.331135x |
| TopK | Native vLLM | 13.920 | 13.584 | 1.024735x |
| Pipeline | Native vLLM | 62.224 | 22.160 | 2.807942x |
| Score | Production | 12.528 | 12.176 | 1.028909x |
| Pipeline | Production | 22.688 | 22.256 | 1.019410x |
| Pipeline,swapped banks | Production | 22.608 | 22.368 | 1.010730x |

`score_matched` was inapplicable for this BF16 fixture. No measurement was
substituted into that row.

### Decision and limits

This fixed-tile short_batch3 diagnostic supports retaining the combined
candidate: score improved about3%,and sequential score+TopK about1.5%,
with self controls centered near1. The previous normal P/C pipeline0.884312x
result was not reproduced. Its cause remains unresolved;the current run
cannot reconstruct its device/cache state and does not invalidate or replace
that archived observation.

This run measures only observed_short_batch3_bfloat16. The acceptance JSON
therefore correctly has measurement_valid=false,target_met=null: both planned
and measured sets differ from the frozen30. That ineligibility is not a
correctness or diagnostic-run failure. The previous30-case result remains
1.164420x native pipeline arithmetic mean with10 cases below0.98;this run
neither updates that mean nor closes the per-case performance target.

Keep the candidate private and resume score optimization of the remaining
long-context cases,especially the four upstream102400-context cases and FP8
long100k. No further identical self-control rerun is needed before resuming
that work. No production integration,TopK change or new optimization was
made for this report audit.

Raw evidence: [stage summary](https://github.com/cheersluvs/FlagGems-vllm/blob/9e9f617d3e4562eeba7bff28f755ec6f762d5bdd/results/qsa_select_paged_prefill/h800/flagtree134-combined-controls-20261022-023240/stage-summary.json),
[all supplemental comparisons](https://github.com/cheersluvs/FlagGems-vllm/blob/9e9f617d3e4562eeba7bff28f755ec6f762d5bdd/results/qsa_select_paged_prefill/h800/flagtree134-combined-controls-20261022-023240/observed_short_batch3_bfloat16.combined-controls.tsv),
[case JSON and telemetry](https://github.com/cheersluvs/FlagGems-vllm/blob/9e9f617d3e4562eeba7bff28f755ec6f762d5bdd/results/qsa_select_paged_prefill/h800/flagtree134-combined-controls-20261022-023240/observed_short_batch3_bfloat16.json),
[correctness](https://github.com/cheersluvs/FlagGems-vllm/blob/9e9f617d3e4562eeba7bff28f755ec6f762d5bdd/results/qsa_select_paged_prefill/h800/flagtree134-combined-controls-20261022-023240/accuracy-full.json),
[source snapshots](https://github.com/cheersluvs/FlagGems-vllm/blob/9e9f617d3e4562eeba7bff28f755ec6f762d5bdd/results/qsa_select_paged_prefill/h800/flagtree134-combined-controls-20261022-023240/sources.json),
[acceptance metadata](https://github.com/cheersluvs/FlagGems-vllm/blob/9e9f617d3e4562eeba7bff28f755ec6f762d5bdd/results/qsa_select_paged_prefill/h800/flagtree134-combined-controls-20261022-023240/overall-acceptance.json).
