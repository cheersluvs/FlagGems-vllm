# Single-stage score pipeline trial

## Project truth (before implementation)

| Field | Frozen choice |
| --- | --- |
| Current op/reference | Existing qsa_select_paged_prefill; pinned native vLLM7a877ae and canonical production SHA256 f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c; no single Torch API |
| Editable | New private stage-config factory,YAML group,GPU/CPU tests,runner,this document; optional candidate/profile/source/archive routing and combined previous-baseline identity |
| Read-only | Production/export/TopK,kernel math and all existing kernels/factories/YAML groups;30 workload definitions; original timer,guard,reference,tolerances and validation helpers |
| Build | Python syntax and pre-commit locally; existing FlagTree JIT on H800, no package rebuild/compiler installation |
| Validation |62-node iteration profile:40 canonical,14 existing mode/page boundary nodes rebound to the new factory,and8 every-config H/D nodes; all must pass. Full canonical +40 candidate profile remains available before integration |
| Benchmark | Same verified FlagTree/ptxas13.4/CUDA13.4compat,--mode kernel; score,TopK,sequential pipeline us and baseline/candidate ratios |
| Active/aggregation | Frozen30 scored-prefill cases; equal-weight arithmetic mean of30 native pipeline ratios>1.1,every case near1 (proposed floor0.98); experimental,not production acceptance |
| Same-run controls | Keep native and canonical score/pipeline comparisons; add original combined stage2 score/pipeline using existing previous-tuning machinery and separate buffers; keep existing fixed query-mode and short_batch3 swap rows |
| Timeout/runtime |2 rounds,200ms warmup/100ms rep,initial80C; case1800s,accuracy7200s,cooldown900s/10s; automatic partial/complete upload |
| Autotune | New qsa_select_paged_prefill_loop_stage_score group: exact17 original joint entries with only launch num_stages2->1; existing key/strategy/pruner,10 BF16/17 FP8 choices; compact6 unchanged |
| Profiler | Prior compiled resources: BF16 TR64/BN64/KT16/W4 uses240 registers/106496 shared bytes; FP8 split equivalent144 registers/57344 bytes. H800 resource records/configs collected again; no hardware counters locally,resource pressure is a hypothesis,not proven occupancy bottleneck |
| Fallback | No Torch compute fallback,new workspace,asm,policy or TopK modification |

## Semantic contract

| Field | Frozen behavior |
| --- | --- |
| Inputs | Existing BF16/FP8 e4m3fn,H/D,query/cache/page-table strides,integer metadata,query bases,chunks,visible bounds and invalid pages |
| Outputs | Same int32 indices,-1 padding,caller buffers and None return |
| Math/promotion | Exact unchanged combined full-D FP32 dot,per-head ReLU then real-head sum; no reduction split,scaling or tolerance change |
| Layout/alias/boundaries | Exact canonical host bytecode and dispatch; existing early returns,compact route and caller buffer rules |
| Scope | Forward only; no new autograd,special-value,dtype or unsupported guarantees |
| Torch | Existing test/benchmark reference only; no production import of trial |

## Paths

| Path | Trigger | Implementation/config | Validation/risk |
| --- | --- | --- | --- |
| Rectangular | Canonical routing | Exact audited combined kernel,independent entry/cache; new stage1 YAML group | All10 BF16/17 FP8 choices,H/D,modes,page and chunk tails; stage1 may lose memory/computation overlap |
| Compact | Canonical ragged routing | Exact canonical kernel and6 configs | Existing candidate compact checks/full-entry path,unchanged stage2 |
| Early/unsupported | Canonical metadata gates | Exact canonical host | Existing iteration/full contract checks |
| Previous baseline | Explicit combined previous-tuning choice | Original pinned combined kernel and original17 stage2 YAML choices | Same TopK,independent rectangular/compact caches,own output/logits buffers,source/config records |
| Report | New explicit candidate | Existing timing/validation/telemetry plus stage1/config/source metadata | Native acceptance unchanged; do not label experimental result production |

Only software-pipeline depth changes in this trial. The matrix geometry,query
mode set,entry key,host dispatch and kernel source stay fixed. The separate
config group prevents editing historical or production choices. Pipeline depth
is a compiler scheduling control,not a CUDA-specific primitive;benefit and
compiler interpretation must be verified separately on each backend.

The configuration uses triton.Config.num_stages,not a new tl.range attribute.
See [official Config documentation](https://triton-lang.org/main/python-api/generated/triton.Config.html)
and [range documentation](https://triton-lang.org/main/python-api/generated/triton.language.range.html)
for that distinction. Lower depth can reduce buffering but also overlap; no
performance or occupancy gain is assumed before the H800 result. The existing
compiled registers/shared/spill/static-site records provide the first check of
whether scheduling actually changed. optimization.md/deep_opt.md are absent in
this checkout; this board records the deep-optimization hypothesis and audit.

## Implementation and delivery

The factory calls the original combined factory,then replaces only its
rectangular entry with an independent LibTuner/entry using the new YAML group.
It binds the exact same JIT function,pruner,key and host globals. The compact
entry remains independently cloned by the original combined factory. Production
exports and every existing kernel/group remain unchanged. Runtime preparation
rejects a config count or any non-stage difference,checks the combined kernel
SHA256,records the actual stage1 choices and selected compiled configs,and
requires identical metadata after timing. No new compute implementation or
Torch fallback is introduced.

Explicit candidate name is query-loop-stage. The runner requires the same-run
previous baseline combined,using an independent original combined module with
its own captured logits/output banks. score_vs_previous_tuning and
pipeline_vs_previous_tuning carry baseline_kind=combined_query_page_stage2.
Their original validation/timer/AB-BA/guard logic is reused unchanged. Existing
native score/TopK/pipeline,canonical score/pipeline,selected FP8 mode replay and
short_batch3 swapped comparison remain. All sides use the verified same compiler
profile. The report column says gems_loop_stage_trial_us; it does not label the
private candidate as production.

The default62-node correctness profile includes every new rectangular config
for BF16/FP8 and H/D4/128,3/24,5/96,1/8,alongside forced mode/page/chunk
boundaries and full-entry checks. The inherited fixed-score helpers normally
request stage2; the new test wrappers explicitly override that request to stage1.
This prevents boundary checks from accidentally testing the previous scheduling.
Full canonical plus40 private-candidate correctness is available with
--accuracy-profile full before any production integration. Iteration results
remain labeled full_suite_validated=false and formal_acceptance_eligible=false.

The default entry is:

```bash
bash tools/run_qsa_prefill_loop_stage_h800.sh --flagtree /opt/flagtree
```

Use the existing QSA vLLM0.31 activation and QSA_STOCK_PYTHON. The existing parent
runner uploads complete/partial reports automatically under flagtree134-loop-stage-*.
Invoke in a child shell so failure preserves the interactive container. No
self-control cycles or extra benchmark shapes are added. Timing remains2 rounds,
200ms warmup/100ms rep,initial80C;compilation/autotune remain untimed. The report
retains TopK data although TopK is outside this optimization's scope.

The deep-optimization decision is pending: compare same-run stage1 against
combined stage2,inspect registers/shared/spills and selected geometry,and check
all30 native sequential pipeline ratios. Independently tuned tile choices can
differ between stage1/stage2;these measurements evaluate the scheduling variant
with its tuner,not a claim of isolated instruction-level causality. Revert if
there is no stable gain or important shapes regress. Keep all original timings,
including any renewed short-batch anomaly. No H800 result is claimed at delivery.

Local validation:103 related CPU checks passed,including6 new stage-config,
factory/cache/source,previous-baseline,acceptance,coverage/CLI tests. Python and
shell syntax,changed-file pre-commit and diff checks passed. Source-byte checks
confirm canonical production,combined kernel/factory,timer,guard,reference and
workloads are unchanged; a normalized YAML pin confirms every pre-existing
config group is unchanged. No local Torch/Triton/CUDA is available. Actual
stage1 JIT,numerics,compiled resources,selected configurations and30-case
performance are pending H800 execution. Status: private deep-optimization trial,
not production integration or a claimed performance gain.

## Audited H800 result: 2026-10-22 03:44:02 UTC

Report branch codex/qsa-h800-results-20261022-034402,commit
`f72b4b7a6dc5a43dd0b7e271dcfe28346c7a08ce`,tested clean source
`86319a3831303b0be44e3592d8e1e25610c3c46b`. The run passed,but the
compound performance target is not met. Do not adopt stage1 globally.

### Verification and scope

All75 archived files (45 unique Git blobs) match their blob hashes/sizes.
All105 source snapshots match recorded SHA256 and tested commit,including the
pinned native reference. All241 stage TSV rows,880 raw baseline/candidate timing
samples and660 guard observations were checked. TSV,grouped means and acceptance
metadata recompute exactly. Per-stage validation before/after,AB/BA order,
common TopK input fingerprints,source/config identities and actual chunk traces
passed. The previous combined scorer was instantiated independently,used its
original stage2 group and captured separate public logits/output buffers.
Every rectangular candidate trace and selected record used stage1;compact
retained its original stage2 configs. Production source,combined kernel and
TopK remain unchanged.

62 iteration correctness nodes passed in237.73s:40 canonical and22 candidate
nodes,including all rectangular choices for the four H/D pairs and both dtypes.
This is not a full-suite run; full_suite_validated and formal_acceptance_eligible
remain false. All30 benchmark cases and their before/after checks passed.
The measured set is exactly the frozen30,so measurement_valid=true is valid
for this experimental report; it is not formal production acceptance.

Preflight,accuracy and all worker runtime identities agree. Both scoring sides
used the same verified FlagTree3.6.0/ptxas13.4.59/CUDA13.4 compatibility profile,
Driver API13040. Sampled temperature31–53C,SM clock1965–1980MHz;all sampled
process/clock guards passed with no foreign PID observations. Benchmark telemetry
spans1817.05s (30m17.05s). Adding correctness gives2054.78s (34m14.78s),
excluding setup/gaps outside that span;this is not full runner wall time.

### Group results

Ratios are baseline median/candidate median; groups use equal-weight arithmetic
means of individual case ratios. Pipeline is timed sequential score+TopK,
not a sum of isolated stage medians. Native TopK is the vLLM CUDA baseline;
candidate TopK is unchanged repository persistent_topk.

| Cases | Count | Score/native | TopK/native | Pipeline/native | Score/combined stage2 | Pipeline/combined stage2 | Score/canonical | Pipeline/canonical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream BF16 | 12 | 1.209502 | 0.898795 | 1.031001 | 1.086321 | 1.041904 | 1.112280 | 1.050957 |
| Observed BF16 | 8 | 2.680292 | 0.907325 | 1.635317 | 0.978314 | 0.988311 | 1.002070 | 0.996035 |
| FP8 | 10 | 1.116805 | 0.923984 | 0.993872 | 1.035635 | 1.011425 | 1.097329 | 1.035218 |
| All active | 30 | 1.570814 | 0.909466 | 1.179776 | 1.040624 | 1.017453 | 1.077907 | 1.031065 |

The30 native pipeline mean1.179776 exceeds1.1,but11/30 remain below1 and
six below0.98;minimum0.907251 is FP8 long100k. The compound objective is not
met. Native score minimum0.990656 is FP8 compact;all score ratios exceed0.98,
but score-only results cannot substitute for the whole-pipeline target.

Compared with the original combined scorer in this same run,the scheduling
variant averages score1.040624/pipeline1.017453. Upstream12 score1.086321/
pipeline1.041904,and FP8ten score1.035635/pipeline1.011425,show useful gains.
Observed BF16 eight average score0.978314/pipeline0.988311,so the overall mean
must not conceal this group's regressions. Across30,score wins both paired
rounds in23 cases and loses both in5;pipeline wins both in24 and loses both in3.
Non-stage configuration choices differ from the combined baseline in18 cases.
These are scheduling variants with independently tuned tiles/modes,not a
fixed-geometry timing isolation of pipeline depth. The past native mean1.164420
is a different run and must not be used to infer the incremental gain;use the
same-run combined comparisons above.

### Regressions versus combined stage2

All three substantial pipeline regressions repeat in both rounds. Do not promote
the all-stage1 variant or add fixture-name dispatch to hide them.

| Observed BF16 case | Score ratio | Score paired rounds | Pipeline ratio | Pipeline paired rounds |
| --- | ---: | --- | ---: | --- |
| observed_tail32k_bfloat16 | 0.914083 | 0.917800/0.910377 | 0.945355 | 0.952055/0.938692 |
| observed_unequal2_bfloat16 | 0.920022 | 0.918454/0.921592 | 0.975133 | 0.975470/0.974799 |
| observed_chunk_tail64_bfloat16 | 0.912754 | 0.909044/0.916475 | 0.962348 | 0.958084/0.966644 |

For unequal2,the selected TR32/BN64/KT16/W8/QUERY_PARTS1 is identical except
pipeline depth. Shared memory drops73728->49152 bytes,registers remain115,
spills remain0,yet score and pipeline regress. This directly shows that reducing
buffering alone is not universally beneficial. It is consistent with losing
useful overlap,but hardware counters are absent,so that cause is not proven.
Tail32k and chunk_tail64 also change selected tile geometry in part of the run.

Matched non-stage resources elsewhere show the intended scheduling effect:
BF16 TR64/BN64/KT16/W4/parts1 shared106496->81920 bytes,registers240->234;
FP8 equivalent parts2 shared57344->49152,registers144->154. Both retain0 spills.
Shared memory decreases while FP8 registers increase;do not claim universal
register reduction or improved occupancy from these records alone.

### Remaining native pipeline floor failures

Times are microsecond medians from the native comparison.

| Case | Native us | Stage1 us | Ratio |
| --- | ---: | ---: | ---: |
| perf_n1_q16384_p102400_bfloat16 | 3149.888 | 3222.048 | 0.977604 |
| observed_unequal2_bfloat16 | 1091.064 | 1132.792 | 0.963164 |
| fp8_tail64_float8_e4m3fn | 920.696 | 940.640 | 0.978797 |
| fp8_tail65_float8_e4m3fn | 903.984 | 925.960 | 0.976267 |
| fp8_two_skewed_float8_e4m3fn | 904.272 | 930.800 | 0.971500 |
| fp8_long100k_float8_e4m3fn | 2516.976 | 2774.288 | 0.907251 |

### Short-batch context variation remains visible

For short_batch3,the canonical/stage1 normal pipeline is1.032871 (canonical
22.624us,stage121.904us);swapped is1.024000 (22.528/22.000us).
However,the combined-stage2/stage1 pipeline comparison records:

| Side | Round1 us | Round2 us |
| --- | ---: | ---: |
| Combined stage2 | 32.400 | 30.368 |
| Stage1 | 30.896 | 31.648 |

Its median ratio1.003582 has paired ratios1.048679/0.959555. Both sides
become slower together relative to other stages,with opposite paired directions.
This small mean is not a stable performance improvement. It demonstrates
continued timing/context sensitivity without identifying its cause. Keep the
row in all aggregates;do not replace it with the normal or swapped measurement.
The previous short-batch anomaly is not declared resolved.

### Decision

Reject stage1 as a universal replacement;keep the original combined stage2
candidate as the retained baseline. Canonical production was never modified,
so no production rollback is needed. Preserve this private stage1 group/report
as evidence of useful upstream and FP8 scheduling choices.

The next single change should combine the exact17 stage2 and17 stage1 choices
in one private LibTuner group,retaining existing query-mode pruning,key,strategy,
kernel,host and TopK. This allows selection from actual rows,width,dtype,H/D
and strides rather than fixture-name or ad hoc backend dispatch. Validate the
expanded choice set and rerun the frozen30 with the same stage breakdown and
same-run baseline. Autotuning is not a guarantee against noisy selection;inspect
actual configs and repeated regressions. Full correctness and formal public
entry validation remain required before production integration. No new
optimization/config or production change is made in this report-only commit.

### All30 stage measurements

Times are microsecond medians. Native-stage cells show native/candidate/ratio.
The last columns show same-run combined stage2/stage1 ratios.

| Case | Score us/us/ratio | TopK us/us/ratio | Pipeline us/us/ratio | Score/combined | Pipeline/combined |
| --- | --- | --- | --- | ---: | ---: |
| perf_n1_q8192_p8192_bfloat16 | 145.296/111.488/1.303243 | 204.912/213.184/0.961198 | 345.584/324.784/1.064043 | 1.092126 | 1.027711 |
| perf_n1_q8192_p51200_bfloat16 | 520.000/441.448/1.177942 | 384.272/421.928/0.910753 | 889.552/859.616/1.034825 | 1.103133 | 1.052326 |
| perf_n1_q8192_p102400_bfloat16 | 1044.672/860.048/1.214667 | 571.296/708.936/0.805850 | 1561.992/1575.488/0.991434 | 1.074777 | 1.042307 |
| perf_n1_q16384_p8192_bfloat16 | 342.240/288.848/1.184845 | 456.576/458.232/0.996386 | 781.696/730.008/1.070805 | 1.083047 | 1.038945 |
| perf_n1_q16384_p51200_bfloat16 | 1169.120/957.040/1.221600 | 792.088/891.336/0.888653 | 1910.288/1844.640/1.035589 | 1.085620 | 1.037947 |
| perf_n1_q16384_p102400_bfloat16 | 2071.840/1723.376/1.202198 | 1163.064/1478.696/0.786547 | 3149.888/3222.048/0.977604 | 1.112395 | 1.055800 |
| perf_n2_q8192_p8192_bfloat16 | 257.424/218.432/1.178509 | 399.696/415.856/0.961140 | 649.088/620.160/1.046646 | 1.058832 | 1.024590 |
| perf_n2_q8192_p51200_bfloat16 | 1085.424/885.104/1.226324 | 760.776/834.336/0.911834 | 1787.392/1723.072/1.037329 | 1.097927 | 1.045776 |
| perf_n2_q8192_p102400_bfloat16 | 1972.064/1667.008/1.182996 | 1137.864/1399.056/0.813308 | 3061.088/3092.672/0.989787 | 1.100680 | 1.053287 |
| perf_n4_q4096_p8192_bfloat16 | 226.368/185.648/1.219340 | 388.968/384.664/1.011189 | 609.632/562.064/1.084631 | 1.082191 | 1.036181 |
| perf_n4_q4096_p51200_bfloat16 | 1065.840/860.888/1.238070 | 757.616/819.584/0.924391 | 1757.424/1660.632/1.058286 | 1.084146 | 1.043550 |
| perf_n4_q4096_p102400_bfloat16 | 1893.504/1626.312/1.164293 | 1131.120/1389.088/0.814290 | 2980.336/3037.944/0.981037 | 1.060978 | 1.044433 |
| observed_short96_bfloat16 | 51.424/8.112/6.339250 | 8.288/11.328/0.731638 | 56.376/28.048/2.009983 | 1.001961 | 1.004999 |
| observed_tail16k_bfloat16 | 54.128/15.824/3.420627 | 17.168/18.736/0.916311 | 66.016/29.152/2.264544 | 0.994980 | 1.004923 |
| observed_tail32k_bfloat16 | 70.096/27.184/2.578576 | 21.440/25.840/0.829721 | 84.952/46.608/1.822691 | 0.914083 | 0.945355 |
| observed_short_batch3_bfloat16 | 52.688/11.840/4.450000 | 13.888/13.712/1.012835 | 62.144/21.840/2.845421 | 1.028188 | 1.003582 |
| observed_ragged54_bfloat16 | 78.096/73.968/1.055808 | 181.312/161.568/1.122202 | 248.944/228.192/1.090941 | 1.001625 | 0.998146 |
| observed_unequal2_bfloat16 | 593.816/520.272/1.141357 | 524.336/635.408/0.825196 | 1091.064/1132.792/0.963164 | 0.920022 | 0.975133 |
| observed_chunk_boundary_bfloat16 | 490.736/408.944/1.200008 | 523.440/538.032/0.972879 | 993.624/948.440/1.047640 | 1.052901 | 1.011999 |
| observed_chunk_tail64_bfloat16 | 626.032/498.152/1.256709 | 536.960/633.344/0.847817 | 1163.104/1120.360/1.038152 | 0.912754 | 0.962348 |
| fp8_first_chunk_float8_e4m3fn | 114.984/108.576/1.059019 | 351.512/345.232/1.018191 | 460.032/445.152/1.033427 | 1.086025 | 1.019942 |
| fp8_budget_exact_float8_e4m3fn | 290.640/288.560/1.007208 | 521.824/534.344/0.976569 | 813.536/825.008/0.986095 | 1.031097 | 1.011092 |
| fp8_dominant_long_float8_e4m3fn | 338.400/309.632/1.092910 | 540.120/561.744/0.961506 | 874.248/867.392/1.007904 | 1.034643 | 1.012877 |
| fp8_tail64_float8_e4m3fn | 386.280/315.336/1.224979 | 538.776/629.472/0.855917 | 920.696/940.640/0.978797 | 1.018234 | 1.008472 |
| fp8_tail65_float8_e4m3fn | 385.360/311.264/1.238049 | 531.680/625.456/0.850068 | 903.984/925.960/0.976267 | 1.021731 | 1.008473 |
| fp8_two_balanced_float8_e4m3fn | 395.296/353.064/1.119616 | 545.152/590.888/0.922598 | 936.656/941.088/0.995291 | 1.035610 | 1.012080 |
| fp8_two_skewed_float8_e4m3fn | 382.408/316.024/1.210060 | 533.088/627.664/0.849321 | 904.272/930.800/0.971500 | 1.012424 | 1.004660 |
| fp8_short_tail_float8_e4m3fn | 74.880/63.664/1.176175 | 92.384/103.120/0.895888 | 162.744/161.712/1.006382 | 1.092697 | 1.022820 |
| fp8_compact_float8_e4m3fn | 53.432/53.936/0.990656 | 180.448/160.656/1.123195 | 223.080/207.360/1.075810 | 0.995558 | 1.000386 |
| fp8_long100k_float8_e4m3fn | 1327.616/1265.144/1.049379 | 1162.304/1477.656/0.786586 | 2516.976/2774.288/0.907251 | 1.028334 | 1.013448 |

Raw evidence: [summary](https://github.com/cheersluvs/FlagGems-vllm/blob/f72b4b7a6dc5a43dd0b7e271dcfe28346c7a08ce/results/qsa_select_paged_prefill/h800/flagtree134-loop-stage-20261022-034402/stage-summary.json),
[all stage rows](https://github.com/cheersluvs/FlagGems-vllm/blob/f72b4b7a6dc5a43dd0b7e271dcfe28346c7a08ce/results/qsa_select_paged_prefill/h800/flagtree134-loop-stage-20261022-034402/triton-stage-ab.tsv),
[correctness](https://github.com/cheersluvs/FlagGems-vllm/blob/f72b4b7a6dc5a43dd0b7e271dcfe28346c7a08ce/results/qsa_select_paged_prefill/h800/flagtree134-loop-stage-20261022-034402/accuracy-full.json),
[source snapshots](https://github.com/cheersluvs/FlagGems-vllm/blob/f72b4b7a6dc5a43dd0b7e271dcfe28346c7a08ce/results/qsa_select_paged_prefill/h800/flagtree134-loop-stage-20261022-034402/sources.json),
[acceptance](https://github.com/cheersluvs/FlagGems-vllm/blob/f72b4b7a6dc5a43dd0b7e271dcfe28346c7a08ce/results/qsa_select_paged_prefill/h800/flagtree134-loop-stage-20261022-034402/overall-acceptance.json).
