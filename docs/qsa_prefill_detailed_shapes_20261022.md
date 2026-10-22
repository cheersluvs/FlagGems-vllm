# Detailed inference capture audit:2026-10-22

Upload branch `codex/qsa-inference-shapes-20261022-070202`,commit
`9937ec6e910db658ee2871ac1743eead0698b7c4`. Archive
`results/qsa_select_paged_prefill/inference-shapes/20261022-070202/inference-shapes.tar.gz`,
SHA256 `f2d4ae225613f20b9625dc4e5f0b5279815e11064eb55fbb23c5a4441db05fcc`.
Three downloaded Git blobs passed size/blob-hash verification;archive checksum
matches SHA256SUMS. Only regular archive members with safe relative paths were
read. The three capture files are TXT,YAML and count YAML for
Qwen3.8-Flash-Next-FP8-p32768d1024.

All three formats agree exactly on128 distinct prefill signatures and6528
recorded occurrences. Counts are capture occurrences,not measured latency or
production traffic weights. The operator also has126 scoring signatures and two
select-all signatures under its existing metadata dispatch. Compared with the
previous FP8 capture's126 six-tuples,only six projected tuples match exactly;
this is a new sampling run,not simply added fields on identical records.
Do not silently combine counts across runs.

| Captured field | Observed value |
| --- | --- |
| Q | [T,4,128] |
| k_cache | [2712,196,1,128] |
| page_table | [N,88] |
| query_start_loc | [N+1] |
| visible_blocks | [T] (shape only) |
| token_topk / compress_ratio | 2048 /4 |
| block_indices | [T,512] |
| N | 1:62 signatures/3360 occurrences;2:64/3072;451:1/48;2048:1/48 |
| max_seq_len of scoring signatures | 16384–33844 |

The model filename says FP8,but q/cache tensor dtypes are not logged. Neither
are metadata dtypes,strides/storage offsets,page-table contents,visible-block
values,query-start values,per-request context lengths or logits workspace budget.
Retain previous explicit assumptions for unavailable fields per user instruction;
label dtype as a benchmark choice rather than a newly observed runtime fact.
Observed tensor dimensions do not establish contiguous memory layout. Two-request
query lengths are inferable as an unordered pair from total T and max_query_len;
request order is still an assumption. Per-request contexts cannot be reconstructed
from batch max_seq_len. A cache of2712 pages does not imply every page is read.

## Consequences for the existing eight representative capture fixtures

Use actual cache/page-table capacities instead of tight synthetic capacities.
Keep H4/D128,page196,K512 (=2048/4),and update small differences in sampled
Q/max_query_len/max_seq_len. The table below derives score chunking from current
production host code with an explicitly assumed512MiB FP32 logits budget. It is
not a trace of actual workspace allocation. Existing and proposed geometries
were recomputed using the benchmark geometry function and cross-checked against
the host formula:

```
capacity = page_table_columns * page_size
visible_bound = min(capacity, ceil(max_seq_len / compress_ratio))
logits_width = min(capacity, round_up(max(64, visible_bound), 64))
rows_per_chunk = min(T, max(1, floor(budget_bytes / (4 * logits_width))))
```

| Existing representative | Proposed query lengths | New max_seq_len | Cache pages old/new | Table columns old/new | Logits width old/new | Chunk rows old -> proposed |
| --- | --- | ---: | --- | --- | --- | --- |
| fp8_first_chunk_float8_e4m3fn | [16384] | 16384 | 21/2712 | 21/88 | 4096/4096 | [16384] -> [16384] |
| fp8_budget_exact_float8_e4m3fn | [16384] | 32768 | 42/2712 | 42/88 | 8192/8192 | [16384] -> [16384] |
| fp8_dominant_long_float8_e4m3fn | [16331] | 33842 | 44/2712 | 44/88 | 8512/8512 | [15768, 563] -> [15768, 563] |
| fp8_tail64_float8_e4m3fn | [16368] | 32853 | 42/2712 | 42/88 | 8232/8256 | [16304, 64] -> [16256, 112] |
| fp8_tail65_float8_e4m3fn | [16369] | 32851 | 42/2712 | 42/88 | 8232/8256 | [16304, 65] -> [16256, 113] |
| fp8_two_balanced_float8_e4m3fn | [10371, 5958] | 32932 | 86/2712 | 43/88 | 8256/8256 | [16256, 73] -> [16256, 73] |
| fp8_two_skewed_float8_e4m3fn | [16330, 54] | 32822 | 84/2712 | 42/88 | 8232/8256 | [16304, 80] -> [16256, 128] |
| fp8_short_tail_float8_e4m3fn | [2316] | 33843 | 44/2712 | 44/88 | 8512/8512 | [2319] -> [2316] |

In particular,old tail64/tail65 use a tight42-column table:capacity8232 clamps
the otherwise aligned8256-wide logits. Actual88 columns remove that clamp.
At512MiB,the main chunk shrinks16304->16256 and the tails become112/113.
The same capacity issue changes two-skewed width8232->8256 and tail80->128,
with the newly observed query pair16330/54. For the first case,the native H800
TopK dispatch also changes from cooperative at64 rows to persistent at112 rows
(the recorded native source chooses cooperative only at rows<=64 with its other
SM90/stride guards). Thus this is relevant to whole-pipeline and TopK comparisons,
not only scoring. Previous reports remain valid for their original synthetic
fixtures;they must not be retroactively relabeled as these new workloads.

## Two additional select-all signatures

| T | N | max_query_len | max_seq_len | Count | Suggested role |
| ---: | ---: | ---: | ---: | ---: | --- |
| 16384 | 2048 | 8 | 8 | 48 | Many-request select-all correctness/public-entry coverage |
| 902 | 451 | 2 | 2 | 48 | Many-request select-all correctness/public-entry coverage |

With ratio4 and K512,the visible bounds are2 and1,so the score+TopK pipeline is
bypassed by the current operator. These do not belong in the30 active scoring
performance cases. Total T=N*max_query_len determines uniform query lengths
under the packed-request contract,but starts/visible contents remain unrecorded.

Recommendation:correct/replace the eight recorded FP8 representative fixtures
with the observed capacities and new envelopes before the next inference-led
optimization experiment,rather than adding all128 signatures. Keep the12 upstream
BF16 fixtures and eight observed BF16 fixtures frozen;this capture contains no
new BF16 run. Keep FP8 compact and long100k as explicitly synthetic dtype variants,
not recorded tuples. This preserves30 active scoring cases and controls runtime.
Add the two select-all captures to functional coverage separately. Changing the
eight fixtures changes benchmark input construction:establish a fresh same-run
native/canonical/joint-stage baseline and do not splice old speedups into new
means. Do not change production kernels or the frozen benchmark yet on the basis
of this analysis-only delivery. Remaining dtype/value/layout assumptions must
remain visible in any later fixture update.

Validation:archive/TXT/YAML/count cross-checks,128-signature consistency,old/new
projection comparison,all eight proposed geometry/capacity checks,and fast-path
classification completed on CPU. No GPU execution or performance claim. No
production,benchmark fixture,timer or guard was modified for this audit.

## Approved fixture update plan

| Project truth | Frozen scope |
| --- | --- |
| Editable | FP8 workloads/provenance manifest,FP8-only shared builder routing,CPU/GPU fixture tests and docs |
| Read-only | Production and all private scorers,TopK,YAML choices,native reference,20 active BF16 fixtures,timers/guards/acceptance thresholds |
| Active set | Same30 scored cases:12 upstream BF16 +8 observed BF16 +8 updated captured FP8 +2 unchanged FP8 variants |
| Validation/build | CPU archive/manifest/geometry/dtype/remap/CLI invariants,Python syntax/pre-commit; existing10 FP8 native GPU tests +2 new full-suite select-all nodes |
| Benchmark | Existing kernel-mode native/canonical/private stage comparisons,2 rounds; fresh baseline for new fixture definitions; no old performance reuse |
| Aggregation/timeout | Existing equal-weight arithmetic pipeline mean>1.1 and proposed0.98 floor; existing runner timeouts unchanged |
| Autotune/profiler | No kernel/performance parameter changes;existing config groups/keys;resource records on H800,no local GPU profiler |
| Fallback | Test/benchmark construction only;no production torch compute/fallback added |

| Contract | Decision |
| --- | --- |
| Reference | Existing pinned native QSA,not a standalone torch API |
| Captured | Exact17-field records including H4/D128,cache2712x196x1x128,table Nx88,K512,ratio4 and output Tx512 |
| Assumed | Explicit FP8 dtype,contiguous layouts/random physical mapping,request order,equal per-request context maxima,causal visible values,512MiB budget |
| Outputs/numerics | Existing FP32 logits,int32 indices/ties/FP8 normalization;no tolerance changes |
| Training/unsupported | Forward-only;production contract and errors unchanged |

| Path | Implementation | Verification/risk |
| --- | --- | --- |
| Eight captured scoring fixtures | Reuse existing inference builder's full-cache remap/table padding with factor1;update recorded envelopes/capacities;rename tail64/65 to tail112/113 | Capacity/manifest/native checks |
| Two synthetic variants | Preserve definitions and construction;not assigned new observed capacities | Exact parent input-definition comparison |
| BF16 | Existing input builder path unchanged except new FP8-only early branch | Existing BF16 matrix pin |
| New select-all captures | Separate functional_cases,not added to any performance suite | Assert public entry bypasses score/TopK and matches native/explicit indices |

The revised inference fixtures require a new baseline. Old reports keep their
original meanings and old case names;they are not migrated or recomputed.

## Fixture update delivered

The shared FP8 workload source and archived schema2 manifest now bind upload
9937ec6e and all128 captured17-field signatures. Eight performance fixtures
use physical_pages2712,table_columns88,cache_page_stride_factor1. The existing
inference builder performs the real allocation/remap/padding;the change is not
limited to reported geometry. Its BF16 and variant input paths remain unchanged.
The eight new fixture geometries are:

| Current FP8 case suffix | Query lengths (chosen order) | max_seq_len | Logits width | Derived chunks at assumed512MiB |
| --- | --- | ---: | ---: | --- |
| first_chunk | [16384] | 16384 | 4096 | [16384] |
| budget_exact | [16384] | 32768 | 8192 | [16384] |
| dominant_long | [16331] | 33842 | 8512 | [15768,563] |
| tail112 | [16368] | 32853 | 8256 | [16256,112] |
| tail113 | [16369] | 32851 | 8256 | [16256,113] |
| two_balanced | [10371,5958] | 32932 | 8256 | [16256,73] |
| two_skewed | [16330,54] | 32822 | 8256 | [16256,128] |
| short_tail | [2316] | 33843 | 8512 | [2316] |

The functional-only captures are supplied by functional_cases() and exercised
by two full-suite test_captured_many_request_select_all nodes;the iteration
profile is unchanged. They assert public entry/output identity,bypass of our
score/TopK,explicit logical indices/-1 padding and direct native output agreement.
Existing ten FP8 native tests now also assert actual Q/cache/table shapes and
strides match declared geometry. All unknown values remain marked synthetic;
observed page size,token_topk and compress_ratio are no longer marked synthetic.

Validation:96 CPU regression checks passed under Python3.12 with NumPy. This
includes actual input-builder execution with opaque uint8 page contents at the
captured capacities for one- and two-request cases,checking all live key rows,
causal visibility,request starts,table padding,cache strides and unused page
initialization. This checks byte mapping,not FP8 numerical computation. The
uploaded TXT/YAML/count records match the updated manifest and selected capture
lines;the original BF16 matrix pin and both synthetic FP8 definitions match.
Changed-file pre-commit,diff and syntax checks pass. Production kernels,all
private candidate kernels/factories,YAML,TopK,runner,timer and guard sources match
the parent byte-for-byte. H800 JIT/numerical tests and fresh30-case speedups remain
pending. Do not use the rejected global unroll2 candidate as the retained baseline;
use production or the retained joint-stage candidate with these revised fixtures.

## First revised-matrix H800 attempt:22d3c2ae

Report branch `codex/qsa-h800-results-20261022-074232`,commit
`22d3c2ae86ec28da5d8d4fab25a528f0c49efd69`,tested clean source
`8b173a8e94793d94908484ba3c2688a5e759b083`,exit2. Eighteen archived
files passed Git blob hash/size checks;all118 source snapshots match the tested
commit and manifest. The selected manifest is exactly the new30-case matrix.
Preflight,accuracy and first worker agree on compiler/GPU/runtime identity.

All62 existing iteration correctness nodes passed in313.72s. This is not full
suite validation;the two new functional-only select-all nodes are not included
in that iteration profile. The first performance case was the unchanged upstream
BF16 perf_n1_q8192_p8192_bfloat16. At score:before_timing,the guard identified:

| PID | Program | Parent PID | GPU memory MiB | Role |
| ---: | --- | ---: | ---: | --- |
| 2726747 | python | 2696274 | 2020 | Positively identified benchmark worker |
| 2727727 | python | 1468207 | 1162 | Separate process,outside allowed worker PIDs |

The other process remained after the benchmark worker exited:the next initial
check observed1172MiB total GPU memory and PID2727727 still holding1162MiB.
It began after the selected worker started (reported process start times15:42:02
and15:41:55 respectively;these are GPU-host log timestamps,not normalized UTC).
This record therefore distinguishes the two processes;it is not evidence that
NVML merely reported the same worker under a second namespace PID. Process owner
and workload arguments were not established. Do not stop or attribute ownership
to the other job from this report alone.

Timing was rejected at33C,well below80C. No benchmark round completed and no
usable score,TopK or pipeline latency/speedup exists. The one case record is
failed;the other29 cases,including all ten FP8 cases,were never reached. The
new eight captured FP8 fixture capacities/native comparisons remain pending on
H800. Overall measurement_valid=false,target_met=null is correct. There is no
candidate keep/revert or new-matrix performance conclusion from this attempt.

The interruption calls for a GPU reserved for the test duration,then an explicit
UUID selection;auto selection only observes idleness when choosing a device.
Changing kernels,raising temperature limits or accepting the occupied-card data
would not address this failure. All30 timings remain to be measured;retain this
report as passing iteration-correctness evidence with its precise scope. This
incident-recording change modifies no executable code,guard or timer.

## Revised30 benchmark-only result:f6a41ae3

Report branch `codex/qsa-h800-results-20261022-084414`,commit
`f6a41ae371cbc0907bb2b2aa4e2eca2c49966a18`,tested clean source
`c79389e9720c997d0bbccdfa78a047730f2d4166`,exit0. Candidate is the retained
private joint-stage scorer,not global-unroll2 and not a production integration.
Verified74 archived files by Git blob hash/size,120 source snapshots against the
tested commit,241 TSV rows,880 raw timing samples and660 guard observations.
All30 cases passed every required before/after numeric check,AB/BA timing and
runtime/source/config checks. Raw medians,paired ratios,group arithmetic means
and exported table were recomputed. Common TopK logits fingerprints are stable.
Both scoring sides use the same verified FlagTree/ptxas13.4/compat runtime;
native TopK is the CUDA extension,candidate TopK is unchanged persistent_topk.

Sampled temperatures32–52C,SM clocks1965–1980MHz;no sampled foreign process
or guard rejection. Benchmark telemetry spans1789.86s (29m49.86s),excluding
setup/preflight and time outside that span. New eight captured FP8 cases really
allocated cache [2712,196,1,128],table [N,88],with the assumed contiguous strides,
Q/cache FP8 dtype and captured2048/4 scalar parameters. Revised112/113/128-row
chunk tails match the manifest. Two synthetic variants remain unchanged.

Independent pytest was skipped by explicit request:accuracy status is
skipped_by_request,zero counts,full_suite_validated=false. The report's overall
measurement_valid=false,target_met=null has exactly one reason:
`run or full accuracy incomplete`. This is the retained accuracy gate,not a
failed timing or numeric check. Do not manufacture a passed pytest record.
The prior62-node iteration pass is report22d3c2ae;it is not this run's full-suite
validation. New functional-only select-all nodes remain pending. This report
supports a benchmark-only baseline and stage comparisons,not formal production
acceptance.

Equal-weight arithmetic means of per-case speedups:

| Group | Cases | Native/score | Native/TopK | Native/pipeline | Canonical/score | Canonical/pipeline | Combined-stage2/score | Combined-stage2/pipeline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream BF16 | 12 | 1.204163 | 0.898472 | 1.030793 | 1.105511 | 1.050943 | 1.087813 | 1.040490 |
| Observed BF16 | 8 | 2.699727 | 0.909993 | 1.828757 | 1.032020 | 1.015757 | 1.010222 | 1.025039 |
| FP8 | 10 | 1.088767 | 0.957090 | 1.010881 | 1.088108 | 1.033124 | 1.036758 | 1.014970 |
| All | 30 | 1.564515 | 0.921084 | 1.236946 | 1.080112 | 1.035620 | 1.050104 | 1.027863 |

The numeric pipeline mean1.236946 exceeds1.1,but eight cases remain below1 and
two below the proposed0.98 floor. Thus even a timing-only assessment does not
satisfy the all-case goal. All score ratios exceed0.98;FP8 compact score0.993193
is the only score ratio below1. Remaining pipeline floor failures:

| Case | Native/score | Native/TopK | Native/pipeline |
| --- | ---: | ---: | ---: |
| perf_n1_q16384_p102400_bfloat16 | 1.197827 | 0.786434 | 0.976673 |
| fp8_long100k_float8_e4m3fn | 1.056790 | 0.786253 | 0.908443 |

FP8 long100k medians illustrate the remaining gap (standalone stages are not
additive predictions of pipeline latency):

| Stage | Native us | Candidate us | Speedup |
| --- | ---: | ---: | ---: |
| score | 1338.784 | 1266.840 | 1.056790 |
| topk | 1162.528 | 1478.568 | 0.786253 |
| pipeline | 2522.136 | 2776.328 | 0.908443 |

This round establishes new fixture baselines;it contains no new kernel optimization.
Old tail64/tail65/80-row skewed measurements cannot be subtracted from revised
112/113/128-row measurements to claim a speedup. Current corresponding pipeline
ratios are1.026961,1.025278 and1.026017;keep them as separate workloads with
recorded capacity and scalar differences.

## Context-sensitive observations retained

Short96 is unchanged as an input definition. This run's native/candidate pipeline
medians are56.576/16.368us,ratio3.456501 (paired3.458824/3.454191). Earlier
e861c93c had ratio1.985365 with candidate pipeline near28us. Same-run canonical
and combined pipeline comparisons here are1.002915 and0.997093 respectively.
All scoring algorithms/TopK are frozen across these reports;this discrepancy
must not be presented as a new kernel improvement. Prior shared-context/timing
sensitivity remains unresolved;no cause is established by this report.

Short_batch3 native/candidate pipeline is62.432/22.208us,ratio2.811239,with
paired2.817656/2.804878. Canonical comparison1.030129 and buffer-swap1.029391
have both positive rounds. However,combined-stage2 comparison1.175036 contains
paired1.332378/1.017241:its baseline rounds are29.760/22.656us while candidate
rounds are22.336/22.272us. Preserve the raw values and the1.027863 all-case
combined comparison mean,but do not treat this short-batch contribution as a
stable17.5% incremental win or silently drop its slow baseline round. No timing
sample was replaced. Two rounds support the reported measurements,not an
unconditional assertion that all context sensitivity is resolved.

Per-case stage ratios and same-run combined-stage2 incremental comparison:

| Case | Native/score | Native/TopK | Native/pipeline | Combined/score | Combined/pipeline |
| --- | ---: | ---: | ---: | ---: | ---: |
| perf_n1_q8192_p8192_bfloat16 | 1.293876 | 0.957666 | 1.062389 | 1.085550 | 1.030371 |
| perf_n1_q8192_p51200_bfloat16 | 1.180496 | 0.910543 | 1.038379 | 1.103044 | 1.052494 |
| perf_n1_q8192_p102400_bfloat16 | 1.200868 | 0.803954 | 0.996337 | 1.086128 | 1.039363 |
| perf_n1_q16384_p8192_bfloat16 | 1.194575 | 0.996521 | 1.069819 | 1.067213 | 1.016995 |
| perf_n1_q16384_p51200_bfloat16 | 1.219746 | 0.888792 | 1.035578 | 1.081163 | 1.039581 |
| perf_n1_q16384_p102400_bfloat16 | 1.197827 | 0.786434 | 0.976673 | 1.135895 | 1.059679 |
| perf_n2_q8192_p8192_bfloat16 | 1.166679 | 0.963283 | 1.046312 | 1.058468 | 1.026395 |
| perf_n2_q8192_p51200_bfloat16 | 1.220711 | 0.912315 | 1.036034 | 1.086436 | 1.041949 |
| perf_n2_q8192_p102400_bfloat16 | 1.175726 | 0.813220 | 0.992125 | 1.102136 | 1.055058 |
| perf_n4_q4096_p8192_bfloat16 | 1.213752 | 1.010537 | 1.087797 | 1.096696 | 1.036043 |
| perf_n4_q4096_p51200_bfloat16 | 1.221449 | 0.924388 | 1.045970 | 1.083051 | 1.042695 |
| perf_n4_q4096_p102400_bfloat16 | 1.164248 | 0.814011 | 0.982102 | 1.067972 | 1.045253 |
| observed_short96_bfloat16 | 6.173996 | 0.731638 | 3.456501 | 0.992322 | 0.997093 |
| observed_tail16k_bfloat16 | 3.397773 | 0.926684 | 2.219578 | 1.002000 | 1.004412 |
| observed_tail32k_bfloat16 | 2.818770 | 0.828377 | 1.935390 | 0.996127 | 1.000363 |
| observed_short_batch3_bfloat16 | 4.352394 | 1.023419 | 2.811239 | 1.037185 | 1.175036 |
| observed_ragged54_bfloat16 | 1.066017 | 1.124290 | 1.089366 | 1.001940 | 1.001680 |
| observed_unequal2_bfloat16 | 1.233254 | 0.825079 | 0.993017 | 0.996583 | 1.000021 |
| observed_chunk_boundary_bfloat16 | 1.186965 | 0.972561 | 1.055911 | 1.054850 | 1.021843 |
| observed_chunk_tail64_bfloat16 | 1.368643 | 0.847899 | 1.069056 | 1.000771 | 0.999867 |
| fp8_first_chunk_float8_e4m3fn | 1.044671 | 1.018818 | 1.029618 | 1.079663 | 1.021866 |
| fp8_budget_exact_float8_e4m3fn | 1.007343 | 0.976336 | 0.987445 | 1.029688 | 1.012120 |
| fp8_dominant_long_float8_e4m3fn | 1.090977 | 0.960705 | 1.008467 | 1.031549 | 1.011414 |
| fp8_tail112_float8_e4m3fn | 1.135569 | 0.963480 | 1.026961 | 1.028764 | 1.013646 |
| fp8_tail113_float8_e4m3fn | 1.134128 | 0.963112 | 1.025278 | 1.032236 | 1.018554 |
| fp8_two_balanced_float8_e4m3fn | 1.118373 | 0.922277 | 0.994891 | 1.028155 | 1.012340 |
| fp8_two_skewed_float8_e4m3fn | 1.135161 | 0.963444 | 1.026017 | 1.035933 | 1.011981 |
| fp8_short_tail_float8_e4m3fn | 1.171475 | 0.895306 | 1.027317 | 1.072645 | 1.033112 |
| fp8_compact_float8_e4m3fn | 0.993185 | 1.121170 | 1.074372 | 1.000000 | 1.000615 |
| fp8_long100k_float8_e4m3fn | 1.056790 | 0.786253 | 0.908443 | 1.028946 | 1.014055 |

Decision:retain the revised30-case definitions and joint-stage private candidate
as the current baseline. The all-case target remains open,primarily FP8 long100k
and the slightly sub-floor upstream BF16 long-prefix case. Future scoring-only
experiments must retain TopK measurements,these two cases,and short-batch
controls;TopK remains frozen. No production integration or new optimization is
made by this result-recording change. Full standalone correctness and formal
public-entry validation are still required before production integration.
