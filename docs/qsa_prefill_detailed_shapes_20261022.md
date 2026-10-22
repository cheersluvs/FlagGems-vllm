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
