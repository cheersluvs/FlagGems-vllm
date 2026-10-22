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
