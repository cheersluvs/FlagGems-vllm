# QSA prefill: FP8 benchmark extension

2026-10-22 update: the eight captured FP8 fixtures now use the detailed upload
9937ec6e,cache [2712,196,1,128] and table [N,88]. tail64/tail65 are renamed
tail112/tail113;new native/canonical/private baselines are required. The30-case
active set is preserved. Two new select-all captures are functional-only. See
[the detailed shape audit and update](qsa_prefill_detailed_shapes_20261022.md).
The preceding fixture values and GPU reports below remain historical results.


Current status, 2026-10-21: the standard benchmark and scored-prefill acceptance
set now have30 scored cases (20 BF16 and10 FP8), excluding the two select-all
fixtures. combined-prefill retains32 diagnostic fixtures. The primary goal is
complete GPU pipeline speedup including TopK, with a30-case arithmetic mean
above1.1 and each case at or near1. See [round1](qsa_prefill_tuning_round1.md).
The sections below record the preceding32-case extension and its validation.

Pre-coding plan, 2026-10-20. Add the agreed eight envelopes from upload
7835b9fd94c6f0c75339a9e156d2fc2cb4183186 plus two explicit FP8 dtype variants.
Archive SHA256: 857381f56826cf7735b7acf54c158fcf4dc850bffd9347faeb220e60b394e106.
The old12 files are unchanged; the new FP8 capture has126 prefill tuples.

## Project truth

| Field | Frozen choice |
| --- | --- |
| API/reference | Existing qsa_select_paged_prefill and pinned native vLLM 7a877ae34dcf4a7d46de08eb16db4699e1e0047c; no equivalent single Torch op |
| Editable | FP8 workload definitions/provenance manifest, canonical benchmark/runner dtype plumbing, CPU/GPU tests and docs |
| Read-only | Production SHA256 6db2918b5b7dc1798244548972458a8c59745f967341a10663e2d704cf35687a, TopK, existing tuning choices, native source, old22 BF16 case definitions and numerical tolerances |
| Build | Local Python/shell syntax and pre-commit; GPU JIT on card outside timing |
| Validation | CPU matrix/provenance/geometry/dtype/aggregation/CLI contracts; official GPU suite plus ten FP8 envelope/native/FP32 stage checks |
| Benchmark | --candidate production --mode kernel, same FlagTree/ptxas13.4/compat runtime; score-only NV acceptance, TopK and sequential pipeline still reported |
| Active set | combined-prefill32 = upstream12 BF16 + observed10 BF16 + fp8-prefill10; two BF16 select-all cases skip score/TopK |
| Aggregation | Separate upstream_perf / observed_perf / fp8_perf geometric means, native/production microseconds, >1 faster; never pool dtypes |
| Timing/timeout | Existing3 rounds, warmup200ms, rep100ms, initial80C; worker1800s, accuracy7200s, cooldown900s/10s; actual thermal/foreign PID guards unchanged |
| Profiler | Existing static resources and ncu/nsys discovery; no hardware-counter inference |
| Autotune/fallback | Production eight rectangular/six compact choices unchanged; no new kernel, no Torch compute fallback; tests/reference Torch only |

## Contract and provenance

| Field | Choice |
| --- | --- |
| Inputs | Explicit Q/cache float8_e4m3fn, H4 D128; ratio4, block_k512, page196, 512MiB logits, compact layout/base0 as before |
| Output/numerics | Existing int32 logical block indices and -1 padding; FP32 logits, ReLU/head sum unchanged; reuse existing SM90 FP8 normalization and tolerances |
| Recorded | Eight actual (T,N,H,D,max_query_len,max_seq_len) tuples and counts; N=2 length multiset inferred from T/max_query_len |
| Synthetic | Actual indexer dtype is not captured; explicit FP8 is a benchmark choice. Request order, per-request contexts, cache/table extent/strides, visible counts and values retain prior construction assumptions |
| Boundaries | 64/65 tail chunks depend on tight M=42 table capacity; not an observed runtime chunk trace. Larger table changes tails |
| Unsupported/autograd | Existing SM90 inference-only and unsupported behavior unchanged |
| References | FP32/native only in tests/benchmark. TopK uses identical frozen native logits and strict selection checks |

## Paths and integration

| Path | Work | Check |
| --- | --- | --- |
| Original BF16 | Keep exact12+10 definitions | Frozen semantic digest and existing correctness |
| FP8 recorded | Eight scalar-envelope fixtures | Archive-derived manifest membership, exact geometry and native/FP32 comparison |
| FP8 supplemental | ragged54 and n1/q16384/p102400 dtype variants | Mark as variants, never as new captured shapes; compact/long-prefix coverage |
| Pipeline validation | Select existing FP8 reference contract from actual Q dtype/D | BF16 behavior preserved; no tolerance changes |
| TopK validation | Same FP32 logits for both sides, actual dtype/D for direct alignment | No TopK algorithm changes or selection-check weakening |
| Pytest benchmark | Each named case only runs its declared dtype | 22 BF16 plus10 FP8 inputs, not32x2 |
| Runner | fp8-prefill and combined-prefill; separate group and full accuracy first | Source/export/compiler identity and automatic report upload retained |

No performance result is claimed by adding a shape. GPU validation remains
pending until the user executes the card command and returns the report.


## Added matrix

Every new case explicitly uses `float8_e4m3fn` Q/cache, H4/D128 and block_k512.
The table contains constructed metadata; only the eight recorded six-tuples
and their counts come from the new capture. Missing fields retain the prior
assumptions described above. FP32 logits still use four bytes per element.

| Case | Query lengths | Batch max_seq_len | Logits W | Chunk rows | Origin |
| --- | --- | ---: | ---: | --- | --- |
| fp8_first_chunk_float8_e4m3fn | [16384] | 16384 | 4096 | [16384] | recorded_envelope |
| fp8_budget_exact_float8_e4m3fn | [16384] | 32768 | 8192 | [16384] | recorded_envelope |
| fp8_dominant_long_float8_e4m3fn | [16331] | 33842 | 8512 | [15768, 563] | recorded_envelope |
| fp8_tail64_float8_e4m3fn | [16368] | 32852 | 8232 | [16304, 64] | recorded_envelope |
| fp8_tail65_float8_e4m3fn | [16369] | 32850 | 8232 | [16304, 65] | recorded_envelope |
| fp8_two_balanced_float8_e4m3fn | [10375, 5954] | 32931 | 8256 | [16256, 73] | recorded_envelope |
| fp8_two_skewed_float8_e4m3fn | [16331, 53] | 32821 | 8232 | [16304, 80] | recorded_envelope |
| fp8_short_tail_float8_e4m3fn | [2319] | 33843 | 8512 | [2319] | recorded_envelope |
| fp8_compact_float8_e4m3fn | [3200]+[112]*38+[96]*15 | 4100 | 1088 | [8896] | dtype_variant |
| fp8_long100k_float8_e4m3fn | [16384] | 118784 | 29696 | [4519, 4519, 4519, 2827] | dtype_variant |

For tail64/tail65, a tight42-column table caps W at8232, so512MiB holds16304
rows. A64-column table instead gives W8256 and tails112/113; the actual runtime
page-table width is unknown. The manifest records source tuples/counts/line
numbers without claiming that derived chunks or the FP8 dtype were captured.
Supplemental variants have no fabricated new-capture count.

## Suites and measurements

| Suite | Cases | Groups |
| --- | ---: | --- |
| upstream-prefill | 12 | upstream_perf BF16; unchanged CLI default |
| observed-prefill | 10 | observed_perf BF16 |
| fp8-prefill | 10 | fp8_perf; explicit FP8 |
| combined-prefill | 32 | Three separate groups above |

The normal pytest benchmark now runs22 BF16 inputs and10 FP8 inputs. Each
named shape is filtered by its declared dtype in get_input_iter; it does not
run32 shapes twice. Its score/TopK tests have30 inputs after excluding the two
select-all cases. Core and comprehensive share this same matrix.

The same-compiler runner validates the actual Q/cache dtype against the case,
checks exact shapes/layout/chunks, captures the actual formal public entry,
and stores source snapshots including the new manifest. Preflight now checks
both BF16/FP8 with8 and128 rows before the full correctness gate. All ten new
FP8 cases have named official functional tests against native vLLM and sampled
independent FP32 references, including real public chunk capture. Missing or
skipped required checks block performance timing.

Pipeline selection validation dispatches by actual Q dtype and head_dim;
FP8 uses existing assert_upstream_selection normalization/tolerances. BF16
checks remain unchanged. Common-logits TopK retains its strict selection
checks, and native alignment receives the actual input dtype/D. Neither the
production math nor any numerical tolerance is changed.

At3 rounds, fp8-prefill measures30 stages/180 raw samples/120 telemetry points.
The full32 measures92 stages/552 raw samples/368 telemetry points. Both retain
score excluding TopK, common-logits TopK, and independently timed sequential
pipeline. Do not merge FP8/BF16 geometric means. An incomplete group does not
receive an aggregate. Run fp8-prefill for the first added-case performance
validation, or combined-prefill for the entire matrix; full official accuracy
runs before either suite. Production and native use the same verified FlagTree
runtime, ptxas13.4 and CUDA13.4 compatibility library.

Run the existing automatic-upload wrapper as a child bash in the verified
vLLM0.31 H800 environment:

```bash
bash tools/run_qsa_prefill_flagtree_ab_h800.sh \
  --flagtree /opt/flagtree --candidate production \
  --suite fp8-prefill --mode kernel --rounds 3 \
  --kernel-warmup-ms 200 --kernel-rep-ms 100 \
  --max-initial-temp-c 80 --timeout 1800 --accuracy-timeout 7200 \
  --cooldown-timeout 900 --cooldown-interval 10
```

The wrapper uploads existing reports on success and failure. Real thermal and
foreign-process guards remain active. No virtualenv packages, GPU jobs or
host drivers are replaced.

## Local verification and pending card validation

178 CPU tests ran:177 passed and one optional Torch-dependent test skipped.
The new contracts cover all ten cases, archive-derived tuple membership,
unchanged BF16 semantic digest, exact geometry, table-capacity dependence,
dtype input construction, pytest32-vs64 routing, actual FP8 selection-helper
routing, strict common TopK inputs and actual dtype/D, separate/incomplete
aggregation, isolated CLI and missing-FP8-coverage rejection. Python/shell
syntax and pre-commit are checked before publication.

Production source, TopK, tuning YAML, native reference and BF16 numerical
contracts remain unchanged from the accepted production run708f30e. The new
H800 correctness/performance run is pending; these CPU checks are not GPU
numerical or performance results. The full GPU suite is expected to add ten
passes to the previous209, with the same existing second-device skip when
one GPU is exposed; actual results must come from the card report.

## H800 validation returned

Report cd132fad331f065721b54dc17432b7ca5caa0226 completed the pending run:219 passed, one second-device skip, all ten FP8 cases and30 stages completed. FP8 score/TopK/pipeline geometric means are0.98231x/0.91918x/0.94879x. These are valid same-compiler measurements, not an overall FP8 performance gain. See [the audited H800 results](qsa_prefill_fp8_h800_results.md) for latency tables and runtime identity. Earlier pending statements describe the pre-run state.
