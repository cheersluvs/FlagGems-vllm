# QSA prefill tuning round 1

Pre-coding plan, 2026-10-21. Hypothesis: the current rectangular tuning set
omits the native FP8 TR32/BN64/KT16/warps8 choice. Add that choice and a
TR32/BN64/KT4/warps8 control, preserving the previous eight choices.
The three-round H800 run has completed in report4f7bfe3:219 passed, one skip,
all30 cases measured. Overall pipeline arithmetic mean1.19464x exceeds1.1,
but11 cases are below0.98, so the full goal is not met. See
[the audited results and round-count analysis](qsa_prefill_round1_h800_results.md).
Subsequent routine runs default to two alternating rounds; boundary cases need
additional confirmation. The tables below preserve the original three-round plan.

## Project truth

| Field | Frozen choice |
| --- | --- |
| API/reference | Existing qsa_select_paged_prefill; pinned native vLLM 7a877ae34dcf4a7d46de08eb16db4699e1e0047c, existing SHA256; no equivalent single Torch op |
| Editable | New NVIDIA rectangular tuning group and production decorator selection, tuning pin, every-config tests, old-eight test/benchmark clone, scored30 suite, stage runner/report plumbing and docs |
| Read-only | Device arithmetic, host dispatch and chunking, compact six configs, previous rectangular eight configs, TopK, native source, all32 fixture definitions/geometry, numerical tolerances and timer |
| Build | Local Python/shell syntax and pre-commit; H800 JIT/autotuning outside reported stage timings |
| Validation | Existing official native/FP32/unsupported/alias/device coverage; every rectangular config across BF16/FP8/H/D/layout/chunk boundaries; actual public entry, previous-eight equivalence and CPU report contracts |
| Benchmark | Same verified FlagTree/ptxas13.4/CUDA13.4compat; --candidate production --suite scored-prefill --mode kernel; native score/TopK/pipeline plus old-eight score/pipeline paired in the same worker |
| Active set | 30 scored cases: upstream12 BF16 + observed8 BF16 + FP8 ten; select-all single and batch18 remain functional/diagnostic fixtures but leave the standard benchmark and acceptance set |
| Metric | Primary: native complete sequential GPU pipeline / production pipeline, including TopK; score and common-logits TopK remain separately reported; Python validation/allocation excluded as before |
| Aggregation | Equal-weight arithmetic mean over exactly30 native/production pipeline ratios >1.1; per-case >=1 or near1; report >=0.98 as the proposed near1 threshold explicitly, not a statistical significance test; retain subgroup geometric means for diagnosis |
| Timeout/timing | Three alternating rounds, warmup200ms/rep100ms, worker1800s, full accuracy7200s, initial80C, cooldown900s/10s; unchanged foreign-PID/thermal guards |
| Profiler | Existing ncu/nsys discovery, compiled resources/PTX/CUBIN and actual config snapshots; tuner scans are diagnostics, not acceptance timing or hardware counters |
| Autotune | New qsa_select_paged_prefill_tuning_round1_score: original8 + native FP8 choice + KT4 control =10; existing keys including IS_FP8/default policy retained; compact group6 and historical group8 unchanged |
| Fallback | NO_TORCH_COMPUTE_FALLBACK; existing generic rectangular/compact paths and unsupported errors; no native kernel substituted into production |

## Alignment contract

| Field | Preserved contract |
| --- | --- |
| Device/input | SM90, matching BF16/FP8 E4M3FN Q/cache, parameterized H/D, existing strided metadata/page bounds/nonzero query base |
| Output | Caller-owned contiguous int32 indices and -1 padding; same None API, visible bounds and tie/numerical policy |
| Math | Same FP32 dot result, ReLU before summing real heads, FP32 logits; no casts or layout conversions introduced |
| Chunking/TopK | Same512MiB budget, chunk boundaries/order/workspaces and persistent_topk; native CUDA TopK unchanged |
| Unsupported/autograd | Existing inference-only and resource/device/dtype/layout limits unchanged |
| Reference use | Torch compute only in tests/benchmark references, never production |

## Paths and evidence

| Path | Change | Gate |
| --- | --- | --- |
| Rectangular | Only decorator config-group name changes;10 choices | Exact old source recovered by reverting that literal, every-config numerical validation and new tuning-cache identity |
| Compact/empty/select-all | None | AST/body/config equality and official tests |
| Previous-eight | Test-only clone of exact current JIT/public host with frozen-eight tuner, separate LibEntry caches | Old production source SHA recovered, same TopK identity, actual old/new chunk geometry and pre/post FP32/native checks |
| Benchmark | New scored-prefill30, native three stages and previous-eight two stages | No native-reference changes, missing/failed/duplicate/select-all cases cannot produce full acceptance aggregate |
| Report | Overall arithmetic mean and per-case floor separately from technical run status | Incomplete/shared measurements cannot pass acceptance; primary complete pipeline timed independently, never summed stages |

The first card run tests the hypothesis; adding a candidate does not prove an
improvement. Retain/revert only after inspecting actual native and old-eight
pipeline ratios, each slow case, and all30 correctness/performance results.

## Implementation and local validation

The production file differs from the previous accepted file only by the
rectangular decorator's config-group literal. Replacing that literal recovers
the exact previous SHA256
6db2918b5b7dc1798244548972458a8c59745f967341a10663e2d704cf35687a.
The round1 production SHA256 is
d345e5e0eeec5de84958f0ffe81aa2ce286ce982448f6d167f94d87e02f7c849.
The compact six and historical rectangular eight config groups are unchanged.
No production Torch call, new dispatch/validation/registration, cache hint,
copy/cast, extra workspace or TopK change is introduced.

The previous-eight comparison is benchmark-only. It reuses the exact JIT and
public host bytecode, with fresh rectangular and compact LibEntry caches and
the same persistent_topk callable. Its source-equivalence pin rejects changes
beyond the config-group literal. Both sides' real public chunk sequences are
captured and checked; old-eight score and pipeline are validated against the
existing FP32/native-selection contract before and after timing. Each pipeline
has its own output/workspace, and no timings are derived by subtracting stages.

Tuner evidence records the ten/eight/six candidates and last tuning event,
outside acceptance timing. Actual compiled config/resource snapshots remain
separate. LibEntry may reuse one choice across chunks; the last tuning event
is not presented as a per-chunk exhaustive scan. Historical six/eight-choice
experiment fixtures retain their original groups and source pins.

Local validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/qsa-round1-pycache PYTHONPATH=tools \
  python3 -m unittest discover -s tools -p 'test_qsa*.py'
```

186 tests ran:185 passed, one optional Torch-dependent test skipped. Config
and source pins, independent previous JIT/host binding, exact30 case selection,
20 BF16/10 FP8 dtype routing, arithmetic-vs-geometric aggregation, per-case
floor and mean gates, partial/duplicate/changed/failed/shared/nonfinite report
rejection and CLI forwarding passed. Pre-commit and syntax checks pass. Local
CPU checks do not validate GPU compilation, numerics or speedup.

## H800 run

Use the existing verified vLLM0.31 virtualenv, /opt/flagtree, ptxas13.4 and
CUDA13.4 compatibility-library wrapper. Run as a child bash, not source:

```bash
bash tools/run_qsa_prefill_flagtree_ab_h800.sh \
  --flagtree /opt/flagtree --candidate production \
  --suite scored-prefill --mode kernel --compare-previous-tuning \
  --rounds 2 --kernel-warmup-ms 200 --kernel-rep-ms 100 \
  --max-initial-temp-c 80 --timeout 1800 --accuracy-timeout 7200 \
  --cooldown-timeout 900 --cooldown-interval 10
```

Full official accuracy runs once before timing, including all ten rectangular
configs across the existing parameterized BF16/FP8 fixtures. The expected
official count remains219 passes and one second-device skip, but actual card
results are pending. The worker then separately validates and measures:

1. Native score versus formal production score.
2. Native TopK versus unchanged repository persistent_topk on common logits.
3. Native pipeline versus formal production pipeline (primary acceptance).
4. Previous-eight score versus formal production score (incremental evidence).
5. Previous-eight pipeline versus formal production pipeline (incremental evidence).

At three rounds, all30 cases produce150 stage records,900 raw samples and600
timing telemetry points. Only one full accuracy run is needed; previous-eight
diagnostics add stage timing, not a second full pytest run. Standard timing
and thermal/foreign-PID safeguards are unchanged. Shared-GPU runs remain
exploratory and cannot pass acceptance.

The existing wrapper automatically uploads JSON/TXT/TSV reports on success or
failure. It now also uploads overall-acceptance.json. Technical exit0 means
the run completed; performance success is separately reported by target_met,
requiring a complete same-run30-case set, arithmetic mean >1.1 and every
pipeline ratio >=0.98 under the explicitly proposed near-one threshold.
Subgroup geometric means remain diagnostic and do not replace this new goal.

The original run above used three rounds. Routine commands now use two rounds
based on the observed variation; one/two-round estimates in the results document
are recomputed subsets of that original report, not additional GPU runs.
The configuration remains under evaluation: the old-eight comparison contains
a0.97582x pipeline regression and the full30 goal is not yet met. No production
or tuning change is made as part of reducing the routine round count.
