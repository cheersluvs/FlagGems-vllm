# QSA prefill: integrate the validated candidate into production

Pre-coding plan, 2026-10-20. This transfers the retained unhinted candidate
validated by report 4b77a6f, without changing its numerical or routing policy.
The evict_first experiment 4436f76 was rejected and is not integrated.

## Project truth

| Field | Frozen choice |
| --- | --- |
| Operator/reference | Existing qsa_select_paged_prefill API; native vLLM 7a877ae34dcf4a7d46de08eb16db4699e1e0047c, existing SHA256-pinned source; no single equivalent Torch API |
| Editable | Production scorer/dispatch; exact old-production snapshot and its test-only loader; production integration tests; runner identity/coverage/resource wiring and historical fixture routing; docs |
| Read-only | Native reference, TopK algorithms, validated trial kernel arithmetic/routing, original six plus expanded eight tuning choices, all22 workload definitions and existing tolerances |
| Build | Python syntax compile and pre-commit locally; GPU JIT/autotuning on card outside timing |
| Validation | Existing native/FP32/unsupported/alias/device suites; every production rectangular eight/compact six config, BF16/FP8, H/D/layout/chunk boundaries; full-entry routes and retained-candidate comparison |
| Benchmark | Actual exported production entry, --candidate production --mode kernel --suite combined-prefill; same FlagTree/ptxas13.4/CUDA13.4 for native and production; score excluding TopK is NV acceptance |
| Active set | Unchanged upstream12 plus observed10 BF16 H4 D128, page196, block_k512, budget512MiB; two select-all cases have only pipeline timing |
| Aggregation | Baseline/candidate latency in microseconds, >1 faster; separate upstream12 and observed8 score/TopK, observed10 pipeline geometric means; no cross-report ratios |
| Timing / timeout | 3 alternating rounds, warmup200ms, rep100ms; worker1800s, accuracy7200s; initial limit80C, cooldown900s/10s; retain actual thermal/co-resident rejection |
| Profiler | Existing ncu/nsys availability detection plus untimed compiled resources, actual configs/route and PTX/CUBIN hashes; no dynamic hardware-counter claims without a profiler |
| Autotune | Rectangular uses existing qsa_select_paged_prefill_reuse_score eight choices; compact uses original qsa_select_paged_prefill_score six; identical keys/default strategy/do_not_specialize; YAML unchanged |
| Fallback | NO_TORCH_COMPUTE_FALLBACK; noncompact routes to production key-major; no compute/copy/cast fallback |

## Alignment contract

| Field | Preserved contract |
| --- | --- |
| Target | NVIDIA SM90 H800/H100; other platforms require separate validation |
| Inputs | Matching BF16/FP8 E4M3FN Q [T,H,D], cache [pages,page_size,1,D], positive H/D, supported nonoverlapping strides; int32/int64 starts/table and int32 visible counts |
| Output | Same None API and caller-owned contiguous int32 [T,token_topk/compress_ratio], logical compressed indices and -1 padding; alias checks unchanged |
| Numerics | FP32 dot result, ReLU per real head then sum; unchanged masks/padding and numerical/tie tolerances, finite inputs; no NaN/Inf or ordering promise added |
| Metadata | Device-resident starts may include empty requests and nonzero base; preserve chunk intersection, invalid-page masks, table stride and visible bounds |
| Autograd/unsupported | Inference only; existing dtype/device/layout/TopK unsupported errors remain |
| Torch use | Production metadata, torch.empty and true views only; all independent Torch/native checks confined to tests/benchmark |

## Paths and integration

| Path | Implementation | Tuning / validation |
| --- | --- | --- |
| Empty / select all | Existing public body and fill unchanged | Existing fixed fill exemption; official boundary/native tests |
| Low-utilization ragged | Exact compact mapping and query-major math from retained trial | Six configs; gate requests>1, T<requests*max_query_len, chunk_rows>max_query_len and requests*max_query_len>=4*chunk_rows; every-config/chunk tests |
| Remaining scored cases | Exact unhinted key-major rectangular scorer | Eight existing configs; every-config BF16/FP8 and arbitrary-H/D boundary tests |
| Public host / TopK | Existing validation, chunking, per-call metadata/workspaces and persistent_topk unchanged | Export/registration already exist; identity and AST contracts; full-entry correctness |
| Historical experiments | Exact old production saved under tools; historical trials explicitly receive it | SHA256 stays d70faeb6928c2c7a56636ded833f3359368871e353ecf4c66c6dfd35cae93e73; never production-imported |
| Benchmark | Native score/TopK/pipeline paired against actual canonical module | Record canonical path, SHA256, public-export identity, route/config/resource evidence; automatic report upload on success/failure |

At planning time, production-entry H800 validation was pending. Historical
candidate measurements do not count as measurements of the new production
entry. The completed run is recorded below. No new algorithm, cache hint,
staging change or benchmark shape is part of this integration.


## Local implementation and validation

Production SHA256 after integration:
`6db2918b5b7dc1798244548972458a8c59745f967341a10663e2d704cf35687a`.
The rectangular kernel matches the retained key-major AST exactly apart from
its name/config decorator, and the compact kernel/gate match their retained
ASTs exactly. The public host, validators, metadata preparation, fill and
persistent_topk call remain unchanged. Both existing YAML config groups and
all22 workload definitions remain unchanged; no new registration is needed
for the existing exported direct-call API.

The old production file is retained byte-for-byte under
`tools/qsa_prefill_frozen_production.py`. Historical compact/key-major/reuse/
eviction/vector tests explicitly use that module, preserving their source
pins. This snapshot is test/benchmark-only and is not imported by production.
Legacy pre-preparation host comparisons likewise explicitly take the frozen
module; their existing numerical assertions remain intact.

Local command:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/qsa-integrate-pycache PYTHONPATH=tools \
  python3 -m unittest discover -s tools -p 'test_qsa*.py'
```

170 CPU tests ran, 169 passed and one optional Torch-dependent test skipped.
The differential integration tests execute actual production/retained public
host bytecode over all22 envelopes plus chunked/strided BF16/FP8 metadata
fixtures, comparing grids, arguments, allocations and TopK calls. They do not
simulate GPU numerical correctness. Syntax, shell syntax, source pins and pre-commit passed. Mac GPU compilation/numerical/performance validation is unavailable.

Production-path Torch audit:

| Call | Role | Decision |
| --- | --- | --- |
| torch.empty | Same uninitialized logits and TopK scratch allocations | Allowed; unchanged |
| torch.cuda.get_device_capability / torch.cuda.device | Existing device checks and context | Metadata/runtime only; unchanged |
| Tensor shape/stride/dtype/device/is_contiguous/data_ptr/element_size | Existing validation and launch metadata | Metadata only; unchanged |
| Tensor slicing/view | Existing contiguous buffer views | No data copy; unchanged |
| .to(tl.int64) | Existing Triton device pointer arithmetic | Not a Torch Tensor cast |

No Torch compute fallback, tools/tests import, new copy/cast, cache hint or
extra device workspace has been added to the production path.

## H800 run and report interpretation

Use the existing verified vLLM0.31 environment and automatic-upload wrapper:

```bash
bash tools/run_qsa_prefill_flagtree_ab_h800.sh \
  --flagtree /opt/flagtree --candidate production \
  --mode kernel --suite combined-prefill --rounds 3 \
  --kernel-warmup-ms 200 --kernel-rep-ms 100 \
  --max-initial-temp-c 80 --timeout 1800 --accuracy-timeout 7200 \
  --cooldown-timeout 900 --cooldown-interval 10
```

The wrapper records and verifies the actual `/opt/flagtree` compiler, ptxas
13.4.59, CUDA13.4 compatibility library, native TopK extensions and one idle
H800 UUID. Both native and production scorers use that same compiler. It
uploads existing reports on success or failure; no GPU jobs are killed.
Run it in a child bash, not by sourcing it into the interactive container.

`--candidate production` uses the actual canonical module and checks its
public function identity against `flaggems_vllm.qsa_select_paged_prefill`.
A private-module clone, different export or source pin fails before timing.
`runtime.operator_binding` records both paths/hashes, canonical-module and
export identity flags, and function module. This binding participates in the
preflight/accuracy/worker runtime identity comparison and is printed to the
terminal. Compile configs, compact chunk offsets and PTX/CUBIN resources are
captured outside timing, including the production compact kernel.

Full accuracy runs `tests/test_qsa_select_paged_prefill.py` without --quick.
That official file includes existing dtype/layout/unsupported/native gates and
new actual-production observed-envelope, eight rectangular/six compact config,
chunk/layout guard and retained-candidate full-entry comparisons. Every config
check uses independent FP32 references and the existing tolerances. Production
runs require the integration test names in the saved correctness report; a
missing/skipped required check blocks all timed workers. Historical trial
suites remain separately runnable, without repeating their full test matrix
in the production profile.

The unchanged matrix has20 scored cases and2 direct-fill cases, yielding62
timed stages (20 each score/TopK/pipeline, plus2 fill pipelines),372 raw timing
samples and248 before/after-round telemetry points with3 rounds. Score capture
and stage binding derive the real chunks from the actual public invocation;
the measured candidate uses no private factory or TopK substitution in this
FlagTree production profile. Correctness tests may construct historical
candidates as additional references.
TopK still reads common frozen native logits; pipeline is independently timed
as score then each side's TopK. NV acceptance uses score only. Two select-all
cases must report score/TopK N/A. Keep upstream12 and observed8/10 means separate.

A successful historical-candidate report is not a successful production report.
The new H800 correctness/22-case performance result has now been returned and
audited; see the completed-run section below.


## Completed production run, 2026-10-20

Report `708f30eba6278b18cdc7b7020cfabe59e1682965` ran source
`cd7195cfcd2aedac70b6879f8e81ecd31e240829`: 209 passed, one existing
second-device skip and all22 benchmark cases completed. All37 new official
integration fixtures passed, including12 full-entry checks against the
retained candidate. The canonical-module/export identity flags and reviewed
production SHA256 match in preflight, accuracy and every timed worker.

Native/production score-only geometric means are1.05553x for upstream12 and
2.00593x for observed8. TopK is0.89554x /0.89676x; pipeline is0.96716x for
upstream12 and1.93485x for observed10. The upstream score minimum is0.98206x
(two cases below1); the observed score minimum is0.98664x (one below1).
These are direct same-card measurements of production against native vLLM,
not an integration increment inferred from earlier reports. TopK stays frozen.

Keep the integration and use this production version as the next optimization
baseline. Full paired samples, all22 score/TopK/pipeline results, actual
configs/resources and runtime evidence are documented in
[qsa_prefill_production_h800_results.md](qsa_prefill_production_h800_results.md).
No production/config/workload changes are included in recording this result.
