# QSA prefill: align the score entry cache with its existing tuner key

Pre-coding plan, 2026-10-21. The previous three-round H800 report4f7bfe3
validated round1's ten choices. It did not meet the complete30 per-case goal.
This experiment changes only the rectangular outer cache key, so different
existing LibTuner keys can reach the tuner instead of reusing the first entry.

## Project truth

| Field | Frozen choice |
| --- | --- |
| API/reference | Existing qsa_select_paged_prefill; pinned native vLLM 7a877ae34dcf4a7d46de08eb16db4699e1e0047c and unchanged source SHA256; no equivalent single Torch op |
| Editable | Operator-local LibEntry subclass/decorator, production source pin, benchmark-only source-equivalent previous10 factory and per-chunk trace, focused tests, baseline-selection CLI plumbing and docs |
| Read-only | Global utils/libentry.py, LibTuner key/strategy, kernel arithmetic/signature/do_not_specialize, all tuning YAML choices, compact path, public host/chunks/TopK/workspaces, native source,30 fixtures, tolerances and timer |
| Build | Local Python/shell syntax and pre-commit; H800 JIT/autotuning before reported timing |
| Validation | Full official accuracy, native/FP32 comparison, key/host/JIT invariants, actual first/second chunk cache entries, baseline/source pins and default2 propagation |
| Benchmark | Formal production versus native; same-worker previous round1 ten choices versus formal production; score, common-logits TopK and independent sequential pipeline |
| Active set | scored-prefill30: upstream12 BF16 + observed8 BF16 + FP8 ten; all fixture definitions unchanged |
| Acceptance | Full-pipeline per-case >=1 or near1 (explicit report floor0.98), equal-weight arithmetic mean >1.1; score and TopK continue separately; incomplete/shared reports cannot pass |
| Timing/timeout | Two alternating rounds, warmup200ms/rep100ms, initial80C; worker1800s, full accuracy7200s, cooldown900s/10s; unchanged thermal/foreign-PID guards |
| Profiler | Untimed actual per-chunk returned compiled kernel/config, entry count, PTX/CUBIN and static resources; no hardware-counter claims |
| Autotune | Same rectangular10, compact6, exact existing keys/default strategy; new outer key appends the exact existing LibTuner.get_key result; no new kernel parameters or constexpr offset |
| Fallback | NO_TORCH_COMPUTE_FALLBACK; existing early-return/compact/rectangular and unsupported behavior unchanged |

## Alignment contract

| Field | Preserved contract |
| --- | --- |
| Inputs | Existing SM90 BF16/FP8, H/D/stride/page/metadata/base contract; no host tensor-value reads |
| Output | Same caller-owned int32 indices, -1 padding, visible-block/tie/alias rules and None return |
| Math | Exact unchanged score kernel body, FP32 dot result, ReLU per real head then sum; FP32 logits |
| Chunks/TopK | Same workspace budget, width, chunk offsets/order, persistent_topk object and native CUDA TopK |
| Unsupported | Existing device/dtype/layout/resource and inference-only limits; no new compute fallback |
| Torch use | Existing production metadata/empty allocations/views; all reference compute remains in tests/benchmarks |

## Paths and evidence

| Path | Change | Gate |
| --- | --- | --- |
| Rectangular entry | Operator-local subclass appends full existing tuner key to the original entry key | Actual LibEntry/Tuner key and run contracts; different row counts have separate outer entries while JIT do_not_specialize is unchanged |
| Device and host | None | Exact AST/body/signature/keys equality and differential execution over all32 diagnostic fixtures;30 measured fixtures unchanged |
| Compact/empty/select-all | None | Existing functional and configuration gates |
| Previous round1 | Same current JIT/host/TopK with standard LibEntry and the same10 choices | Source normalization recovers exact d345e5e0... round1 SHA; separate outer caches and identical captured geometry |
| Historical previous8 | Existing factory retained with source-equivalence normalization | Original eight group and original6db... pin still required; historical trials retain their frozen source |
| Timing | Five stages as before; previous-tuning selector explicitly chooses round1 | Same worker/toolchain/GPU, before/after FP32 checks; full-pipeline ratio is not stage-sum arithmetic |
| Trace | Capture actual kernel/config returned by each entry during public observation | No additional GPU launch, no inferred last-best-config assignment to all chunks; old and new traces both saved |

This round deliberately preserves the existing exact-value tuner key instead
of introducing row-count buckets or dispatch heuristics at the same time.
There can be more outer entries and tuning events for distinct existing keys;
this does not turn runtime row/offset arguments into JIT specialization values.
Compiled artifacts remain reusable when configs and compiler signatures match.
Query offsets remain outside the tuner key, exactly as before. A later separate
trial may bucket the key if measured tuning/cache overhead warrants it.

## Implementation and local validation

The formal exported operator now uses `_QSAScoreEntry` only for rectangular
scoring. Source normalization removes that class/import/decorator and recovers
the previous round1 source exactly (SHA256 d345e5e0eeec5de84958f0ffe81aa2ce286ce982448f6d167f94d87e02f7c849).
The current production SHA256 is f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c.
Global LibEntry, tuning YAML, score bodies, host dispatch, fixtures, persistent
TopK and timer are byte-for-byte unchanged from29ef6bf.

CPU tests execute the actual LibEntry.key/run and LibTuner.get_key methods
with a fake device runtime: standard entry tunes once for64/17/64 rows;
the local entry tunes twice, returns the correct separate configurations,
and reuses the first entry for a different query offset. The same tests
cover existing key fields, dtype ordering, baseline isolation, trace capture
and CLI propagation. All32 diagnostic fixture metadata routes remain covered
by the existing differential integration test.

Local validation: `PYTHONPATH=tools python3 -m unittest discover -s tools -p
'test_qsa*.py'` ran191 tests:190 passed,1 skipped. Changed-file pre-commit passed.
These are CPU orchestration/source-contract tests; no GPU result is implied.
Two added H800 BF16/FP8 tests compare actual64/17-row cache miss/hit launches
against FP32 logits with existing tolerances; they run in the full accuracy gate.

The H800 run selects `--candidate production --suite scored-prefill --mode kernel
--compare-previous-tuning --previous-tuning round1 --rounds 2`, preserving the
same10 previous choices and TopK. It records five stages (native score/TopK/pipeline,
plus previous-production score/pipeline) for all30 cases. Per-case JSON adds
`cache_granularity.production` and `.previous`: actual returned compiled kernel
resources, constants, outer entry count and requested tuner key per chunk.
The requested key alone is not proof that the old entry reached the tuner.

H800 report1d9ccf7 completed:221 passed/1 skipped; all30 measurements valid.
Decision: keep the cache alignment for verified key consistency and targeted
tail gains. Native pipeline mean1.1943,12 cases below0.98: target not met.
Same-worker previous10 pipeline mean1.0018; this is not broad acceleration.
See [the full result](qsa_prefill_cache_h800_results.md) for all stage latencies,
TopK data, actual tail configs, jitter, duration and limitations.
Exact existing keys can increase first-call tuning time and outer-cache memory;
kernel-mode timing excludes Python cache-key construction and cannot establish
host end-to-end latency improvements.
