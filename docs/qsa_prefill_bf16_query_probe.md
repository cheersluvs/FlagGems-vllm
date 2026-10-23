# BF16 fixed-configuration query subdivision probe

Pre-coding board, 2026-10-23. Evidence: capture3621d9ca (36 scorer launches)
shows BF16 long candidate registers234/shared80KiB/active warps12.06–12.30%,
yet faster independent scoring than the canonical36% active-warp case. FP8
query subdivision and stages1 coexist with154 rather than226 registers.
Hypothesis: subdividing the BF16 query tile may reduce the live accumulator
footprint without reducing key reuse. Occupancy is an observation, not a target.
This follows the Claude session workflow: attribute -> one-hypothesis probe ->
representative A/A+A/B -> full regression -> integration. optimization.md and
deep_opt.md are absent. No production change or gain is claimed before H800 data.

| Project truth | Frozen choice |
| --- | --- |
| Operator/reference | Existing QSA prefill; SHA-pinned native vLLM7a877ae; original measured candidate source1ef12d43 and canonicalf1c5e71a |
| Editable | New private probe kernel, standalone coordinator/worker, wrapper, CPU checks, this board |
| Read-only | Production/candidate sources, YAML, registered tests/benchmark30, reference/math/tolerances/timer/GPU guards |
| Build | JIT only; Python compilation and changed-file pre-commit locally |
| Validation | Two small forced-mode boundary fixtures, sampled FP32-reference score and selection checks before/after timing, frozen binary/config/input checks |
| Benchmark | Existing kernel-mode graph timer, baseline_us/candidate_us; score, identical production TopK, actual sequential pipeline |
| Active set | One existing BF16 n1/q16384/p102400 fixture, all four original chunks4519/4519/4519/2827, width29696; H4/D128/page196 |
| Aggregation | Two raw rounds, alternating side order; per-stage median ratio and paired range; A/A controls for each scorer and pipeline; no full30 mean |
| Timeout | Whole isolated worker1800s; no repeated native benchmark, full pytest or NCU replay in this initial probe |
| Profiler | Previous full NCU capture informs hypothesis; this probe records compiled resources/cubin/PTX hashes only, no dynamic counter claim |
| Autotune | Explicit diagnostic exemption: exact measured R64/N64/KT16/W4/S1/CTA1, only QUERY_PARTS1vs2; bypass tuner in private FixedEntry, no YAML change |
| Fallback | No torch compute in scorer; torch input/reference/canary operations only outside timing; no fallback |

| Contract | Preserved |
| --- | --- |
| API | Existing qsa_select_paged_prefill host and private score launcher cloned for diagnostics, inference-only |
| Device/dtype/layout | H800 SM90, BF16 Q/cache H4/D128, page196; small padded/global-base fixtures check strides and tails |
| Output | FP32 logits, visible-column masking and real stores; existing int32 TopK selection and padding |
| Math | Full-D dot to FP32; per-head ReLU then head sum; same reference tolerances; no reduction-dimension split |
| Boundaries | Query tails31/32/33/63/64/65, request/chunk boundaries, nonzero query base, zero visibility, partial last key block |
| Unsupported | Other shapes/dtypes are outside probe scope; fail closed rather than infer general acceptance; production coverage unchanged |
| Torch reference | Only setup/validation; inference inputs and measured scorer do not use a torch compute fallback |

| Path | Implementation | Config/test/risk |
| --- | --- | --- |
| Baseline | Original private combined scorer with fixed P1 and unchanged host/TopK | Exact SHA/config/input/chunk pins; original stores retained |
| Candidate | Source-identical private copy except split-query constexpr gate admits BF16 | Fixed P2, two full-D dot/reduce/store subtiles, same page lookup/key load/loop/grid |
| Small validation | Direct score calls with canary logits, all rows against reference and baseline | Same fixed large tile, padded/global-base and split/chunk tails; not timed |
| Main measurement | Publicly observed baseline chunks, same logits/workspace/output addresses for both sides | Compile first; immediate per-side validation; freeze and compare resources before/after |
| Controls | Same callable and buffers A/A for P1/P2 score and pipeline; TopK P1vsP2 is identical-call control | Report all samples; differences within noise are not a win |
| Production | No import, dispatch, config, export or registration edits | Full correctness and30-case regression required after a useful probe, before integration |

## Run and delivery

Use the existing vLLM0.31 virtualenv Python with -I. The wrapper explicitly
routes Triton/TLE to /opt/flagtree and loads the pinned CUDA13.4 compatibility
library in a fresh process. It reuses the SHA-verified ptxas path and native
reference in the saved profiling report. Compiler/library hashes must match
that capture; extracted library paths and GPU addresses need not match.
No native scorer launch or new native benchmark is required for this probe.
The original input fingerprints must match exactly.

```bash
QSA_STOCK_PYTHON=/tmp/qsa-vllm031-h800.PQnVpV/venv/bin/python \
  bash tools/run_qsa_prefill_bf16_query_probe_h800.sh \
  --flagtree /opt/flagtree \
  --baseline-report /tmp/qsa-profiling-h800.RFNO3i
```

Default rounds2, warmup200ms/rep100ms, initial temperature limit80C and bounded
cooldown120s. Both sides use exactly the same score/workspace/output buffers.
Standalone TopK uses frozen original P1 logits and the identical callable on
both sides, so its ratio is a control, not an optimization claim. Four A/A
controls cover P1/P2 score and pipeline. Every timed comparison retains
before/after numerical and foreign-process/thermal checks. The long case uses
sampled reference rows, including request/chunk boundaries; each small fixture
checks all289 rows plus both canaries and every unwritten column. These are
probe guards, not the full production correctness suite.

Console tables show score/TopK/pipeline latencies and ratios, paired ranges,
A/A data and per-chunk registers/shared/spills. JSON retains all raw samples,
input/runtime/source fingerprints and pre/post compiled-resource records.
Static PTX site counts are never interpreted as executed instruction counts.
No launch is removed, no store is elided and no timer is changed.
The child-shell wrapper tees terminal output, uploads JSON/TXT/TSV even after
a probe failure, and preserves the measured exit code. terminal-summary.txt
is a saved copy of the console log. Push failures print the exact retry
command and local report path, without rerunning the probe.

Local validation:38 affected CPU checks passed (eight new probe checks),
including the complete mocked worker success/failure paths, two-round order,
TopK visibility, source-only gate verification, baseline/config rejection,
compiler-content identity, same-buffer score-to-TopK chaining and unchanged
fixed-entry/full-D split contracts. Changed-file pre-commit, shell syntax and
CLI help passed. The actual3621d9ca BF16 report passes the baseline loader and
compiler-identity extraction locally. H800 compilation, numerical correctness
and performance of BF16 P2 remain unmeasured until the card run returns.

## Upstream12 expansion: pre-coding board (2026-10-23)

Probeed62e5dc: score1.028196, TopK0.999924, pipeline1.016504 for the long
BF16 fixture; both paired pipeline samples positive. Registers234->176,
shared80->88KiB. This supports retaining P2 privately, not production integration.

| Project truth | Frozen choice |
| --- | --- |
| Editable | Existing probe coordinator/worker and wrapper, new upstream helper/config manifest and CPU checks, this document |
| Read-only | Canonical and both private kernel sources, native reference, YAML/policies, original upstream12 definitions, timer/tolerances/GPU guards |
| Active set | Exact upstream12 BF16 cases, every actual public chunk |
| Configuration provenance | Reuse per-chunk selected P1 configs and input fingerprints from verified reportf6a41ae371cbc0907bb2b2aa4e2eca2c49966a18; freeze a SHA-pinned manifest, no retuning |
| Metric | Same-run P1/P2 kernel-mode score/TopK/pipeline, rounds2,200ms/100ms; raw samples and paired ranges retained |
| Aggregation | Arithmetic mean, minimum and cases below1 per stage; full12 completeness explicit; diagnostic only, not native-vLLM/full30 acceptance |
| Validation/build | Source/manifest/runtime/input pins; first-case small all-row boundary guards; per-case pre/post numerical checks and compiled/config/common-logit freeze; CPU/pre-commit/shell checks |
| Profiler/timeout | Static compiled resources only; no new NCU replay or native timing;1800s per case,120s cooldown,80C initial limit |
| Autotune exemption | Private matched-config probe uses each original P1 tile/warps/stages/CTA, changing only query parts; no YAML/import/registration changes |
| GPU delivery | Existing virtualenv + explicit FlagTree + pinned ptxas/CUDAcompat; isolated child per case, partial reports uploaded on failure |

| Contract | Preserved |
| --- | --- |
| Inputs | Exact original upstream BF16 H4/D128/page196 inputs, strides, starts, visible lengths,512MiB budget and physical page maps; fingerprint equality mandatory |
| Math/output | Full-D FP32 dot, per-head ReLU/head sum, real masked FP32 stores and original int32 selection semantics |
| Boundaries | Actual request/chunk tails from each case; two existing small boundary fixtures once before the suite |
| Unsupported/training | No new supported API/dtype/layout; inference only; no production torch fallback |

| Path | Behavior | Gate |
| --- | --- | --- |
| Frozen input/config loading | Verified old case blob SHA/provenance captured in manifest; no old timing included | Original fixture/config source and compiler content must match |
| P1/P2 measurement | Original public host discovers chunks under fixed P1; shared buffers, identical TopK callable and all real chunks | Unknown/mismatched geometry rejected, all numerical/telemetry guards retained |
| Suite coordination | Sequential isolated workers, per-case live tables and cumulative stage means/minima, JSON/TSV | First failure stops, incomplete12 not presented as full12 success |
| Integration | Not part of this change | Full correctness/full30 same-run validation required later |

Enable the expansion with `--suite upstream-prefill` on the same wrapper.
The46KB manifest contains only case/input/config/compiler fingerprints and
original report blob identities, never historical timings. All12 original
case blobs were verified against the Git tree atf6a41ae before extraction.
It freezes26 chunks, including mixed R64/KT16 and R32/KT4 configurations within
one shape; baseline per-chunk warps/stages/CTA remain exact. No autotune runs.
The original single-case mode remains the default.

Each case has an isolated worker and independent JSON/TSV. Small boundary
checks run once in the first requested case; later records identify that case.
Every case retains all numerical guards, four A/A controls and environmental
checks. Cumulative terminal tables/summary report arithmetic mean, minimum and
below-one names for score/TopK/pipeline. Only all12 validated records can mark
full_upstream12_complete; explicit `--cases NAME ...` subsets stay partial.
A failure stops later workers and preserves completed and failing raw files.
Use `upstream-probe-timings.tsv` and terminal-summary.txt for the combined view.
These P1/P2 ratios are not native-vLLM speedups and cannot be multiplied by
old report ratios to infer new native-vLLM performance.

Validation:45 affected CPU checks passed, including frozen-manifest integrity,
real mixed chunk configs, compiler/content matching, suite-worker case selection,
first-case boundary provenance, full12 coordination, partial-failure stopping,
numerical/order/telemetry rejection and unweighted aggregation. Changed-file
pre-commit, CLI help, shell syntax and diff checks passed. Kernel sources,
production/YAML/benchmark definitions and the original timer remain unchanged.
The expanded H800 run is pending; no new production optimization is accepted.
