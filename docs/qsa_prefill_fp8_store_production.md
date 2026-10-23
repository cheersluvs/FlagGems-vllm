<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# FP8 bulk/tail production integration

Pre-code boards, 2026-10-23. `optimization.md` and `deep_opt.md` are absent.
Evidence: report d50b3e0d1ee0d853b82640529c76448fb3fa82a2, code e93e496.
Private combined baseline versus FP8-only bulk/tail: ten-case arithmetic mean
score 1.0755083, TopK 0.9999689, pipeline 1.0278342. This is not native-vLLM
performance or production acceptance. Nine rectangular FP8 fixtures benefit;
the compact control uses identical binaries.

## Project truth

|Field|Frozen decision|
|---|---|
|Current op|Existing exported `flaggems_vllm.qsa_select_paged_prefill`, SM90 H800|
|Editable|Production op, dedicated NVIDIA config group, tests, harness diagnostics/pin, runner and this document|
|Read-only|Native reference 7a877ae / SHA4d09eb2f; workload manifests and all existing config groups; private validated kernel source|
|Build|Python/Triton JIT; Python syntax/lint on Mac; GPU compilation on H800|
|Validation|Actual public entry, native-vLLM comparison, FP8 split modes/generic H/D, tail canaries, misalignment fallback and runtime cache keys|
|Benchmark|`--candidate production --mode kernel --suite scored-prefill --rounds 2`; score/TopK/pipeline microseconds and native/production speedups|
|Active set|Frozen 30 scored cases:12 upstream BF16,8 observed BF16,10 FP8; skip no shapes|
|Aggregation|Per-shape paired results and arithmetic means; score excluding TopK is NV acceptance metric; also report TopK and pipeline|
|Timeout|Worker1800s, accuracy7200s, cooldown120s with80C starting ceiling|
|Profiler|Existing compiled PTX/cubin resource capture, including new FP8 entry; prior hardware profiling is already in reports|
|Autotune|Dedicated `qsa_select_paged_prefill_fp8_store_score`, exact34 joint-stage choices, standard libtuner/entry keys and query-mode prune|
|Fallback policy|Original Triton scoring for BF16 or ineligible alignment; compact remains priority; no torch compute fallback|

## Alignment contract

|Field|Contract|
|---|---|
|API/reference|Existing vLLM0.31.0 pinned QSA contract; no standalone torch equivalent|
|Device/backend|NVIDIA SM90, FlagTree with verified PTXAS13.4 and CUDA compatibility library; native and production in same process/compiler|
|Input|Original validated BF16/E4M3FN queries/cache, paged or strided table, generic supported H/D; metadata values retain documented preconditions|
|Output|Original int32 block indices, FP32 logits workspace; public allocation/alias/chunk rules unchanged|
|Dtype/promotion|Dot FP32 accumulation, ReLU per head then head sum; no scales added|
|Semantics|Same finite-input score and TopK membership/tie rules; NaN/Inf not newly promised; invalid input raises original errors|
|Autograd|Forward only|
|Unsupported|Same public validation and unsupported types/layouts as before|
|Reference torch|Tests/benchmarks only; production permits empty allocation and metadata/view operations|

## Implementation paths

|Path|Trigger|Implementation/config|Validation/benchmark|Risk|
|---|---|---|---|---|
|Early|Empty/select-all|Original fixed kernel, no tune|Existing boundary tests; outside scored30|None introduced|
|Compact|Existing compact gate|Exact original kernel/config|FP8 compact + BF16 ragged controls|Gate unchanged|
|FP8 rectangular fast|E4M3FN, contiguous FP32 logits, width multiple4, pointer16B aligned|Transfer validated bulk/tail scorer with joint query modes/stages; dedicated34 YAML configs|Native observed FP8 fixtures, public-entry boundary/canary tests and generic H/D forced choices|Production autotune can choose different tiles than frozen probe; must remeasure|
|General|All other supported inputs|Original rectangular kernel/config|BF16 upstream12, misaligned/odd-width fallback|No new performance claim|
|Unsupported|Original public rejection conditions|Original exceptions; no external compute|Existing errors|Scope unchanged|
|TopK|All selected paths|Repository persistent_topk, unchanged|Stage timing + membership checks|No TopK optimization|

Old production source f1c5e71a is retained as an explicit historical snapshot;
private historical trials must not silently compare against this new integration.
No export/registration change: the existing public entry is the one being edited.

## Validation and H800 run

Mac:340 CPU tool/host/integration checks passed; changed files pass pre-commit,
`git diff --check` and runner `bash -n`. The production AST contains only
`torch.empty`, device-context and capability calls; no torch compute/copy path
or test/tool import. Numerical GPU validation and native speedups remain pending.

Run `bash tools/run_qsa_prefill_fp8_store_production_h800.sh --flagtree /opt/flagtree`
with `QSA_STOCK_PYTHON` pointing at the existing vLLM0.31 environment. The runner
routes both implementations through FlagTree and the verified CUDA13.4 tools;
it does not install FlagTree into the virtualenv. Default correctness uses45
representative nodes including the five new production-store checks, then all30
cases receive native/production score, selection and timed pre/post validation.
This reduced profile is not labeled full-suite acceptance. Append
`--accuracy-profile full` for the full production test file.

Default measurement:2 rounds,200ms warmup,100ms repetition per timing call,
80C initial ceiling and120s cooldown timeout. Actual autotune is outside measured
kernel timings and is not replaced by the probe's frozen choice manifest.
Each completed case prints score/TopK/pipeline latency and speedup; the same
values are retained in JSON/TSV. Existing idle/external-process guards stay on.
A test failure still uploads diagnostics. Upload uses a child shell and identity
fallback; a push failure prints a retry command so tests need not be rerun.

## Bootstrap failure b974e738 and recovery

Report b974e738f186f251cc893948f928d9323985c91b failed before preflight,
correctness or timing: the newly downloaded NVIDIA ptxas archive did not match
its original pinned SHA256. GPU selection and isolated libcuda loading passed.
The old log has no byte count, actual digest or response metadata, so an
incomplete download versus an unexpected response cannot be distinguished yet.
No operator correctness/performance conclusion follows from this report.

Tool acquisition now first reuses prior `qsa-flagtree134-tools.*/<name>.tar.xz`
archives after checking the original digest and the copied bytes. Downloads
receive up to three attempts with the same hash requirement. The source URL,
HTTP status/type/length/range, byte count, actual hash and attempts are saved in
`toolchain-downloads.json`; bootstrap exports this file to the uploaded report
even on failure. Original archive/version/binary pins stay unchanged. This is
a tooling-only change; production SHA8e4ad799 and all workloads are unchanged.
Local validation:27 relevant toolchain/orchestration tests and changed-file
pre-commit passed. H800 compilation, correctness and timings remain pending.

Report08b29c742ffba6e4dab5d35e7f6b3567d018527f confirmed successful reuse of
both pinned tool archives, but the benchmark rejected the newly exported
`toolchain-downloads.json` as an existing result. The FlagTree bootstrap-only
allowlist now explicitly includes this diagnostic filename. The real wrapper
handoff regression test creates/exports it before calling benchmark main; it
also proves existing stage summaries and case results still reject the run
without overwriting their contents.25 relevant orchestration/production-binding
checks passed. No GPU correctness checks or performance measurements ran in
this failed report; production/workloads/toolchain pins remain unchanged.
