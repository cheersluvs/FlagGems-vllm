# QSA prefill production stage baseline

After host report e760e9e, the current validation-only candidate and paired
old/new runner are documented in the [validation trial](qsa_prefill_host_validation_trial.md).
The diagnostic-only phase below remains recorded as its baseline rationale.

## Host investigation and balanced timing (2026-10-15)

Report [9665cf3](https://github.com/cheersluvs/FlagGems-vllm/blob/9665cf30f03ca8aaaf8a46c099ab3e6cb081519c/results/qsa_select_paged_prefill/h800/stages-20261015-022905/stage-summary.json)
completed 104 independent correctness passes, one expected second-GPU skip,
and all 22 performance cases. Small-case wall/graph differences motivate
profiling host validation, outer allocation and dispatch before changing
production code. The largest case heated from 68 to 85 degrees Celsius;
its mode-to-mode timing drift does not establish a specific throttling cause.

The following tables freeze this investigation before code changes.

| Project field | Choice |
| --- | --- |
| Editable | Stage benchmark, CPU harness tests, this guide |
| Read-only | All production code, Top-K, six tuning candidates, existing functional tests/references and legacy pytest benchmark |
| Hypothesis | Validation, allocation and Python launch costs explain part of small-case wall latency; repeated overlap metadata is a candidate, not an established bottleneck |
| Build / validation | Python syntax, CPU routing/unit tests, pre-commit; card runs unchanged full independent correctness gate before timing |
| Active set | Existing 22 cases; focused run core_0/core_1/core_2/core_3, upstream BF16/FP8 slack=0, mixed_padded and zero_visible |
| Measurement | Schema 2, balanced calls per round and rotating (stage, mode) order. Rebuild baseline before any future candidate comparison; do not mix protocols for speedup |
| Profiler | Optional stdlib cProfile on real validation and public entry; CPU elapsed microbench of validation, device guard and hot outer allocation. All run after formal timing, never in timed batches |
| Timeout / scope | Same 900 s per case; large cases (>512 rows) omit host profiling to avoid heavy extra launches. No package install, clock/power change, compiler patch or production optimization |
| Autotune / fallback | Original six candidates and policy; no new kernel or autotune exemption. No production Torch compute/fallback introduced |

| Contract field | Preserved choice |
| --- | --- |
| API / output | Original QSA prefill signature, BF16/FP8 H/D inference, caller-owned int32 output, unordered tie-compatible selection and -1 padding |
| Metadata / layout | Existing stride/device/alias validation, nonzero query base, ragged and nonmonotonic visibility, compressed pages and budget=0 one-row chunks |
| Numerics / support | Existing FP32 score accumulation and reference tolerances; finite Q/K, inference-only, current SM90 restriction |
| Reference-only Torch | Input generation, independent output checks, allocation microbench only; production unchanged |

| Path | Implementation | Checks / risk |
| --- | --- | --- |
| Original full/score/Top-K/pipeline | Execute original callables and observed chunks | Existing correctness guards; no production path replacement |
| Graph timing | Prepare optional graphs before rounds; every mode executes the same effective call count | Captured loops have no per-call Python validation; graph failures remain explicit |
| Host diagnostics | Profile actual callables after formal measurements | Instrumentation distorts absolute latency; microbench values are not additive components of full wall |
| Device monitoring | Untimed NVIDIA-SMI clock and performance-state queries before/after rounds | Query errors saved explicitly; monitoring does not prove steady state or change clocks |

Schema 1 report 9665cf3 remains immutable. Its graph batches invoked the stage
800 times versus 100 eager calls and all graph measurements followed eager
measurements. Schema 2 rounds up `--iterations` to a multiple of graph batch
size (default 100 becomes 104), uses 13 replays of eight calls, and uses 104
calls for each eager mode as well. With `--no-graph`, no rounding is needed.
All timing samples are normalized by the actual number of calls.

`--host-profile` adds unprofiled CPU microbenchmarks and cProfile records for
up to 512 query rows. Validation uses the actual `_validate_inputs`; the
allocation probe uses the actual observed outer workspace shapes/dtypes and
a hot allocator, without their in-flight kernel lifetimes. A device-guard
probe measures only entering/exiting the current backend's device context.
No-op call timing is included to expose loop/clock overhead. Profiles list
primitive/total call counts, self/cumulative CPU wall time and source locations;
these instrumented times are diagnostic, not production latency or GPU time.
Clock/performance snapshots are taken with the selected physical GPU UUID
outside timed intervals, using [NVIDIA-SMI queries](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
They are boundary snapshots, not a continuous in-kernel clock trace; an
inactive limit reason after synchronization does not exclude an earlier limit.

For the focused follow-up, use the safe fresh-checkout command below and add:

```bash
CUDA_VISIBLE_DEVICES=0 QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
  "$PYTHON" /absolute/checkout/tools/benchmark_qsa_prefill_stages.py \
  --cases core_0 core_1 core_2 core_3 upstream_bfloat16_s0 \
    upstream_float8_e4m3fn_s0 mixed_padded zero_visible --host-profile
```

The default correctness gate remains enabled. Return all top-level reports,
including the new `host-summary.tsv`, `*.host-profile.txt` and per-case JSON
`host_diagnostics`, `timing_plan`, `telemetry` and `measurement_order` fields.
No native-vLLM performance comparison or production speedup is claimed by this
investigation. Baseline and candidate must both use schema 2 in subsequent A/B.

This measures the unchanged production implementation before selecting an
optimization. No production
kernel, Top-K algorithm, dispatch, tuning candidate or tolerance is changed.
The executable scope is currently NVIDIA SM90; using runtime device APIs in
the harness does not establish support on another backend.

| Field | Frozen choice |
| --- | --- |
| Editable | Standalone stage benchmark, its CPU harness checks, documentation |
| Read-only | Production QSA and persistent_topk, six tuning candidates, existing tests/input builders/references and pytest benchmark |
| Reference | Production self baseline in one selected compiler. No native vLLM import, original-source patch, Torch performance baseline or cross-compiler speedup |
| Build | Python-only, no installation; existing H800 Python + explicit FlagTree/TLE runtime |
| Correctness gate | Full independent suite (`-k 'not vllm'`) before timing; missing records, failures and unexpected skips block timing. The second-SM90 device-guard test may skip when only one GPU is visible; its count/reason remains in accuracy-full.json |
| Active set | Core: six existing BF16 benchmark tuples, four upstream BF16/FP8 fixtures (slack 0/1792), zero-visible fixture. Comprehensive additionally covers H/D=(1,32),(3,64),(5,96),(8,256) in both dtypes, two one-row-chunk fixtures and mixed nonmonotonic visibility with padded strides |
| Tuning | Original libtuner policy and candidates; fresh caches per case, JIT/autotuning excluded from timing, selected cache entries and PTX/CUBIN hashes recorded |
| Timing | 25 warmup calls/stage, seven balanced rounds. Defaults: 104 eager calls or 13 graph replays of eight calls per round. All numbers normalized to microseconds per complete invocation of that stage |
| Isolation | One fresh worker per case; timeout 900 seconds including JIT. A failed case cannot contaminate the next case's CUDA context |
| Completion | Schema 1 report 9665cf3 passed; schema 2 and host diagnostics have local harness checks only and require the next H800 report |

## What is measured

The harness observes one untimed production invocation, records actual score
and Top-K calls, and restores all observation wrappers before warmup/timing.
This derives chunks from executed production code rather than a duplicated
chunk-width formula. Every stage processes all observed chunks, including the
tail, with the original output slices, widths and Top-K scratch arguments.

| Stage | Scope |
| --- | --- |
| `full` | Unchanged public production entry point, including validation, allocation, chunk dispatch, scoring and Top-K |
| `score` | Original `_prefill_logits` calls over every observed chunk, with the original reusable logits allocation |
| `topk` | Original `persistent_topk` calls over frozen per-chunk production logits; includes its existing internal dispatch, initialization and allocations |
| `pipeline_preallocated` | Original score + Top-K calls using the observed allocations; excludes outer public-entry validation/allocation. Diagnostic only, not a proposed production implementation |

For the select-all fast path, only `full` executes. Other stages are recorded
as `not_executed_select_all`, without artificial zero latency.

Frozen per-chunk logits occupy separate allocations, whereas the production
pipeline reuses its logits buffer. Separate stage runs also have different
cache states and Python dispatch costs. **Do not add stage latencies, infer an
exact host cost by subtraction, or treat their ratios as profiler percentages.**
`pipeline_preallocated` still includes allocations internal to persistent_topk.

| Metric | Meaning |
| --- | --- |
| `wall_us` | Host perf_counter around a batch, with device synchronization before and after: includes Python dispatch and device execution |
| `event_us` | Device events around an eager batch: a device-stream interval which can include host supply gaps; not pure kernel execution time |
| `graph_us` | Device events around repeated graph replays, divided by captured calls per replay. Reduces repeated Python dispatch, while retaining graph scheduling and any remaining launch gaps |

Graph capture is optional: capture failures and their reasons are recorded
as `capture_unavailable`. An unhealthy CUDA context, replay failure or wrong
output fails the case. `--no-graph` explicitly records `disabled`. Warmup on
the capture stream and capture itself are untimed. Graph full-operator timing
does not include executing its Python validation/allocation on each replay.
All prepared graphs and their pools remain alive through the balanced rounds;
reported peak allocation therefore describes the benchmark process, not the
production operator's minimal workspace requirement.
See [PyTorch events](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html)
and [graph capture](https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph.html).

All modes reuse warm inputs without explicit cache flushing. Stage and mode
order rotate each round. JSON retains every batch average plus median, mean,
min/max and standard deviation; these are not per-call tail percentiles.
No GPU clock or power setting is modified. NVIDIA-SMI snapshots accompany
the run so contention, power and thermal state can be considered during review.

Untimed guards validate up to eight complete rows per case against independent
FP32 Torch scores using existing BF16/FP8 selection tolerances. Replayed output
is poisoned before validation to catch missing writes. Score-only guards
check the last chunk's valid sampled rows exactly against observed logits;
earlier chunks share/overwrite that allocation. These checks supplement the
full correctness gate and do not replace direct native-vLLM alignment tests.

## Run on H800

Use a fresh checkout; these commands can run outside any Git repository and
leave existing card checkouts untouched. The activation path is the previously
validated vLLM 0.31 environment; it supplies Torch and CUDA compatibility libs.

```bash
if bash <<'QSA_RUN'
  set -euo pipefail
  source /tmp/qsa-vllm031-h800.PQnVpV/activate_qsa.sh
  qsa_stage_repo=$(mktemp -d /tmp/flaggems-qsa-stage.XXXXXX)
  git clone --depth 1 --single-branch \
    --branch codex/qsa-select-paged-prefill-h800 \
    https://github.com/cheersluvs/FlagGems-vllm.git "$qsa_stage_repo"
  git -C "$qsa_stage_repo" status --short
  git -C "$qsa_stage_repo" rev-parse HEAD
  unset CUDA_LAUNCH_BLOCKING
  CUDA_VISIBLE_DEVICES=0 QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
    "$PYTHON" "$qsa_stage_repo/tools/benchmark_qsa_prefill_stages.py"
QSA_RUN
then
  echo "QSA run completed."
else
  echo "QSA run failed; this shell remains open. Check the Reports path above." >&2
fi
```

The default comprehensive run has 22 cases. `--suite core` selects 11 cases;
`--list-cases` needs no Torch/Triton/GPU. For a timing-harness smoke check,
select a small set and explicitly skip the full correctness suite:

```bash
CUDA_VISIBLE_DEVICES=0 QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
  "$PYTHON" /absolute/checkout/tools/benchmark_qsa_prefill_stages.py \
  --cases core_0 upstream_bfloat16_s0 zero_visible \
  --warmup 2 --iterations 5 --rounds 2 --graph-batch 2 --skip-accuracy
```

Smoke results are labeled accordingly and do not establish a correctness-gated
baseline. Do not set launch-blocking for performance runs: the parent rejects
it. A nonempty `--report-dir` is also rejected to preserve existing reports.

## Reports and later A/B

The parent prints `Reports: /tmp/qsa-stage-h800.…`. Return **all top-level
JSON/TXT/TSV files**, including `sources.json`, `manifest.json`,
`accuracy-full.*`, every case report/log and `stage-summary.*`. Per-case compiler
and autotuner caches under `cache/` need not be uploaded. JSON snapshots include
source text/hashes, Git commit/status, actual Torch/Triton paths, compiler and
assembler file hashes/version, GPU identity, input shapes/strides/value hashes,
observed chunks, score configurations and timing samples.

For example, upload through a new report worktree, using the printed report
directory and the fresh code checkout from the run. Replace both `.REPLACE`
paths before executing. Copy the entire `if ... fi` block: the upload runs in
a separate Bash process and its failure is handled, even if the interactive
parent shell already has `set -e` enabled. It reports the failed command and
keeps the current session open.

```bash
if bash <<'QSA_UPLOAD'
  set -euo pipefail
  trap 'qsa_exit=$?; printf "Upload failed (rc=%s, line=%s): %s\n" "$qsa_exit" "$LINENO" "$BASH_COMMAND" >&2; exit "$qsa_exit"' ERR
  qsa_code_repo=/tmp/flaggems-qsa-stage.REPLACE
  qsa_report=/tmp/qsa-stage-h800.REPLACE
  if [[ ! -d "$qsa_code_repo" || ! -f "$qsa_report/stage-summary.json" ]]; then
    echo "Replace qsa_code_repo and qsa_report with the actual checkout and Reports paths." >&2
    exit 2
  fi
  test -f "$qsa_report/stage-summary.json"
  qsa_stamp=$(date -u +%Y%m%d-%H%M%S)
  qsa_branch="codex/qsa-h800-results-$qsa_stamp"
  qsa_result_tree=$(mktemp -d /tmp/flaggems-qsa-stage-results.XXXXXX)
  git -C "$qsa_code_repo" worktree add -b "$qsa_branch" "$qsa_result_tree" HEAD
  qsa_dest="results/qsa_select_paged_prefill/h800/stages-$qsa_stamp"
  mkdir -p "$qsa_result_tree/$qsa_dest"
  find "$qsa_report" -maxdepth 1 -type f \
    \( -name '*.json' -o -name '*.txt' -o -name '*.tsv' \) \
    -exec cp -- {} "$qsa_result_tree/$qsa_dest/" \;
  git -C "$qsa_result_tree" add -- "$qsa_dest"
  git -C "$qsa_result_tree" commit -m "Report H800 QSA production stage baseline"
  git -C "$qsa_result_tree" push \
    https://github.com/cheersluvs/FlagGems-vllm.git "HEAD:refs/heads/$qsa_branch"
  echo "Result branch: $qsa_branch"
  git -C "$qsa_result_tree" rev-parse HEAD
QSA_UPLOAD
then
  echo "Report upload completed."
else
  echo "Report upload failed; this shell remains open. Check the error above." >&2
fi
```

The earlier subshell-only example could still terminate an interactive parent
shell that already had `set -e` enabled when a report command failed. This is
a shell-session exit; whether the container stops depends on whether that
shell is the container's main process. It does not prove that the container
itself was stopped. A missing report path, Git identity, credentials or remote
access must be diagnosed from the actual command output.

Later optimized runs must preserve the same GPU, compiler/assembler, input
hashes, budget, timing settings, TF32 policy and correctness gate. Compare the
same stage and metric, review raw variation and alternate baseline/candidate
runs to check drift. A changed compiler requires a new baseline. Source/config
hash changes must be attributable to the intended optimization. This report
does not establish native-vLLM speedup, even if vLLM is installed in the venv.

Local CPU checks:

```bash
python tools/test_qsa_prefill_stage_baseline.py -v
```
