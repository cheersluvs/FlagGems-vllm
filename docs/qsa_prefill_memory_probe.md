# QSA scorer memory-pipeline diagnosis

Pre-coding board, 2026-10-23. Selective BF16 subdivision remains private;
report059fcb6b gives only1.008309 whole-pipeline incremental mean. Do not
integrate or further tune that small gain in this round. Return to the original
36-launch profiling evidence3621d9ca and investigate output layout/epilogue.
Candidate BF16 long L1TEX LSU wavefront utilization59.50–62.39% (elapsed),
FP8 long65.44–68.43%; FP8 DRAM cycles42.50–44.46% elapsed and tensor
19.87–20.17% active. These different denominators/units do not establish a
store bottleneck. Loads, shared layout conversions and stores are not separated.

| Project truth | Frozen choice |
| --- | --- |
| Editable | New memory diagnostic/CPU checks, optional explicit-metric plumbing in profiling driver, wrapper and this document |
| Read-only | All kernels, production/private configs, builders, old worker/reference/cache contents, timer/guards/numerical tolerances and measured independent benchmark files |
| Question | Are memory-pipeline costs concentrated in global loads, logits stores, shared layout conversions/bank conflicts, or scoreboards? |
| Active set | Original BF16 n1/q16384/p102400 and FP8 long100k; all4 real chunks, native/canonical/retained combined candidate,24 captured launches total |
| Build/validation | No changed kernel/JIT policy; strict original source/runtime/input/config/cubin matching and existing pre/post numerical checks; CPU/pre-commit/shell checks locally |
| Benchmark | No benchmark/A/A/full pytest rerun; old score/TopK/pipeline timing remains provenance only; replay durations are not speedups |
| Profiler | Existing pinned NCU, kernel replay/cache-control none/clock-control none; query actual metric availability, select explicit required traffic and optional shared/stall/utilization counters |
| Static evidence | Export actual cubin/PTX-matched cache entry PTX/TTGIR/TTIR/LLIR; optional verified nvdisasm output; no reconstructed or newly optimized kernels |
| Timeout |1800s per capture; same original GPU UUID,80C idle gate; fail closed and preserve partial evidence |
| Autotune/fallback | Original warmed/cached choices; no tuning expansion, torch fallback,TopK or production edits |

| Contract | Preserved |
| --- | --- |
| Inputs/output | Original BF16/FP8 H4/D128/page196, starts/visible/strides/budget/workspaces and real observable FP32 logits stores |
| Math | Full-D dot, per-head ReLU/head sum, native and repository numerical checks unchanged |
| Runtime | Original vLLM0.31/FlagTree/ptxas13.4/CUDAcompat source and library paths, hashes and compiler caches |
| Scope | Diagnostics only; neither new correctness coverage nor an optimization speedup claim |

| Path | Evidence | Follow-up decision |
| --- | --- | --- |
| Global reads/writes | Per-kernel requests/sectors and DRAM bytes, kept separate by side/chunk | Inspect coalescing/vector width/predication before a load/store-layout probe |
| Shared movement | Wavefronts/bank conflicts and matching TTGIR/PTX/SASS | Probe layout/epilogue traffic only if evidence supports it |
| Scheduler | Optional long/short scoreboard, MIO/LG throttle metrics | Distinguish dependency/latency from bandwidth; no causal inference from one percentage |
| Missing optional metric | Explicit N/A/list of unavailable names | Never fabricate zero or claim collected |
| Missing cache artifact | Preserve counters, record missing artifact | No substitution with a differently compiled kernel |
| Integration | Deferred | One-hypothesis probe with real stores and same-run timing, then regression, then production |

Metrics/queries follow NVIDIA's [CLI reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
and [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
Sectors/request depends on instruction width and active lanes; do not treat4
as a universal optimum. Store-sector bytes divided by useful output bytes is
an L1 transaction indicator, not DRAM write amplification or a speedup ceiling.

## Run and validation

Use a fresh checkout of the pushed commit. Keep the original measured checkout
`/tmp/flaggems-qsa-profile.2H9z8e`, report
`/tmp/qsa-profiling-h800.Ji2cwS`, its per-case caches, and original toolchain
intact. The resume preflight checks their recorded hashes and runtime paths;
it does not replace them with the fresh diagnostic checkout.

```bash
export QSA_STOCK_PYTHON=/tmp/qsa-vllm031-h800.PQnVpV/venv/bin/python
bash tools/run_qsa_prefill_memory_h800.sh \
  --resume-report /tmp/qsa-profiling-h800.Ji2cwS \
  --flagtree /opt/flagtree \
  --ncu /tmp/qsa-ncu-portable.5qd8fj5x/nsight_compute-linux-x86_64-2026.3.1.2-archive/ncu
```

The wrapper prints traffic/stall tables and historical score/TopK/pipeline
measurements (explicitly labelled historical), then commits and pushes the
new report even after a capture failure. A failed push prints a retry command;
no recapture is required. PTX/IR/SASS are exported as top-level text files and
included in the report. Missing matching PTX/TTGIR yields exit2 and preserves
any completed counters; no differently compiled artifact is substituted.
Replay may execute each of the24 selected launches several times to collect
metrics. No GPU run or optimization gain is claimed by this change.

Local validation:34 CPU checks passed across memory-probe, profiling,
profile-resume, report and portable-NCU modules; pre-commit and `bash -n` passed.
Hardware validation remains pending the user's H800 run. Inspect requests and
sectors alongside active masks/vector widths and matching IR; shared bank
conflicts/scoreboards can direct a different probe if stores are not dominant.
