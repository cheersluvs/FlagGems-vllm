# H800 long-prefill score baseline: first complete report

## Result

All 12 published-layout cases completed with exit 0. The fresh full correctness
suite passed 161 cases and skipped one optional second-device test. Every new
long-prefill case passed sampled FP32 score checks before and after timing and
sampled public selection checks outside timing. This establishes the current
production score baseline for the new matrix; no production change or speedup
is accepted by this report.

- Report branch: `codex/qsa-h800-results-20261016-034303`.
- Report commit: `88dee1c4be295fb2b2a43162e440963b6528af4a`.
- Report directory: `results/qsa_select_paged_prefill/h800/stages-20261016-034303/`.
- Code: `7fd2c432c901d3833262b6fd1139542a04df4c6b`, clean tree.
- Baseline source pin: `454074da0c079cc649d139489ce11fd3141585b7`;
  operator SHA256 `d70faeb6928c2c7a56636ded833f3359368871e353ecf4c66c6dfd35cae93e73`.
- BF16, H=4,D=128, page=196, compression ratio=4, selection K=512,
  logits workspace budget 512 MiB. Prefix counts original tokens before query.
- H800 UUID `GPU-2d17872f-4327-2beb-8049-604ecf6a3d66`.
- Actual compiler `/opt/flagtree/triton`, version 3.6.0; ptxas 12.8.93;
  Python/Torch virtual environment unchanged, Torch 2.13.0+cu129.
- Actual shared `Benchmark.get_latency` in `--mode kernel`: score launches
  only, no Top-K in the timer; warmup 1000 ms, rep 100 ms, seven alternating
  baseline/candidate rounds, latency medians in microseconds, unchanged default
  do_bench cache policy.

The vector trial is retained as a control. Every observed chunk has more than
one query and `score_path=matmul`; vector configs are empty. Both timed sides
therefore execute the same production dot kernel. Differences near 1x are
control variation and are not claimed as algorithmic improvement. This is not
an original native-vLLM performance comparison or a comparison to the PR's
GB300 measurements. The PR did not disclose full H/D/dtype/builder details;
see the [matrix contract](qsa_prefill_upstream_perf_matrix.md).

## Production scoring medians

K denotes 1024 tokens. Columns specify the original-token prefix; each sequence
also includes that row's query length. Times exclude Top-K.

| Query layout | Prefix 8K us | Prefix 50K us | Prefix 100K us |
| --- | ---: | ---: | ---: |
| 1x8K | 176.768 | 717.664 | 1366.784 |
| 1x16K | 466.576 | 1546.368 | 2840.512 |
| 2x8K | 348.992 | 1440.960 | 2768.192 |
| 4x4K | 296.160 | 1406.816 | 2615.632 |

## Full paired results and dispatch

Configs below are the actual cached/tuned entries captured by libentry,
not evidence that each chunk independently selected its optimum config.
All cases use BLOCK_N=64, four warps and two stages. Nine cases record
TILE_R=32,K_TILES=4 (shared memory 73728 bytes); three record
TILE_R=64,K_TILES=16 (shared memory 106496 bytes). Register counts, occupancy,
cache hit rate, achieved memory bandwidth and tensor-core utilization were
not captured, so the report cannot attribute the observed latency to a
particular resource bottleneck.

| Case | Baseline us | Candidate us | Ratio of medians | Paired median | Chunk rows | TILE_R / K_TILES |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| perf_n1_q8192_p8192_bfloat16 | 176.768 | 176.512 | 1.001450x | 1.002357x | 8192 | 32 / 4 |
| perf_n1_q8192_p51200_bfloat16 | 717.664 | 717.728 | 0.999911x | 1.000000x | 8192 | 32 / 4 |
| perf_n1_q8192_p102400_bfloat16 | 1366.784 | 1359.872 | 1.005083x | 1.000070x | 4854/3338 | 32 / 4 |
| perf_n1_q16384_p8192_bfloat16 | 466.576 | 463.808 | 1.005968x | 1.000840x | 16384 | 64 / 16 |
| perf_n1_q16384_p51200_bfloat16 | 1546.368 | 1547.392 | 0.999338x | 0.998987x | 7943/7943/498 | 32 / 4 |
| perf_n1_q16384_p102400_bfloat16 | 2840.512 | 2840.368 | 1.000051x | 0.999966x | 4519/4519/4519/2827 | 32 / 4 |
| perf_n2_q8192_p8192_bfloat16 | 348.992 | 349.024 | 0.999908x | 0.999954x | 16384 | 32 / 4 |
| perf_n2_q8192_p51200_bfloat16 | 1440.960 | 1440.512 | 1.000311x | 1.000555x | 9039/7345 | 64 / 16 |
| perf_n2_q8192_p102400_bfloat16 | 2768.192 | 2768.688 | 0.999821x | 1.000046x | 4854/4854/4854/1822 | 32 / 4 |
| perf_n4_q4096_p8192_bfloat16 | 296.160 | 296.160 | 1.000000x | 0.999785x | 16384 | 32 / 4 |
| perf_n4_q4096_p51200_bfloat16 | 1406.816 | 1399.488 | 1.005236x | 0.999909x | 9709/6675 | 32 / 4 |
| perf_n4_q4096_p102400_bfloat16 | 2615.632 | 2615.536 | 1.000037x | 1.000349x | 5041/5041/5041/1261 | 64 / 16 |

## Environment and integrity audit

- All 21 captured source hashes match the tested source tree. The canonical
  12-case checkpoint validator passed; source/runtime/compiler/timer/GPU
  fingerprints agree across every case. Seven-round medians, speedups and
  paired ratios were independently recomputed by the validator.
- Observed Q/cache/table shapes, logits widths and chunk rows match the
  recorded original-token/compressed/page geometry for all 12 cases.
- All 96 timing guard samples passed: sampled temperature 31–51 C,
  SM clock 1980 MHz, no rejected process or thermal sample.
- There are 14 idle-check records: 13 passed, one initial check before the
  last case reported 7% utilization (limit 5%) with no active compute PID.
  The bounded retry passed before measurement. No rejected timing was reused.
- Neither ncu nor nsys was available. No hardware profiler result is claimed.
- The host is shared; passing sampled guards does not prove continuous
  isolation. Keep all raw rounds. For example, the 4x4K/100K baseline has a
  2675.888 us round versus its 2615.632 us median; no outlier was deleted.
- Raw JSON/TSV files retain full precision, input fingerprints, expected and
  observed geometry, configs, PTX/CUBIN hashes, compiler files and telemetry.

## Next optimization target

Focus the next experiment on the multi-query dot score path, preserving this
production baseline, Top-K and kernel-mode metric. A generic hypothesis is to
increase reuse of K/page-table data across query rows by changing the query
and column work assigned to a program. Larger blocks also increase shared
memory/register demand; the current 72/104 KiB shared-memory entries make
that tradeoff material. Capture resource/candidate timing evidence before
choosing a regime, rather than claiming a bandwidth or occupancy bottleneck
from aggregate timings alone. Hardware-specific intrinsics are unnecessary
for this first hypothesis.

Retain the existing 38-case regression suite. The 12 long-prefill cases are a
separate BF16 matrix and do not replace FP8, parameterized H/D, padded layout,
zero-visible or small-query correctness/regression coverage. Any new candidate
must be timed against current production in the same compiler/process on
both sets. No operator/config changes are made in this report-only archive.
