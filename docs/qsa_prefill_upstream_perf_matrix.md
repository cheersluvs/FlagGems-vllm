# QSA long-prefill performance matrix

Current matrix (2026-10-19): the repository pytest benchmark and default H800 runners
use only these 12 upstream layouts. The stock-Triton stage A/B accepts only
`--suite upstream-prefill` and produces 36 stage rows. Historical inference probes
remain diagnostic fixtures outside the official benchmark matrix.

The first H800 run completed all 12 cases with exit 0 in report
`88dee1c4be295fb2b2a43162e440963b6528af4a`. The fresh correctness suite
passed 161 cases with one optional second-device skip. See
[results and production score baseline](qsa_prefill_upstream_perf_results.md).
The tables and command below describe the frozen measurement contract.

## Scope frozen before implementation

Source: [vLLM PR #54513, Microbenchmarks / Prefill](https://github.com/vllm-project/vllm/pull/54513).
The public table specifies four query layouts crossed with three prefix lengths,
and page size 196 on GB300. It does not publish the complete microbenchmark
builder, dtype, H/D or timing protocol. This change reconstructs that published
shape matrix on H800; it does not claim byte-identical inputs or reproduce the
published GB300 speedups.

| Project field | Choice |
| --- | --- |
| Current op | qsa_select_paged_prefill; benchmark coverage only, no operator change |
| Editable | New benchmark workload module; stage runner suite selection/metadata; suite-aware checkpoint validation; relevant CPU tests; this guide |
| Read-only | Production operator, vector prototype, Top-K, tuning YAML, reference helpers and tolerances, shared benchmark timer, original 38 case definitions |
| Build | No native build; Python syntax and cached pre-commit hooks |
| Validation | CPU checks for original-token prefix conversion, causal visibility, page geometry, full 12-case coverage, CLI forwarding and resume isolation; card runs existing full correctness then independent FP32 scores before/after each case |
| Benchmark | --suite upstream-prefill --single-row-ab --mode kernel; actual shared do_bench, score-only us, smaller is better, Top-K excluded |
| Active set | 12 BF16 cases, H=4,D=128, page=196, ratio=4,K=512, budget=512 MiB; original 38 cases remain in their existing suite |
| Aggregation | Seven alternating rounds, per-case latency medians and paired ratios; no aggregate across unlike shapes |
| Timeout | 1800 s for full accuracy and each isolated worker |
| Profiler | Record ncu/nsys availability; no profiler execution is claimed; chunks, launch paths, tuned configs and memory recorded |
| Autotune | Existing six dot configs and four trial vector configs frozen; no new kernel/config, no new autotune needed |
| Fallback | Existing production dot path for these multi-row chunks; no Torch production fallback |

| Contract field | Choice |
| --- | --- |
| Reference | Existing QSA API and FP32 benchmark references; no corresponding ATen API |
| Device | NVIDIA H800 SM90; same actual /opt/flagtree for both sides |
| Query / prefix | K=1024 units; Prefix means original tokens preceding this query; sequence length = prefix + queries per request |
| Compression | Cache extent = ceil(sequence length / 4); visible compressed blocks for row j = floor((prefix + j + 1) / 4) |
| Inputs | Packed contiguous BF16 Q[sum(query lengths),4,128], cache[physical pages,196,1,128], shuffled int32 page table, int32 starts/visible |
| Output | Existing int32 [total queries,512], logical block indices, unchanged -1 padding |
| Accumulation / semantics | Existing FP32 score accumulation/ReLU/sum, unchanged finite-input inference contract and score/selection tolerances |
| Unknown upstream details | BF16,H=4,D=128,ratio=4,K=512 and 512 MiB are explicit local choices consistent with the pinned correctness fixture/default; not facts disclosed by the PR performance table |
| Reference-only Torch | Input creation and independent validation only; no production imports of benchmark helpers |

| Path | Trigger | Implementation / validation | Benchmark / risk |
| --- | --- | --- | --- |
| Long prefill | New upstream-prefill suite | New builder, existing public preparation and observed score chunks | Only new 12 cases; prefix interpretation and resource use recorded |
| Existing coverage | core/comprehensive | Original case dictionaries and builders unchanged | Existing 38 remain reproducible; never mix suite checkpoint identities |
| Candidate control | Chunk has more than one query | Trial falls back to same production dot kernel | Establish production latency/config baseline; ratios are controls, not a new optimization claim |
| Empty / unsupported | Existing API contract | Existing validation/return paths unchanged | No registration or export change required |

This round establishes long-prefill score latency/configuration evidence before
changing the main score kernel. Top-K may execute in untimed public setup and
selection checks; it never enters the timed score callable. The original vector
trial is retained as a control and is not enabled in production.

## Exact local geometry

All lengths below use token counts, not labels in K. Prefix is uncompressed;
compressed context includes the current query. Chunk rows are the expected
production 512 MiB split, checked against the observed launches before timing.

| Requests | Query/request | Prefix | Sequence/request | Compressed context | Logits width | Score chunk rows |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 8192 | 8192 | 16384 | 4096 | 4096 | 8192 |
| 1 | 8192 | 51200 | 59392 | 14848 | 14848 | 8192 |
| 1 | 8192 | 102400 | 110592 | 27648 | 27648 | 4854/3338 |
| 1 | 16384 | 8192 | 24576 | 6144 | 6144 | 16384 |
| 1 | 16384 | 51200 | 67584 | 16896 | 16896 | 7943/7943/498 |
| 1 | 16384 | 102400 | 118784 | 29696 | 29696 | 4519/4519/4519/2827 |
| 2 | 8192 | 8192 | 16384 | 4096 | 4096 | 16384 |
| 2 | 8192 | 51200 | 59392 | 14848 | 14848 | 9039/7345 |
| 2 | 8192 | 102400 | 110592 | 27648 | 27648 | 4854/4854/4854/1822 |
| 4 | 4096 | 8192 | 12288 | 3072 | 3072 | 16384 |
| 4 | 4096 | 51200 | 55296 | 13824 | 13824 | 9709/6675 |
| 4 | 4096 | 102400 | 106496 | 26624 | 26624 | 5041/5041/5041/1261 |

## Run on H800 with automatic report upload

Use the existing Python/Torch virtual environment and the actual external
`/opt/flagtree` runtime. This is a fresh 12-case suite, not a resume of the
completed 38-case report. Full correctness runs first; each new shape then
gets independent sampled score checks before and after timing and public
selection checks outside timing. No native-vLLM benchmark is invoked.

```bash
if bash <<'QSA_LONG_PREFILL'
set -euo pipefail
source /tmp/qsa-vllm031-h800.PQnVpV/activate_qsa.sh
qsa_repo=$(mktemp -d /tmp/flaggems-qsa-long-prefill.XXXXXX)
git clone --depth 1 --single-branch \
  --branch codex/qsa-select-paged-prefill-h800 \
  https://github.com/cheersluvs/FlagGems-vllm.git "$qsa_repo"
git -C "$qsa_repo" rev-parse HEAD
CUDA_VISIBLE_DEVICES=GPU-2d17872f-4327-2beb-8049-604ecf6a3d66 \
QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
QSA_PREFILL_AB_KIND=single-row \
  bash "$qsa_repo/tools/run_qsa_prefill_validation_ab_h800.sh" \
    --suite upstream-prefill --require-idle-gpu --mode kernel --timeout 1800
QSA_LONG_PREFILL
then
  echo "长 prefill 测试和报告回传均完成。"
else
  echo "请检查上方退出原因和结果分支；当前会话已保留。" >&2
fi
```

The runner uploads top-level JSON/TXT/TSV even after a benchmark rejection
when a summary exists. New `upstream-prefill-shapes.tsv` and manifest.workloads
record the conversion above; per-case `workload` and `chunks` record expected
and observed geometry. `score-kernel-ab.tsv` retains per-case latency/ratios.
All 12 cases have chunk sizes greater than one, so both sides execute the
shared production dot kernel; ratios are controls and do not represent new
algorithmic gains. The actual full correctness and first card timing results
are linked above.

If a shared GPU job interrupts this suite, its new report can be resumed with
`--suite upstream-prefill --resume-from <stage-summary.json>` on the same GPU
and unchanged sources/runtime/timer. Full-suite reuse requires explicit
`--skip-accuracy`; prior correctness evidence remains separately labelled.
The suite identity is checked in both the parent and every worker. A checkpoint
from core/comprehensive cannot be reused as this 12-case dataset.

Local verification: 82 unittest methods passed, including the worker-side
checkpoint definition check and suite mismatch rejection; all applicable
pre-commit hooks passed; documented shell syntax passed. Eighteen previously
captured frozen sources remain byte-identical, and eight existing functions
(including the original 38-case generator and timing code) retain identical
ASTs. Production math, Top-K and tuning remain unchanged. `optimization.md`
and `deep_opt.md` are absent; this change is coverage/measurement preparation,
not a new kernel optimization trial. The completed H800 run is linked above.
