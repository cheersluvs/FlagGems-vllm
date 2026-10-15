# QSA prefill: prepare chunk-invariant host metadata and workspace

Trial 3 starts from `ab50d7a03ec3d5e683691e0f2f8bb46c57307de8`, retaining
both measured host validation optimizations. The prior profile shows 36
score-launcher calls for the 36-row budget=0 fixture. Each call reconstructs
six strides, three runtime layout dimensions, four constexpr values, a flat
workspace slice and a two-dimensional view. Instrumented view costs include
Top-K internals too and are not a predicted production saving.

The two tail benchmark fixtures reuse the existing
`test_workspace_reuse_and_cross_request_chunks` request lengths (39,34),
compressed lengths (4096,8192), query base 113 and 1 MiB budget; FP8 adds the
same fixture for the other supported dtype. They are regression fixtures,
not additional claims about upstream model traffic.

These tables freeze scope before production changes.

| Project field | Choice |
| --- | --- |
| Current op | qsa_select_paged_prefill; same pinned vLLM QSA semantics and public API |
| Editable | Public chunk setup, private score launcher and metadata helper; regression tests; frozen old host entry/launcher and same-compiler A/B support; documentation |
| Read-only | All validation, score/fill kernels, persistent_topk, shared libentry, tuning configs/keys, numerical references/tolerances, logits budget and scratch size |
| Hypothesis | Prepare invariant score arguments once per call and allocate a two-dimensional logits workspace once; full chunks reuse it directly, only a partial tail creates a row slice |
| Build / validation | Python syntax/lint and CPU launch-argument differential tests; H800 full independent correctness suite before timing |
| Benchmark | Same-process full host-entry A/B against the retained version; same actual compiled GPU kernel objects and runtime, same inputs and chunk sizes; full wall us is primary |
| Active / aggregation | Existing 22 cases plus BF16/FP8 73-row width-8192 fixtures with budget=1 MiB, chunks 32/32/9; per-case medians and paired-round ratios, no aggregate hiding regressions |
| Timing / timeout | Schema 2, seven rounds and 104 calls per mode/round by default; existing isolated 900 s worker timeout; warmup/JIT untimed |
| Profiling | Real candidate and frozen baseline public-entry CPU profiles after timing; score stage includes private host launcher, Top-K stage unchanged; ncu is not needed for this host-only hypothesis |
| Autotune | Original six score candidates, original keys, unchanged kernel signature/grid math; no new GPU kernel requiring autotune |
| Fallback | No Torch compute/copy/cast; metadata, empty allocation and no-copy slicing only |
| Acceptance | Candidate until correctness and full-entry A/B show useful benefit without important regression; otherwise revert host production changes |
| Docs / environment | workflow.md and AGENTS.md apply; optimization.md/deep_opt.md remain absent. Mac has no GPU/Torch; user runs H800 |

| Contract field | Preserved behavior |
| --- | --- |
| API / dtype | Same ten arguments and None return, BF16/FP8 Q/cache, int32 output, inferred H/D, NVIDIA SM90 and inference-only |
| Layout / errors | Both retained validators unchanged, same dtype/stride/alias checks and error order |
| Device metadata | No reads of values to host, no assumptions of uniform query lengths or monotonic visible counts |
| Chunking / memory | Same capacity/visible bound/width/budget/rows_per_chunk, same allocation bytes and Top-K scratch; two-dimensional workspace is contiguous with stride (width,1) |
| Tail / reuse | Full chunks return the same workspace tensor object; tail uses workspace[:num_queries], always starting at workspace offset zero, exact logical shape and no stale-tail reads |
| Lifetime / streams | Per-call setup only, no global cache or cached tensor pointer; same current-device context and launch order |
| Semantics | Same score/ReLU/head reduction, selection/ties/padding; no Top-K change |
| Reference use | Torch references and frozen old host implementation are tests/benchmark only, never imported by production |

| Path | Implementation / verification |
| --- | --- |
| Empty / select-all | Return/dispatch unchanged; do not prepare score metadata or allocate score workspace |
| Scored full chunks | Precompute scalar launch metadata once after validation; reuse 2-D workspace without per-chunk slice/view |
| Partial final chunk | Exact prefix-row view; unchanged score mask, Top-K rows, lengths/output slices and scratch reuse |
| Private helper compatibility | Existing direct calls with flat workspace and no prepared metadata continue to work using the original slice/view path and local metadata construction |
| Baseline | Exact old public entry and _prefill_logits sources pinned from ab50d7a, cloned into a private namespace sharing current validator, score kernel, fill kernel and Top-K; public/launcher bytecode intentionally differs |
| Verification | Differential old/new kernel arguments and grids, allocation bytes, chunk offsets/counts, metadata identity per call, fresh metadata on new inputs, failure/no-global-mutation guards; actual BF16/FP8 score/selection checks |

Status: candidate implemented with local checks; H800 correctness and performance acceptance are pending.

Implementation uses a two-dimensional empty allocation directly, so constructing
its complete 2-D view requires no separate view operation. `_prefill_logits`
accepts optional keyword-only `launch_metadata`; existing standalone flat-buffer
calls still work. Production observers and existing score tests now forward
that keyword. Per-call metadata contains only scalar strides/layout/constexpr
values and no tensor pointers. Query offset/count remain per-chunk arguments.

Local checks: 29 CPU test methods pass, including 36 differential executions
across BF16/FP8 metadata, three budgets, contiguous/padded strides and three H/D
pairs. The captured score arguments and all six candidate grids, Top-K arguments,
chunk offsets/shapes and allocation bytes match the frozen old entry. Additional
checks cover same-object workspace reuse, exact tail prefix, fresh next-call
metadata/allocations, legacy flat-buffer calls, empty/select-all/error paths,
old and candidate source pins and no production-global mutation. These tests capture host launches;
they do not test GPU numerical behavior. Prior randomized validation comparisons
continue to pass. Pre-commit and syntax checks passed. AST/source comparison confirms that
validators, GPU kernels, shared runtime, Top-K, tuning and numerical
references are unchanged from the retained baseline.

Six new H800 tests run both frozen old host code and the candidate against the
existing Torch score/selection references. They cover BF16/FP8 with chunks
73, 32/32/9 and one row at a time, checking actual workspace pointer/shape/stride,
metadata identity, visible score values and final selection. With one visible
GPU the expected independent gate is **139 passed, 1 second-device skip**, with
33 native-vLLM cases excluded. Existing cross-request/stale-workspace and alias
coverage remains in the gate. Unexpected failures/skips prevent timing.

Use **host A/B**, not the historical validation-only A/B: the public entry and
score launcher have changed, so those older builders deliberately reject this
public body. `QSA_PREFILL_AB_KIND=host` selects `--host-ab` in the existing runner.
The new `host_ab` JSON and `host-ab.tsv` compare `full` against
`full_host_baseline`, pinned to ab50d7a. `same_public_bytecode` and `shared_score`
(launcher identity) are expected false; `shared_score_kernel`, `shared_fill_kernel`,
`shared_topk`, `shared_validator_bytecode` and `shared_overlap_helper` must be true.
Every case validates baseline output before timing, and both full stages are
validated after timing and optional Graph replay. Chunk records expose workspace
shape, prepared metadata and whether returned logits is the workspace itself.

The candidate score and preallocated-pipeline stages use the observed prepared
metadata as production does. Their timings therefore differ in included host
work from earlier reports and must not be compared across reports for gains.
Only this run's full old/new A/B is the acceptance metric. CPU profiles now also
include the complete frozen baseline entry, outside formal timing. Peak memory
includes benchmark observation buffers and retained Graph pools; it is not the
operator's minimum memory requirement. Kernel code, numerical tolerance,
Top-K and tuning configuration remain frozen.

Run and automatically upload all top-level report files in one command:

```bash
if bash <<'QSA_CHUNK_AB'
set -euo pipefail
source /tmp/qsa-vllm031-h800.PQnVpV/activate_qsa.sh
qsa_repo=$(mktemp -d /tmp/flaggems-qsa-chunk.XXXXXX)
git clone --depth 1 --single-branch \
  --branch codex/qsa-select-paged-prefill-h800 \
  https://github.com/cheersluvs/FlagGems-vllm.git "$qsa_repo"
CUDA_VISIBLE_DEVICES=0 QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
QSA_PREFILL_AB_KIND=host \
  bash "$qsa_repo/tools/run_qsa_prefill_validation_ab_h800.sh"
QSA_CHUNK_AB
then
  echo "运行和报告回传均完成。"
else
  echo "运行或回传失败，当前会话已保留，请检查上方输出。" >&2
fi
```

Defaults run the mandatory independent correctness gate and all 24 cases with
balanced timing. Failed tests with an available summary are uploaded too, with
their original failure status preserved. The parent shell's options are not
changed. H800 acceptance and a keep/revert decision remain pending; the prior
long-case drift still limits any claimed gain for that workload. Other vendors
are untested, the supported device gate remains SM90, and no native-vLLM or
cross-compiler speed comparison is made.
