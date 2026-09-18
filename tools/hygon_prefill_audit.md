# Hygon prefill: controlled final-selection and slot-counter experiment

Base: `3138faddeebe6fac93c314c72a7a8f9143785a6a` (`topk-metax`).
This is a probe, not a production override. No baseline, tests or benchmark
entry points are modified. `optimization.md` and `deep_opt.md`, referenced by
`workflow.md`, are absent at this revision. The task is prefill, not the stale
conv1d example in that workflow document.

## Project truth (G1)

| Field | Frozen value |
| --- | --- |
| Current op | `flaggems_vllm.top_k_per_row_prefill`; reference `torch.topk` on each valid row interval |
| Editable files | New `tools/hygon_prefill_audit*` probe, source builder and CPU checks |
| Read-only files | Generic and Hygon operators, existing tests, benchmarks and historic reports |
| Build | Python source/AST compilation locally; Triton JIT on BW1000; no package installation or `setup.sh` |
| Validation | Every arm: bounds, uniqueness, exact selected values, padding; normal/tied full shapes and partial/short/padded-row cases; nonzero exit on any failure |
| Benchmark | Paired, order-balanced rounds; profiler events for the single Triton kernel, microseconds; speedup = control/candidate |
| Active set | Seven existing fp32 benchmark shapes, strides and k; seeds 42 and 43 |
| Aggregation | Per-seed paired ratios and raw samples; geometric mean across seven shapes only with complete coverage; no best-of-arm minimum comparison |
| Timeout | Suggested 20 min for first compilation plus validation and timing; individual trials must finish before further changes |
| Profiler | Torch profiler, count/name-checked device events; raw kernel count and durations recorded; no profiler ratio against vLLM |
| Autotune | Exempt: diagnostic-only Hygon arms, production geometry fixed; compare rank tiles 8/16, never pick a per-round winner |
| Fallback | No torch compute in generated kernels; original rank path for candidate sets above 256 in tiled arms |

## Alignment contract (G2)

| Field | Contract |
| --- | --- |
| Reference | [torch.topk](https://docs.pytorch.org/docs/2.10/generated/torch.topk.html); installed torch build printed remotely |
| Device | Hygon BW1000, CUDA-compatible torch device, wave64; require Hygon backend |
| Input | fp32 2-D logits, caller strides; int32 row starts/ends with `0 <= start <= end <= vocab`; positive k |
| Output | Caller-owned int32 `[rows,k]`; indices relative to each start; unique valid indices, `-1` padding for spans shorter than k; output order unspecified |
| Dtypes/promotion | This experiment covers fp32 logits and int32 metadata/output only, matching current active benchmark; no promotion |
| Semantics | Exact value multiset; ties may select different unique indices. Include signed zero and infinities in correctness cases. NaN semantics are not established and not part of acceptance |
| Autograd | Forward-only index selection |
| Unsupported | Other dtypes, invalid ranges, NaNs and concurrent reuse of a probe's scratch are outside this diagnostic scope; no new public dispatch or fallback is installed |
| Torch use | Input generation, oracle, validation and report statistics only, outside timed paths |

## Paths (G3)

| Path | Trigger | Implementation/config | Validation and benchmark | Risk |
| --- | --- | --- | --- | --- |
| Control | All supported cases | Exact shipped one-scan source plus dense helper where production would select it | Compare against public shipped op; full matrix | Source drift detected by exact replacement counts |
| Rank8/rank16 | `final_cnt <= 256` | Same comparisons and tie rule, process j in groups of 8/16; reduce local comparisons | Full matrix; original scalar rank above 256 | Register pressure/layout changes; no assumed speedup |
| Register carry | Dense route only | Carry definitely-selected output count through existing tile loops; publish once before step return | Aligned/head/tail/strided and fallback cases | Cross-step count visibility and barriers |
| General | Large candidate set/sparse counter arm | Original rank/sparse collection respectively | Tied and adversarial ranges | No torch fallback |
| Diagnostic | Separate untimed module | Reuse unused one-scan scalar scratch to record final executed step and threshold-bin size | Compare candidate stats across seeds/cases | Never time instrumented modules |
| Unsupported/external | Not applicable | No external kernel or public API added | Outside acceptance | GPU execution remains required |

## Measurement rules

- Build input/oracle and validate before measuring; guard output bounds before
  gathering, check uniqueness even on ties, require exact value equality.
- Each comparison is ABBA or BAAB, alternated by round; rotate candidate order.
  Preserve every duration and pair ratio. Do not divide separately chosen minima.
- Timed plans own identical preallocated scratch; all arms use the same cached
  direct launcher. Also measure the shipped public entry point as a control
  check. These are device-kernel results, not Python wall-time claims.
- Statistics use another compiled module and are not included in timed runs.
- On an idle HCU, record device identity, visible-device variables, source hashes,
  versions and occupancy snapshots. Stop on profiler event-count mismatch.
- Report a kernel failure explicitly and return nonzero. No production changes
  or performance claims until the card reports back.

## Decision order

1. Check correctness, control agreement, candidate statistics and timing spread.
2. Evaluate rank arms and register carry independently on the unchanged 2048 bins.
3. Only then design the narrowed-histogram/final-selector combination; the
   outcome of an isolated arm does not logically rule out a combined design.
4. Revisit resident/sampled/stage-split pipelines only with measured cost evidence.

Production acceptance later uses the unmodified functional suite and benchmark
in two passes, `SpeedUp >= 0.9` on key shapes and no unexplained regressions.

## First remote run

The Mac publishes `codex/hygon-prefill-audit`. From the existing Hygon checkout,
create a separate worktree so pending work on `topk-metax` is preserved:

```bash
cd /data/wuyuqing/workspace/FlagGems-vllm &&
git fetch https://github.com/cheersluvs/FlagGems-vllm.git codex/hygon-prefill-audit &&
git worktree add -b codex/hygon-prefill-audit ../FlagGems-vllm-hygon-prefill FETCH_HEAD &&
cd ../FlagGems-vllm-hygon-prefill &&
HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 bash tools/hygon_prefill_audit_run.sh hygon_prefill_audit_v1
```

Select a currently idle HCU with `hy-smi`; the `1` above is an example, not a
claim about current occupancy. The report is `reports/hygon_prefill_audit_v1.txt`.
The runner pushes to the explicit cheersluvs URL, independent of the card's
`origin`. Credentials and identity use the existing card setup. It never
rewrites history. Repeated runs need a new report name. The runner reports
GPU/profiler failures and returns their nonzero exit code even after pushing
the failure report. Each shape subprocess has a 900-second timeout.

Local checks completed before publication: seven CPU construction/semantic
checks, Python syntax, shell syntax, and all applicable repository pre-commit
hooks. Mac has no torch/Triton/Hygon execution environment; GPU compilation,
validation and timings are pending the returned report.
