# BW1000 prefill full-row and wave64 scan trial

`optimization.md` and `deep_opt.md` are absent in this checkout. This trial
changes no production route or official benchmark. It copies the currently
selected Hygon non-TLE module, changes one mechanism, and runs the whole op.

| Project fact | Decision |
| --- | --- |
| Op | `flaggems_vllm.top_k_per_row_prefill`, fp32 logits, int32 bounds/output |
| Editable | These private probe tools and run-stage registration only |
| Read-only | Generic and Hygon production op, functional tests, official benchmark, vLLM C++ baseline |
| Build | Generated Python source parse locally; Triton JIT on BW1000 |
| Validation | Exact selected value multiset, distinct relative indices, -1 short padding, output guards, repeated calls; two normal seeds plus tied/constant/partial/short/special |
| Benchmark | Paired CUDA-event full-kernel microseconds, same current production module, launch geometry, scratch and inputs; A-B-B-A/BAAB over three rounds and seeds 42/43 |
| Active set | `fullrow`: official shape IDs 1–6; `wave64`: IDs 2,4,5,6 |
| Aggregation | Per-shape median of six paired ratios; report min/max and both seeds, no cross-shape claim without correctness |
| Timeout | 1800 s per worker, 14400 s outer runner |
| Profiler | JIT metadata/AMDGCN hash, registers, spills, `ds_bpermute`, barriers; CUDA events for timing |
| Autotune | Exempt: Hygon-only fixed source-copy comparison using shipped geometry |
| Fallback | These are probes only; no production fallback changed; source drift fails the worker |

The torch reference is used **only** by the probe oracle, never by the
production path. Column-contiguous rows with padding are the timed contract;
partial/short rows exercise the existing generic branches. Special values and
ties are correctness cases. NaNs are outside the current measured contract.

`fullrow` extends the existing aligned-row arm to any complete row with a
VEC-aligned base. It adds a masked tail to both histogram and collection. The
existing arm demanded `vocab % BLOCK_SIZE == 0`, which only official shape 0
satisfies at its current block size. Partial rows and unaligned bases still
use the original branches. Shape 0 is not timed because this patch cannot
change its already-aligned path.

`wave64` rewrites only the carried dense output-slot scan into 64-element local
scans and a scan over wave totals. This tests whether explicit hierarchy helps
the full operator. It is **not** an AMD ballot instruction yet; if the hierarchy
passes, inspect generated code before attempting a lower-level ballot/popcount
version. It excludes the sparse path, where the carried helper is not used.

On the Mac: `python3 tools/hygon_prefill_scan_paths_check.py` and static/style
checks. On BW1000, on the dedicated audit branch:

```bash
tools/hygon_prefill_next_run.sh scan_paths hygon_prefill_scan_paths_v1
```

The report is committed and pushed even if a worker fails. Stop either stage
on wrong results, spill growth, or stable regressions. An isolated improvement
does not authorize production routing until the unchanged official benchmark
also improves; specifically protect the already-fast 4100x1025 shape.
