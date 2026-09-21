# Hygon prefill algorithm probes

Measurement-only tools; no production registration or dispatch changes.
`optimization.md` and `deep_opt.md` are absent in this checkout. These probes
follow the existing audit contract, with the current public API as baseline.

## Frozen project facts

| Item | Contract |
| --- | --- |
| Operation | `top_k_per_row_prefill`, exact per-row top-k relative indices |
| Editable | New `tools/hygon_prefill_algorithms*` files and report runner |
| Read-only | Production sources, existing correctness helpers and benchmarks |
| Build | Python syntax/source-construction checks locally; Triton JIT on BW1000 |
| Validation | Exact selected-value multiset, index bounds/uniqueness, -1 padding, output guards; no timing after failure |
| Benchmark | ABBA paired profiler kernel totals (us, control/candidate >1 is better); separately allocation-inclusive synchronized wall time |
| Active set | Existing seven audit shapes, each algorithm restricted to its documented targets |
| Aggregation | Per shape/config and per seed, no cross-distribution speedup claim |
| Timeout | 600 seconds per isolated worker, 13000 seconds for a suite |
| Profiler | PyTorch device events, all candidate kernels included; fail if event counts differ |
| Autotune | Exempt: fixed Hygon experiment matrix, no NVIDIA/production integration |
| Fallback | Original Triton final selector for large network tails; no torch compute in candidates |

## Input/output contract

| Item | Contract |
| --- | --- |
| Reference | Existing `hygon_prefill_audit.oracle/check_output` using torch.topk |
| Device/type | Hygon HIP wave64, fp32 logits, int32 starts/ends/output |
| Layout | Column-contiguous, contiguous or padded rows; arbitrary valid starts/ends |
| Output | Relative indices, unordered exact top-k value multiset; any distinct tied indices allowed |
| Short rows | All valid indices followed by -1; empty rows entirely -1 |
| Specials | Finite values, ties, +/-Inf, +/-0; normalize zero before key conversion |
| Unsupported | NaNs (not silently interpreted), non-fp32, column-strided input; forward-only |
| Torch use | Input generation, oracle, diagnostics and scratch allocation only |

## Experiment paths

| Arm | Target shape IDs | Algorithm / parameters | Principal risk |
| --- | --- | --- | --- |
| threshold | 6,2,4,5 | Row-resident ordered-key search, binary or three pivots, 4/8 warps; one final compaction | Register pressure, reduction count |
| streaming | 0,1,3 | Exact k-entry queue, merge each k-sized chunk, with/without threshold filtering; 8 warps | k=512/1024 sorting cost, ascending adversary |
| final | 0..6 | Actual production source with small 64/128/256 network or common-prefix key search final selector; scalar original fallback | Final stage has insufficient share of total time |
| delegate | 0 | Block maxima (32/64 elements), exact delegate kth bound, streaming tail skips blocks below bound; 8 warps | Three launches and weak pruning |

Threshold searches operate on an integer ordered key, not approximate floating
point pivots. Each iteration strictly reduces the interval. No distribution
assumption or approximate stopping is used. The streaming prototype sorts a
2k merge tile when new elements can enter the queue; it is not the full AMD
ballot/staging-buffer implementation. Delegate filtering retains all blocks
whose maxima equal the bound and falls back to bound=0 when there are fewer
than k nonempty blocks. No candidate capacity truncation is permitted.

Final probes include `remaining=0/all/1` shortcuts. Network sorts value+position
keys and preserves the existing final selector's later-position tie rule.
Prefix search starts at the actual candidate key min/max and never allocates
radix counters. It still checks the full exact top-k result.

Default timings use full normal shapes with seeds 42 and 43. Validation also
uses tied, constant, partial, short, Inf/zero, padded rows, ascending, descending,
heavy-tail and clustered data. Small adversarial cases additionally get full
operator timings, clearly marked as validation-size timings. Iteration counts,
merge counts and delegate survivor fractions are collected in separate launches
so diagnostic stores do not contaminate timed kernels. Resource metadata is
static compiler information, not measured occupancy.

## Running

On the dedicated `codex/hygon-prefill-audit` Hygon worktree after pulling:

```bash
tools/hygon_prefill_next_run.sh algorithms hygon_prefill_algorithms_v1
```

Independent report stages: `alg_threshold`, `alg_streaming`, `alg_final`,
`alg_delegate`. After the first full Hygon report, use `alg_threshold` and
`alg_final_network` to repeat only the two initially failed compilation arms.
Workers run serially in subprocesses. Faults/timeouts are
reported as failures; the parent retains completed results. Only validated
workers can emit performance summaries. A local `--check` needs no torch/Triton
and validates source construction and scalar algorithm models; it does not
claim HIP compilation or device correctness.

## First Hygon report

`reports/hygon_prefill_algorithms_v1.txt` shows streaming at 0.072–0.139x
device ratio and delegate at 0.062–0.066x on their full normal target shapes.
Final prefix search reaches 1.05–1.21x device ratio on the two four-row shapes,
but its allocation-inclusive wall ratio is 0.79–0.81x there; on other full
normal shapes it is slower on both measures. These ratios are control divided
by candidate, so larger than one is faster. Threshold and final network had
Triton branch-variable type errors before measurement. This revision renames
those conflicting variables and adds a network-only runner; Hygon validation
is still required for both arms.

The subsequent `alg_threshold_v3` and `alg_final_network_v3` reports pass all
correctness cases. Threshold search is slower on every full target shape
(best device ratio 0.72x). The network improves device time by 4–6% on the
large-row dense shapes and by 12–30% on the two four-row shapes. The four-row
allocation-inclusive wall ratio is only 0.79–0.82x because the candidate
allocated its six scratch tensors per call while the public baseline reused
them. `alg_final_cached` gives the candidate a separate one-shape scratch
cache with the same six tensors and leaves the baseline's existing cache
untouched. It measures whether the network benefit survives comparable host
reuse. It still allocates the guarded output for each call in both arms.
