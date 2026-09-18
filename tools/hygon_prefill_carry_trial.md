# Hygon prefill register-carry production trial

This continues the frozen G1–G3 contract in `hygon_prefill_audit.md` and the
exact v3 report, `reports/hygon_prefill_audit_v3.txt`. The v3 report passed all
seven shapes and its 200 validation records. The carried-counter arm improved
the four dense shapes by 1.066–1.075x in paired device-kernel timing, with both
seeds agreeing. Rank8 was at most a small dense gain; rank16 regressed overall.

## G1 project truth for this trial

| Field | Frozen value |
| --- | --- |
| Current op | Hygon `flaggems_vllm.top_k_per_row_prefill` |
| Editable | Hygon override, its pure-stdlib source builder, audit construction check, this trial runner/document |
| Read-only | Generic op, existing functional suite, existing benchmark, vLLM baseline, v3 report |
| Build | Python syntax/source construction locally; Triton JIT on BW1000; no package installation |
| Validation | Audit source-byte equality, all existing functional tests, candidate enabled and source hash checked |
| Benchmark | Existing seven-shape fp32 benchmark in kernel mode, baseline/candidate/candidate/baseline order; SpeedUp = vLLM/device latency |
| Active set | The seven shapes, int32 metadata/output, column-contiguous logits with row padding |
| Aggregation | Per-shape comparison and seven-shape geometric mean; keep both passes; no best-of-run cherry-picking |
| Timeout | 20 minutes per remote test/benchmark invocation; stop on correctness failure |
| Profiler | v3 torch-profiler paired single-kernel events; unchanged benchmark provides vLLM comparison |
| Autotune | Exempt: one Hygon-specific algorithmic arm with existing geometry; no new tunable parameter or NVIDIA path |
| Fallback | Flag off by default; source drift or import failure keeps the already shipped dense module; no torch compute fallback |

## G2 input/output contract

| Field | Contract |
| --- | --- |
| Reference | `torch.topk` value multiset per valid row interval, as in `hygon_prefill_audit.md` |
| Device/dtype | BW1000, fp32 logits, int32 row bounds and indices; this trial does not expand dtype support |
| Layout | `stride1=1`; `stride0>=vocab` may include row padding; `stride1=2` is outside the current operator contract |
| Output | Caller-owned `[rows,k]` int32 indices relative to row start; unique in-range indices and `-1` for short spans |
| Semantics | Ties, signed zero, infinities, partial/empty spans checked; NaNs remain unsupported/unestablished |
| Autograd | Forward-only index selection; no promotion or backward change |
| Torch usage | Inputs, oracle and benchmark baseline only; production source builder and kernel contain no torch compute |

## G3 paths and decision

| Path | Trigger | Change | Verification/risk |
| --- | --- | --- | --- |
| Dense candidate | `vocab <= 10*k` and `FLAGGEMS_HYGON_TOPK_CARRY=1` | Carry selected count in registers through all seven collection sites; publish it once per histogram step | Exact audited source digest, existing suite, 4 dense benchmarks; cross-step synchronization |
| Dense shipped | Candidate flag off or source cannot be built | Existing prefix-sum/tile-atomic helper | Baseline arm; retain as fallback |
| Sparse | `vocab > 10*k` | Unchanged source and routing | Three sparse benchmarks must not regress |
| General/short/tied | Same host route; existing in-kernel fallback | No new torch fallback or public API | Full functional suite and v3 adversarial cases |
| Unsupported | NaNs, non-fp32, column stride other than one | No expanded promise | Explicitly outside this trial |

## Decision: keep as default

`hygon_prefill_carry_v1` met the promotion rule on BW1000: the exact audited
source hash was loaded, the existing suite passed **19 passed / 1 skipped**,
and the B-C-C-B benchmark showed the four dense shapes improve by 1.06-1.08x.
The three sparse shapes use the unchanged module. The carry copy is therefore
the default dense path; `FLAGGEMS_HYGON_TOPK_CARRY=0` remains the reversible
fallback for diagnosis.

On the Hygon experiment worktree, after fetching the promotion commit, run
the ordinary functional suite and benchmark with no carry environment variable
to verify the default route. Set visible-device variables before invoking it
if other HCUs are occupied.
