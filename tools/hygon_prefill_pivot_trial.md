# BW1000 exact pivot / partition cost gate

After pulling the probe commit in the dedicated Hygon worktree, run:

```bash
tools/hygon_prefill_next_run.sh pivot hygon_prefill_pivot_v1
```

The runner commits and pushes the full report, including any worker errors.
It runs four representative shapes as isolated workers. Each worker checks
normal seeds 42 and 43, plus tied, constant, partial, short and special rows.
The pivot is estimated from 256 row samples using a normal-quantile heuristic;
**no correctness decision relies on the heuristic**. The exact `>` and `==`
counts are checked against torch. Only rows with `k <= count_gt <= CAP` may
continue to an exact selection of the compacted candidates; other rows require
a fallback, which this probe does not implement or time. Eligible rows are
checked against the exact top-k value multiset.

Timings use alternating ABBA/reverse order and device-kernel events. The
reported two-kernel cost is only a *lower bound* for a complete algorithm:
candidate top-k, overflow/fallback, scratch allocation and any inter-kernel
gap are absent. Stop if either seed has many fallback rows, or if the first
two kernels already cost as much as the current complete non-TLE operator.
If they leave a material budget, the next experiment is a complete exact
selection plus fallback, measured end-to-end against the compiled vLLM
baseline and current production path. Do not promote this diagnostic as an
operator speedup.
