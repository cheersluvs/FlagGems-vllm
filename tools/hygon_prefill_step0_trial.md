# BW1000 STEP-0 code-size upper bound

Pull the probe commit in the dedicated Hygon worktree, then run:

```bash
tools/hygon_prefill_next_run.sh step0 hygon_prefill_step0_only_v1
```

This changes only a generated experiment module: the four unrolled refinement
steps become one. The shipped operator and source files are untouched. The
STEP-0-only candidate is **not** a correct general top-k operator: constant,
ties and other inputs can overflow its final bucket. It is only timed after
the exact torch oracle accepts the output for each normal seed. A working
version would need a safe, timed fallback for every exceptional row.

The seven current benchmark shapes run in separate workers. Each records
production-versus-production A/A and production-versus-STEP-0 ABBA/reverse
device-time ratios, plus AMDGCN code size, registers, spills and static
barrier count. Ratios above 1 favor the STEP-0-only upper bound. Static
barrier counts are not dynamic execution counts. If a shape fails validation,
do not use its timings. Even a clean normal speedup is not a deployable gain:
an extra launch and fallback may consume it. Only a material advantage beyond
the A/A spread warrants implementing the complete guarded path.
