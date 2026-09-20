# BW1000 prefill VEC layout sweep

On the dedicated Hygon branch, pull this commit and run:

```bash
tools/hygon_prefill_next_run.sh vec hygon_prefill_vec_v1
```

The runner pushes the complete report, including worker failures. Seven
production benchmark shapes run in separate workers. Each compares the exact
production non-TLE `VEC=4` kernel with variants `VEC=1/2/8`, changing both
the alignment calculation and the histogram/collection tile's vector width.
Routing, BLOCK, warps, output and scratch allocation stay unchanged. The
source must match the shipped kernel byte-for-byte at `VEC=4` or the worker
stops. Full normal seeds 42/43 and tied, constant, partial, short, special,
and legal padded-row inputs are checked against exact top-k before timings.

Device-only ABBA/reverse paired timing includes an independent production A/A
control for each seed. Reported ratio is production/candidate; greater than 1
favours the candidate. Codegen reports registers, spills, static barriers and
target code size; those are diagnostics, not evidence of runtime savings.
One rejected variant does not suppress results for the other variants. This
is a layout sweep of the current algorithm, **not** a production change.

Only retain a variant for further work when both seeds improve beyond the
same-shape A/A spread, with exact adversarial correctness. If no material
full-operator gain remains, close the VEC sub-item and move to A2/B1 rather
than broadening a blind compiler-parameter sweep.
