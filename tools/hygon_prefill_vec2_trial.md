# BW1000 dense VEC=2 production-route confirmation

The isolated VEC sweep (`reports/hygon_prefill_vec_v1.txt`) passed exact
checks for every arm and found VEC=2 faster than VEC=4 by about 3–6% on the
four dense benchmark shapes. Sparse shapes did not reproduce a gain. This
follow-up enables the candidate only in a child process via
`FLAGGEMS_HYGON_TOPK_VEC2=1`; the default remains the shipped VEC=4 path.

On the dedicated Hygon branch, pull the trial commit and run:

```bash
tools/hygon_prefill_next_run.sh vec2 hygon_prefill_vec2_trial_v1
```

The runner commits and pushes the full report. It verifies the loaded VEC=2
source is exactly the carried production source with only the two expected
VEC constants changed, verifies the compiled vLLM C++ baseline registration,
runs the existing functional suite, then runs the existing seven-shape kernel
benchmark in VEC4–VEC2–VEC2–VEC4 order. Each child invocation has a 20-minute
timeout; a failed check stops further benchmark runs. `SpeedUp` is vLLM over
this operator, not a cross-operator result. Judge each shape from both passes;
do not promote on one chosen minimum or on the isolated probe alone.

No production default changes in this commit. Sparse routing is unchanged.
