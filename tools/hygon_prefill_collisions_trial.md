# BW1000 histogram collision opportunity

Run `tools/hygon_prefill_next_run.sh collisions hygon_prefill_collisions_v1` in
the dedicated Hygon worktree after pulling this commit. The runner commits and
pushes `reports/hygon_prefill_collisions_v1.txt`.

The probe samples up to eight rows for the long-row, dense, and four-row
regimes, at two seeds. It reports normal, tied, and constant data. `flat64`
counts adjacent 64 input elements; `vec4_component` counts 64 elements at
stride four (one component from each logical VEC=4 lane). `ideal_reduction`
is only the theoretical fraction of histogram updates that could be removed
if identical addresses in each group were perfectly combined. It does not
measure actual physical wave mapping, instruction support, merge overhead, or
operator performance.

Decision: if normal data offers little reduction, stop A1. If it offers a
large reduction, inspect the generated layout/ISA and implement one isolated
wave-aggregation variant, then validate exact output before paired timings.
Do not change the production geometry: focus-v1 found `[256,2]` incorrect at
16,385 columns, while `[1024,16]` did not repeat the earlier large gain.
