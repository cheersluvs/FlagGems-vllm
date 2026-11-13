# Two-dimensional NZ key packing probe

## Frozen project scope

| Field | Scope |
| --- | --- |
| Evidence | c7712cb8d03570d099af70014d479fa889c631c4: long K pack 4219.99 us vs same-grid copy121.40 us; aiv_scalar_ratio0.991; extracted per-element memory loop |
| Editable | New isolated K packing helper and probe mode, CPU tests and this note |
| Read-only | Raw C++/Cube chain, Q/W packing, finish, current INT8/production implementation, harness, installed toolchain |
| Compiler | Same independent final-task-mode copy, common to all measured arms |
| Active set | compact128/W1088 and long100k112/W29696 first chunks; support budget_exact via existing case flag |
| Validation | Entire packed K bit-exact to unchanged packer, all chunk scores/masks at rtol=atol=1e-3, CPU reference, postchecks |
| Timing | Two interleaved rounds, warmup3/active6; original and two candidates, K stage and complete raw pipeline; strict host/device task counts |
| Profiler | Compile/IR evidence and PipeUtilization/Memory counters in the same invocation; two calls per metric |
| Autotune | Exempt evidence-driven API probe, no production config changes |
| Timeout | Worker1800 seconds, existing automatic report upload |

## Contract

| Field | Scope |
| --- | --- |
| Tensor | Original INT8 physical K[pages,192,1,128], immutable shuffled shared table; output [ceil(max_visible/64),8192] NZ |
| Part mapping | Each original1024-element part becomes a 32-row by32-column load; scalar tile/page/dimension-block, contiguous innermost32 |
| Address | tile=part//8; row_half=part%2; dim_block=part%8//2; each N64 tile remains inside one page192 |
| Mask | Invalid final key rows write zero; all output elements written, no alias or fallback |
| Numerical | Packing contains no arithmetic/quantization change; native W.T/Fixpipe path and FP32 output comparison unchanged |
| Resources |1024 byte data tile plus two-dimensional indices/masks; no increased data tile size |
| Boundaries | CPU coverage at32/64/192 and their adjacent values; card cases include real tail257 and25628, all page/tile boundaries |

## Paths

| Arm | Meaning |
| --- | --- |
| original | Unmodified flat-index packer and original tiles*8 grid |
| two_d_grid | Only address/layout expression changes; same grid,1024 values per program |
| two_d_capped | Same two-dimensional tile with at most20 programs and grid-stride loop |
| complete | Q/W prep + selected K prep + unchanged raw Cube + unchanged finish; all four device tasks charged |
| current | Existing routed INT8 scoring entry, no TopK; reference comparison only |
| IR | Record extracted access markers; do not equate a remaining scheduling loop with a scalar gather loop |

This remains an unregistered probe. A K-only speedup is not a whole-operator
speedup; unchanged finish/QW/Cube costs remain charged in complete timing.
The formal benchmark harness acceptance mean is not measured in this experiment.

Mac validation: fifteen raw/vector/packing CPU checks plus the five existing
compiler-copy checks; generated-kernel syntax, Python/Bash syntax, and repository
pre-commit hooks. The two-dimensional address test evaluates expressions from
the actual generated source against the earlier flat NZ layout, including
part3207 and mask transitions around32/64/192. Device compilation, vectorization,
correctness and timings remain pending the single user-run card invocation.
