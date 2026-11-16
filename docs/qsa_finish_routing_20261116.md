# Raw score finish routing probe

## Project truth

| Field | Contract |
| --- | --- |
| Goal | Explain/reduce finish cost after the ten-input raw score run |
| Frozen | Production, TopK, native query16, QW/K/KS preparation, scratch, current INT8 adapter, isolated compiler correction |
| Editable | Diagnostic finish generator, optional suite mode, tests/runner |
| Inputs | Frozen first_chunk, budget_exact, tail113, long100k; all requests/chunks, unchanged metadata |
| Timing | Two reversed rounds, three active sweeps; exact host/device task accounting; all four raw stages charged |
| Evidence | Same capture supplies stage sums; save generated source and compiled IR; no inference that masked source proves no DMA |
| Build | Existing pinned FlagTree copy/CANN/custom bitcode; runtime key bounds unchanged |
| Validation | New boundary fixtures plus all-row pre/post exact candidate/baseline check and current1e-3 comparison |
| Autotune | Fixed diagnostic, no production selection or tuning |
| Exclusions | TopK/CANN/formal harness acceptance; no production change |

## Contract

| Field | Contract |
| --- | --- |
| Output | FP32 logical-width output, visible values unchanged, invisible columns -inf |
| Logical KS | Prepared by the same fused K/KS kernel for every candidate |
| Safe addressing | Invalid raw/KS columns use index0 plus masked loads; output stores mask logical width |
| Empty block | Only block-start>=visible; directly masked-store -inf, including visible0 |
| Boundary | Rows17, width2057, varying visibility0/1/63/64/65/1023/1024/1025/2048/2049/2057, reversed pages |
| Precision | Identical final FP32 multiply; no changed Cube rounding or coefficient preparation |

## Paths

| Arm | Change from logical1024 |
| --- | --- |
| current | Existing INT8 score reference |
| logical1024 | Passing fused KS raw baseline |
| empty1024 | Empty block direct fill; partial block keeps physical KS/page path |
| tail1024 | Partial block uses masked logical KS; empty block follows same masked helper |
| combined1024 | Empty direct fill plus logical KS partial block |

All candidate preparation and finish launches are timed. Reuse the passing
all-chunk planner and charge request splits. Save per-stage device sums from
the same trace, including finish, Cube and packing; no separate hardware pass.
The four-case mean is exploratory and does not replace the ten-shape acceptance
set. Only a demonstrated candidate advances to ten-input validation.

Local checks: 67 CPU tests passed (34 probe, 24 raw-address, 5 compiler-mode,
4 ten-input planner), CLI incompatible-mode guard, Python/Bash syntax and
focused pre-commit. Device compilation and measurements are pending. Boundary
outputs and candidate validation outputs are poisoned with NaN outside timing.
