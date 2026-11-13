# Page-aligned raw score output probe

## Frozen project scope

| Field | Scope |
| --- | --- |
| Evidence | 5bd407387914599cb92237c7d3a73b18ff14a3da: after two-dimensional capped K packing, long finish3460.39us is85.3% of the4057.50us raw pipeline |
| Editable | Isolated page64 finish helper, probe flag/runner, CPU tests and this note |
| Read-only | Raw C++ math, Q/W packing, current INT8/production scorer, harness and installed packages |
| Frozen K baseline | Exact raw_k_2d_capped from the passing previous probe; at most20 programs,32x32 data tile |
| Active set | compact128/W1088 and long100k112/W29696; optional existing budget_exact flag |
| Build/validation | Compile all variants, compare all output values/masks, run timing/counters in one invocation |
| Timing | Original finish256 vs page64 vs page64-capped; individual stage and four-task complete raw pipeline; two interleaved rounds, warmup3/active6 |
| Profiler | Existing IR capture and PipeUtilization/Memory counters, two calls per group |
| Autotune | Exempt fixed evidence-driven probe; no production dispatch/config changes |
| Timeout | Whole worker1800 seconds; automatic report upload retained |

## Contract

| Field | Scope |
| --- | --- |
| Inputs | Existing Raw[group4_query,key_tile,16,64] FP32, positive FP16 key scale, immutable page192 table, per-row visibility |
| Formula | Same FP32 Raw*float(KS), output -inf at every invisible coordinate; no arithmetic reordering |
| Address | Each64-key tile lies inside one page192 and one contiguous Raw row; load page ID once, use contiguous64-element Raw/KS addresses |
| Bounds | Full visible mask on Raw/KS loads; scalar page lookup masked for wholly invisible tiles; dense output store mask handles width tail |
| Validation | New finish must equal old finish elementwise exactly; also compare whole scores/masks to current at rtol=atol=1e-3 |
| K preparation | Rebuild packed K every complete invocation, bit-exact check before/after; not cached for free |
| Preparation cost | Q/W packing, fixed K packing, unchanged raw Cube and selected finish all counted; allocations outside device-task timing |
| Unsupported | Same underlying H4/D128/page192 raw feasibility scope; no TopK or model-forward claim |

## Paths

| Arm | Interpretation |
| --- | --- |
| original | Unchanged256-column finish, fixed fast K packing |
| page64_grid | Single-page and single-Raw-tile addresses, grid(rows,ceil(width/64)) |
| page64_capped | Identical64-element body, at most20 programs with grid-stride loop |
| complete | Four tasks with the same QW/K/Cube work for all finish arms |
| IR gate | Record disappearance/persistence of extracted scalar access; a scheduling loop is not a scalar gather |
| Result | Keep as candidate only after correctness and measured benefit; formal harness acceptance is separate |

Mac checks: eighteen raw/vector/packing/finish CPU tests and five compiler-copy
tests, generated source and Python/Bash syntax, repository pre-commit hooks.
Finish tests evaluate actual source address/mask expressions against the original
formula, including partial pages, zero visibility, widths not divisible by64,
partial query groups, and logical output tiles beyond allocated Raw tiles.
Device compilation, actual vectorization, exact finish equality and speedups
remain pending the user-run card invocation. No production code is changed.
