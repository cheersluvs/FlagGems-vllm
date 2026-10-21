# QSA prefill correctness profiles

Plan, 2026-10-21: reduce repeated optimization checks without deleting tests.
The completed packed-grid report07993da spent713.06s in230 collected tests
(229 passed,1 second-GPU skip), plus1264.65s between first/last benchmark
telemetry. Reducing correctness cases cannot remove benchmark worker startup,
imports, compilation, tuning, validation and timing costs. No elapsed saving
is promised before a new H800 profile run.

| Scope | Decision |
| --- | --- |
| Editable | CPU-safe fixed iteration node manifest, runner profile/child propagation/coverage gate, focused CPU tests, result and workflow docs |
| Frozen | Production and candidate kernels, configs, TopK, native reference, numerical tolerances,30 benchmark fixtures, timing and GPU guards |
| Modes | iteration:40 exact production nodes; each current private trial adds its8 exact candidate nodes. full:original complete suites and mandatory coverage unchanged |
| Default | FlagTree production/current-trial CLI uses iteration; stock and historical private trials retain full profile behavior; explicit --accuracy-profile full always available |
| Scope marker | Reports record profile and full-suite-validation state; iteration data never carries formal production acceptance eligibility |

| Iteration coverage | Cases |
| --- | ---: |
| Visibility/ties and ReLU-before-head-reduction | 3 |
| Native cooperative/persistent TopK dispatch across row/K branches, BF16/FP8 | 10 |
| Native empty/select-all BF16/FP8 | 2 |
| Cross-request workspace reuse and pinned native chunk comparison | 2 |
| Upstream BF16/FP8 prefill, clipped width and normal/one-row chunks | 8 |
| Split-prefill Torch/native and no-complete-block boundary | 10 |
| Formal public entry identity | 1 |
| BF16/FP8 runtime-row cache separation | 2 |
| BF16 and FP8 compact native comparison | 2 |
| Total production | 40 |
| Current packed-grid candidate boundary/full-entry extras | 8 |

| Gate | Rule |
| --- | --- |
| Collection | Explicit frozen pytest node IDs; a missing/renamed case fails collection |
| Iteration results | Exact40/48 node set, no duplicate-normalized nodes, every result passed; missing/extra/skipped nodes fail |
| Full results | Existing complete-suite targets, direct native/config/FP8 requirements and only documented second-GPU skip retained |
| Runtime | Same compiler, GPU and native-extension identity before/after accuracy and benchmark; profile identity must match requested profile |
| Benchmark | All30 cases and per-stage before/after correctness remain; retain score, TopK and actual pipeline data |
| Release/integration | Full profile required before retaining an integrated production change; use full when changing math, H/D/layout contract, tuning space, host allocation or public interface |
| Packed-grid decision | Production unchanged; completed experiment has no demonstrated stable gain, so do not integrate it |

Usage on the current FlagTree runner: `--accuracy-profile iteration` is now
the CLI default for production and current private trials. To select complete production
validation explicitly, pass `--accuracy-profile full`. Both run the original
unchanged30-case benchmark and per-stage before/after checks. Older private
trials and stock profiles retain full-suite selection behavior.

CPU-only selection inspection (no Torch/Triton required):

```bash
python3 tools/benchmark_qsa_prefill_triton_stages.py \
  --compiler flagtree --candidate production --suite scored-prefill \
  --list-accuracy-tests
```

The fixed node manifest is `tools/qsa_prefill_accuracy_profiles.py`. All40
production and48 packed-grid nodes were found and passed in the completed
07993da report. This verifies membership and prior results, not a new reduced
suite GPU duration. The source test files are not modified or deleted.
General H/D matrices, all-config sweeps, invalid-input/alias/device-placement
matrices and duplicate benchmark fixtures remain in the full suite.

For report compatibility, existing accuracy-full.json/txt filenames remain;
their contents' scope is identified by manifest options, accuracy.profile,
accuracy.full_suite_validated, and overall-acceptance.accuracy_profile and
formal_acceptance_eligible. An iteration benchmark can validly report its
performance target result without claiming complete production correctness.

Local validation:204 CPU orchestration/contract tests ran,203 passed and1
skipped; changed-file pre-commit and syntax checks passed. Exact-node coverage
was rechecked against the returned full report. Reduced-profile execution and
elapsed time still need an H800 run in a future optimization iteration; this
change does not require repeating the completed packed-grid benchmark.

The next `page-table` trial also has8 exact candidate nodes plus the same40
production nodes (48 total), selected by `PAGE_TABLE_CASES`. Its full profile
runs its8 candidates before the complete production file. The previously
verified48 nodes in report07993da refer specifically to packed-grid; the new
page-table candidates still require their own on-card validation.
