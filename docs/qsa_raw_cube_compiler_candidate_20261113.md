# Isolated final-task-type compiler candidate

This is an experimental FlagTree compiler compatibility change, separate from
operator optimization. It is not part of production or formal QSA acceptance.
The original installed toolchain and production operator are read-only.

## Frozen plan

| Project field | Scope |
| --- | --- |
| Editable | Preparation helper, its CPU tests, the raw probe/runner and this note |
| Read-only | Original FlagTree/CANN/OPP/Python, operator, formal benchmark table |
| Build | Copy FlagTree into the fresh report directory; patch only the copy's compiler.py |
| Patch point | Immediately after the final task-type callback, in both existing binary compilation routes |
| Validation | CPU source-transform/mapping tests, separate normal Vector and Dot/Vector smoke kernels, existing QK/chain fixtures |
| Timing | Same candidate compiler for custom/reference; existing two paired micro rounds; not full operator/harness acceptance |
| Profiler | Existing native inputs, LLVM, binary and callback/launch metadata capture |
| Autotune | Exempt fixed-shape compatibility probe, no production config change |
| Timeout | Existing 900-second device worker bound; local preparation returns errors for upload |
| Fallback | No compute fallback or manual CompiledKernel.metadata rewrite |

| Contract field | Scope |
| --- | --- |
| Metadata | Final callback bs_task_type is authoritative for loader/core-count mode |
| Mapping | kind 1 -> aiv; 2 -> aic; 3/4 -> mix; kind is bs_task_type // 10 |
| Preservation | bs_task_type including ratio digit, workspace, lock, binary, IR and all unrelated fields remain unchanged |
| Unknown/missing callback | Preserve existing behavior; do not guess a mode |
| Device math | Existing QK/half/ReLU/second-Cube contract unchanged; smoke references are CPU-only |
| Coverage limit | Mapping tests cover MIX_AIC and MIX_AIV/ratios; device smoke records actual inferred kinds, not guaranteed coverage of both MIX leads |

| Path | Trigger | Result |
| --- | --- | --- |
| Original toolchain | No --compiler-corefix | Existing reproduction unchanged |
| Candidate | --compiler-corefix | Materialized independent copy, reviewed source patch, fresh Triton cache, manifest/hash/import checks |
| Smoke | Verified candidate import | Separate Vector and Dot/Vector normal-entry numerical checks |
| QSA micro | Both control/chain launch modes coherent | Existing correctness then timing in the same worker |
| Unsupported source | Callback sites/AST differ, already patched, or copy escapes output | Fail preparation and return diagnostics; never edit original |

The candidate reconciles final compiler metadata inside compiler.py before a
CompiledKernel is created. It does not change the probe's runtime metadata,
normal loader, generated IR, native task-type callback, launch geometry or
library implementation. The original core guard remains enabled.

The helper recognizes the two task callback sites by AST, validates their
enclosing compilation routes and generates a unified diff for audit. CPU tests
exercise both routes, already-consistent modes, all four known task kinds,
ratios, missing/unknown callbacks and preservation of unrelated metadata.
Original compiler bytes are checked before/after preparation. Imports must
resolve into the candidate copy and match the patched compiler hash.

The review patch is [flagtree-final-core-mode.patch](../tools/patches/flagtree-final-core-mode.patch).
It uses zero context; direct git application requires --unidiff-zero. The runner
uses the fingerprint-checked AST transformer rather than blindly applying a diff.

The candidate copy is retained below the report directory for inspection. Only
the manifest, diff and normal top-level report artifacts are uploaded; the
entire copied package is excluded. A successful micro experiment would establish
compatibility only, and still needs full operator regression before adoption.

## Local validation and device run

Mac checks passed: five CPU tests, patch application against the captured
compiler with the frozen SHA256, syntax checks for all four wrapper/ABI pairs,
Black, isort with the repository's Black profile, Flake8 and Bash syntax.
No NPU execution or performance result is available for this candidate yet.

Run the existing runner with --compiler-corefix as its first argument. It
prepares the independent copy, checks imports, runs both normal-entry smokes,
then runs the QK/two-Cube correctness and timing checks in the same invocation.
Reports include the source/patched compiler, review diff and manifest. Source
fingerprint mismatches stop preparation and retain diagnostics for upload.
