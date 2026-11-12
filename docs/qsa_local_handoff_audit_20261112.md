# Operator-level local handoff audit

Hypothesis: the query16 second-Cube path loses more time moving intermediate
results than it saves in Vector reduction. The previous report a209a710 shows
half+Vector scoring close to current, but second-Cube scoring slower with higher
Fixpipe/MTE counters. Counters alone do not identify concrete buffer edges.

This is an evidence collection step, with no kernel/autotune/production change.
It reuses the completed 88912117 diagnostic report and replays budget_exact's
saved compiler inputs for current and second-Cube arms. No model data or full
benchmark suite needs to be regenerated.

| Evidence | Scope |
| --- | --- |
| Installed API | Resolve the actual Triton/TLE paths, signatures and exported local memory/custom-op symbols |
| Source inventory | Capture relevant installed Python sources/hashes with explicit size limits |
| Post-solver IR | Original compiler input/options plus the existing print-after-sync-solver flag; input/output paths relocated to fresh audit directories |
| Binary identity | Compare emitted object/shared-library hashes to the successful ordinary compilation; differing hashes prevent assuming exact binary equivalence |
| Prior measurements | Preserve the source report/hash and successful ordinary correctness/timing record |

No API symbol is taken as proof of legal L0C-to-L1 transfers, fused conversion,
or second-Cube execution. Missing Python APIs do not prove that no custom-op or
compiled backend facility exists. The full IR must be checked through producer,
consumer and memory-space types; static op counts do not represent dynamic
traffic or individual operation timing. Diagnostic binaries are not executed.

The source report must have code-commit.txt equal to
889121172dbe56da4cba2bc97520f72b0f913e0b and completed reduction diagnostics.
By default the script searches retained /tmp/qsa-ascend910b-int8-harness.* reports;
QSA_SOURCE_REPORT can specify one explicitly. It does not download new compilers
or modify the installed environment.

Decision after collection: use an existing operator-level interface only if its
semantics can express the required handoff; then build a correctness/timing probe.
If that requires compiler modifications, close this route under the user's
operator-only constraint. Do not infer that a generic to_buffer(L1) reproduces
CANN Fixpipe(ReLU/dequant) directly into L1.
