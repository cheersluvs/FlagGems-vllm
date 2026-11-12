# Resume full INT8 acceptance after device startup timeout

Report da80530e has seven complete correctness/stage/combined case records.
int8_short_tail failed in torch.npu.set_device(0), before any operator call,
with 507033/E39007 device subprocess startup timeout; compact and long100k were
not reached. This is not an observed numerical or kernel execution failure.
The report does not establish the device/HDC timeout's underlying cause.

QSA_RESUME_REPORT=auto with QSA_HARNESS_ACCEPTANCE=1 selects the newest retained
report with measured code commit c49ca4ef7e5d3b1657144638115fc3efa326eebc.
An explicit report directory is also accepted. Source/benchmark changes other
than this resume helper, runner, and this document are rejected by Git diff.
Each reusable case must have complete full-input kernel records for score,
TopK and combined, a successful parent pytest case, correctness metadata and
matching scorer/workload/TopK hashes. An incomplete case is rerun; changed
measured code aborts the resume instead of mixing results.

A fresh report copies the original top-level artifacts. Original summary,
exit code and failed-case records are preserved with before-resume prefixes.
resume-provenance.json records reused/pending case names and source hashes.
The original code-commit.txt remains intact; resume-code-commit.txt records
this runner version. Existing completed cases are never rerun. Empty pending
sets are summarized without starting an empty pytest worker.

Use the original warmup20/iter50 settings and pinned environment/device.
The full ten-case table and arithmetic mean are computed only once all ten
records are complete. Results are paired within each case but may come from
different sessions; this is not a repeated-run stability measurement. No
NPU/process reset, environment upgrade, or scoring/TopK change is performed.
If device startup still fails, preserve the new report and investigate the
runtime/device service rather than altering operator math.
