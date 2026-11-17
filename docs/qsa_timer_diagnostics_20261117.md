# Preserve failed timer captures during two-case recovery

| Contract | Decision |
| --- | --- |
| Editable | Optional diagnostic launcher, its tests, master dispatch and this note |
| Frozen | Kernels, inputs, baseline, stock harness timer selection and sample parameters |
| Scope | Resume 080738 report; retain eight completed cases and original policy; run compact and long100k |
| Acceptance | Ten complete stock kernel-mode CANN/candidate ratios; never average missing or infinite results |
| Production | Unchanged |

The uploaded timer deletes profiler directories in a finally clause with keep_res=False. Thus the prior postmortem had no raw files. Opt-in QSA_TIMER_DIAGNOSTICS=1 runs pytest through an instrumented launcher. The stock outer do_bench_npu is untouched. MSPTI calls retain all parameters, returns and exceptions. Only an already-selected profiler fallback receives keep_res=True and an explicit unique output directory; warmup, active, cache clearing, target filter and mismatch handling remain unchanged.

Each fresh worker writes timer source and JSON-line events as root text files, and a root compressed JSON evidence file. The latter contains all file names/sizes, with at most 128 KiB per log/CSV and 8 MiB total text. Raw captures stay on the card. Normal Python exit, including pytest failure, collects evidence; forced process termination can bypass collection, so already-flushed events and raw captures remain useful. Existing report upload includes the root evidence files.

Twelve timer/runner checks and six history-policy checks pass locally, together with pre-commit. Device execution remains pending. No conclusion about MSPTI failure or profiler task loss is asserted yet. Collection-time npu-smi showed other processes on device0; this is a possible interference source, not proof of causality.
