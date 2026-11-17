# Measure packing before implementing within-call reuse

| Project truth | Contract |
| --- | --- |
| Editable | Diagnostic stage closures, optional benchmark branch, packing statistics/driver/tests and wrapper |
| Frozen | Arithmetic, layouts, kernel bodies, two slots, history threshold and stock kernel timer defaults |
| Active set | first_chunk, budget_exact, compact, long100k, complete original inputs |
| Reference | Existing native/current/raw/candidate pre/post checks and selected-score contracts |
| Production/autotune | Unchanged; unregistered fixed diagnostic, no production autotune change |
| Metric | Whole-call and isolated packing ms plus sampled prepared mixed-kernel ms; no CANN acceptance mean |
| Timeout | 1800 seconds per case, outer 7200 seconds |

The existing closure is split into QW packing, K/KS packing and mixed execution,
retaining precisely this order for normal calls. No packing is removed or cached.
The probe uses the inherited validated history policy and complete input schedule.
It runs all-row pre/post correctness checks; selected first/middle/last pieces also
verify exact prepared score traces and tie-aware ranking against raw scores.

Whole-schedule QW and K/KS stages are measured independently with the stock
get_latency. They reuse scratch safely because these diagnostic calls do not
consume the scratch. Prepared mixed execution is measured one piece at a time,
after preparing that piece; shared scratch prevents a correct all-piece
prepared-only batch. Sample timings must not be extrapolated as a measured
whole-input compute time. Stage measurements are non-additive and not internal
pipeline durations. Existing timer diagnostics retain fallback evidence.

Traffic accounting compares physical page prefixes for repeated pieces of each
request, independently of table pointers. It reports requested valid K/KS input
bytes, padded packed output bytes, repeated prefix bytes, and storage for all
request maxima. Counts describe logical elements, excluding page-table reads,
cache hits and physical DMA transactions. Requests are counted separately even
if they share physical pages. A changed prefix fails the reuse-feasibility gate.
Zero-visible inputs read zero K/KS elements but still prepare one padded block.
Within-call maximum-prefix storage is an estimate, not an implemented cache.

Progress JSON and terminal output retain traffic and completed timings even if a
later measurement fails. No new performance or reuse benefit is claimed until
the card report is available. Local validation: 81 existing raw tests, six policy
tests and three new packing checks passed, plus pre-commit and Python/Bash syntax.
