# Tune short-loop fallback then run ten complete inputs

| Project truth | Contract |
| --- | --- |
| Goal | Validate prefix-only history merge with short-loop fallback on full inputs |
| Acceptance | Ten-case arithmetic mean of stock harness kernel-mode CANN/candidate whole-call ratios |
| Frozen | Fixed local TLE sort,block512 score,query16,two slots,fixtures,packing,splits,outer state commit,compiler and sampling |
| Editable | Boundary preflight/policy,prepared dispatch,optional harness mode,summary and tests |
| Production | Unchanged;candidate remains unregistered |

| Contract | Value |
| --- | --- |
| Boundary set | rows16/128 at1/2/4/8 updates plus rows17,width8256 and rows112,width29696 anchors |
| Policy | Smallest tested threshold with >=1.02 speedup in every paired round of every tested case at/above threshold |
| Upper range | Prefix only through largest measured update count;outside range falls back |
| Gate | All boundary numerical/guard/count/IR checks;no threshold means stop before full benchmark |
| Runtime | CPU integer block count selects prefix or original helper;real history preserved,packing charged |
| Control | Same fixed sort/block512 entry always using original snapshot helper |
| Correctness | Original exhaustive pre/post full-input rank,trace,guard and primitive checks for candidate/control |
| TopK | Persistent,always-snapshot online and dispatched online timings from staged raw scores;diagnostics only |

| Path | Behavior |
| --- | --- |
| --history-prefix-harness | Requires serial,mixed and tle-sort harness flags;preflight and full ten-case test in one command |
| --history-boundary | Micro worker option for the frozen boundary set and snapshot/prefix pair |
| Provenance | Boundary artifacts and selected policy copied to report root;each case records actual piece dispatch |
| Failure | Automatic report upload retained;no silent fallback from failed correctness or absent policy |
| Resume | Reuse only already-validated full cases with same policy and code gate;no mixing another policy's results |

The boundary samples select an experimental threshold, not a guarantee for
unmeasured update counts or row counts. The full ten-input run evaluates those
actual request fragments and retains their dispatch decisions. Compiler path,
candidate hash and installed-source hash are verified in both phases through
the same inherited manifest. No compiler change is introduced by this step.

Local validation: 79 existing raw-probe checks, six policy/integration checks
and five complete-input scheduling checks passed (90 total). Pre-commit,
Python syntax, shell syntax and diff-whitespace checks passed. NPU compilation,
correctness and performance remain to be measured on the card; production is
unchanged.
