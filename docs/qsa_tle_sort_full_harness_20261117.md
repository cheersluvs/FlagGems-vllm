# Full ten-input harness for fixed TLE local sorting

| Project truth | Contract |
| --- | --- |
| Goal | Test whether the3-4% prepared micro gain survives full inputs and packing |
| Acceptance | Arithmetic mean of ten stock harness kernel-mode CANN V1/fixed whole-call speedups |
| Frozen | Ten INT8 fixtures,global chunks/request pieces,query16,block512 native score,two slots,history helper,compiler and stock sampling |
| Editable | Runtime-bound source wrappers,prepared kernel selection,optional benchmark/master mode and tests |
| Production | Unchanged,unregistered candidate; no FlagTree or native custom-op changes |

| Contract | Value |
| --- | --- |
| fixed | Proven sort32 plus two mrgsort calls,full evolving history and final unpack |
| generic control | Same block512 score/schedule with original sort_1d_pack; not the older eight64-key-call mixed entry |
| Charged | QW/K/KS packing per piece,request splits,all kernels and final indices |
| Runtime | At least one padded update,otherwise ceil(max visible/512),same as prior full-input mixed adapter |
| Correctness | Native/current/raw/fixed/generic ranked scores; exact raw-versus-candidate trace; guard checks before/after timing |
| TopK diagnostics | Staged raw logits,persistent TopK plus generic and fixed online TopK separately; not fused internal stage durations |
| Timing | Stock harness CANN/fixed; same get_latency generic whole-call and auxiliary stages |
| Averages | Require all ten workers and pre/post checks; no partial acceptance mean |

| Path | Behavior |
| --- | --- |
| --serial-harness --mixed-harness --tle-sort-harness | New fixed candidate mode and explicit op/route identity |
| Existing serial/mixed modes | Preserve defaults and previous sources |
| Resume | Preserve code-change gate and selected-mode identity; never reuse old generic/mixed measurements as fixed results |
| Generic comparison | Auxiliary generic/fixed ratio and mean reported separately from acceptance |

Local validation:74 raw/ABI/boundary/aggregation/source-wrapper checks and five
complete-input schedule checks passed. New regressions bind actual runtime
signatures,verify block512 control identity and primitive networks,and reject
missing generic correctness/IR evidence or incomplete acceptance tables.
Python compilation,Bash syntax,focused pre-commit and diff checks passed.
No NPU execution performed locally;full-input correctness/performance pending.
