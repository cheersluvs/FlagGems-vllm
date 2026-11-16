# Raw mixed kernel query-group and key-tail coverage

## Project truth

| Field | Contract |
| --- | --- |
| Goal | Extend proven one/two-slot mixed pair to realistic chunk dimensions and tails before full harness |
| Frozen | Native query16 C++ and .cube bridge, online sort/UB snapshot helper, event protocol, precision, production and compiler |
| Editable | New derived boundary wrapper/padded packer, optional probe mode, geometry/tests |
| Cases | (rows,width)=(16,4096),(1,1088),(15,1088),(17,8256),(112,8512),(128,29696) |
| Mapping | One program per16-query group; fixture rows share one shuffled page mapping; no cross-request-sharing claim |
| Timing | Prepared QW/K/KS, same one/two-slot mixed kernels, two reversed rounds/three active/strict task count/5ms capture guard |
| Validation | Independent CPU half oracle; exact raw materialized score trace and selected score values; pre/post; poisoned outputs plus guard regions |
| Hardware | No repeated hardware collection in boundary run; prior counters remain evidence only for prior16-row case |
| Acceptance | Feasibility/paired device times only, not full-input/CANN harness mean |
| Autotune | Fixed diagnostic exemption; no production tuning/registration |

## Contract

| Field | Contract |
| --- | --- |
| Query groups | ceil(rows/16)<=20; base addresses cast before explicit group byte offsets; each program owns its own ring |
| Row tail | QW packing zeros invalid rows; Vector visible loads and all output stores mask row<ROWS; tail still follows all ready/free events |
| Key tail | Cube executes ceil(width/512) full512 updates; padded packed K/KS are zero; sort sees out-of-width columns as -inf |
| Page table | Retain exact logical page count; padded packer masks page-table loads beyond width instead of extending caller metadata |
| Output | FP32 scores/values and int32 indices; no stores outside actual row/width rectangle; guard storage accounts for whole padded rows/columns |
| No deadlock shortcut | Inactive rows do not skip block communication; each program releases/drains every slot |
| Supported fixtures | Single shared request map per case, forward INT8 H4/D128/page192; not arbitrary production arguments |

## Paths

| Path | Implementation |
| --- | --- |
| Old --raw-mixed-online | Unchanged16-row whole512 reference experiment |
| New --raw-mixed-online --boundaries | Same native/merge/event math, query-group offsets and bounds-masked padded wrapper |
| Reference | Same QW and padded K/KS preparation; standalone raw Cube materializes all padded groups/tiles, existing finish writes logical output |
| CPU | Packing and half-chain oracle independent of mixed output; offsets/masks/padded table behavior checked separately |

The previous report80a036a023151459b0691c581735662392a15a02 passed mixed
linking and exact pre/post checks with1.410x eight-update pair speedup.
This expansion tests new address/bounds cases, not a production adoption.

Local verification: 45 focused CPU tests passed (4 boundary, 5 mixed ownership/
address, 3 ABI including host Clang, 24 raw address, 5 compiler-mode and 4
capture-guard tests), generated-source/Python syntax, CLI help and Bash syntax.
Focused pre-commit passed after import formatting. Device compilation, bounds,
postchecks and timing for these expanded dimensions remain pending.
