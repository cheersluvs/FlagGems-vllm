# Fused logical KS preparation probe

## Project truth

| Field | Contract |
| --- | --- |
| Evidence | Report1952cec1: full1024 long output251.45/full421.62us; Raw regular, page/KS extraction loops remain |
| Editable | New fused-KS source/tests, per-arm preparation support in diagnostic runner and entry |
| Frozen | Native query16 Cube, QW, old K packing leaf, full1024 reference and masked tail, production/TopK/harness/compiler |
| Build | Existing bitcode/isolated metadata correction; no new compiler patch |
| Validation | Packed K exact, logical FP16 KS exact including zero padding, output exact, full/tail boundaries, postchecks |
| Active/timing | compact and long100k; original64/full1024/logical1024; two reversed rounds/six active, device us/round medians |
| Complete | QW + original/fused K pack + native Cube + selected output; four tasks, each arm's actual preparation charged |
| Hardware | Existing output-stage counters plus original/fused packing stages; source/IR/liveness/store evidence |
| Config/timeout | Diagnostic only, no registration/autotune change;1800s/upload; torch only diagnostic oracles, no production fallback |

## Contract

| Field | Contract |
| --- | --- |
| Inputs | Frozen shared-map INT8 K/page192 and FP16 positive KS, Raw16 FP32, H4/D128/finite guards |
| Logical buffer | tiles*64 FP16 in logical key order; last tile after max_visible exactly zero; no requantization |
| Fused preparation | Run unchanged K32x32 leaf, once per tile (part%8==0) copy64 scales with same page map; no extra launch |
| Full output | Contiguous1024 logical scales, FP16->FP32, multiply regular16x64 Raw, flat1024 store |
| Tail | Keep physical KS/page-table masked helper exactly as previous full1024; no packed-KS tail shortcut |
| Resources | Additional tiles*64*2 GM bytes; account scale indices/masks/local buffers in compiler artifacts |
| Boundaries | Shuffled pages, partial65 and2049 key lengths, visible0/1/1023/1024/1025/2048/2049, rows17/width2057 |
| Unsupported | Existing nonshared/H/D/nonfinite/overflow/backward guards; reject, no fallback |

## Paths

| Arm | Preparation | Output |
| --- | --- | --- |
| original | Frozen K |64-key |
| full1024 | Frozen K | Previous full-block candidate |
| logical1024 | Fused K+KS | New logical scale full block, unchanged physical tail |

A precomputed KS buffer is used only for isolated output timing; the complete
arm always recomputes it with fused K packing. CPU layout construction is an
oracle, never a substitute in the timed candidate. Required KS/LogicalKS/Raw/
page/visible input reads and output sink are audited; static branch counts
are not executed instruction counts. No CANN score split or TopK/production
speedup is claimed. Local device compilation and performance remain pending.

Local checks add four tests to the existing56: execute the generated fused KS
stores to prove unique coverage/logical order/zero padding; freeze the K leaf
AST; execute full and tail output paths with bounded pointers and shuffled
pages; require LogicalKS in the live-read audit. Device checks additionally
compare full packed K to the old packer and full logical KS to a CPU oracle,
including rows17/width2057 and partial2049-key preparation, before timing.
