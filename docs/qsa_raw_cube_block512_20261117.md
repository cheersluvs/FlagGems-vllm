# Reuse Q and weights across eight native Cube tiles

| Project truth | Contract |
| --- | --- |
| Goal | Determine whether eight64-key custom calls per512-key update waste material preparation time |
| Frozen | Query16 packing, two Cube computations, FP16/ReLU intermediate, ring layout, two slots, Vector TopK, cross-core events, compiler and production |
| Editable | New native512-key body/export, paired micro source, probe flag, CPU tests and runner |
| Timing | Paired device-timeline timing on prepared inputs; three cases, two rounds; no CANN or acceptance mean |
| Profiler | Existing expected-host/count checks; compile artifacts and exact trace checks in same execution |
| Autotune | Fixed diagnostic candidate on910B, no production registration or NVIDIA tuning changes |

| Contract | Value |
| --- | --- |
| Native ABI | Five64-bit scalar addresses/bits, normal and .cube exports in one translation unit |
| Input | Packed INT8 Q/K, FP16 weights and KS; sixteen query rows per Cube group |
| Work | Eight64-key tiles with one TPipe/local allocation and one Q/weight GM load per512-key block |
| Accumulation | Original INT32 first Cube, Fixpipe FP16+ReLU to L1, FP32 second Cube unchanged |
| Output | Eight consecutive16x64 FP32 tiles in original ring; padding and masks unchanged |
| Reuse | Q and weights remain in L1; reload L0 Q after each second Cube overwrites L0A |
| Synchronization | Original per-tile fences and PIPE_ALL retained; no explicit prefetch or barrier removal |
| Lifecycle | Private allocator remains confined to pointer-only AIC wrapper; no enclosing Cube local allocations |
| Unsupported | No other dtype/layout or custom-op generalization in this probe |

| Path | Behavior |
| --- | --- |
| baseline64 | Existing two-slot kernel, eight original raw calls per update |
| block512 | Same two-slot kernel and exact Vector body, one new raw call per update |
| Fixture | rows16,width512;rows17,width8256;rows112,width29696; zero/partial/full visibility and row/key tails |
| Validation | Exact K/KS packing; independent CPU oracle; pre/post trace, selected scores, output guards; mixed binary metadata and expected device host counts |
| Delivery | Compile, correctness and paired timing in one card run; automatic report upload; production unchanged |

Static source/LLVM evidence is separate from machine instruction execution.
Neither call counts nor timing improvements alone establish hardware overlap.

Local validation:53 raw/ABI/boundary/schedule/summary checks and five complete
input schedule checks passed. The new host C++ model executes the actual native
scheduling body with counting stubs:one allocator, seven buffer declarations,
ten GM input copies (one Q,one weight,eight K),sixteen Cube calls,eight Fixpipe
stages and eight disjoint output tiles. This is not CANN numerical validation.
Python compilation,Bash syntax,focused pre-commit and diff checks passed.
Device compilation,exact numerical checks and timing remain the next card run.
