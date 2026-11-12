# Explicit Cube buffer handoff feasibility

The previous to_buffer(L1)->to_tensor->tl.dot candidate crashed during compilation.
It did not test the lower-level tile_cube_launch API with an explicit L0C
accumulator and L1 destination. This is a new small feasibility experiment, not
an operator optimization or a compiler modification.

| Stage | Intent and gate |
| --- | --- |
| launch_control | INT8 M16/K128/N64 -> explicit INT32 L0C -> GM output; require retained Cube op and exact CPU QK equality |
| direct_half_l1 | Same first launch with FP16 L1 destination -> second half-input Cube; require INT32 L0C Fixpipe directly into FP16 cbuf, no tile-sized INT32/FP16 GM intermediates, compare to CPU half-rounded QK then weighting |
| relu_half_l1 | Express maximum on the half L1 tensor before rebuilding L1 input; require the compiler to fuse ReLU into the first Fixpipe, rather than accept a Vector/GM detour; same CPU checks |

The third expression intentionally leaves fusion to the existing compiler.
Its source-level maximum is not itself proof of a Fixpipe activation. Only
post-sync-solver IR with a positive ReLU attribute on the same INT32-to-FP16
L1 Fixpipe is accepted. Any missing/truncated dump, unknown representation,
extra intermediate GM tile, or missed fusion rejects the structural claim.
L0C->UB is not requested; no Ascend910_95 predicates are bypassed.

Use existing generic DSA alloc/copy/cube_launch/cube_wait/to_tensor/to_buffer.
The public empty mma descriptor default is retained, with no guessed backend
attributes. Missing exports, an unsupported descriptor or failed ordinary launch
control means this particular API expression is blocked; it does not prove all
possible generic TLE programs impossible. Save source and compiler inputs/errors.

Input uses signed INT8 values in [-8,8], a fixed seed, nonzero positive and
negative QK values, half block-diagonal weights and FP32 second-Cube output.
QK is checked exactly; half-rounded paths use rtol/atol1e-3. This tests unit-scale
conversion, not the exact CANN DEQF16 scale flag or saturation/overflow contract.
There is no public prequant parameter in this generic call; do not guess or
silently adjust the reference to fit an unknown conversion scale.

Compilation uses the print-after-sync-solver option, and executes only binaries
whose same-process IR has passed. A successful full chain is compared against a
plain tensor two-dot/ReLU reference using the existing harness kernel timer,
two reverse-order rounds. Both compile with the same print-only option.
Timing is diagnostic single-program latency, with prepared weights, no paging,
no TopK and no full-input acceptance mean. There is no claim of QSA speedup.

QSA_EXPLICIT_CUBE_HANDOFF=1 selects a dedicated tiny pytest node and summary.
Exit0 indicates collection completed; pipeline_proven is true only after all
three stages pass and paired timing exists. Device failures return nonzero.
No production kernels, TopK implementation, installed packages or compiler source
are changed. The existing autotune regime is not applicable to this fixed
feasibility micro experiment.

## Output-pointer correction after the first preflight

Report 90d7c4f5 had all explicit DSA APIs but no
al.ascend_address_space.GM. The initial probe incorrectly required this enum,
so no compilation or device test occurred. This is a probe preflight bug, not
evidence against 910B or generic TLE.

The corrected expression uses a caller-owned output block pointer directly as
tile_cube_launch's dst. The installed frontend intentionally validates a/b/acc/
stage_a/stage_b as buffers but forwards dst.handle without that buffer check.
The ordinary launch_control checks whether the installed backend accepts this
output representation; no claim of backend support precedes that compile test.
This also removes an unnecessary temporary GM allocation and GM-to-output copy.
No numeric reference, L0C/L1 candidate or structural acceptance gate changes.
