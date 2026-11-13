# Ascend 910B raw custom-op Cube chain feasibility

This is an isolated experiment, not a production QSA implementation. No compiler
or installed package is edited. Source inspection used FlagTree Ascend commit
394d50493a21dc99bdbfc527b9625dbbb5f327da and the existing local CANN arch22
snapshot recorded in qsa_910b_cann_v1_alignment_plan.md. optimization.md and
deep_opt.md are absent in this checkout.

## Project truth before implementation

| Field | Frozen scope |
| --- | --- |
| Current op | QK INT8 -> L0C INT32 -> Fixpipe DEQF16/ReLU -> L1 half -> weighted half Cube -> GM FP32 |
| Editable | New tools/probe_qsa_raw_cube_chain_910b.py, tools/qsa_raw_cube_chain_910b.cpp, tools/run_qsa_raw_cube_chain_910b.sh, this note |
| Read-only | Production, existing tests, benchmark harness, CANN snapshots, installed compiler/packages |
| Build | ccec ordinary AIC bitcode; pass directly to final kernel link through installed @dialect, otherwise explicit process-local registration |
| Validation | Raw QK exact CPU INT32; identity and grouped-weight chains rtol=atol=1e-3; signed, zero, half-rounding fixtures |
| Benchmark | Two reverse-order rounds, installed do_bench_npu (same timer as harness kernel mode), microseconds; larger ratio is faster |
| Active set | One program M16/K128/N64, identity and grouped weights, three seeded fixtures, unit and 1/1024 prequant scales |
| Aggregation | Paired samples/medians per fixture, no QSA acceptance average |
| Timeout | Whole probe default 900 seconds; compilation commands bounded; failed device subprocess does not prevent report upload |
| Profiler | Save bitcode disassembly, available final IR/assembly and custom-link commands; no hardware traffic claim |
| Autotune | Exempt: fixed-shape API feasibility probe; no production tuning parameters or registrations |
| Fallback | No compute fallback; missing new object entry uses labelled local registration of the same bitcode; missing toolchain returns 2 |

## Reference contract

| Field | Contract |
| --- | --- |
| Torch API | CPU int32 matmul, half cast, clamp and FP32 matmul; reference use only |
| Device/backend | Ascend910B, existing pinned FlagTree/CANN environment |
| Input | CPU logical Q[16,128], K[64,128] INT8 [-8,8], W[16,16] half; device inputs prepacked NZ independently on CPU |
| Output | Fresh GM [16,64], INT32 QK control or FP32 chain; no alias to inputs |
| Promotion | Exact QK INT32; positive prequant scalar; half intermediate after ReLU; half second inputs, FP32 accumulation |
| Semantics | W @ half(relu(float(Q@K.T)*scale)); includes +/-4099 to expose half rounding; values finite, no overflow |
| Unsupported | Other shapes, paging, TopK, arbitrary scales/strides, NaN/Inf, gradients and full QSA numerical equivalence |
| Preparation | CPU packing/transfers excluded from diagnostic timing; must be charged in any future full-input comparison |

## Implementation paths

| Path | Trigger | Implementation | Config/testing | Risk |
| --- | --- | --- | --- | --- |
| Cube control | Always | Private TPipe, packed Q/K load, one INT8 Mmad, GM INT32 sink | Exact full tile | NZ interpretation, external function ABI |
| Cube chain | QK control passes | Same first Cube, explicit M/FIX/MTE1 events, half L1, second Cube | Identity and grouped weights at two scales; basic LLVM gate at completion | CANN APIs in extern functions, memory/sync correctness |
| Unsupported | Required imports/tools/headers missing | Report missing prerequisites and exit 2 | No device execution | Installed wheel may lack source-JIT entry |

The custom Cube function owns all local allocation via one TPipe. Its enclosing
Triton wrapper forwards scalar GM pointers (rank-one memref C ABI), with no local
tensor calculation or other Cube work. This restricts allocation/event ownership
to the micro probe; it does not establish safe composability with other local
Triton allocations or a mixed Cube/Vector production kernel.

The basic structural gate checks exported symbols, at least two Mmad call
sites and CC-to-CBUF evidence in ccec LLVM. The LLVM, link commands and wrapper
IR/available final assembly are retained. DEQF16/ReLU argument interpretation,
final memory traffic and overlap still require review; pipeline_proven remains
false even if the basic gate and all numerical checks pass. Unknown intrinsic
spelling is reported as unproven, not silently treated as success. This avoids
claiming that numerical equivalence proves exact hardware dataflow.

Exit 0 requires the selected entry to compile/link, correctness, basic structural
evidence and timing. It does not mean full CANN pipeline equivalence. Exit 1 means an attempted stage failed or the dataflow
remains unproven; exit 2 means prerequisites are absent. JSON records each stage
separately. Console prints numerical results and timing, not only PASS counts.

Mac verification: Python and generated-kernel syntax, formatter/linter, shell
syntax, negative LLVM evidence cases and prerequisite-report handling. Native
CANN compilation, device correctness and timing require the user-run 910B test.

## First environment report and LLVM text fallback

Result 84631951872b9516aaea52555063cc9a257feffb (20261113-002651) confirms
FlagTree 3.5.1 in the pinned environment exposes CANNJITFunction and accepts the
bitcode object binding. ccec and llvm-link are available, but llvm-dis was not
found. The initial probe returned prerequisite exit 2 before invoking ccec;
there was no native compilation, device correctness result or timing.

llvm-dis is now optional. If absent, ccec emits textual LLVM using -S
-emit-llvm and the same source/target/include/options as the plain bitcode
build. The report marks llvm_text_method=ccec-source-sibling. That text is a
sibling compilation, not a disassembly of the linked bitcode. Preserve this
evidence distinction; the actual linked bitcode, hash, compiler commands and
final wrapper artifacts are still retained. No package installation or compiler
modification is needed for this fallback.

## Native header globals and the AIC-only module

Report d33c12b65306ee578a8ce720b7422d554d73e310 (20261113-004453) compiled
both the ordinary and mix C++ modules with ccec successfully. llvm-link then
rejected their merge: g_tilingKey was multiply defined. There was no Triton
wrapper compilation, device correctness check or timing. The successful C++
compilations alone do not prove execution or the requested hardware dataflow.

This experiment has only a Cube custom op and a pointer-only Triton wrapper,
so it now supplies the single ordinary AIC bitcode directly to the final
kernel compiler. It does not compile/merge the unused mix variant or require
llvm-link. CANN globals are not stripped, renamed or patched. A future mixed
AIC/AIV integration must independently resolve its module/ABI requirements;
this probe does not establish that such an integration is supported.

Save the successful bitcode archive before requesting LLVM text, so a failure
later in text emission or Triton lowering/link still returns that artifact.
The C++ math, memory allocation, scales and numerical comparisons are unchanged.

## Inspect the existing bitcode instead of re-entering the CCE frontend

Report 59a9f3ed4174586a42e4a9e4c6bfd7918ff0da44 (20261113-005133) generated
the ordinary AIC module (105952 bytes, SHA256
54fad612a0d6627a789e58e67914e6e62a429b462f36169b155855babff4318d).
The proposed sibling source-to-text CCE invocation failed because this driver
rejects -S -emit-llvm. That was an artifact-collection failure, before any
Triton wrapper compilation or device execution.

Use llvm-link -S with the single existing bitcode file when llvm-dis is absent.
This writes LLVM text without combining ordinary/mix modules or recompiling
the source. The final kernel continues to consume the original .bc. See the
[LLVM command guide](https://llvm.org/docs/CommandGuide/llvm-link.html).
Text-inspection failure is recorded but no longer suppresses device diagnostics;
the final structural gate still cannot pass without evidence.

The returned bitcode was also read on the Mac using the installed host clang
as an LLVM reader. Its output changes the target triple, so it is used only for
inspection and parser checks, never as device input. The actual module contains
both exported functions and calls to llvm.hivm.MAD.s8,
llvm.hivm.FIX.L0C.TO.L1.s322f16 (AS5 to AS2), llvm.hivm.SET.QUANT.PRE and
llvm.hivm.MAD.f162f32. These are compiler-level evidence, not device execution
or proof of ReLU mode. The parser now recognizes those observed intrinsic names
and requires both matrix types and the explicit INT32-to-half L1 edge, instead
of relying on a generic Mmad count.

## Destination pointer is an external argument, not a DPS output

Report 974036b6dd9824a5e7100ba8510a246fe47d7bb3 (20261113-010029) passed
native C++ compilation and single-module LLVM text inspection. Both exports,
both matrix input types and the direct L0C INT32 -> L1 half edge were detected.
The first QK Triton wrapper then failed TTIR verification: operand 4 in outs
was a scalar !tt.ptr<i32>, while a DPS output must be a tensor or memref.
No wrapper binary ran and there are still no timing or numerical results.

The corrected call puts all five C++ arguments, including the destination GM
pointer, in custom-op inputs and requests no DPS outputs/results. The registry
variant calls its fifth ordinary parameter dst rather than the reserved out.
This is a pointer-writing external function, not a tensor-returning primitive;
the original ranked GM memref C ABI and C++ signature are unchanged.

Do not annotate this opaque call no_side_effect. The inspected HIVM custom-op
implementation conservatively models resource-level reads/writes in this form.
The installed toolchain's behavior still needs checking: retain available IR,
and require the sentinel-initialized GM destination to match all CPU QK and
full-chain fixtures. A deleted/no-op custom call must fail those checks. This
isolated wrapper contains no other device work with which it could be reordered.
This does not establish alias/scheduling correctness for a future mixed kernel.
