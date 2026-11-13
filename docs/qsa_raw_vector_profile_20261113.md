# Profiling the raw scorer's Vector preparation and output

## Frozen project scope

| Field | Scope |
| --- | --- |
| Editable | Isolated profiling helper, probe/runner flags and CPU checks |
| Read-only | Raw C++ math, current INT8/production scorer, benchmark harness, installed toolchain |
| Compiler | Same independent final-task-mode copy as the passing raw-score run |
| Active set | compact first 128 rows/W1088 and long100k first 112 rows/W29696; existing inputs and masks |
| Build/validation | Unchanged raw chain plus profiling-only copies; compile, full scoring/packing check, trace and counters in one invocation |
| Benchmark | Two interleaved rounds, warmup3/active6; strict task/name counts, sums exclude host gaps; no formal acceptance average |
| Profiler | Torch-NPU/msprof-backed PipeUtilization and Memory, two invocations per group; original units retained, exported tables archived |
| Autotune | Exempt diagnostic controls, no production configs |
| Timeout | Whole worker 1800 seconds, existing report upload handles failures |

## Contract

| Field | Scope |
| --- | --- |
| Original | Exact raw_pack_k and raw_finish from passing e301e93564cc4a6d1d979d8d025876c323743c5e report |
| Same-grid controls | Copy frozen packed K or score output into fresh storage with the original grid and output tile size |
| Capped controls | Same byte-exact copy with at most 20 programs and a grid-stride loop; diagnostic scheduling comparison only |
| Equality | Copies match entire source, including output -inf padding; no tolerance change, no output supplied to production |
| Traffic scope | Controls match output payload, not all table/scale/index work; warm-cache logical byte rate is not HBM bandwidth or speed of light |
| IR evidence | Count extracted scalar-access loops/markers and retain original text; BCMLIR is not final instruction or runtime evidence |

## Paths

| Path | Interpretation |
| --- | --- |
| Original vs same-grid copy | Layout, address and auxiliary-data cost; not an additive isolated gather cost |
| Same-grid vs capped copy | Scheduling/block-geometry sensitivity without changing copy values |
| Hardware counters | Separate from timing; require numeric rows for the intended kernel and exact iteration count |
| Missing counters | Preserve correctness/timing/artifacts, record unavailable status and exit failure; do not call collection complete |
| Completion | Check packing/output again after profiling, then upload results |

Existing report shows scalar memref.load/tensor.insert loops marked
DiscreteMemAccess and ExtractedLoadOrStore in both K preparation and output.
This is the evidence motivating the targeted profile. It is not yet a claim
that grid geometry or scalar work alone explains the device timings.
The 16-query native Cube redesign is deferred until this diagnostic returns.

Mac checks pass: twelve CPU tests (including grid/payload coverage, capped-loop
tails, unique control names and IR-marker counting), the existing five compiler
copy tests, generated source/Python/Bash syntax and repository pre-commit hooks.
The parser was also run against actual compact/long100k artifacts: pack_k has
one extracted-access loop and finish has three. These are static counts only.
No NPU profiling has run on the Mac.

Use --score-chunks --profile-vectors --score-cases int8_compact int8_long100k.
The vector profiling mode retains correctness and replaces the ordinary
seven-arm chunk timing with the targeted six-arm timing and counter passes.
The controls write separate buffers; unchanged production and raw outputs are
checked again afterward. Exact Copy equality is required, including padding.
