# QSA single-query trial: guarded uncontended repeat

The report `96d3f8c32956d789e83998f39f0f6ad2e8475786` passed correctness but
cannot establish a performance result: a competing GPU process and thermal
slowdown were present throughout. This step changes measurement guards only.

| Field | Frozen choice |
| --- | --- |
| Editable | Benchmark environment guard, benchmark plumbing/tests and this document |
| Read-only | Production operator, vector prototype, all tuning sections, Top-K, input/reference helpers, correctness tolerances, the 38-case active set and timer/methodology |
| Hypothesis | An unoccupied GPU without thermal slowdown permits a meaningful repeat of the existing A/B; no new kernel optimization hypothesis |
| Selection | Explicit full GPU UUID through CUDA_VISIBLE_DEVICES; no automatic card selection or job termination |
| Before correctness / each worker | Reject visible compute processes, utilization above 5%, memory above 256 MiB, temperature above 75 C, active thermal slowdown or missing/unparseable telemetry |
| During measurement | Outside timed intervals, check existing before-timing/after-round telemetry plus a process query; allow only this worker's PID or a PID recorded in its Linux NSpid chain; unknown PIDs reject the run |
| Failure | Preserve diagnostic JSON and any correctness results, return nonzero and automatically upload the report; stop launching cases after an environment rejection |
| Limits | Sampling cannot detect every transient co-tenant or replace GPU reservation; unknown host/container PID mapping is rejected rather than guessed |
| Tests | CPU tests with nvidia-smi responses and benchmark/runner integration; GPU repeat executed by user |
| Decision | Candidate remains benchmark-only pending valid returned data |

Both sides must warm up and retune on the selected GPU in the same process
and actual FlagTree compiler. The guarded run rebuilds both baselines; do
not compare its timings against the rejected report. No clock/power/fan
settings or other processes are changed. There is no hardware profiling claim.

## Guard behavior

Enable `--require-idle-gpu` on the existing runner. Without this flag, existing
modes retain their prior behavior. The guard requires a full UUID rather than
a CUDA ordinal, avoiding ambiguity between CUDA and nvidia-smi numbering.
The command below resolves the physical nvidia-smi index once and then binds
correctness, both A/B entries and telemetry to that UUID.

The initial temperature limit is 75 C; during measurement the guard rejects
active SW/HW thermal slowdown. Temperature and SM clocks remain recorded for
manual assessment. When the only initial rejection is a trailing utilization
sample and the process list is empty, it retries twice at one-second intervals.
All attempts are saved. Occupied, hot or unqueryable cards are rejected directly.

`stage-summary.json` stores initial/per-case `idle_gpu_checks`. A rejection
sets `environment_rejected` and exit code 2. Each worker stores a process
snapshot with every existing telemetry point and a `performance_guard` verdict.
Before raising on a rejected measurement point it persists the evidence.
The parent stops launching subsequent cases; incomplete samples are not
converted to speedups. The shell runner automatically uploads the diagnostic
report, including an early rejection before correctness, and preserves exit 2.

The identified worker PID set contains `os.getpid()` plus any Linux `NSpid`
chain visible in `/proc/self/status`. If NVML reports a host PID outside that
set, ownership is unverified and the run is rejected. A container may need a
host-visible PID namespace for a valid repeat; do not whitelist a new GPU PID
based only on its appearance after startup. GPU sampling alone does not prove
exclusive access or detect every transient job; returned data still need review.

## Run and automatically return the report

The index 1 below is a proposed candidate based on the last returned snapshot;
the guard checks current state before proceeding. Change `qsa_device` to an
available physical nvidia-smi index if needed. No automatic selection occurs.

```bash
if bash <<'QSA_IDLE_REPEAT'
set -euo pipefail
source /tmp/qsa-vllm031-h800.PQnVpV/activate_qsa.sh
qsa_device=1
qsa_gpu_uuid=$(nvidia-smi -i "$qsa_device" --query-gpu=uuid --format=csv,noheader)
qsa_repo=$(mktemp -d /tmp/flaggems-qsa-idle.XXXXXX)
git clone --depth 1 --single-branch \
  --branch codex/qsa-select-paged-prefill-h800 \
  https://github.com/cheersluvs/FlagGems-vllm.git "$qsa_repo"
git -C "$qsa_repo" rev-parse HEAD
CUDA_VISIBLE_DEVICES="$qsa_gpu_uuid" \
QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
QSA_PREFILL_AB_KIND=single-row \
  bash "$qsa_repo/tools/run_qsa_prefill_validation_ab_h800.sh" --require-idle-gpu
QSA_IDLE_REPEAT
then
  echo "运行和报告回传均完成。"
else
  echo "运行或回传未成功；当前会话已保留，请检查上方诊断和结果分支。" >&2
fi
```

The Mac has no GPU runtime. Local verification covers policy parsing, real CLI
rejection before GPU imports, early/late benchmark rejection, bounded retries,
namespace uncertainty and run/upload forwarding. The GPU repeat completed;
see the returned result below.

Local result: 55 unittest methods passed across host validation, chunk setup,
vector candidate isolation, GPU guard, benchmark and upload suites. Replaying
the previous report's actual thermal telemetry rejects all 304 snapshots.
Seven frozen code/config/test files and seven case/timer/stage functions were
verified identical to the original vector trial. Pre-commit and shell syntax
checks passed. These results do not validate nvidia-smi behavior on the card;
the completed guarded GPU run and saved diagnostics provide that evidence.

## Returned GPU result

Report `0025302fefd2edbdb18edf8d07f6f29cd6dc3dbe` completed with exit 0:
161 passed, 1 second-device skip, 33 native cases deselected, 38 performance
cases completed, all 39 idle checks and all 304 timed-run guard checks passed.
No foreign GPU PID or thermal slowdown was observed; sampled temperatures
were 29–45 C and SM clocks were 1980 MHz. The candidate did not improve the
primary full-entry wall metric, so production routing stays unchanged. See
[qsa_prefill_single_row_repeat_results.md](qsa_prefill_single_row_repeat_results.md).
