"""The eager baseline against the operator, ONE SHAPE PER PROCESS.

WHY. The single-process version died with

    rtDeviceSynchronizeWithTimeout execution failed, reason=aicpu timeout
    runtime result = 507017.  The aicpu execution times out.

which is a device-level failure, not an out-of-memory. Everything after it in
that process repeated the same error from inside `npuSynchronizeDevice` and
`empty_cache` -- the process was poisoned, so no measurement taken after the
first failure would have been trustworthy even if the run had continued.

The likely source is the eager composition itself: its paged-cache insert is an
advanced-indexing scatter, which lands on AICPU, and at 131072 tokens that is a
131072 x 448 `index_put` on the slow scalar unit. But "likely" is not measured,
and the single-process run could not say which shape or which side of the
comparison died, because the error spam buried the last good line.

So: each shape gets a fresh process. A shape that faults the card takes only
itself down, the shapes that work still report, and the log says exactly which
is which. This is the same isolation the compiler probes in this directory use,
for the same reason -- output to a FILE rather than a pipe, because an aborted
Ascend process leaves workers holding the pipe's write end, and the whole
process group swept afterwards so strays cannot linger into the next shape.

What is NOT changed to make this work: the baseline. Replacing its scatter with
something friendlier to this card would be hand-optimising the thing the
operator is being compared against. If the eager composition cannot complete a
shape, that is a property of doing this work with framework operators here, and
it belongs in the result.

Each child drives the repo's own Benchmark -- same do_bench, same kernel mode,
same make_input -- restricted to one shape, with the eager composition bound
into the empty torch_op slot.

    REPO=/path/to/FlagGems-vllm python3 run_eager_isolated.py
    (child mode is entered by the driver: ... run_eager_isolated.py <n> <h>)

`eager_baseline.py` must sit beside this file.
"""

import importlib
import os
import re
import signal
import subprocess
import sys
import time
import traceback

REPO = os.environ.get("REPO", "/home/secure/wuyuqing/workspace/FlagGems-vllm")
TIMEOUT = int(os.environ.get("SHAPE_TIMEOUT", "900"))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

SHAPES = [
    (n, h)
    for n in (1, 4, 17, 64, 1024, 2048, 8192, 32768, 65536, 98304, 131072)
    for h in (64, 128)
]

# ONLY_SHAPES="98304x128,131072x128" re-runs a subset without editing the file.
if os.environ.get("ONLY_SHAPES"):
    SHAPES = [
        tuple(int(v) for v in part.split("x"))
        for part in os.environ["ONLY_SHAPES"].split(",")
    ]


def patch_randn(torch):
    """Build large bfloat16 tensors without an fp32 temporary; see the other
    runners. It matters more here, since the eager baseline needs the room."""
    real_randn = torch.randn
    cache = {}

    def randn_no_fp32_temp(*size, **kw):
        dtype = kw.get("dtype")
        dev = kw.get("device")
        if dtype is not torch.bfloat16 or dev is None:
            return real_randn(*size, **kw)
        shape = (
            size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
        )
        t = torch.empty(*shape, dtype=dtype, device=dev)
        n = t.numel()
        chunk = min(n, 1 << 22)
        key = (chunk, str(dev))
        if key not in cache:
            cache[key] = (
                real_randn(chunk, dtype=torch.float32).to(torch.bfloat16).to(dev)
            )
        seed = cache[key]
        flat = t.view(-1)
        for off in range(0, n, chunk):
            end = min(off + chunk, n)
            flat[off:end].copy_(seed[: end - off])
        return t

    torch.randn = randn_no_fp32_temp


def child(n, h):
    from benchmark import conftest as cf
    from benchmark import consts

    assert cf.Config is None
    cf.Config = cf.BenchConfig()
    cf.Config.mode = consts.BenchMode.KERNEL
    cf.Config.bench_level = consts.BenchLevel.CORE
    cf.Config.query = False

    import torch
    import torch_npu  # noqa: F401

    patch_randn(torch)

    from eager_baseline import eager_fused_deepseek_v4

    mod = importlib.import_module(
        "benchmark.test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    )
    cls = mod.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark
    cls.get_performance_test_params = staticmethod(
        lambda: [
            mod.TestParam(
                n, h, num_tokens_insert=n, block_size=64, max_pos=4096, eps=1e-6
            )
        ]
    )

    bench = cls()
    bench.torch_op = eager_fused_deepseek_v4

    import benchmark.base as base

    base.pytest.fail = lambda *a, **k: None

    real_latency = bench.get_latency
    state = {"i": 0}

    def timed(op, *args, **kwargs):
        state["i"] += 1
        which = "eager" if state["i"] % 2 == 1 else "fused"
        # PRINT THE EXCEPTION HERE. The single-process runner did this and it
        # worked; dropping it when this file was written cost a round trip. The
        # harness will not do it for you: the timing except block stores the
        # message in `metric.error_msg` and prints nothing, and the table shows
        # only FAILED / N/A. (The input-iterator except block a few lines above
        # it does print, but its second string literal is not an f-string, so it
        # emits `err=<<<{e}>>>` verbatim -- a separate repo bug, and not the one
        # that fires here.)
        try:
            ms = real_latency(op, *args, **kwargs)
        except Exception as e:
            print("[FAIL] {} {}".format(
                which, " | ".join(str(e).splitlines()[:2])[:220]), flush=True)
            raise
        # One machine-readable line per measurement, flushed, so the driver can
        # recover whichever side completed even if the other kills the process.
        print("[MEAS] {} {:.6f}".format(which, ms), flush=True)
        return ms

    bench.get_latency = timed
    bench.run()
    print("[RESULT] SHAPE_OK", flush=True)


def run_shape(n, h):
    log = "/tmp/eager_{}x{}.log".format(n, h)
    t0 = time.time()
    timed_out = False
    with open(log, "w") as lf:
        p = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), str(n), str(h)],
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=dict(os.environ, REPO=REPO),
        )
        try:
            rc = p.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            timed_out = True
            rc = None
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    if timed_out:
        rc = p.wait()
    elapsed = time.time() - t0

    with open(log, errors="replace") as f:
        out = f.read()

    meas = dict(
        (m.group(1), float(m.group(2)))
        for m in re.finditer(r"\[MEAS\] (eager|fused) ([0-9.]+)", out)
    )
    if timed_out:
        note = "TIMED OUT after {}s".format(TIMEOUT)
    elif "aicpu timeout" in out or "507017" in out:
        note = "AICPU TIMEOUT (507017) -- device fault, not OOM"
    elif "out of memory" in out.lower():
        note = "OUT OF MEMORY"
    elif rc is not None and rc < 0:
        note = "killed by signal {}".format(-rc)
    elif "[RESULT] SHAPE_OK" in out:
        note = ""
    else:
        note = "no result line (rc={})".format(rc)

    fails = re.findall(r"\[FAIL\] (eager|fused) (.*)", out)
    if fails:
        note = "; ".join("{} raised: {}".format(w, m) for w, m in fails)
    elif not note and len(meas) < 2:
        # Exited cleanly having measured nothing. That is a contradiction, and
        # calling it "incomplete" is how it went undiagnosed for a round trip.
        note = "EXITED CLEANLY WITH NO MEASUREMENT -- diagnostic is missing"
    return meas, note, elapsed, log


def driver():
    print("The operator vs an eager torch_npu composition, one shape per process.")
    print()
    print("SpeedUp is eager / fused. The baseline is CONSTRUCTED -- there is no")
    print("vendor kernel on this card and vLLM's portable Triton does not compile")
    print("here -- so this measures what framework operators cost, not a margin")
    print("over vLLM. Per-shape logs: /tmp/eager_<n>x<h>.log")
    print()
    print("  {:>7} {:>5} {:>12} {:>12} {:>10} {:>7}  {}".format(
        "tokens", "heads", "eager ms", "fused ms", "speedup", "secs", "note"))

    for n, h in SHAPES:
        meas, note, elapsed, _ = run_shape(n, h)
        e, f = meas.get("eager"), meas.get("fused")
        if e is not None and f is not None:
            print("  {:>7} {:>5} {:>12.4f} {:>12.4f} {:>9.2f}x {:>7.0f}  {}".format(
                n, h, e, f, e / f, elapsed, note), flush=True)
        else:
            got = "eager only" if e else ("fused only" if f else "neither")
            print("  {:>7} {:>5} {:>12} {:>12} {:>10} {:>7.0f}  {} [{}]".format(
                n, h, "{:.4f}".format(e) if e else "-",
                "{:.4f}".format(f) if f else "-",
                "-", elapsed, note or "incomplete", got), flush=True)

    print("\n[RESULT] EAGER_ISOLATED_DONE")


try:
    if len(sys.argv) == 3:
        child(int(sys.argv[1]), int(sys.argv[2]))
    else:
        driver()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
