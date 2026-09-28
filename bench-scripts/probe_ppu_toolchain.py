"""Why does this PPU box compile nothing?

A plain `y = x * 2` kernel fails in the vendor's hgbin stage with no shim
installed at all, so the fp8/#1116 question cannot be asked yet. Every failure
there is masked as `NameError: name 'log_file' is not defined` (the vendor's own
failure path opens a name its module never defines), and the assembly is a
NamedTemporaryFile that is deleted before the error is read.

This probe repairs both -- and patches NOTHING else, no options, no semantic --
then prints ppu-llc's real message plus the versions on both sides of it.

    REPO=/path/to/worktree PYTHONPATH=$REPO/src:$PYTHONPATH \
        python3 bench-scripts/probe_ppu_toolchain.py
"""

import glob
import os
import subprocess
import sys
import time
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def k_trivial(s, d, n, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    m = o < n
    tl.store(d + o, tl.load(s + o, mask=m, other=0.0) * 2.0, mask=m)


def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return "({}: {})".format(type(e).__name__, e)


def main():
    import ppu_fp8_shim

    print("=" * 84)
    print("VERSIONS -- the two sides of the failing step")
    print("=" * 84)
    print("  triton {}  from {}".format(getattr(triton, "__version__", "?"), os.path.dirname(triton.__file__)))
    print("  torch  {}".format(torch.__version__))
    for name in ("triton", "flagtree", "torch"):
        out = sh("{} -m pip show {} 2>/dev/null | head -3 | tr '\\n' ' '".format(sys.executable, name))
        print("  pip {}: {}".format(name, out or "(not installed)"))
    others = sh("ls -d /usr/local/lib/python3.12/site-packages/triton 2>/dev/null")
    print("  another triton outside this venv: {}".format(others or "no"))
    print("  ppu-llc: {}".format(sh("ls -l /usr/local/PPU_SDK/bin/ppu-llc 2>&1 | head -1")))
    print("  ppu-llc md5: {}".format(sh("md5sum /usr/local/PPU_SDK/bin/ppu-llc 2>&1 | head -1")))
    print("  ppu-llc --version: {}".format(sh("/usr/local/PPU_SDK/bin/ppu-llc --version 2>&1 | head -4 | tr '\\n' ' | '")))
    try:
        from triton.backends.ppu import compiler as ppuc
        print("  backend's own get_ppu_llc_version(): {}".format(ppuc.get_ppu_llc_version()))
    except Exception as e:
        print("  backend's own get_ppu_llc_version(): failed ({}: {})".format(type(e).__name__, e))
    print("  PPU_SDK contents: {}".format(sh("ls /usr/local/PPU_SDK/bin 2>&1 | tr '\\n' ' '")[:300]))

    # When did a compile last succeed in this cache? If never, the install was
    # never usable here; if recently, something changed under it.
    cache = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
    hits = glob.glob(os.path.join(cache, "*", "*"))
    hits = [(os.path.getmtime(h), h) for h in hits if os.path.isfile(h)]
    print("\n  triton cache {}: {} files".format(cache, len(hits)))
    for t, h in sorted(hits, reverse=True)[:5]:
        print("    {}  {}".format(time.strftime("%Y-%m-%d %H:%M", time.localtime(t)), os.path.basename(h)))

    print("\n" + "=" * 84)
    print("THE REAL ERROR for a plain y = x * 2 kernel (nothing patched but the")
    print("vendor's own masked failure path)")
    print("=" * 84)
    log = []
    keep = ppu_fp8_shim.preserve_vendor_asm(log, keep_dir="/tmp/ppu_keep_trivial")
    ppu_fp8_shim.repair_vendor_error_path(log)
    for line in log:
        print("  " + line)

    dev = torch.device("cuda")
    x = torch.randn(256, device=dev)
    y = torch.empty_like(x)
    try:
        k_trivial[(1, )](x, y, 256, BLOCK=256, num_warps=1)
        torch.cuda.synchronize()
        print("\n  COMPILES AND RUNS. correct: {}".format(bool(torch.equal(y, x * 2))))
        print("\n[RESULT] TOOLCHAIN_OK")
        return
    except Exception as e:
        msg = str(e)
        print("\n  failed: {}".format(type(e).__name__))
        for line in [ln for ln in msg.splitlines() if ln.strip()][:30]:
            print("    " + line[:160])

    err = os.path.join("/tmp/ppu_keep_trivial", "stderr.txt")
    if os.path.exists(err):
        print("\n  ppu-llc stderr, first 20 lines:")
        for line in open(err, errors="replace").read().splitlines()[:20]:
            print("    " + line[:160])
    tix = sorted(glob.glob("/tmp/ppu_keep_trivial/*.trans"))
    if tix:
        print("\n  the assembly it rejected is kept at {} ({} bytes)".format(
            tix[0], os.path.getsize(tix[0])))
        print("  rerun by hand with the reproduce command in stderr.txt")
    print("\n[RESULT] TOOLCHAIN_BROKEN")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
