"""Enable the f32 -> fp8e4nv cast of FlagTree PR #1116 on a flagtree that predates it.

Why this exists
---------------
PR #1116 (merged into flagtree main on 2026-09-24, in no release or 0.7.0-rc
branch) lets capability 80-88 use fp8e4nv through a *software* cast.  On the
PPU the two directions are implemented in different layers:

  fp8e4nv -> f16   assembly, third_party/ppu/lib/.../ElementwiseOpToLLVM.cpp
  f32     -> fp8e4nv   pure Python, third_party/ppu/language/ppu/utils.py
                       ("downcasts to f8e4m3nv are implemented in the frontend")

fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert only ever *writes* fp8, so
only the downcast matters -- and that one needs no rebuilt compiler.  This
module installs it as a monkey patch: the same numerical code as #1116
(copied from third_party/ppu/language/ppu/utils.py on main, MIT, T-Head
copyright), plus the two declarations #1116 adds around it.

It is a measurement tool, never something to ship: it makes the generic
FlagGems kernel compile so its cost can be compared against the vendor
override.  install() verifies the cast byte-for-byte against torch before
reporting success, so a silent numerical difference cannot reach a timing.
"""

import dataclasses
import os
import tempfile

import torch
import triton
import triton.language as tl
from triton.language import core

_INSTALLED = {}


# --------------------------------------------------------------------------
# the numerical cast, from flagtree main third_party/ppu/language/ppu/utils.py
# (@core.builtin dropped: these are called from inside semantic.cast, which
# already holds the semantic object, not from kernel source)
# --------------------------------------------------------------------------
def _rounding_is_rtz(fp_downcast_rounding):
    if fp_downcast_rounding is None:
        return False
    if isinstance(fp_downcast_rounding, str):
        return fp_downcast_rounding.lower() == "rtz"
    try:
        from triton._C.libtriton import ir
        return fp_downcast_rounding == ir.ROUNDING_MODE.RTZ
    except Exception:
        return False


def _upcast_e4nv_to_f16(sem, arg):
    """Software OCP E4M3FN -> fp16; exact for every one of the 256 encodings."""
    u = arg.to(core.uint8, bitcast=True, _semantic=sem).to(core.uint16, _semantic=sem)
    s = sem.shl(sem.and_(u, 0x0080), 8)
    em = sem.and_(u, 0x007F)
    e = sem.lshr(em, 3)
    m = sem.and_(em, 0x0007)
    normal = sem.or_(sem.shl(sem.add(e, 8, False), 10), sem.shl(m, 7))
    sub_f = sem.mul(m.to(core.float32, _semantic=sem), 2.0**-9, False)
    sub_bits = sub_f.to(core.float16, _semantic=sem).to(core.uint16, bitcast=True, _semantic=sem)
    body = sem.where(sem.equal(e, 0), sub_bits, normal)
    body = sem.where(sem.equal(em, 0x007F), 0x7E00, body)
    return sem.or_(body, s).to(core.float16, bitcast=True, _semantic=sem)


def _f64_to_f32_round_to_odd(sem, arg):
    b = arg.to(core.int64, bitcast=True, _semantic=sem)
    sign = sem.shl(sem.and_(sem.lshr(b, 63), 1), 31)
    e64 = sem.and_(sem.lshr(b, 52), 0x7FF)
    m64 = sem.and_(b, (1 << 52) - 1)
    is_nan = sem.and_(sem.equal(e64, 0x7FF), sem.not_equal(m64, 0))
    e32 = sem.sub(e64, 1023 - 127, False)
    sticky = sem.not_equal(sem.and_(m64, (1 << 29) - 1), 0).to(core.int64, _semantic=sem)
    m32 = sem.or_(sem.lshr(m64, 29), sticky)
    bits = sem.or_(sem.shl(e32, 23), m32)
    bits = sem.where(sem.greater_equal(e32, 0xFF), 0x7F800000, bits)
    bits = sem.where(sem.less_equal(e32, 0), 0, bits)
    bits = sem.where(is_nan, 0x7FC00000, bits)
    return sem.or_(bits, sign).to(core.int32, _semantic=sem).to(core.float32, bitcast=True, _semantic=sem)


def _downcast_f32_to_e4nv(sem, arg, rtz):
    """Software fp32 -> OCP E4M3FN, single rounding (RTNE or RTZ), satfinite."""
    b = arg.to(core.int32, bitcast=True, _semantic=sem)
    sign = sem.shl(sem.and_(sem.lshr(b, 31), 1), 7)
    ab = sem.and_(b, 0x7FFFFFFF)
    is_nan = sem.greater_than(ab, 0x7F800000)
    e32 = sem.lshr(ab, 23)
    m32 = sem.and_(ab, 0x7FFFFF)
    E = sem.sub(e32, 120, False)
    pn = core.PropagateNan.NONE
    k = sem.add(sem.minimum(sem.maximum(sem.sub(1, E, False), 0, pn), 6, pn), 20, False)
    sig = sem.or_(m32, 0x800000)
    keep = sem.lshr(sig, k)
    if not rtz:
        rem = sem.and_(sig, sem.sub(sem.shl(1, k), 1, False))
        half = sem.shl(1, sem.sub(k, 1, False))
        up = sem.or_(sem.greater_than(rem, half), sem.and_(sem.equal(rem, half), sem.equal(sem.and_(keep, 1), 1)))
        keep = sem.add(keep, up.to(core.int32, _semantic=sem), False)
    expmant = sem.where(sem.greater_equal(E, 1), sem.sub(sem.add(sem.shl(E, 3), keep, False), 8, False), keep)
    expmant = sem.minimum(expmant, 0x7E, pn)
    res = sem.or_(expmant, sign)
    res = sem.where(is_nan, 0x7F, res)
    return res.to(core.uint8, _semantic=sem).to(core.float8e4nv, bitcast=True, _semantic=sem)


# --------------------------------------------------------------------------
# the two declarations #1116 adds: the dtype is legal, and its cast is custom
# --------------------------------------------------------------------------
def _active_backend_cls(log):
    """The backend class triton would compile with.

    NOT `backends[target.backend]`: that dict is keyed by backend *directory*
    ("ppu", "nvidia"), while a vendor that aliases the device onto cuda reports
    `target.backend == "cuda"` -- which is how the first run of this shim died
    with KeyError: 'cuda'. `make_backend` scans `supports_target` instead, which
    is the same resolution the compiler performs.
    """
    target = triton.runtime.driver.active.get_current_target()
    try:
        from triton.compiler.compiler import make_backend
        cls = type(make_backend(target))
        log.append("backend {} (target.backend={!r}, via make_backend)".format(cls.__name__, target.backend))
        return cls
    except Exception as e:
        log.append("make_backend failed ({}), scanning supports_target".format(str(e)[:60]))
    from triton.backends import backends
    for name, be in backends.items():
        ok = False
        try:
            ok = bool(be.compiler.supports_target(target))
        except Exception:
            try:
                ok = isinstance(triton.runtime.driver.active, be.driver)
            except Exception:
                ok = False
        if ok:
            log.append("backend {} (dir {!r}, target.backend={!r})".format(be.compiler.__name__, name, target.backend))
            return be.compiler
    raise RuntimeError("no registered backend claims target {!r}; registered: {}".format(
        target, sorted(backends)))


def repair_vendor_error_path(log):
    """Un-hide the vendor compiler's real error.

    `triton/backends/ppu/compiler.py:make_hgbin` opens `log_file` on its failure
    path, and that name is defined nowhere: every binary-stage failure on this
    box surfaces as `NameError: name 'log_file' is not defined` instead of the
    reason. Python resolves the free variable in the function's *module*
    globals, so defining it there lets the vendor's own handler finish and raise
    (or log) the real message. Returns the path it will be written to, if any.
    """
    import sys as _sys
    try:
        cls = _active_backend_cls(log)
    except Exception as e:
        log.append("cannot locate the backend module ({})".format(str(e)[:60]))
        return None
    mod = _sys.modules.get(cls.__module__)
    if mod is None or hasattr(mod, "log_file"):
        return getattr(mod, "log_file", None)
    fn = getattr(cls, "make_hgbin", None)
    names = getattr(getattr(fn, "__code__", None), "co_names", ())
    if "log_file" not in names:
        return None
    path = os.path.join(tempfile.gettempdir(), "ppu_hgbin_error.log")
    open(path, "w").close()
    mod.log_file = path
    log.append("{}.log_file was undefined (its own failure path raises NameError); "
               "pointed it at {}".format(cls.__module__, path))
    return path


def preserve_vendor_asm(log, keep_dir="/tmp/ppu_keep"):
    """Keep the assembly ppu-llc rejected.

    make_hgbin feeds ppu-llc a NamedTemporaryFile, so the file is gone by the
    time the error is read and the reproduce command it prints cannot be run.
    Wrap the *module's* `subprocess`: on CalledProcessError, copy every existing
    path in the command line into keep_dir, with the stderr beside it.
    """
    import shutil
    import sys as _sys
    cls = _active_backend_cls(log)
    mod = _sys.modules.get(cls.__module__)
    sp = getattr(mod, "subprocess", None)
    if sp is None or getattr(sp, "_shim_proxy", False):
        return keep_dir
    os.makedirs(keep_dir, exist_ok=True)

    class _Proxy(object):
        _shim_proxy = True

        def __getattr__(self, k):
            return getattr(sp, k)

        def run(self, cmd, *a, **kw):
            try:
                return sp.run(cmd, *a, **kw)
            except sp.CalledProcessError as e:
                toks = cmd.split() if isinstance(cmd, str) else list(cmd)
                for t in toks:
                    if os.path.isfile(t):
                        try:
                            shutil.copy(t, os.path.join(keep_dir, os.path.basename(t)))
                        except Exception:
                            pass
                try:
                    open(os.path.join(keep_dir, "stderr.txt"), "w").write(
                        (e.stderr or "") + "\n---cmd---\n" + (cmd if isinstance(cmd, str) else " ".join(cmd)))
                except Exception:
                    pass
                raise

    mod.subprocess = _Proxy()
    log.append("vendor subprocess wrapped: rejected assembly is kept in " + keep_dir)
    return keep_dir


def _patch_backend_options(log):
    backend_cls = _active_backend_cls(log)
    orig = backend_cls.parse_options

    def parse_options(self, opts, _orig=orig):
        o = _orig(self, opts)
        fp8 = tuple(getattr(o, "supported_fp8_dtypes", ()) or ())
        if "fp8e4nv" in fp8:
            return o
        want = fp8 + ("fp8e4nv", )
        try:
            object.__setattr__(o, "supported_fp8_dtypes", want)
        except Exception:
            pass
        if "fp8e4nv" not in tuple(getattr(o, "supported_fp8_dtypes", ()) or ()):
            o = dataclasses.replace(o, supported_fp8_dtypes=want)
        return o

    backend_cls.parse_options = parse_options
    log.append("backend {}.parse_options declares fp8e4nv".format(backend_cls.__name__))
    return backend_cls, orig


def _patch_semantic_cast(log):
    from triton.language import semantic as semantic_mod
    cls = None
    for name in ("TritonSemantic", "Semantic"):
        cls = getattr(semantic_mod, name, None)
        if cls is not None and hasattr(cls, "cast"):
            break
    assert cls is not None, "no semantic class with a cast method (unexpected triton layout)"
    orig_cast = cls.cast

    def cast(self, input, dst_ty, *a, **kw):
        try:
            src = input.type.scalar
            dst = dst_ty.scalar
            rounding = a[0] if a else kw.get("fp_downcast_rounding")
            if dst.is_fp8e4nv() and src.is_floating() and not src.is_fp8e4nv():
                if src.is_fp32():
                    x = input
                elif src.is_fp64():
                    x = _f64_to_f32_round_to_odd(self, input)
                else:
                    x = input.to(core.float32, _semantic=self)
                return _downcast_f32_to_e4nv(self, x, _rounding_is_rtz(rounding))
            if src.is_fp8e4nv() and dst.is_floating() and not dst.is_fp8e4nv():
                up = _upcast_e4nv_to_f16(self, input)
                return up if dst.is_fp16() else up.to(dst, _semantic=self)
        except AttributeError:
            pass
        return orig_cast(self, input, dst_ty, *a, **kw)

    cls.cast = cast
    log.append("{}.cast routes fp8e4nv through the software cast".format(cls.__name__))
    return cls, orig_cast


# --------------------------------------------------------------------------
# verification: the shim is only usable if it agrees with torch bit for bit
# --------------------------------------------------------------------------
@triton.jit
def _shim_probe(src, dst, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    x = tl.load(src + off, mask=m, other=0.0)
    tl.store(dst + off, x.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=m)


def _edge_values(n=4096):
    """Everything the encoder can get wrong: ties, the subnormal ladder, the
    saturation boundary, zeros and signs -- plus random values."""
    fixed = [0.0, -0.0, 1.0, -1.0, 448.0, -448.0, 447.9, 448.1, 500.0, 1e30,
             2.0**-9, 2.0**-10, 2.0**-6, 1.5 * 2.0**-9, 0.5 * 2.0**-9,
             1.0625, 1.09375, 1.03125, 1.046875, 1.0234375,  # RTNE ties at k=20
             260.0, 264.0, 268.0, 272.0, 240.0, 232.0]
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(max(0, n - len(fixed)), generator=g) * torch.exp(
        torch.rand(max(0, n - len(fixed)), generator=g) * 12 - 6)
    return torch.cat([torch.tensor(fixed, dtype=torch.float32), rnd.float()])[:n].contiguous()


def probe_cast(device, n=4096, sync=None, block=256):
    """Compile and run the cast; return (ok, mismatches, detail)."""
    x = _edge_values(n)
    src = x.to(device)
    dst = torch.zeros(n, dtype=torch.uint8, device=device)
    grid = ((n + block - 1) // block, )
    _shim_probe[grid](src, dst, n, BLOCK=block, num_warps=1)
    if sync is not None:
        sync()
    else:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    got = dst.cpu()
    ref = x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    bad = (got != ref).nonzero().flatten().tolist()
    detail = ""
    if bad:
        i = bad[0]
        detail = "first: x={!r} got 0x{:02X} torch 0x{:02X} ({} of {} differ)".format(
            float(x[i]), int(got[i]), int(ref[i]), len(bad), n)
    return (not bad), len(bad), detail


def native_cast_works(device, sync=None):
    try:
        ok, bad, detail = probe_cast(device, 256, sync=sync)
        return True, ("bit-exact vs torch" if ok else "COMPILES BUT DIFFERS: " + detail)
    except Exception as e:
        return False, str(e).splitlines()[0][:100]


def install(device, verbose=True, sync=None):
    """Install the #1116 downcast on a flagtree that lacks it.

    Returns a dict: mode is "native" (nothing was patched), "shim" (patched and
    verified byte-exact) or "unavailable" (with reason).
    """
    if _INSTALLED:
        return _INSTALLED
    info = {"mode": None, "log": [], "reason": ""}
    native, detail = native_cast_works(device, sync=sync)
    if native:
        info["mode"] = "native"
        info["reason"] = detail
        _INSTALLED.update(info)
        return info
    info["log"].append("native cast unavailable: " + detail)
    info["vendor_log"] = repair_vendor_error_path(info["log"])
    try:
        _patch_backend_options(info["log"])
        _patch_semantic_cast(info["log"])
    except Exception as e:
        import traceback
        info["mode"] = "unavailable"
        info["reason"] = "patching failed: {}: {}".format(type(e).__name__, e)
        info["log"].append("traceback:\n" + "".join(traceback.format_exc()[-800:]))
        _INSTALLED.update(info)
        return info
    try:
        ok, bad, detail = probe_cast(device, sync=sync)
    except Exception as e:
        import traceback
        info["mode"] = "unavailable"
        info["reason"] = "shim compiled/ran with an error: {}: {}".format(
            type(e).__name__, str(e).splitlines()[0][:160])
        info["log"].append("traceback:\n" + "".join(traceback.format_exc()[-1200:]))
        vlog = info.get("vendor_log")
        if vlog and os.path.exists(vlog) and os.path.getsize(vlog):
            info["log"].append("vendor compiler log {} (tail):\n{}".format(
                vlog, open(vlog, errors="replace").read()[-2000:]))
        _INSTALLED.update(info)
        return info
    if not ok:
        info["mode"] = "unavailable"
        info["reason"] = "shim is numerically WRONG, refusing to time it -- " + detail
    else:
        info["mode"] = "shim"
        info["reason"] = "byte-exact vs torch on {} values (ties, subnormals, saturation)".format(4096)
    _INSTALLED.update(info)
    if verbose:
        for line in info["log"]:
            print("    " + line)
    return info


if __name__ == "__main__":
    import flaggems_vllm
    dev = flaggems_vllm.device
    got = install(torch.device(dev) if not isinstance(dev, torch.device) else dev)
    print("mode: {}\nreason: {}".format(got["mode"], got["reason"]))
