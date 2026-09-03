"""Load the TLE dialect into the Ascend backend's MLIRContext.

The dialect is compiled into libtriton and has a registration binding
(`libtriton.tle.load_dialects`), but the Ascend backend's load_dialects only
registers its own:

    def load_dialects(self, ctx):
        ascend.load_dialects(ctx)          # nvidia's also calls nvidia + instrumentation

so every tle op fails to build with a hard abort, not a diagnostic:

    LLVM ERROR: Building op `tle.local_pointers` but it isn't known in this
    MLIRContext: the dialect may not be loaded or this operation hasn't been
    added by the dialect

This adds the missing call.  It patches the backend class reached through the
backend registry rather than importing it by name, so it does not depend on the
class's name staying put.  Nothing in the generic operator changes.

Registration is necessary, not sufficient: if the ops register but the Ascend
lowering pipeline has no pattern for them, the failure moves later.  That is the
point of trying it.
"""


def apply() -> str:
    from triton._C import libtriton

    if not hasattr(libtriton, "tle") or not hasattr(libtriton.tle, "load_dialects"):
        return "libtriton has no tle.load_dialects -- the dialect is not in this build"

    import triton.backends as tb

    backend = tb.backends.get("ascend")
    if backend is None:
        return f"no ascend backend registered (have {sorted(tb.backends)})"

    cls = backend.compiler
    if getattr(cls, "_tle_dialect_patched", False):
        return "already patched"

    original = cls.load_dialects

    def load_dialects(self, ctx):
        original(self, ctx)
        libtriton.tle.load_dialects(ctx)

    cls.load_dialects = load_dialects
    cls._tle_dialect_patched = True
    return f"patched {cls.__module__}.{cls.__qualname__}.load_dialects"
