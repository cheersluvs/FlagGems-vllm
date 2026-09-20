"""Construct exact non-TLE production variants with only VEC changed."""

from hygon_prefill_audit_source import build, function_text, replace_once


FUNCTIONS = ("_process_histogram_step", "non_tle_top_k_per_row_prefill")


def variants(root, dense):
    control = build(root, dense, "carry" if dense else "control")
    result = {4: control}
    for vec in (1, 2, 8):
        source = control
        for name in FUNCTIONS:
            old = function_text(source, name)
            new = replace_once(old, "    VEC: tl.constexpr = 4", f"    VEC: tl.constexpr = {vec}")
            source = replace_once(source, old, new)
        compile(source, f"<hygon-prefill-vec{vec}>", "exec")
        result[vec] = source
    return result
