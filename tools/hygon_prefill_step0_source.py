"""CPU-only, fail-closed source construction for a STEP-0-only upper bound.

This deliberately is NOT an operator implementation. A row whose STEP-0
boundary bin exceeds the final-buffer capacity needs an exact fallback.
"""

import hashlib
from pathlib import Path

from hygon_prefill_audit_source import build, function_text, replace_once


def variants(root, dense):
    root = Path(root)
    control = build(root, dense, "carry" if dense else "control")
    job = function_text(control, "_top_k_per_row_job")
    shortened = replace_once(
        job,
        "for step_idx in tl.static_range(0, 4):",
        "for step_idx in tl.static_range(0, 1):",
    )
    fast = replace_once(control, job, shortened)
    compile(fast, "<step0-only-upper-bound>", "exec")
    return control, fast


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for dense in (False, True):
        control, fast = variants(root, dense)
        assert control != fast
        assert fast.count("for step_idx in tl.static_range(0, 1):") == 1
        print(f"dense={dense} control={digest(control)} step0={digest(fast)}")
