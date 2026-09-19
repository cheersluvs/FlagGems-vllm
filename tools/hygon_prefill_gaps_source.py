"""CPU-only, fail-closed construction of the remaining Hygon experiments."""

import ast

from hygon_prefill_audit_source import OVERRIDE, build, function_text, replace_once


def atomic_to_scan(source, function, counter, mask):
    old = function_text(source, function)
    calls = [
        n
        for n in ast.walk(ast.parse(old))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "tl"
        and n.func.attr == "atomic_add"
        and isinstance(n.args[0], ast.Name)
        and n.args[0].id == counter
    ]
    if len(calls) != 1:
        raise ValueError(f"Expected one atomic for {function}/{counter}")
    call = calls[0]
    actual_mask = next(k.value for k in call.keywords if k.arg == "mask")
    if not isinstance(actual_mask, ast.Name) or actual_mask.id != mask:
        raise ValueError("Atomic mask drifted")
    new = replace_once(
        old, ast.get_source_segment(old, call), f"_alloc_slots({counter}, {mask})"
    )
    return replace_once(source, old, new)


def variant(root, dense, arm):
    source = build(root, dense, "carry" if dense else "control")
    if arm == "control":
        return source
    if arm.startswith("radix"):
        gate = int(arm.removeprefix("radix"))
        if gate not in (0, 64, 256):
            raise ValueError("Unsupported radix crossover")
        old = function_text(source, "_final_select_radix")
        start = old.index("    s_radix_counts = tle.gpu.alloc(")
        end = old.index("    radix_count_vec_ptr", start)
        new = (
            old[:start] + "    s_radix_count_ptr = s_histogram_ptr + 2048\n" + old[end:]
        )
        new = replace_once(
            new,
            "                prefix_sum, _ = tle.cumsum(counts, axis=0, reverse=False)",
            "                prefix_sum = tl.cumsum(counts, axis=0) - counts",
        )
        if "tle." in new:
            raise ValueError("Unported TLE dependency")
        source = replace_once(source, old, new)
        source = replace_once(
            source,
            "        if USE_RADIX_FINAL and HAS_TLE:",
            f"        if USE_RADIX_FINAL and tl.load(s_final_cnt_ptr) >= {gate}:",
        )
        old = function_text(source, "non_tle_top_k_per_row_prefill")
        new = replace_once(
            old, "        USE_RADIX_FINAL=False,", "        USE_RADIX_FINAL=True,"
        )
        new = replace_once(
            new,
            "    s_histogram_ptr += row_id * NUM_BINS",
            "    s_histogram_ptr += row_id * 2304",
        )
        source = replace_once(source, old, new)
    elif arm in ("found_scan", "both_scan", "final_scan"):
        if dense and arm != "final_scan":
            raise ValueError("Dense production already carries the found counter")
        if not dense:
            source += (
                "\n\n"
                + function_text((root / OVERRIDE).read_text(), "_alloc_slots")
                + "\n"
            )
        function = "_process_bins_slotscan" if dense else "_process_bins"
        if arm in ("found_scan", "both_scan"):
            source = atomic_to_scan(
                source, function, "found_topk_values_ptrs", "take_lt"
            )
        if arm in ("final_scan", "both_scan"):
            source = atomic_to_scan(source, function, "final_cnt_ptrs", "take_eq_final")
    else:
        raise ValueError(arm)
    compile(source, f"<hygon-gap-{arm}>", "exec")
    return source
