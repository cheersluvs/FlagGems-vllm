"""Fail-closed final-selector patching; stdlib only."""

from hygon_prefill_audit_source import function_text, replace_once


def final_variant(source, mode):
    if mode not in ("network", "prefix"):
        raise ValueError(mode)
    job = function_text(source, "_top_k_per_row_job")
    start = job.index("            sort_chunks = tl.cdiv(final_cnt, BLOCK_SIZE)")
    end = job.index("            tl.debug_barrier()", start)
    original = job[start:end]
    # Use the exact original scalar selector if the network capacity is exceeded.
    branch = (
        "            remain = tl.minimum(tl.maximum(TOPK - base_idx, 0), final_cnt)\n"
    )
    caps = (64, 128, 256) if mode == "network" else (64, 128, 256, 2048)
    for i, cap in enumerate(caps):
        branch += f"            {'if' if i == 0 else 'elif'} final_cnt <= {cap}:\n"
        branch += (
            f"                _probe_final_{mode}(\n"
            "                    s_final_logits_ptr, s_histogram_ptr,\n"
            "                    s_out_indices_ptr, final_cnt, base_idx, remain,\n"
            f"                    CAP={cap},\n"
            "                )\n"
        )
    branch += "            else:\n"
    branch += "".join("    " + line + "\n" for line in original.splitlines())
    changed = job[:start] + branch + job[end:]
    result = replace_once(source, job, changed)
    result += (
        "\nfrom hygon_prefill_algorithms_kernel import "
        f"final_{mode} as _probe_final_{mode}\n"
    )
    compile(result, f"<prefill-final-{mode}>", "exec")
    return result
