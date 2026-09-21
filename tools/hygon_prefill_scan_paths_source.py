"""Source-only variants for the Hygon full-row and wave64 scan probes.

These transforms fail closed on source drift and never modify the shipped op.
"""

import ast


def function_text(source, name):
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return "\n".join(source.splitlines()[first - 1 : node.end_lineno])


def replace_once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected one block, got {count}: {old[:90]!r}")
    return source.replace(old, new, 1)


def fullrow_variant(source):
    """Allow full, aligned rows with a masked tail into the fast layout path."""
    job = function_text(source, "_top_k_per_row_job")
    new_job = replace_once(
        job,
        """        & (stride1 == 1)
        & ((vocab_size % BLOCK_SIZE) == 0)
""",
        """        & (stride1 == 1)
        & (skip_elems == 0)
""",
    )
    new_job = replace_once(
        new_job,
        """        tl.assume(stride1 == 1)
        vocab_size = tl.multiple_of(vocab_size, BLOCK_SIZE)
""",
        """        tl.assume(stride1 == 1)
        tl.assume(skip_elems == 0)
""",
    )
    source = replace_once(source, job, new_job)

    step = function_text(source, "_process_histogram_step")
    old_rem = """        rem_tiles = (vocab_size - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
"""
    new_rem = """        rem_tiles = (vocab_size - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        rem_elems = vocab_size % BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
"""
    if step.count(old_rem) != 2:
        raise ValueError("Expected histogram and collection aligned loops")
    new_step = step.replace(old_rem, new_rem, 2)
    # The aligned arm occurs once in the histogram pass and once in the
    # collection pass. The remaining branches are byte-for-byte unchanged.
    new_step = replace_once(
        new_step,
        """                s_histogram_ptr,
                STEP=STEP,
            )
    elif stride1 == 1:
""",
        """                s_histogram_ptr,
                STEP=STEP,
            )
        if rem_elems > 0:
            offs = (n_vec_full * VEC + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(logits_ptr + offs, mask=in_range, other=float("-inf"))
            _distribute_to_bins(
                x, in_range, ones, logit_pattern, s_histogram_ptr, STEP=STEP
            )
    elif stride1 == 1:
""",
    )
    new_step = replace_once(
        new_step,
        """                MERGE_BLOCKS=MERGE_BLOCKS,
            )
    elif stride1 == 1:
""",
        """                MERGE_BLOCKS=MERGE_BLOCKS,
            )
        if rem_elems > 0:
            offs = (n_vec_full * VEC + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(logits_ptr + offs, mask=in_range, other=float("-inf"))
            _process_bins(
                x,
                in_range,
                ones,
                offs,
                found_ptrs,
                final_cnt_ptrs,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                use_final,
                row_start,
                indices_ptr,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_out_indices_ptr,
                s_out_logits_ptr,
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
    elif stride1 == 1:
""",
    )
    source = replace_once(source, step, new_step)
    compile(source, "<hygon-prefill-fullrow>", "exec")
    return source


def wave64_variant(source):
    """Replace dense carried full-tile scan with two explicit wave64 scans."""
    step = function_text(source, "_process_bins_slotscan")
    new_step = replace_once(
        step,
        """    offsets = tl.cumsum(flat_take, axis=0) - flat_take
    out_pos_lt = slot_base + tl.reshape(offsets, take_int.shape)
""",
        """    WAVE: tl.constexpr = 64
    N_WAVES: tl.constexpr = take_int.numel // WAVE
    by_wave = tl.reshape(flat_take, (N_WAVES, WAVE))
    wave_counts = tl.sum(by_wave, axis=1)
    wave_base = tl.cumsum(wave_counts, axis=0) - wave_counts
    local = tl.cumsum(by_wave, axis=1) - by_wave
    offsets = tl.reshape(local + wave_base[:, None], (take_int.numel,))
    out_pos_lt = slot_base + tl.reshape(offsets, take_int.shape)
""",
    )
    source = replace_once(source, step, new_step)
    compile(source, "<hygon-prefill-wave64>", "exec")
    return source
