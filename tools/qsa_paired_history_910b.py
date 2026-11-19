# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Probe a three-way TopK update for pairs of 512-key blocks."""

import ast
import copy
import json
from collections import Counter
from functools import partial

from probe_qsa_raw_cube_chain_910b import record_launch_mode
from qsa_history_policy_910b import dispatcher, use_prefix
from qsa_raw_history_merge_910b import ir_evidence as history_ir
from qsa_short_history_910b import measure_arms

CASES = ("int8_budget_exact", "int8_long100k")

MERGE = """
@triton.jit
def _history_three_way(history, first, second):
    packed = tl.zeros((4096,), tl.float32)
    packed = tle.dsa.insert_slice(packed, history, (0,), (1024,), (1,))
    packed = tle.dsa.insert_slice(packed, first, (1024,), (1024,), (1,))
    packed = tle.dsa.insert_slice(packed, second, (2048,), (1024,), (1,))
    # Exhaustion can occur after more than 1024 pairs have been merged.
    merged, consumed = tle.dsa.ascend.raw(
        "merge_exhaust_sort4", packed, 3, 0, 512, 1024, 1536, 512, 512, 512, 0,
        out=[tl.zeros((4096,), tl.float32), tl.zeros((4,), tl.int32)],
    )
    return al.extract_slice(merged, (0,), (1024,), (1,)), consumed

@triton.jit
def raw_history_check_three_way(Input, Output, Consumed):
    row = tl.program_id(0)
    word = tl.arange(0, 1024)
    base = Input + row * 3072 + word
    history = tl.load(base)
    first = tl.load(base + 1024)
    second = tl.load(base + 2048)
    prefix, consumed = _history_three_way(history, first, second)
    tl.store(Output + row * 1024 + word, prefix)
    tl.store(Consumed + row * 4 + tl.arange(0, 4), consumed)
"""


def direct_history_pack_source(*, join_pack=False):
    helper = ast.parse(MERGE).body[0]
    helper.name = (
        "_history_three_way_join_pack" if join_pack else "_history_three_way_direct_pack"
    )
    assert len(helper.body) == 6
    assert all(
        isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "packed"
        for node in helper.body[:4]
    )
    if join_pack:
        helper.body[:4] = ast.parse(
            """
left = tl.reshape(tl.trans(tl.join(history, first)), (2048,))
right = tl.reshape(tl.trans(tl.join(second, tl.zeros((1024,), tl.float32))), (2048,))
packed = tl.reshape(tl.trans(tl.join(left, right)), (4096,))
"""
        ).body
        return "\n" + ast.unparse(ast.fix_missing_locations(helper)) + "\n"
    helper.body[:4] = ast.parse(
        """
workspace = tle.dsa.alloc([4096], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
history_dst = tle.dsa.subview(workspace, [0], [1024], [1])
first_dst = tle.dsa.subview(workspace, [1024], [1024], [1])
second_dst = tle.dsa.subview(workspace, [2048], [1024], [1])
padding_dst = tle.dsa.subview(workspace, [3072], [1024], [1])
tle.dsa.to_buffer(history, space=tle.dsa.ascend.UB, bind_buffer=history_dst)
tle.dsa.to_buffer(first, space=tle.dsa.ascend.UB, bind_buffer=first_dst)
tle.dsa.to_buffer(second, space=tle.dsa.ascend.UB, bind_buffer=second_dst)
tle.dsa.to_buffer(tl.zeros((1024,), tl.float32), space=tle.dsa.ascend.UB, bind_buffer=padding_dst)
packed = tle.dsa.to_tensor(workspace)
"""
    ).body
    return "\n" + ast.unparse(ast.fix_missing_locations(helper)) + "\n"


def source(history_source):
    kernel = next(
        node
        for node in ast.parse(history_source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "raw_history_prepared_prefix"
    )
    kernel.name = "raw_history_prepared_paired"
    vector = kernel.body[1].body
    vector.insert(
        1,
        ast.parse(
            "pending = tle.dsa.alloc([8 * 1024], dtype=tl.float32, "
            "mem_addr_space=tle.dsa.ascend.UB)"
        ).body[0],
    )
    tile = next(
        node
        for node in vector
        if isinstance(node, ast.For) and node.target.id == "vector_block"
    )
    update = next(
        node
        for node in tile.body
        if isinstance(node, ast.For) and node.target.id == "update_query"
    )
    update_index = tile.body.index(update)
    tile.body[update_index:update_index] = ast.parse(
        """
complete_pair = vector_block % 2 == 1
save_pending = (vector_block % 2 == 0) & (vector_block + 1 < BLOCKS)
"""
    ).body
    start = next(
        i
        for i, node in enumerate(update.body)
        if isinstance(node, ast.Assign) and node.targets[0].id == "update_state_slot"
    )
    # Keep the trailing barrier shared by both paths. The first block can
    # return its ring slot as soon as its sorted pairs are saved.
    update.body[start:-1] = ast.parse(
        """
pending_slot = tle.dsa.subview(pending, [update_query * 1024], [1024], [1])
if save_pending:
    tle.dsa.to_buffer(pairs, space=tle.dsa.ascend.UB, bind_buffer=pending_slot)
else:
    state_slot = tle.dsa.subview(state_storage, [update_query * 1024], [1024], [1])
    previous = tle.dsa.to_tensor(state_slot)
    if complete_pair:
        first = tle.dsa.to_tensor(pending_slot)
        next_state, consumed = _history_three_way(previous, first, pairs)
    else:
        next_state = _history_prefix(previous, pairs, 512, 512)
    tle.dsa.to_buffer(next_state, space=tle.dsa.ascend.UB, bind_buffer=state_slot)
"""
    ).body
    return MERGE + "\n" + ast.unparse(ast.fix_missing_locations(kernel)) + "\n"


def sort_barrier_source(*, sort32_boundary=False):
    from qsa_raw_history_merge_910b import harness_source

    history = harness_source()
    paired = source(history)
    control = no_merge_barrier_source(paired)
    candidate = no_sort32_barrier_source(control) if sort32_boundary else ""
    return history + paired + control + candidate


def no_sort32_barrier_source(control_source):
    return _remove_sort_barrier(
        control_source,
        kernel_name="raw_history_prepared_paired_no_merge_barrier",
        candidate_name="raw_history_prepared_paired_no_sort_barriers",
        producer="sort32",
        output="local32",
    )


def no_merge_barrier_source(paired_source):
    return _remove_sort_barrier(
        paired_source,
        kernel_name="raw_history_prepared_paired",
        candidate_name="raw_history_prepared_paired_no_merge_barrier",
        producer="mrgsort",
        output="local128",
    )


def _remove_sort_barrier(source_text, *, kernel_name, candidate_name, producer, output):
    control = next(
        node for node in ast.parse(source_text).body
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name
    )
    candidate = copy.deepcopy(control)
    candidate.name = candidate_name
    update = next(
        node for node in ast.walk(candidate)
        if isinstance(node, ast.For) and node.target.id == "update_query"
    )
    i = next(
        i for i, node in enumerate(update.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == output
    )
    first, barrier, second = update.body[i : i + 3]
    assert ast.literal_eval(first.value.args[0]) == producer
    assert ast.unparse(barrier) == "tl.debug_barrier()"
    assert ast.unparse(second.value.args[0]) == "'mrgsort'"
    assert ast.unparse(second.value.args[1]) == output
    del update.body[i + 1]
    return "\n" + ast.unparse(ast.fix_missing_locations(candidate)) + "\n"


def sort_barrier_ir(kernel, label, *, sort32_boundary=False):
    evidence = ir_evidence(kernel)
    if record_launch_mode(kernel, require_cube=False)["mix_mode"] == "aiv":
        return evidence
    text = kernel.asm["bcmlir"]
    if isinstance(text, bytes):
        text = text.decode()
    lines = text.splitlines()
    sort, first_merge, second_merge = [
        i for i, line in enumerate(lines)
        if "hivm.hir.custom" in line and ('"sort32"' in line or '"mrgsort"' in line)
    ]
    sort_interval = lines[sort : first_merge + 1]
    merge_interval = lines[first_merge : second_merge + 1]
    sort_barriers = sum("gpu.barrier" in line for line in sort_interval)
    merge_barriers = sum("gpu.barrier" in line for line in merge_interval)
    expected = (int(label == "full"), 0) if sort32_boundary else (1, int(label == "full"))
    if (sort_barriers, merge_barriers) != expected:
        raise RuntimeError(
            f"Unexpected explicit sort barriers: {(sort_barriers, merge_barriers)}"
        )
    if not all(
        "#hivm.pipe<PIPE_V>" in lines[i] for i in (sort, first_merge, second_merge)
    ):
        raise RuntimeError("Expected sort32 and both mrgsort calls on PIPE_V")
    for interval in (sort_interval, merge_interval):
        result = interval[0].split(" = hivm.hir.custom", 1)[0].strip()
        if f" ins({result}," not in interval[-1]:
            raise RuntimeError("Local sort SSA dependency changed")
    return dict(
        **evidence,
        sort32_interval=[line.strip() for line in sort_interval],
        merge_interval=[line.strip() for line in merge_interval],
        explicit_sort_barriers=dict(
            sort32_to_merge=sort_barriers, between_merges=merge_barriers
        ),
    )


def vector_profile_source():
    from qsa_raw_history_merge_910b import harness_source

    history = harness_source()
    tree = ast.parse(source(history))
    helper = tree.body[0]
    full = tree.body[-1]
    assert helper.name == "_history_three_way"
    assert full.name == "raw_history_prepared_paired"
    first_config = next(
        i for i, arg in enumerate(full.args.args) if arg.annotation is not None
    )
    full.args.args.insert(first_config, ast.arg(arg="Audit"))
    kernels = []
    for label in ("full", "sort_only", "score_only"):
        kernel = copy.deepcopy(full)
        kernel.name = "raw_paired_vector_" + label
        vector = kernel.body[1].body
        if label != "full":
            vector[:] = [
                node for node in vector
                if not (
                    isinstance(node, ast.Assign)
                    and ast.unparse(node.targets[0]) in (
                        "state_storage", "pending", "word", "initial"
                    )
                    or isinstance(node, ast.For)
                    and node.target.id in ("init_query", "final_query")
                )
            ]
        block = next(
            node for node in vector
            if isinstance(node, ast.For) and node.target.id == "vector_block"
        )
        query = next(
            node for node in block.body
            if isinstance(node, ast.For) and node.target.id == "update_query"
        )
        if label != "full":
            first_removed = "pending_slot" if label == "sort_only" else "local_indices"
            stop = next(
                i for i, node in enumerate(query.body)
                if isinstance(node, ast.Assign)
                and ast.unparse(node.targets[0]) == first_removed
            )
            query.body[stop:-1] = []
            assert [ast.unparse(node.targets[0]) for node in block.body[2:4]] == [
                "complete_pair", "save_pending"
            ]
            block.body[2:4] = []
        # Every arm publishes one checksum per block to retain consumer work.
        values = (
            "audit_values = score"
            if label == "score_only"
            else "audit_values, _ = tl.split(tl.reshape(pairs, (512, 2)))"
        )
        query.body[-1:-1] = ast.parse(
            f"""
{values}
checksum = tl.sum(tl.where(audit_values > -float("inf"), audit_values, 0) * (lane + 1), 0)
tl.store(Audit + update_row * BLOCKS + vector_block, checksum, mask=update_row < ROWS)
"""
        ).body
        kernels.append(kernel)
    return history + "\n" + "\n".join(
        ast.unparse(ast.fix_missing_locations(fn)) for fn in [helper, *kernels]
    ) + "\n"


def vector_profile_ir(kernel, label):
    text = kernel.asm.get("bcmlir", "")
    if isinstance(text, bytes):
        text = text.decode()
    lines = [line for line in text.splitlines() if "hivm.hir.custom" in line]
    calls = {
        name: sum(f'"{name}"' in line for line in lines)
        for name in ("sort32", "mrgsort", "merge_exhaust_sort4", "unpack_sort")
    }
    expected = (0, 0, 0, 0) if label == "score_only" else (1, 2, 0, 0)
    if label == "full":
        expected = (1, 2, calls["merge_exhaust_sort4"], 1)
        if not 1 <= calls["merge_exhaust_sort4"] <= 2:
            raise RuntimeError(f"Missing paired history update: {calls}")
    if tuple(calls.values()) != expected:
        raise RuntimeError(f"Unexpected {label} Vector primitives: {calls}")
    return dict(
        calls=calls,
        sync=[line.strip() for line in text.splitlines() if "sync_block" in line],
    )


def handoff_source(paired_source):
    kernel = next(
        node
        for node in ast.parse(paired_source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "raw_history_prepared_paired"
    )
    kernel.name = "raw_history_prepared_paired_handoff1024"
    for scope, side, payload_name, payload in (
        (
            kernel.body[0],
            "cube",
            "dest",
            """
rbase + (((tl.program_id(0) * SLOTS + cube_slot) * 2 + cube_block % 2) * 8).to(tl.uint64) * 4096
""",
        ),
        (
            kernel.body[1],
            "vector",
            "received",
            """
tl.load(Ring + (((tl.program_id(0) * SLOTS + vector_slot) * 2 + vector_block % 2) * 8
        + tile[:, None]) * 1024 + update_local_row * 64 + col[None, :])
""",
        ),
    ):
        block, pair, slot = (f"{side}_{name}" for name in ("block", "pair", "slot"))
        loop = next(
            node
            for node in scope.body
            if isinstance(node, ast.For) and node.target.id == block
        )
        _, wait, *work, release = loop.body
        assert ast.unparse(wait.value.func) == "al.sync_block_wait"
        assert ast.unparse(release.value.func) == "al.sync_block_set"
        ptr = next(
            node
            for statement in work
            for node in ast.walk(statement)
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == payload_name
        )
        ptr.value = ast.parse(payload.strip(), mode="eval").body
        outer = ast.parse(
            f"""
for {pair} in range(tl.cdiv(BLOCKS, 2)):
    {slot} = {pair} % SLOTS
    for {block} in range({pair} * 2, tl.minimum({pair} * 2 + 2, BLOCKS)):
        pass
"""
        ).body[0]
        inner = outer.body[-1]
        inner.body = work
        outer.body = [outer.body[0], wait, inner, release]
        scope.body[scope.body.index(loop)] = outer
    return "\n" + ast.unparse(ast.fix_missing_locations(kernel)) + "\n"


def early_release_source(paired_source):
    kernel = next(
        node
        for node in ast.parse(paired_source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "raw_history_prepared_paired"
    )
    kernel.name = "raw_history_prepared_paired_early_release"
    vector = kernel.body[1].body
    tile = next(
        node
        for node in vector
        if isinstance(node, ast.For) and node.target.id == "vector_block"
    )
    update = next(
        node
        for node in tile.body
        if isinstance(node, ast.For) and node.target.id == "update_query"
    )
    received = next(
        node
        for node in update.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "received"
    )
    vector.insert(
        vector.index(tile),
        ast.parse(
            "staged = tle.dsa.alloc([4096], dtype=tl.float32, "
            "mem_addr_space=tle.dsa.ascend.UB)"
        ).body[0],
    )
    assert ast.unparse(tile.body[-1].value.func) == "al.sync_block_set"
    release = tile.body.pop()
    assert ast.unparse(tile.body[1].value.func) == "al.sync_block_wait"
    tile.body[2:2] = ast.parse(
        """
offset = tl.arange(0, 4096)
staged_row = al.sub_vec_id() * 8 + offset // 512
staged_tile = offset % 512 // 64
staged_col = offset % 64
ring_tile = (tl.program_id(0) * SLOTS + vector_slot) * 8 + staged_tile
payload = tl.load(Ring + ring_tile * 1024 + staged_row * 64 + staged_col)
tle.dsa.to_buffer(payload, space=tle.dsa.ascend.UB, bind_buffer=staged)
tl.debug_barrier()
"""
    ).body + [release]
    received.value = ast.parse(
        "tle.dsa.to_tensor(tle.dsa.subview(staged, [update_query * 512], [512], [1]))",
        mode="eval",
    ).body
    # Acknowledge from the pipe that reads the slot. Initial credits and Cube
    # waits use the same pipe, including the drain at kernel exit.
    for call in ast.walk(kernel):
        if (
            isinstance(call, ast.Call)
            and ast.unparse(call.func) in ("al.sync_block_set", "al.sync_block_wait")
            and ast.unparse(call.args[0]) == "'vector'"
        ):
            call.args[3] = ast.parse("pipe.PIPE_MTE2", mode="eval").body
    return "\n" + ast.unparse(ast.fix_missing_locations(kernel)) + "\n"


def ir_evidence(kernel, *, early_release=False):
    if record_launch_mode(kernel, require_cube=False)["mix_mode"] == "aiv":
        return history_ir(kernel, complete=False, expect_no_gather=True)
    text = kernel.asm.get("bcmlir", "")
    if isinstance(text, bytes):
        text = text.decode()
    lines = [line for line in text.splitlines() if "hivm.hir.custom" in line]
    calls = {
        name: sum(f'"{name}"' in line for line in lines)
        for name in ("sort32", "mrgsort", "merge_exhaust_sort4", "unpack_sort")
    }
    if calls != dict(sort32=1, mrgsort=2, merge_exhaust_sort4=2, unpack_sort=1):
        raise RuntimeError("Unexpected paired history structure: " + repr(calls))
    if "triton_gather" in text:
        raise RuntimeError("Unexpected gather in paired history")
    sync_ir = [
        line.strip() for line in text.splitlines()
        if "sync_block" in line or "barrier" in line
    ]
    if early_release and not any(
        "sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>]" in line
        for line in sync_ir
    ):
        raise RuntimeError("Expected Vector MTE2 slot release in paired-history IR")
    return dict(
        calls=calls,
        lines=lines,
        extra_pending_ub_bytes_per_vector=8 * 1024 * 4,
        extra_payload_ub_bytes_per_vector=4096 * 4 if early_release else 0,
        sync_ir=sync_ir,
        ub_allocations=[
            line.strip()
            for line in text.splitlines()
            if "alloc" in line and "#hivm.address_space<ub>" in line
        ],
    )


def paired_dispatcher(
    module,
    policy,
    handoff1024=False,
    *,
    fallback=None,
    early_release=False,
    no_merge_barrier=False,
    no_sort32_barrier=False,
    reuse_local32=False,
):
    original = fallback if fallback is not None else dispatcher(module, policy)
    kernel = (
        module.raw_history_prepared_paired_handoff1024
        if handoff1024
        else module.raw_history_prepared_paired
    )
    mode = "paired_handoff1024" if handoff1024 else "paired_prefix"
    assert sum(
        (handoff1024, early_release, no_merge_barrier, no_sort32_barrier, reuse_local32)
    ) <= 1
    evidence = ir_evidence
    if early_release:
        kernel = module.raw_history_prepared_paired_early_release
        mode = "paired_early_release"
        evidence = partial(ir_evidence, early_release=True)
    if no_merge_barrier:
        kernel = module.raw_history_prepared_paired_no_merge_barrier
        mode = "paired_no_merge_barrier"
        evidence = partial(sort_barrier_ir, label="no_merge_barrier")
    if no_sort32_barrier:
        kernel = module.raw_history_prepared_paired_no_sort_barriers
        mode = "paired_no_sort_barriers"
        evidence = partial(sort_barrier_ir, label="no_sort32_barrier", sort32_boundary=True)
    if reuse_local32:
        from qsa_final_mrgsort_ub_910b import prepared_reuse_ir

        kernel = module.raw_history_prepared_paired_reuse_local32
        mode = "paired_reuse_local32"
        evidence = prepared_reuse_ir

    def choose(blocks):
        if blocks >= 4 and use_prefix(blocks, policy):
            return (
                kernel,
                module.raw_history_topk_prefix,
                evidence,
                mode,
            )
        return original(blocks)

    return choose


def measure_combined(
    case_name,
    benchmark,
    native,
    control,
    candidate,
    path,
    *,
    early_release=False,
    no_merge_barrier=False,
    no_sort32_barrier=False,
    reuse_local32=False,
):
    if (
        control.ring_blocks_per_slot != 1
        or candidate.ring_blocks_per_slot != 1
        or control.routes != candidate.routes
        or len(control.dispatch_routes) != len(candidate.routes)
        or len(candidate.dispatch_routes) != len(candidate.routes)
    ):
        raise ValueError(f"Unexpected combined history schedule for {case_name}")
    transitions = Counter()
    paired_blocks = Counter()
    if reuse_local32:
        old_mode = "paired_no_sort_barriers"
        candidate_mode = "paired_reuse_local32"
        label = "FINAL_MRGSORT_REUSE_LONG"
        control_name = "paired history without both local sort barriers, zero-filled mrgsort output"
    elif no_sort32_barrier:
        old_mode = "paired_no_merge_barrier"
        candidate_mode = "paired_no_sort_barriers"
        label = "SORT32_BARRIER_TEN"
        control_name = "two-block history merge with the barrier after sort32"
    elif no_merge_barrier:
        old_mode = "paired_prefix"
        candidate_mode = "paired_no_merge_barrier"
        label = "SORT_BARRIER_TEN"
        control_name = "two-block history merge with the barrier between mrgsort calls"
    elif early_release:
        old_mode = "paired_prefix"
        candidate_mode = "paired_early_release"
        label = "EARLY_RELEASE"
        control_name = "512-key paired history with slot release after TopK"
    else:
        old_mode = "prefix"
        candidate_mode = "paired_prefix"
        label = "COMBINED_HISTORY"
        control_name = "short-history dispatch with per-block prefix updates"
    for old, new in zip(control.dispatch_routes, candidate.dispatch_routes):
        mode = candidate_mode if old["mode"] == old_mode else old["mode"]
        if new != dict(old, mode=mode):
            raise ValueError(f"Unexpected combined history route: {old} -> {new}")
        transitions[f'{old["mode"]}->{mode}'] += 1
        if mode == candidate_mode:
            paired_blocks[new["blocks"]] += 1
    routing = dict(
        route_transitions=dict(transitions),
        paired_pieces_by_blocks=dict(paired_blocks),
        paired_odd_tail_pieces=sum(n for b, n in paired_blocks.items() if b % 2),
        handoff_keys=512,
        slots=2,
    )
    if (early_release or reuse_local32) and not paired_blocks:
        raise ValueError(f"No paired pieces exercised for {case_name}")
    if early_release:
        routing.update(extra_payload_ub_bytes_per_vector=4096 * 4)
    return measure_arms(
        case_name,
        benchmark,
        native,
        control,
        candidate,
        path,
        routing,
        label,
        control_name=control_name,
    )


def check_three_way(module, torch, device, report, save_asm, case_name):
    base = torch.arange(512, 0, -1, dtype=torch.float32)
    low = base - 10000
    # Row 0 forces a long output before exhaustion. The other rows exercise
    # one winning stream, interleaved values, ties, and masked tail keys.
    high = base + 10000
    high[-1] = -20000
    masked = low.clone()
    masked[73:] = -float("inf")
    zeros = torch.zeros_like(base)
    rows = torch.stack(
        (
            torch.stack((base, high, high)),
            torch.stack((low, base, base + 1000)),
            torch.stack((base * 3, base * 3 + 1, base * 3 + 2)),
            torch.stack((zeros, zeros, zeros)),
            torch.stack((low, masked, torch.full_like(base, -float("inf")))),
        )
    ).reshape(5, 1536)
    packed = torch.empty((5, 3072), dtype=torch.float32)
    packed[:, 0::2] = rows
    packed[:, 1::2] = torch.arange(1536, dtype=torch.int32).view(torch.float32)
    inputs = packed.to(device)
    out_storage = torch.full(
        (7, 1024), float("nan"), dtype=torch.float32, device=device
    )
    out = out_storage[1:-1]
    consumed = torch.full((5, 4), -1, dtype=torch.int32, device=device)
    kernel = module.raw_history_check_three_way
    path = report / (case_name + "-three-way-merge.json")
    result = dict(status="compiling")
    path.write_text(json.dumps(result, indent=2))
    compiled = kernel.warmup(inputs, out, consumed, grid=(5,), num_warps=4)
    mode = record_launch_mode(compiled, require_cube=False)
    if mode["mix_mode"] != "aiv" or mode["bs_task_type"] // 10 != 1:
        raise RuntimeError("Three-way merge validation requires AIV")
    result.update(
        status="running",
        launch=mode,
        ir=history_ir(
            compiled, complete=False, local_sort=False, expect_no_gather=True
        ),
        asm=save_asm(compiled, report, case_name + "-three-way-merge"),
    )
    path.write_text(json.dumps(result, indent=2))
    kernel[(5,)](inputs, out, consumed, num_warps=4)
    torch.npu.synchronize()

    pairs = out.cpu()
    values = pairs[:, 0::2]
    indices = pairs[:, 1::2].contiguous().view(torch.int32).long()
    torch.testing.assert_close(values, rows.topk(512, dim=1).values, rtol=0, atol=0)
    assert bool(((indices >= 0) & (indices < 1536)).all())
    ordered = indices.sort(dim=1).values
    assert bool((ordered[:, 1:] != ordered[:, :-1]).all())
    torch.testing.assert_close(rows.gather(1, indices), values, rtol=0, atol=0)
    assert bool(torch.isnan(out_storage[[0, -1]].cpu()).all())
    counts = consumed.cpu()[:, :3]
    assert bool(((counts >= 0) & (counts <= 512)).all())
    assert bool((counts.max(dim=1).values == 512).all())
    assert int(counts[0].sum()) > 1024
    result.update(status="passed", consumed=counts.tolist(), checked_rows=5)
    path.write_text(json.dumps(result, indent=2))
    print("PAIRED_HISTORY_MERGE_CHECK", json.dumps(result), flush=True)
    return result


def measure(case_name, benchmark, native, control, candidate, path):
    handoff1024 = candidate.ring_blocks_per_slot == 2
    control_mode = "paired_prefix" if handoff1024 else "prefix"
    candidate_mode = "paired_handoff1024" if handoff1024 else "paired_prefix"
    blocks = Counter(route["blocks"] for route in candidate.routes)
    if (
        not blocks
        or control.ring_blocks_per_slot != 1
        or candidate.ring_blocks_per_slot not in (1, 2)
        or not any(b % 2 == 1 for b in blocks)
        or not any(b % 2 == 0 for b in blocks)
        or control.routes != candidate.routes
        or len(control.dispatch_routes) != len(candidate.routes)
        or len(candidate.dispatch_routes) != len(candidate.routes)
    ):
        raise ValueError(f"Unexpected paired-history schedule: {blocks}")
    for old, new in zip(control.dispatch_routes, candidate.dispatch_routes):
        if old["mode"] != control_mode or new != dict(old, mode=candidate_mode):
            raise ValueError(f"Unexpected paired-history route: {old} -> {new}")
    routing = dict(
        pieces_by_blocks=dict(blocks),
        odd_tail_pieces=sum(n for b, n in blocks.items() if b % 2),
        control_updates_per_query_by_blocks={
            b: (b + 1) // 2 if handoff1024 else b for b in sorted(blocks)
        },
        paired_updates_per_query_by_blocks={b: (b + 1) // 2 for b in sorted(blocks)},
        handoff_keys=1024 if handoff1024 else 512,
        slots=2,
    )
    if handoff1024:
        routing.update(
            control_handoff_keys=512,
            control_handoffs_by_blocks={b: b for b in sorted(blocks)},
            candidate_handoffs_by_blocks={b: (b + 1) // 2 for b in sorted(blocks)},
            control_ring_bytes_per_query16_group=2 * 8192 * 4,
            candidate_ring_bytes_per_query16_group=2 * 2 * 8192 * 4,
        )
    return measure_arms(
        case_name,
        benchmark,
        native,
        control,
        candidate,
        path,
        routing,
        "PAIRED_HANDOFF" if handoff1024 else "PAIRED_HISTORY",
        control_name=(
            "paired merge with 512-key handoff"
            if handoff1024
            else "original history policy"
        ),
    )
