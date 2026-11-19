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


"""Narrow the local sort while preserving the paired 512-entry merge ABI."""

import ast

from probe_qsa_raw_cube_chain_910b import record_launch_mode
from qsa_raw_history_merge_910b import ir_evidence as history_ir
from qsa_short_history_910b import measure_arms

# The inclusive prefix gives each surviving lane a one-based rank. Binary
# lifting finds the last lane with smaller rank for each of 32 output slots.
COMPACT = """
# All prefix counts are in [0, 512], exactly representable by the FP32 gather.
prefix = tl.cumsum(keep.to(tl.int32), 0).to(tl.float32)
slot = tl.arange(0, 32)
before = tl.full((32,), -1, tl.int32)
for bit in tl.static_range(9):
    probe = before + (256 >> bit)
    below = tl.gather(prefix, probe, axis=0) < slot + 1
    before = tl.where(below, probe, before)
lane32 = before + 1
valid32 = slot < count
scores32 = tl.where(valid32, tl.gather(score, lane32, axis=0), -float('inf'))
indices32 = tl.where(valid32, vector_block * 512 + lane32, -1).to(tl.uint32)
short_pairs = tle.dsa.ascend.raw("sort32", scores32, indices32, 1,
                                 out=tl.zeros((64,), tl.float32))
tl.debug_barrier()
pairs = tle.dsa.insert_slice(initial, short_pairs, (0,), (64,), (1,))
"""


def source(text):
    kernel = next(
        node
        for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "raw_history_prepared_paired_reuse_local32"
    )
    kernel.name = "raw_history_prepared_paired_compact32"
    loop = next(
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.For) and node.target.id == "update_query"
    )
    start = next(
        i
        for i, node in enumerate(loop.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "local32"
    )
    fallback = loop.body[start : start + 3]
    assert [ast.literal_eval(n.value.args[0]) for n in fallback] == [
        "sort32",
        "mrgsort",
        "mrgsort",
    ]
    selection = ast.parse(
        """
if vector_block >= 2:
    cutoff_slot = tle.dsa.subview(state_storage, [update_query * 1024], [1024], [1])
    cutoff = al.get_element(tle.dsa.to_tensor(cutoff_slot), (1022,))
    keep = (score >= cutoff) & (score != -float('inf'))
    count = tl.sum(keep.to(tl.int32), 0)
else:
    keep = tl.full((512,), False, tl.int1)
    count = 513
if count == 0:
    pairs = initial
elif count <= 32:
    pass
else:
    pass
"""
    ).body
    selection[-1].orelse[0].body = ast.parse(COMPACT).body
    selection[-1].orelse[0].orelse = fallback
    loop.body[start : start + 3] = selection
    return "\n" + ast.unparse(ast.fix_missing_locations(kernel)) + "\n"


def evidence(kernel):
    if record_launch_mode(kernel, require_cube=False)["mix_mode"] == "aiv":
        return history_ir(kernel, complete=False, expect_no_gather=True)
    text = kernel.asm["bcmlir"]
    if isinstance(text, bytes):
        text = text.decode()
    sorts = [line.strip() for line in text.splitlines() if '"sort32" ins(' in line]
    for width in (32, 512):
        if not any(f"tensor<{width}xf32>" in line for line in sorts):
            raise ValueError(
                f"Missing {width}-element sort32 arm in compiled mixed kernel"
            )
    return dict(sort32_arms=sorts)


def dispatcher(module, control):
    def choose(blocks):
        kernel, online, inspect, mode = control(blocks)
        if mode == "paired_reuse_local32":
            return (
                module.raw_history_prepared_paired_compact32,
                online,
                evidence,
                "paired_compact32",
            )
        return kernel, online, inspect, mode

    return choose


def measure(case, benchmark, native, control, candidate, path):
    if control.routes != candidate.routes or not candidate.routes:
        raise ValueError("Candidate and control must cover identical pieces")
    for old, new in zip(
        control.dispatch_routes, candidate.dispatch_routes, strict=True
    ):
        if old["mode"] != "paired_reuse_local32" or new != dict(
            old, mode="paired_compact32"
        ):
            raise ValueError(f"Unexpected route: {old} -> {new}")
    return measure_arms(
        case,
        benchmark,
        native,
        control,
        candidate,
        path,
        dict(pieces=len(candidate.routes), rows=candidate.rows),
        "COMPACT32",
        control_name="paired history, final mrgsort out=local32",
    )
