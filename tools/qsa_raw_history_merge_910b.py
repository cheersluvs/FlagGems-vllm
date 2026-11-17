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

"""Use complete two-way merging with real evolving online history."""

import ast
import copy

from qsa_raw_tle_sort512_910b import source as fixed_source

HELPER = """
@triton.jit
def _history_complete(state,tile,TOP:tl.constexpr,CHUNK:tl.constexpr):
    tl.static_assert(TOP==512 and CHUNK==512)
    packed = tl.zeros((8*TOP,),tl.float32)
    packed = tle.dsa.insert_slice(packed,state,(0,),(2*TOP,),(1,))
    packed = tle.dsa.insert_slice(packed,tile,(2*TOP,),(2*CHUNK,),(1,))
    merged = tle.dsa.ascend.raw("mrgsort",packed,0,TOP,0,0,TOP,CHUNK,0,0,False,3,1,out=tl.zeros((4*TOP,),tl.float32))
    return al.extract_slice(merged,(0,),(2*TOP,),(1,))
"""

PREFIX = """
@triton.jit
def _history_prefix(state,tile,TOP:tl.constexpr,CHUNK:tl.constexpr):
    tl.static_assert(TOP==512 and CHUNK==512)
    packed = tl.zeros((8*TOP,),tl.float32)
    packed = tle.dsa.insert_slice(packed,state,(0,),(2*TOP,),(1,))
    packed = tle.dsa.insert_slice(packed,tile,(2*TOP,),(2*CHUNK,),(1,))
    merged,consumed = tle.dsa.ascend.raw(
        "merge_exhaust_sort4",packed,2,0,TOP,2*TOP,3*TOP,TOP,CHUNK,0,0,
        out=[tl.zeros((4*TOP,),tl.float32),tl.zeros((4,),tl.int32)])
    return al.extract_slice(merged,(0,),(2*TOP,),(1,))
"""


def source():
    old = ast.parse(fixed_source()).body[1]
    old.name = "raw_history_snapshot"
    new = copy.deepcopy(old)
    new.name = "raw_history_complete"
    for node in ast.walk(new):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_update_online_topk_ub_snapshot"
        ):
            node.func.id = "_history_complete"
    prefix = copy.deepcopy(new)
    prefix.name = "raw_history_prefix"
    for node in ast.walk(prefix):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_history_complete"
        ):
            node.func.id = "_history_prefix"
    checks = []
    for name, helper in (
        ("snapshot", "_update_online_topk_ub_snapshot"),
        ("complete", "_history_complete"),
        ("prefix", "_history_prefix"),
    ):
        checks.append(
            f"""
@triton.jit
def history_check_{name}(State,Tile,Out):
    row=tl.program_id(0)
    word=tl.arange(0,1024)
    state=tl.load(State+row*1024+word)
    tile=tl.load(Tile+row*1024+word)
    merged={helper}(state,tile,512,512)
    tl.store(Out+row*1024+word,merged)
"""
        )
    return (
        HELPER
        + PREFIX
        + "\n"
        + "\n".join(
            ast.unparse(ast.fix_missing_locations(n)) for n in (old, new, prefix)
        )
        + "\n"
        + "\n".join(checks)
    )


def ir_evidence(kernel, complete, local_sort=True, expect_no_gather=False):
    text = kernel.asm.get("bcmlir", "")
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    lines = [line for line in text.splitlines() if "hivm.hir.custom" in line]
    counts = {
        name: sum('"' + name + '"' in line for line in lines)
        for name in ("sort32", "mrgsort", "merge_exhaust_sort4", "unpack_sort")
    }
    expected = dict(
        sort32=int(local_sort),
        mrgsort=(2 if local_sort else 0) + int(complete),
        merge_exhaust_sort4=int(not complete),
        unpack_sort=int(local_sort),
    )
    if counts != expected:
        raise RuntimeError("History primitive structure mismatch: " + repr(counts))
    if expect_no_gather and "triton_gather" in text:
        raise RuntimeError("History fallback gather unexpectedly retained")
    return dict(
        calls=counts,
        gather_references=text.count("triton_gather"),
        lines=lines,
        scope="Pre-device IR structure,not measured hardware instruction counts",
    )


def transition_checks(module, torch, device, report, save_asm, record_launch_mode):
    import json

    rows = 5
    base = torch.arange(512, 0, -1, dtype=torch.float32)
    initial = base.repeat(rows, 1)
    initial[3].zero_()
    pool = initial.clone()

    def pack(values, offset):
        out = torch.empty((rows, 1024), dtype=torch.float32)
        out[:, 0::2] = values
        out[:, 1::2] = torch.arange(offset, offset + 512, dtype=torch.int32).view(
            torch.float32
        )
        return out

    state0 = pack(initial, 0).to(device)
    states = {name: state0 for name in ("snapshot", "prefix", "complete")}
    buffers = {}
    evidence = {}
    gen = torch.Generator().manual_seed(17)
    for name in states:
        buffers[name] = []
        for _ in range(2):
            storage = torch.empty((rows + 32, 1024), dtype=torch.float32, device=device)
            buffers[name].append((storage, storage[16 : 16 + rows]))
    results = []
    progress = dict(status="running", checks=results, compiled=evidence)

    def save_progress():
        (report / "history-transitions.json").write_text(json.dumps(progress, indent=2))

    save_progress()
    for step in range(4):
        tile = torch.empty((rows, 512), dtype=torch.float32)
        tile[0] = base - 10000 - step * 1000
        tile[1] = base + 10000 + step * 1000
        tile[2] = torch.randint(-1000, 1001, (512,), generator=gen).float()
        tile[3].zero_()
        tile[4] = base * (-1 if step % 2 else 1)
        if step == 1:
            tile[4].fill_(-float("inf"))
        pool = torch.cat((pool, tile), dim=1)
        ordered, order = tile.sort(dim=1, descending=True)
        packed = pack(ordered, 512 * (step + 1))
        packed[:, 1::2] = (
            (order.int() + 512 * (step + 1)).contiguous().view(torch.float32)
        )
        tile_d = packed.to(device)
        expected = pool.topk(512, dim=1).values
        for name in states:
            progress["phase"] = dict(step=step, arm=name)
            save_progress()
            kernel = getattr(module, "history_check_" + name)
            storage, out = buffers[name][step % 2]
            storage.fill_(float("nan"))
            if step == 0:
                compiled = kernel.warmup(
                    states[name], tile_d, out, grid=(rows,), num_warps=4
                )
                launch = record_launch_mode(compiled, require_cube=False)
                if launch["mix_mode"] != "aiv" or launch["bs_task_type"] // 10 != 1:
                    raise RuntimeError(
                        "Transition check expected Vector kernel: " + repr(launch)
                    )
                evidence[name] = dict(
                    launch=launch,
                    ir=ir_evidence(
                        compiled,
                        name == "complete",
                        False,
                        expect_no_gather=name != "snapshot",
                    ),
                    asm=save_asm(compiled, report, "history-transition-" + name),
                )
            kernel[(rows,)](states[name], tile_d, out, num_warps=4)
            torch.npu.synchronize()
            actual = out.cpu()
            values = actual[:, 0::2]
            indices = actual[:, 1::2].contiguous().view(torch.int32).long()
            torch.testing.assert_close(values, expected, rtol=0, atol=0)
            assert bool(((indices >= 0) & (indices < pool.shape[1])).all())
            assert bool(
                (indices.sort(1).values[:, 1:] != indices.sort(1).values[:, :-1]).all()
            )
            torch.testing.assert_close(pool.gather(1, indices), values, rtol=0, atol=0)
            assert bool(torch.isnan(storage[:16].cpu()).all())
            assert bool(torch.isnan(storage[16 + rows :].cpu()).all())
            states[name] = out
            row = dict(step=step, arm=name, passed=True, rows=rows)
            results.append(row)
            save_progress()
            print("HISTORY_TRANSITION", row, flush=True)
    result = dict(
        status="passed",
        checks=results,
        compiled=evidence,
        scenarios=[
            "new_low",
            "new_high",
            "interleaved",
            "ties",
            "negative_and_infinity",
        ],
        scope="Four real state transitions;exact ranked values and valid unique index payloads",
    )
    (report / "history-transitions.json").write_text(json.dumps(result, indent=2))
    return result
