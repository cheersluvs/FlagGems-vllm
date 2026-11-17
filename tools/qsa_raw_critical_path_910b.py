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

"""Matched-event mixed ablations with explicitly different payload transport."""

import ast
import copy
import re
from pathlib import Path

from qsa_raw_cube_block512_910b import source as full_source
from qsa_raw_mixed_abi_910b import SYMBOL, check_alias_llvm

TICK = "_mlir_ciface_qsa_raw_replay_tick"


def source():
    fn = ast.parse(full_source()).body[0]
    first_config = next(
        i for i, arg in enumerate(fn.args.args) if arg.annotation is not None
    )
    fn.args.args[first_config:first_config] = [
        ast.arg(arg="Replay"),
        ast.arg(arg="Audit"),
    ]
    full = copy.deepcopy(fn)
    full.name = "raw_critical_full"
    light = copy.deepcopy(fn)
    light.name = "raw_critical_light"
    vec = light.body[1].body
    vec[:] = [
        n
        for n in vec
        if not (
            isinstance(n, ast.Assign)
            and n.targets[0].id in ("state_storage", "word", "initial")
            or isinstance(n, ast.For)
            and n.target.id in ("init_query", "final_query")
        )
    ]
    outer = next(
        n for n in vec if isinstance(n, ast.For) and n.target.id == "vector_block"
    )
    inner = next(
        n
        for n in outer.body
        if isinstance(n, ast.For) and n.target.id == "update_query"
    )
    end = next(
        i
        for i, n in enumerate(inner.body)
        if isinstance(n, ast.Assign) and n.targets[0].id == "pairs"
    )
    inner.body[end:] = ast.parse(
        """
checksum = tl.sum(tl.where((vector_block*512+lane<visible)&(vector_block*512+lane<WIDTH),score,0),0)
tl.store(Audit+update_row*triton.cdiv(WIDTH,512)+vector_block,checksum,mask=update_row<ROWS)
tl.debug_barrier()
"""
    ).body
    replay = copy.deepcopy(fn)
    replay.name = "raw_critical_replay"
    cube_loop = next(
        n
        for n in replay.body[0].body
        if isinstance(n, ast.For) and n.target.id == "cube_block"
    )
    # Retain the exact first wait and final ready-set; replace only production.
    cube_loop.body[2:-1] = ast.parse(
        """
raw.call(replay_tick_op,[qbase,kbase,wbase,cube_block.to(tl.int64),tl.cast(Audit,tl.uint64)+tl.program_id(0).to(tl.uint64)*4],output_indices=[])
"""
    ).body
    for n in ast.walk(replay.body[1]):
        if (
            isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "received"
        ):
            n.value = ast.parse(
                "tl.load(Replay+((tl.program_id(0)*triton.cdiv(WIDTH,512)"
                "+vector_block)*8+tile[:,None])*1024+update_local_row*64+col[None,:])",
                mode="eval",
            ).body
    return (
        "\n".join(
            ast.unparse(ast.fix_missing_locations(n)) for n in (full, light, replay)
        )
        + "\n"
    )


def bind_tick(report, result, save):
    from probe_qsa_raw_cube_chain_910b import build_bitcode
    from triton.experimental.tle.raw import dialect

    entry = dict(custom_symbols=dict(tick=TICK.removeprefix("_mlir_ciface_")))
    result["replay_tick"] = entry
    path = build_bitcode(
        report,
        entry,
        Path(__file__).with_suffix(".cpp"),
        stem="replay-tick",
        defines=("QSA_QUERY_PACK=16",),
    )
    text = (report / "replay-tick-plain.ll.txt").read_text()
    if not re.search(r'^define[^\n]*@"?' + TICK + r'"?\(', text, re.M):
        raise RuntimeError("Replay ordinary entry missing")
    entry["abi"] = check_alias_llvm(text.replace(TICK, SYMBOL))
    entry["scope"] = "Scalar progress tick; no score computation or payload copying"

    def stub(*args, **kwargs):
        pass

    stub.__name__ = "qsa_replay_tick"
    bound = dialect(
        name="cann",
        file=str(path),
        format="bitcode",
        extern_func_name=TICK.removeprefix("_mlir_ciface_"),
        pipeline={"core": "cube", "pipe": "PIPE_M"},
    )(stub)
    save()
    return bound
