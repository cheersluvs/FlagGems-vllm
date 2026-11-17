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

"""Keep sort math while exposing scratch lifetime and installed implementation."""

import ast
import copy
import gzip
import hashlib
import json
import re
from pathlib import Path

from qsa_raw_cube_block512_910b import source as block_source


def source():
    original = ast.parse(block_source()).body[0]
    result = []
    for mode in ("zero", "uninit", "poison"):
        fn = copy.deepcopy(original)
        fn.name = "raw_sort_" + mode
        if mode != "zero":
            vec = fn.body[1].body
            vec[:0] = ast.parse(
                """
sort_scratch = tle.dsa.alloc([2048],dtype=tl.float32,mem_addr_space=tle.dsa.ascend.UB)
sort_output = tle.dsa.alloc([1024],dtype=tl.float32,mem_addr_space=tle.dsa.ascend.UB)
"""
            ).body
            outer = next(
                n
                for n in vec
                if isinstance(n, ast.For) and n.target.id == "vector_block"
            )
            update = next(
                n
                for n in outer.body
                if isinstance(n, ast.For) and n.target.id == "update_query"
            )
            pos = next(
                i
                for i, n in enumerate(update.body)
                if isinstance(n, ast.Assign) and n.targets[0].id == "pairs"
            )
            call = update.body[pos].value
            call.args[2] = ast.parse(
                "tle.dsa.to_tensor(sort_scratch)", mode="eval"
            ).body
            next(k for k in call.keywords if k.arg == "out").value = ast.parse(
                "tle.dsa.to_tensor(sort_output)", mode="eval"
            ).body
            if mode == "poison":
                update.body[pos:pos] = ast.parse(
                    """
tle.dsa.to_buffer(tl.full((2048,),float("nan"),tl.float32),space=tle.dsa.ascend.UB,bind_buffer=sort_scratch)
tle.dsa.to_buffer(tl.full((1024,),float("nan"),tl.float32),space=tle.dsa.ascend.UB,bind_buffer=sort_output)
tl.debug_barrier()
"""
                ).body
        result.append(fn)
    return "\n".join(ast.unparse(ast.fix_missing_locations(fn)) for fn in result) + "\n"


def sort_ir_evidence(kernel):
    text = kernel.asm.get("bcmlir", "")
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    lines = [
        line
        for line in text.splitlines()
        if '"sort_1d_pack"' in line and "hivm.hir.custom" in line
    ]
    if len(lines) != 1 or not all(
        x in lines[0]
        for x in (
            "custom_sort_1d_pack_float",
            "512xf32",
            "2048xf32",
            "1024xf32",
        )
    ):
        raise RuntimeError("Missing retained sort512 custom call/shape evidence")
    return dict(
        sort_call=lines[0],
        scope="Pre-device-lowering IR,not a hardware fill/copy count",
    )


def capture_installed(report):
    from probe_qsa_raw_cube_chain_910b import command, tool
    from triton.experimental.tle.language.dsa.ascend.custom_ops import registry

    root = Path(registry.__file__).resolve().parent
    result = dict(root=str(root), sources=[], llvm_status="unavailable")
    for relative in (
        "registry.py",
        "sort_ops/sort_1d_pack.cpp",
        "sort_ops/sort_base.h",
        "sort_ops/sort_common.h",
    ):
        path = root / relative
        row = dict(path=str(path), available=path.is_file())
        if path.is_file():
            data = path.read_bytes()
            dest = "installed-sort-" + path.name + ".txt"
            (report / dest).write_bytes(data)
            row.update(sha256=hashlib.sha256(data).hexdigest(), artifact=dest)
        result["sources"].append(row)
    path = Path(registry.CUSTOM_OPS_BITCODE).resolve()
    data = path.read_bytes()
    result["bitcode"] = dict(
        path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
    )
    with gzip.open(report / "installed-sort-custom-ops.bc.gz", "wb") as stream:
        stream.write(data)
    try:
        ccec = tool("ccec")
        dis = tool("llvm-dis", ccec)
        linker = tool("llvm-link", ccec)
        work = report / "sort-audit-work"
        work.mkdir(exist_ok=True)
        llvm = work / "custom.ll"
        if dis:
            command([dis, str(path), "-o", str(llvm)], report, "sort-audit-dis", 120)
        elif linker:
            command(
                [linker, "-S", str(path), "-o", str(llvm)],
                report,
                "sort-audit-dis",
                120,
            )
        else:
            raise RuntimeError("No LLVM text tool")
        text = llvm.read_text()
        functions = re.findall(r"^define[^\n]*\{.*?^}", text, re.M | re.S)
        selected = [
            f
            for f in functions
            if any(
                n in f.splitlines()[0]
                for n in (
                    "sort_1d_pack",
                    "sort_base_impl",
                    "prepare_src_",
                    "block_sort",
                    "merge_sort",
                    "lower_vms_",
                )
            )
        ]
        if not selected:
            raise RuntimeError("No sort definitions in LLVM text")
        excerpt = "\n\n".join(selected)
        with gzip.open(report / "installed-sort-llvm.ll.bin.gz", "wb") as stream:
            stream.write(excerpt.encode())
        # Keep a readable bounded copy; compressed full excerpt remains available.
        (report / "installed-sort-llvm-excerpt.txt").write_text(excerpt[:300000])
        result.update(
            llvm_status="captured",
            definitions=len(selected),
            excerpt_bytes=len(excerpt),
        )
    except Exception as exc:
        result["llvm_error"] = repr(exc)
    (report / "installed-sort-audit.json").write_text(json.dumps(result, indent=2))
    print("INSTALLED_SORT_AUDIT", result, flush=True)
    return result
