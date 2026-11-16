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

"""Bind an operator-owned Cube-suffixed forwarding entry for mixed callers."""

import re
from pathlib import Path

SYMBOL = "_mlir_ciface_qsa_raw_chain_address_query16"


def check_alias_llvm(text):
    match = re.search(
        r'^define[^\n]*\bvoid\s+@"?' + re.escape(SYMBOL) + r'\.cube"?\(([^\n]*)\)',
        text,
        re.M,
    )
    operands = match.group(1).split(",") if match else []
    if len(operands) != 5 or not all(
        re.match(r"\s*i64(?:\s|$)", operand) for operand in operands
    ):
        raise RuntimeError("Missing Cube alias definition with five i64 operands")
    if not re.search(r'^define[^\n]*@"?' + re.escape(SYMBOL) + r'"?\(', text, re.M):
        raise RuntimeError("Original native entry missing from the same module")
    return dict(
        cube_symbol=SYMBOL + ".cube",
        original_symbol=SYMBOL,
        argument_types=["i64"] * 5,
        shared_translation_unit=True,
        scope="LLVM export/signature gate; mixed linking and execution remain separate gates",
    )


def bind_mixed_chain(report, result, save):
    from probe_qsa_raw_cube_chain_910b import build_bitcode
    from triton.experimental.tle.raw import dialect

    entry = dict(custom_symbols=dict(chain="qsa_raw_chain_address_query16"))
    result["mixed_abi_bridge"] = entry
    save()
    path = build_bitcode(
        report,
        entry,
        Path(__file__).with_name("qsa_raw_cube_mixed_alias_910b.cpp"),
        stem="query16-mixed-abi",
        defines=("QSA_QUERY_PACK=16",),
    )
    llvm = (report / "query16-mixed-abi-plain.ll.txt").read_text()
    entry["alias_signature"] = check_alias_llvm(llvm)
    if not (
        entry["bitcode_structure"]["candidate_structure_present"]
        and entry["bitcode_structure"]["exported_symbols"]
    ):
        raise RuntimeError("Native two-Cube structural evidence missing")

    def stub(*args, **kwargs):
        pass

    stub.__name__ = "qsa_raw_chain_mixed_query16"
    # Keep the unsuffixed raw-call registration: mixed lowering requests .cube.
    bound = dialect(
        name="cann",
        file=str(path),
        format="bitcode",
        extern_func_name=SYMBOL.removeprefix("_mlir_ciface_"),
        pipeline={"core": "cube", "pipe": "PIPE_M"},
    )(stub)
    save()
    return bound
