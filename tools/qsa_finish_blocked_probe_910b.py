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

"""Direct regular-stride key-tile reads and contiguous output writes."""

from qsa_finish_merge_probe_910b import WIDTHS
from qsa_finish_merge_probe_910b import run_probe as run_width_sweep


def source():
    text = "import triton\nimport triton.language as tl\n"
    for block in WIDTHS:
        text += f"""
@triton.jit
def raw_finish_merge{block}(Raw, KS, Table, Visible, Out, ROWS:tl.constexpr,
                            WIDTH:tl.constexpr,RAW_TILES:tl.constexpr,CORES:tl.constexpr):
    for unit in range(tl.program_id(0),ROWS*triton.cdiv(WIDTH,{block}),CORES):
        row = unit//triton.cdiv(WIDTH,{block})
        block = unit%triton.cdiv(WIDTH,{block})
        tiles = block*{block//64}+tl.arange(0,{block//64})
        c = tl.arange(0,64)
        visible = tl.load(Visible+row)
        segment_live = (tiles*64<WIDTH)&(tiles*64<visible)
        page = tl.where(segment_live,tiles//3,0)
        physical = tl.load(Table+page,mask=segment_live,other=0)
        safe_tile = tl.where(segment_live,tiles,0)
        live = (tiles[:,None]*64+c[None,:]<WIDTH)&(tiles[:,None]*64+c[None,:]<visible)
        ks = tl.load(KS+physical[:,None].to(tl.int64)*192+safe_tile[:,None]%3*64+c[None,:],
                     mask=live,other=0).to(tl.float32)
        value = tl.load(Raw+((row//16)*RAW_TILES+safe_tile[:,None])*1024+row%16*64+c[None,:],
                        mask=live,other=0)*ks
        result = tl.reshape(tl.where(live,value,-float("inf")),({block},))
        out_col = block*{block}+tl.arange(0,{block})
        tl.store(Out+row*WIDTH+out_col,result,mask=out_col<WIDTH)
"""
    return text


def run_probe(*args):
    from qsa_finish_full_block_probe_910b import VARIANTS, boundary
    from qsa_finish_full_block_probe_910b import source as full_source

    return run_width_sweep(
        *args,
        source_factory=full_source,
        kind="fullblocks",
        variants=VARIANTS,
        boundary_validator=boundary,
    )
