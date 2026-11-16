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

"""Remove conditional tile selection on fully visible 1024-key blocks."""

from qsa_finish_blocked_probe_910b import source as blocked_source
from qsa_finish_merge_probe_910b import check_output, prepare

VARIANTS = dict(
    blocked1024=(1024, "raw_finish_merge1024"), full1024=(1024, "raw_finish_full1024")
)


def source():
    frozen = blocked_source()
    last = frozen[frozen.index("def raw_finish_merge1024") :]
    body = last[last.index("        tiles =") :]
    tail = "\n".join(
        line[4:] for line in body.splitlines() if "visible = tl.load" not in line
    )
    return (
        frozen
        + "\n@triton.jit\ndef raw_full_tail(Raw, KS, Table, Out, row, block, visible,\n"
        "                  WIDTH:tl.constexpr,RAW_TILES:tl.constexpr):\n"
        + tail
        + """

@triton.jit
def raw_full_body(Raw, KS, Table, Out, row, block, WIDTH:tl.constexpr,RAW_TILES:tl.constexpr):
    tiles = block*16+tl.arange(0,16)
    c = tl.arange(0,64)
    physical = tl.load(Table+tiles//3)
    ks = tl.load(KS+physical[:,None].to(tl.int64)*192+tiles[:,None]%3*64+c[None,:]).to(tl.float32)
    value = tl.load(Raw+((row//16)*RAW_TILES+tiles[:,None])*1024+row%16*64+c[None,:])*ks
    result = tl.reshape(value,(1024,))
    col = block*1024+tl.arange(0,1024)
    tl.store(Out+row*WIDTH+col,result)

@triton.jit
def raw_finish_full1024(Raw, KS, Table, Visible, Out, ROWS:tl.constexpr,
                        WIDTH:tl.constexpr,RAW_TILES:tl.constexpr,CORES:tl.constexpr):
    for unit in range(tl.program_id(0),ROWS*triton.cdiv(WIDTH,1024),CORES):
        row = unit//triton.cdiv(WIDTH,1024)
        block = unit%triton.cdiv(WIDTH,1024)
        visible = tl.load(Visible+row)
        if ((block+1)*1024<=WIDTH)&((block+1)*1024<=visible):
            raw_full_body(Raw,KS,Table,Out,row,block,WIDTH,RAW_TILES)
        else:
            raw_full_tail(Raw,KS,Table,Out,row,block,visible,WIDTH,RAW_TILES)
"""
    )


def boundary(module, label, torch, device):
    rows, width, tiles, pages = 17, 2057, 33, 11
    raw = (
        (torch.arange(2 * tiles * 16 * 64, dtype=torch.float32) % 257) - 128
    ).reshape(2, tiles, 16, 64) / 16
    ks = (
        (torch.arange(pages * 192, dtype=torch.float32) / 4096 + 1 / 64)
        .half()
        .reshape(pages, 192, 1)
    )
    table = torch.tensor(list(reversed(range(pages))), dtype=torch.int32).repeat(
        rows, 1
    )
    visible = torch.tensor(
        ([0, 1, 1023, 1024, 1025, 2048, 2049] * 3)[:rows], dtype=torch.int32
    )
    expected = torch.full((rows, width), -float("inf"), dtype=torch.float32)
    for row in range(rows):
        n = int(visible[row])
        col = torch.arange(n)
        scale = ks.reshape(-1)[table[row, col // 192].long() * 192 + col % 192].float()
        expected[row, :n] = raw[row // 16, :, row % 16, :].flatten()[:n] * scale
    values = {
        k: v.to(device)
        for k, v in dict(ks=ks, row_table=table, visible=visible).items()
    }
    arm = dict(raw=raw.to(device), output=expected.to(device))
    trial = prepare(module, arm, values, torch, VARIANTS)
    trial["launches"][label]()
    torch.npu.synchronize()
    check = check_output(torch, trial["targets"][label], expected)
    print("FULL_BLOCK_BOUNDARY", label, check, flush=True)
    if not check["exact"]:
        raise AssertionError(str(check))
    return dict(rows=rows, width=width, visible=visible.tolist(), **check)
