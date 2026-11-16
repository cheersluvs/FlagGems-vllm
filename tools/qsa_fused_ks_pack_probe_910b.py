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

"""Charge fused logical-scale packing in the complete raw score pipeline."""

from qsa_finish_full_block_probe_910b import source as full_source
from qsa_finish_merge_probe_910b import check_output, prepare
from qsa_k_pack_2d_probe_910b import source as k_source

VARIANTS = dict(
    full1024=(1024, "raw_finish_full1024"), logical1024=(1024, "raw_finish_logical1024")
)


def source():
    return (
        full_source()
        + "\n"
        + k_source()
        + """
@triton.jit
def raw_pack_k_ks(K, KS, Table, PK, LogicalKS, MAX_VISIBLE:tl.constexpr,TILES:tl.constexpr,CORES:tl.constexpr):
    for part in range(tl.program_id(0),TILES*8,CORES):
        raw_k_2d_part(K,Table,PK,part,MAX_VISIBLE)
        if part%8 == 0:
            tile = part//8
            c = tl.arange(0,64)
            physical = tl.load(Table+tile//3)
            scale = tl.load(KS+physical.to(tl.int64)*192+tile%3*64+c,
                            mask=tile*64+c<MAX_VISIBLE,other=0)
            tl.store(LogicalKS+tile*64+c,scale)

@triton.jit
def raw_logical_body(Raw, LogicalKS, Out, row, block, WIDTH:tl.constexpr,RAW_TILES:tl.constexpr):
    tiles = block*16+tl.arange(0,16)
    c = tl.arange(0,64)
    col = block*1024+tl.arange(0,1024)
    ks = tl.reshape(tl.load(LogicalKS+col).to(tl.float32),(16,64))
    value = tl.load(Raw+((row//16)*RAW_TILES+tiles[:,None])*1024+row%16*64+c[None,:])*ks
    tl.store(Out+row*WIDTH+col,tl.reshape(value,(1024,)))

@triton.jit
def raw_finish_logical1024(Raw, KS, Table, Visible, Out, LogicalKS, ROWS:tl.constexpr,
                           WIDTH:tl.constexpr,RAW_TILES:tl.constexpr,CORES:tl.constexpr):
    for unit in range(tl.program_id(0),ROWS*triton.cdiv(WIDTH,1024),CORES):
        row = unit//triton.cdiv(WIDTH,1024)
        block = unit%triton.cdiv(WIDTH,1024)
        visible = tl.load(Visible+row)
        if ((block+1)*1024<=WIDTH)&((block+1)*1024<=visible):
            raw_logical_body(Raw,LogicalKS,Out,row,block,WIDTH,RAW_TILES)
        else:
            raw_full_tail(Raw,KS,Table,Out,row,block,visible,WIDTH,RAW_TILES)
"""
    )


def trial(module, arm, values, torch, variants=None):
    tiles = arm["raw"].shape[1]
    logical = torch.full(
        (tiles * 64,),
        float("nan"),
        dtype=values["ks"].dtype,
        device=arm["output"].device,
    )
    packed_k = arm.get("pk")
    if packed_k is None:
        packed_k = torch.empty(
            (tiles, 8192), dtype=values["cache"].dtype, device=logical.device
        )
        ref = torch.empty_like(packed_k)
        n = min(arm["output"].shape[1], int(values["visible"].cpu().max()))
        cores = min(20, tiles * 8)
        module.raw_k_2d_capped[(cores,)](
            values["cache"],
            values["row_table"],
            ref,
            MAX_VISIBLE=n,
            TILES=tiles,
            CORES=cores,
            num_warps=4,
        )
        torch.npu.synchronize()
        expected_k = ref.cpu()
    else:
        expected_k = packed_k.cpu()
    maximum = min(arm["output"].shape[1], int(values["visible"].cpu().max()))
    cores = min(20, tiles * 8)
    ks = values["ks"].cpu().reshape(-1)
    mapping = values["row_table"].cpu()[0].long()
    expected = torch.zeros((tiles * 64,), dtype=ks.dtype)
    col = torch.arange(maximum)
    expected[:maximum] = ks[mapping[col // 192] * 192 + col % 192]

    def packer():
        return module.raw_pack_k_ks[(cores,)](
            values["cache"],
            values["ks"],
            values["row_table"],
            packed_k,
            logical,
            MAX_VISIBLE=maximum,
            TILES=tiles,
            CORES=cores,
            num_warps=4,
        )

    def check():
        result = dict(
            k_exact=bool(torch.equal(packed_k.cpu(), expected_k)),
            ks_exact=bool(torch.equal(logical.cpu(), expected)),
        )
        if not all(result.values()):
            raise AssertionError(str(result))
        return result

    result = prepare(
        module, arm, values, torch, VARIANTS, extra_args={"logical1024": (logical,)}
    )
    result.update(
        packers=dict(logical1024=packer),
        packing_check=check,
        liveness=dict(logical1024="logicalfull"),
        logical_buffer=logical,
        skip_basic_boundary=True,
    )
    return result


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
    cache = (
        ((torch.arange(pages * 192 * 128, dtype=torch.int32) % 255) - 127)
        .to(torch.int8)
        .reshape(pages, 192, 1, 128)
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
        for k, v in dict(cache=cache, ks=ks, row_table=table, visible=visible).items()
    }
    arm = dict(raw=raw.to(device), output=expected.to(device))
    controls = trial(module, arm, values, torch)
    controls["packers"]["logical1024"]()
    torch.npu.synchronize()
    packing = controls["packing_check"]()
    controls["launches"][label]()
    torch.npu.synchronize()
    check = check_output(torch, controls["targets"][label], expected)
    print("FUSED_KS_BOUNDARY", label, packing, check, flush=True)
    if not check["exact"]:
        raise AssertionError(str(check))
    return dict(rows=rows, width=width, packing=packing, **check)
