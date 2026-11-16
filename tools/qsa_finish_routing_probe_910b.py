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

"""Separate invisible-block and logical-scale tail costs without changing Cube."""

VARIANTS = ("empty1024", "tail1024", "combined1024")
CASES = ("int8_first_chunk", "int8_budget_exact", "int8_tail113", "int8_long100k")


def selected_variants(args):
    return ("empty1024",) if getattr(args, "finish_empty_pair", False) else VARIANTS


def source():
    text = """
@triton.jit
def raw_logical_tail(Raw, LogicalKS, Out, row, block, visible,
                     WIDTH:tl.constexpr, RAW_TILES):
    tiles = block*16+tl.arange(0,16)
    c = tl.arange(0,64)
    col = block*1024+tl.arange(0,1024)
    live = (tiles[:,None]*64+c[None,:]<WIDTH)&(tiles[:,None]*64+c[None,:]<visible)
    segment_live = (tiles*64<WIDTH)&(tiles*64<visible)
    safe_tile = tl.where(segment_live,tiles,0)
    ks_col = tl.reshape(tl.where(live,tiles[:,None]*64+c[None,:],0),(1024,))
    ks = tl.reshape(tl.load(LogicalKS+ks_col,mask=tl.reshape(live,(1024,)),other=0).to(tl.float32),(16,64))
    value = tl.load(Raw+((row//16)*RAW_TILES+safe_tile[:,None])*1024+row%16*64+c[None,:],
                    mask=live,other=0)*ks
    result = tl.reshape(tl.where(live,value,-float("inf")),(1024,))
    tl.store(Out+row*WIDTH+col,result,mask=col<WIDTH)

@triton.jit
def raw_empty_block(Out,row,block,WIDTH:tl.constexpr):
    col = block*1024+tl.arange(0,1024)
    tl.store(Out+row*WIDTH+col,tl.full((1024,),-float("inf"),tl.float32),mask=col<WIDTH)
"""
    for name in VARIANTS:
        tail = (
            "raw_full_tail(Raw,KS,Table,Out,row,block,visible,WIDTH,RAW_TILES)"
            if name == "empty1024"
            else "raw_logical_tail(Raw,LogicalKS,Out,row,block,visible,WIDTH,RAW_TILES)"
        )
        if name == "tail1024":
            branch = "        if ((block+1)*1024<=WIDTH)&((block+1)*1024<=visible):\n"
        else:
            branch = (
                "        if block*1024>=visible:\n"
                "            raw_empty_block(Out,row,block,WIDTH)\n"
                "        elif ((block+1)*1024<=WIDTH)&((block+1)*1024<=visible):\n"
            )
        text += f"""
@triton.jit
def raw_finish_{name}(Raw,KS,Table,Visible,Out,LogicalKS,ROWS:tl.constexpr,
                      WIDTH:tl.constexpr,RAW_TILES,CORES:tl.constexpr):
    for unit in range(tl.program_id(0),ROWS*triton.cdiv(WIDTH,1024),CORES):
        row = unit//triton.cdiv(WIDTH,1024)
        block = unit%triton.cdiv(WIDTH,1024)
        visible = tl.load(Visible+row)
{branch}            raw_logical_body(Raw,LogicalKS,Out,row,block,WIDTH,RAW_TILES)
        else:
            {tail}
"""
    return text


def boundary(module, torch, device, variants=VARIANTS):
    rows, width, tiles = 17, 2057, 33
    pages = 11
    raw = ((torch.arange(2 * tiles * 1024, dtype=torch.float32) % 257) - 128).reshape(
        2, tiles, 16, 64
    ) / 16
    ks = (
        ((torch.arange(pages * 192, dtype=torch.float32) + 1) / 4096)
        .half()
        .reshape(pages, 192, 1)
    )
    table = torch.tensor(list(reversed(range(pages))), dtype=torch.int32).repeat(
        rows, 1
    )
    visible = torch.tensor(
        ([0, 1, 63, 64, 65, 1023, 1024, 1025, 2048, 2049, 2057] * 2)[:rows],
        dtype=torch.int32,
    )
    logical = ks[table[0].long(), :, 0].reshape(-1)[: tiles * 64].contiguous()
    expected = torch.full((rows, width), -float("inf"), dtype=torch.float32)
    for row in range(rows):
        n = int(visible[row])
        expected[row, :n] = (
            raw[row // 16, :, row % 16, :].reshape(-1)[:n] * logical[:n].float()
        )
    values = [x.to(device) for x in (raw, ks, table, visible, logical)]
    raw_d, ks_d, bt_d, vis_d, logical_d = values
    output = torch.empty((rows, width), dtype=torch.float32, device=device)
    result = {}
    for name in ("logical1024",) + tuple(variants):
        output.fill_(float("nan"))
        getattr(module, "raw_finish_" + name)[(20,)](
            raw_d,
            ks_d,
            bt_d,
            vis_d,
            output,
            logical_d,
            ROWS=rows,
            WIDTH=width,
            RAW_TILES=tiles,
            CORES=20,
            num_warps=4,
        )
        torch.npu.synchronize()
        exact = bool(torch.equal(output.cpu(), expected))
        result[name] = dict(exact=exact, rows=rows, width=width)
        print("FINISH_ROUTING_BOUNDARY", name, result[name], flush=True)
        if not exact:
            raise AssertionError(str(result))
    return result


def stage_totals(report, samples):
    """Reuse the accepted capture; group device durations without another run."""
    import collections
    import gzip
    import json

    from qsa_int8_stage_trace_timer_910b import stage_windows

    artifact = next(iter(samples.values()))[0]["artifact"]
    data = json.loads(gzip.decompress((report / artifact).read_bytes()))
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    count = sum(len(s["window_indices"]) for arm in samples.values() for s in arm)
    windows = list(stage_windows(events, count))
    result = {}
    for arm, rounds in samples.items():
        indices = [i for s in rounds for i in s["window_indices"]]
        sums = collections.defaultdict(float)
        for i in indices:
            for event in windows[i][1]:
                name = event["name"]
                stage = (
                    "finish"
                    if name.startswith("raw_finish_")
                    else (
                        "cube"
                        if name == "raw_query16_cube"
                        else (
                            "qw_pack"
                            if name == "raw_query16_qw"
                            else "k_ks_pack" if name == "raw_pack_k_ks" else name
                        )
                    )
                )
                sums[stage] += event["dur"] / len(indices)
        expected = sum(s["kernel_us"] for s in rounds) / len(rounds)
        if abs(sum(sums.values()) - expected) > max(1e-6, expected * 1e-9):
            raise AssertionError("Stage sums differ from accepted sweep timing")
        result[arm] = dict(sums)
    return result
