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

"""Prepared request-piece mixed kernel; all packing remains inside each call."""

from qsa_raw_mixed_online_910b import boundary_geometry


def piece_geometry(rows, width, maximum):
    if not 0 <= maximum <= width:
        raise ValueError("Visibility outside logical width")
    shape = boundary_geometry(rows, width, 2)
    shape["blocks"] = max(1, (maximum + 511) // 512)
    shape["active_tiles"] = shape["blocks"] * 8
    shape["maximum"] = maximum
    return shape


class PreparedRawMixedSelection:
    def __init__(
        self, context, module, report, case_name, save_asm, record_launch_mode
    ):
        import torch

        self.context = context
        self.module = module
        adapter = context["adapter"]
        self.rows, self.width, self.cap = adapter.rows, adapter.width, adapter.chunk
        device = adapter.output.device
        max_groups = (self.cap + 15) // 16
        tiles = ((self.width + 511) // 512) * 8
        self.pq = torch.empty((max_groups, 8192), dtype=torch.int8, device=device)
        self.pw = torch.empty((max_groups, 1024), dtype=torch.float16, device=device)
        self.pk = torch.empty((tiles, 8192), dtype=torch.int8, device=device)
        self.ks = torch.empty((tiles * 64,), dtype=torch.float16, device=device)
        self.ring = torch.empty(
            (max_groups * 2 * 8192,), dtype=torch.float32, device=device
        )
        self.trace = torch.empty(
            (self.cap, self.width), dtype=torch.float32, device=device
        )
        self.values = torch.empty((self.cap, 512), dtype=torch.float32, device=device)
        self.storage = torch.empty(
            (self.rows + 32, 512), dtype=torch.int32, device=device
        )
        self.output = self.storage[16 : 16 + self.rows]
        self.jobs = []
        self.by_chunk = {}
        self.artifacts = []
        self.routes = []
        seen = set()
        v = context["values"]
        for entry, pieces in context["jobs"]:
            chunk = []
            for item in pieces:
                first, n, maximum = item["first"], item["count"], item["maximum"]
                g = piece_geometry(n, self.width, maximum)
                groups, blocks = g["groups"], g["blocks"]
                q, qs, w = [v[k][first : first + n] for k in ("q", "qs", "weights")]
                table = v["row_table"][first : first + n]
                vis = v["visible"][first : first + n]
                target = self.output[first : first + n]
                vals = self.values[:n]
                trace = self.trace[:n]
                pq = self.pq[:groups]
                pw = self.pw[:groups]
                pk = self.pk[: g["active_tiles"]]
                lks = self.ks[: blocks * 512]
                ring = self.ring[: groups * 2 * 8192]
                operands = (pq, pk, pw, lks, vis, ring, target, vals, trace)
                meta = dict(
                    WIDTH=self.width, ROWS=n, SLOTS=2, BLOCKS=blocks, num_warps=4
                )
                if len(self.routes) % 16 == 0:
                    print("MIXED_PREPARE", case_name, first, n, blocks, flush=True)
                for traced in (False, True):
                    compiled = module.raw_mixed_prepared.warmup(
                        *operands, TRACE=traced, grid=(groups,), **meta
                    )
                    launch = record_launch_mode(compiled)
                    if launch["mix_mode"] != "mix" or launch[
                        "bs_task_type"
                    ] // 10 not in (3, 4):
                        raise RuntimeError(str(launch))
                    key = (traced, str(compiled.metadata))
                    if key not in seen:
                        seen.add(key)
                        self.artifacts.append(
                            dict(
                                launch=launch,
                                asm=save_asm(
                                    compiled, report, f"{case_name}-mixed-{len(seen)}"
                                ),
                            )
                        )
                online_meta = dict(ROWS=n, WIDTH=self.width, BLOCKS=blocks, num_warps=4)
                online = module.raw_online_topk_from_scores.warmup(
                    trace, vis, target, vals, grid=((n + 7) // 8,), **online_meta
                )
                mode = record_launch_mode(online, require_cube=False)
                if mode["mix_mode"] != "aiv" or mode["bs_task_type"] // 10 != 1:
                    raise RuntimeError(str(mode))
                key = ("online", str(online.metadata))
                if key not in seen:
                    seen.add(key)
                    self.artifacts.append(
                        dict(
                            launch=mode,
                            asm=save_asm(
                                online, report, f"{case_name}-online-{len(seen)}"
                            ),
                        )
                    )

                def execute(
                    traced=False,
                    q=q,
                    qs=qs,
                    w=w,
                    table=table,
                    pq=pq,
                    pw=pw,
                    pk=pk,
                    lks=lks,
                    n=n,
                    g=g,
                    operands=operands,
                    meta=meta,
                ):
                    module.raw_query16_qw[(g["groups"],)](
                        q, qs, w, pq, pw, ROWS=n, QUERIES=16, num_warps=4
                    )
                    cores = min(20, g["active_tiles"] * 8)
                    module.raw_pack_k_ks_padded[(cores,)](
                        v["cache"],
                        v["ks"],
                        table,
                        pk,
                        lks,
                        MAX_VISIBLE=g["maximum"],
                        TILES=g["active_tiles"],
                        CORES=cores,
                        num_warps=4,
                    )
                    return module.raw_mixed_prepared[(g["groups"],)](
                        *operands, TRACE=traced, **meta
                    )

                chunk.append(
                    dict(
                        first=first,
                        count=n,
                        offset=item["offset"],
                        execute=execute,
                        meta=online_meta,
                        visible=vis,
                        target=target,
                        values=vals,
                    )
                )
                self.routes.append(
                    dict(
                        first=first,
                        count=n,
                        maximum=maximum,
                        blocks=blocks,
                        groups=groups,
                    )
                )
            self.jobs.extend(chunk)
            self.by_chunk[entry["first"]] = chunk

    def __call__(self):
        for item in self.jobs:
            item["execute"]()
        return self.output

    def reset_guards(self):
        self.storage[:16].fill_(-3)
        self.storage[16 + self.rows :].fill_(-3)

    def check_guards(self):
        for guard in (self.storage[:16], self.storage[16 + self.rows :]):
            if not bool((guard.cpu() == -3).all()):
                raise AssertionError("Mixed whole-output guard overwritten")

    def check_chunk_scores(self, torch, entry, expected):
        for item in self.by_chunk[entry["first"]]:
            n, offset = item["count"], item["offset"]
            self.trace.fill_(float("nan"))
            self.trace[:n].fill_(-float("inf"))
            self.values.fill_(float("nan"))
            item["execute"](True)
            torch.npu.synchronize()
            torch.testing.assert_close(
                self.trace[:n].cpu(), expected[offset : offset + n], rtol=0, atol=0
            )
            assert bool(
                torch.isnan(self.trace[n:].cpu()).all()
            ), "Mixed trace row-tail store"
            assert bool(
                torch.isnan(self.values[n:].cpu()).all()
            ), "Mixed values row-tail store"
            self.check_guards()

    def topk_all_from_staged(self, staged):
        for item in self.jobs:
            first, n = item["first"], item["count"]
            self.module.raw_online_topk_from_scores[((n + 7) // 8,)](
                staged[first : first + n],
                item["visible"],
                item["target"],
                item["values"],
                **item["meta"],
            )
        return self.output
