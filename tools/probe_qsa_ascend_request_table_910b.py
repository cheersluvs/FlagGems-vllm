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

"""Exercise the image's actual BlockTable CPU snapshot and NPU commit."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def verify_group(torch, module, device, *, block_size, kernel_sizes=None):
    # Select single-rank geometry without starting a distributed model worker.
    with patch.object(
        module,
        "get_dcp_group",
        return_value=SimpleNamespace(world_size=1, rank_in_group=0),
    ):
        table = module.BlockTable(
            block_size=block_size,
            max_num_reqs=4,
            max_num_blocks_per_req=8,
            max_num_batched_tokens=16,
            pin_memory=True,
            device=device,
            kernel_sizes=kernel_sizes,
        )
    cpu = table.get_cpu_tensor()
    cpu.zero_()
    table.add_row([7, 11], 0)
    table.add_row([13], 1)

    def snapshot(n):
        return tuple(tuple(row) for row in cpu[:n].tolist())

    def committed(n):
        torch.npu.synchronize()
        return tuple(tuple(row) for row in table.get_device_tensor(n).cpu().tolist())

    first = snapshot(2)
    table.commit_block_table(2)
    assert committed(2) == first
    device_pointer = table.get_device_tensor(2).data_ptr()

    table.swap_row(0, 1)
    table.move_row(1, 2)
    table.clear_row(1)
    table.add_row([17, 19], 1)
    second = snapshot(3)
    assert second[:2] != first
    assert committed(2) == first, "CPU staging edit changed a committed NPU table"
    table.commit_block_table(3)
    assert committed(3) == second
    assert table.get_device_tensor(3).data_ptr() == device_pointer
    assert first[0][: len(second[2])] == second[2]
    return dict(
        physical_block_size=block_size,
        logical_block_size=table.block_size,
        columns=cpu.shape[1],
        commit_rows=[2, 3],
        snapshots_match_npu=True,
        device_address_reused_across_commits=True,
        old_snapshot_differs_from_new_batch=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401
    from vllm.v1.utils import CpuGpuBuffer
    from vllm_ascend.worker import block_table

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    device_name = torch.npu.get_device_name(args.device)
    if "910B" not in device_name:
        raise RuntimeError(f"Ascend 910B required, got {device_name}")
    source = Path(inspect.getfile(block_table)).resolve()
    buffer_source = Path(inspect.getfile(CpuGpuBuffer)).resolve()
    results = [
        verify_group(torch, block_table, device, block_size=192),
        verify_group(torch, block_table, device, block_size=384, kernel_sizes=[192]),
    ]
    print(
        json.dumps(
            dict(
                status="passed",
                device=device_name,
                source=str(source),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                buffer_source=str(buffer_source),
                groups=results,
                scope="actual BlockTable class, synthetic schedules, no model forward",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
