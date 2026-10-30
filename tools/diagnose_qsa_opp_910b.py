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

"""Collect OPP package and loaded-library evidence for native QSA failures."""

import hashlib
import json
import os
import traceback
from pathlib import Path


def inventory(root):
    root = Path(root)
    result = dict(
        root=str(root), resolved=str(root.resolve()), exists=root.is_dir(), matches=[]
    )
    if not root.is_dir():
        return result
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            if "lightning" not in str(path).lower():
                continue
            item = dict(
                path=str(path), resolved=str(path.resolve()), exists=path.exists()
            )
            if path.is_file():
                item["bytes"] = path.stat().st_size
                exact_op = (
                    "/lightning_indexer/" in str(path)
                    or name == "lightning_indexer.json"
                )
                if exact_op and path.suffix in (".json", ".o"):
                    item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                if path.suffix == ".json" and (exact_op or len(result["matches"]) < 30):
                    item["json_prefix"] = path.read_text(errors="replace")[:20000]
            result["matches"].append(item)
            if len(result["matches"]) >= 1000:
                result["truncated"] = True
                return result
    return result


def libraries():
    maps = Path("/proc/self/maps").read_text(errors="replace")
    return sorted(
        {
            line.split()[-1]
            for line in maps.splitlines()
            if ".so" in line
            and any(
                token in line.lower()
                for token in (
                    "ascend",
                    "cann-",
                    "libnnop",
                    "libopapi",
                    "libcust_opapi",
                    "torch_npu",
                )
            )
        }
    )


def run(args, result):
    import torch
    from qsa_lightning_baseline_910b import qsa_select_paged_prefill

    roots = [os.environ.get("ASCEND_OPP_PATH"), "/usr/local/Ascend/cann-9.0.0/opp"]
    roots.extend(filter(None, os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")))
    result["opp_inventory"] = [inventory(root) for root in dict.fromkeys(roots) if root]
    result["loaded_libraries_before"] = libraries()
    device = f"npu:{args.device}"
    try:
        q = torch.zeros((1, 4, 128), device=device, dtype=torch.bfloat16)
        cache = torch.zeros((3, 192, 1, 128), device=device, dtype=torch.bfloat16)
        table = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
        starts = torch.tensor([0, 1], device=device, dtype=torch.int32)
        visible = torch.tensor([513], device=device, dtype=torch.int32)
        output = torch.empty((1, 512), device=device, dtype=torch.int32)
        qsa_select_paged_prefill(
            q, cache, table, starts, visible, 2048, 4, 1, output, 2304
        )
        torch.npu.synchronize()
        result["native_probe"] = dict(
            status="executed", output_shape=list(output.shape)
        )
    except Exception:
        result["native_probe"] = dict(status="failed", error=traceback.format_exc())
    result["loaded_libraries_after"] = libraries()
    result["collection_note"] = (
        "Passed means diagnostics collected; native execution status is reported separately"
    )
    print(
        "OPP_INVENTORY",
        json.dumps(
            [
                dict(
                    root=item["root"],
                    exists=item["exists"],
                    matching_files=len(item["matches"]),
                )
                for item in result["opp_inventory"]
            ]
        ),
        flush=True,
    )
    print("NATIVE_PROBE", json.dumps(result["native_probe"]), flush=True)
    print("LOADED_LIBRARIES", json.dumps(result["loaded_libraries_after"]), flush=True)
    result["status"] = "passed"
    return 0


def collect_file_trace(raw, target):
    """Keep relevant file accesses in the upload; retain the full trace locally."""
    raw, target = Path(raw), Path(target)
    count, kept = 0, 0
    with target.open("w") as output:
        if not raw.is_file():
            output.write("strace did not produce a trace file; inspect worker stderr\n")
            return dict(status="missing", raw=str(raw))
        with raw.open(errors="replace") as source:
            for line in source:
                count += 1
                text = line.lower()
                if "lightning" in text or ("opp/" in text and "enoent" in text):
                    if kept < 3000:
                        output.write(line)
                    kept += 1
    return dict(
        status="collected",
        total_lines=count,
        relevant_lines=kept,
        uploaded_lines=min(kept, 3000),
        raw=str(raw),
        filtered=str(target),
    )
