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

"""Test-only score tile sweep using the unchanged production Triton kernel."""

import ast
import hashlib
import json
from copy import copy
from pathlib import Path
from unittest.mock import patch


def run(args, result, context):
    import torch
    import triton
    from benchmark_qsa_prefill_stages_910b import aggregate, sample
    from qsa_timeline_910b import collect

    module = context["module"]
    kernel = module._paged_score_910b
    node = next(
        n
        for n in ast.parse(Path(module.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_paged_score_910b"
    )
    digest = hashlib.sha256(
        ast.dump(node, include_attributes=False).encode()
    ).hexdigest()
    assert digest == "8f4c8d07542fe9ac0e83519f31ed95ca68009a68a4fa0ca40ba55eb903b4bad0"
    programs = module._score_programs(args.device)
    entries = {str(n): dict(status="pending", samples=[]) for n in (32, 64, 128)}
    result["tile_probe"] = dict(
        production_source_modified=False,
        production_ast_sha256=digest,
        scope="Only BLOCK_N and corresponding grid change; original per-column page lookup retained",
        programs=programs,
        variants=entries,
        resource_scope="Compiler metadata and textual IR; not a measured peak UB allocation",
    )
    function = context["score_stage"]

    class Launch:
        def __init__(self, block, entry):
            self.block, self.entry = block, entry
            self.compile_only = False
            self.compiled = set()

        def __getitem__(self, original_grid):
            def call(*operands, **options):
                options["BLOCK_N"] = self.block
                grid = (
                    min(
                        programs,
                        options["ROWS"] * triton.cdiv(options["WIDTH"], self.block),
                    ),
                )
                if not self.compile_only:
                    return kernel[grid](*operands, **options)
                compiled = kernel.warmup(*operands, grid=grid, **options)
                signature = (options["ROWS"], options["WIDTH"], options["OFFSET"])
                if signature not in self.compiled:
                    self.compiled.add(signature)
                    metadata = compiled.metadata
                    if hasattr(metadata, "_asdict"):
                        metadata = metadata._asdict()
                    elif hasattr(metadata, "__dict__"):
                        metadata = vars(metadata)
                    record = dict(
                        rows=signature[0],
                        width=signature[1],
                        offset=signature[2],
                        grid=grid,
                        metadata=json.loads(json.dumps(metadata, default=str)),
                    )
                    artifacts = []
                    for kind, content in getattr(compiled, "asm", {}).items():
                        if not isinstance(content, str):
                            continue
                        data = content.encode()
                        # Flat text artifacts are included by the existing report uploader.
                        safe_kind = "".join(c if c.isalnum() else "_" for c in kind)
                        name = (
                            f"{args.worker}-n{self.block}-r{signature[0]}"
                            f"-w{signature[1]}-o{signature[2]}-{safe_kind}.txt"
                        )
                        if len(data) <= 2_000_000:
                            (Path(args.report) / name).write_bytes(data)
                        artifacts.append(
                            dict(
                                kind=kind,
                                bytes=len(data),
                                sha256=hashlib.sha256(data).hexdigest(),
                                file=name if len(data) <= 2_000_000 else None,
                            )
                        )
                    record["artifacts"] = artifacts
                    self.entry.setdefault("compilation", []).append(record)
                    print("TILE_COMPILE", self.block, json.dumps(record), flush=True)
                return compiled

            return call

    launches = {name: Launch(int(name), entry) for name, entry in entries.items()}

    def check(boundaries=False):
        width = 385 if boundaries else context["width"]
        full = torch.empty(
            (context["rows"], width), device=context["q"].device, dtype=torch.float32
        )
        target = context["reference"][:, :width].clone()
        visible = context["visible"]
        if boundaries:
            edge = [
                0,
                1,
                31,
                32,
                33,
                63,
                64,
                65,
                127,
                128,
                129,
                191,
                192,
                193,
                255,
                385,
            ]
            visible = torch.tensor(
                [edge[i % len(edge)] for i in range(context["rows"])],
                device=full.device,
                dtype=torch.int32,
            )
            target[torch.arange(width)[None, :] >= visible.cpu()[:, None]] = -torch.inf
        for first, count in context["chunks"]:
            module._score(
                context["q"],
                context["cache"],
                context["table"],
                context["starts"],
                visible,
                full[first : first + count],
                offset=first,
                owners=context["owners"],
            )
        actual = full.cpu()
        finite = torch.isfinite(target)
        torch.testing.assert_close(actual[finite], target[finite], rtol=1e-3, atol=1e-3)
        assert torch.isneginf(actual[~finite]).all()
        return dict(
            max_abs_error=float((actual[finite] - target[finite]).abs().max()),
            masks_valid=True,
        )

    active = []
    for name, launch in launches.items():
        entry = entries[name]
        with patch.object(module, "_paged_score_910b", launch):
            # JIT warmup compiles without launching this candidate on device.
            launch.compile_only = True
            try:
                function()
            except Exception as exc:
                if name == "32" or not any(
                    cls.__name__
                    in ("MLIRCompilationError", "OutOfResources", "CompilationError")
                    for cls in type(exc).__mro__
                ):
                    raise
                entry.update(
                    status="compile_rejected",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                print("TILE_REJECTED", name, json.dumps(entry), flush=True)
                continue
            finally:
                launch.compile_only = False
            # A compiling candidate still must pass every numerical/masking check.
            entry["correctness_before"] = check()
            if args.worker.startswith("short"):
                entry["boundary_tail_check"] = check(boundaries=True)
            for _ in range(args.warmup):
                function()
            torch.npu.synchronize()
            active.append(name)
            entry["status"] = "ready"
    for round_index in range(args.rounds):
        order = active if round_index % 2 == 0 else list(reversed(active))
        for name in order:
            with patch.object(module, "_paged_score_910b", launches[name]):
                measurement = sample(function, torch, args.iterations)
                entries[name]["samples"].append(measurement)
                entries[name]["correctness_after"] = check()
            print(
                "TILE_TIMING",
                name,
                round_index + 1,
                json.dumps(measurement),
                flush=True,
            )
    for name in active:
        entry = entries[name]
        entry["latency"] = aggregate(entry["samples"])
        entry["speedup_n32_over_candidate"] = {
            field: entries["32"]["latency"][field]["median"]
            / entry["latency"][field]["median"]
            for field in ("event_us", "wall_us")
        }
        profile_args = copy(args)
        profile_args.worker = f"{args.worker}-n{name}"
        with patch.object(module, "_paged_score_910b", launches[name]):
            collect(
                profile_args, {"score": function}, torch, entry, selected=("score",)
            )
        timeline = entry.get("timeline", {})
        stage = timeline.get("stages", {}).get("score", {})
        if "device_task_duration_sum_us" in stage:
            entry["profile_device_us_per_invocation"] = (
                stage["device_task_duration_sum_us"] / timeline["iterations"]
            )
        entry["status"] = "passed"
        print(
            "TILE_SUMMARY",
            name,
            json.dumps(entry["latency"]),
            json.dumps(entry["speedup_n32_over_candidate"]),
            "profile_device_us",
            entry.get("profile_device_us_per_invocation"),
            flush=True,
        )
    result["tile_probe"]["execution_status"] = (
        "completed" if len(active) == 3 else "completed_with_compile_rejections"
    )
    print("TILE_EXECUTION", result["tile_probe"]["execution_status"], flush=True)
    result["status"] = "passed"
    return 0
