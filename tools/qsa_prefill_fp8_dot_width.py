# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Four controlled FP8 widths with fixed query/cache geometry;source JIT only."""

import argparse
import copy
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_fp8_dot_full as full  # noqa: E402

runtime = full.runtime
CASE = "fp8_long100k_float8_e4m3fn"
WIDTHS = (8192, 16384, 24576, 29696)
ROWS = 4519
WORKER, COORDINATE, VERIFY = full.WORKER, full.COORDINATE, full.verify_source
EXTRA = {
    "tools/qsa_prefill_fp8_dot_width.py",
    "tools/test_qsa_prefill_fp8_dot_width.py",
    "tools/run_qsa_prefill_fp8_dot_width_h800.sh",
    "docs/qsa_prefill_fp8_dot_width.md",
}


def visible_plan(width):
    if width not in WIDTHS:
        raise ValueError("Unsupported diagnostic width")
    return [width - (ROWS - 1 - row) // 4 for row in range(ROWS)]


def verify_source():
    return dict(
        VERIFY(),
        diagnostic=dict(
            widths=list(WIDTHS),
            rows=ROWS,
            origin="controlled causal-ramp fixture;not a captured benchmark case",
            seed_case=CASE,
            cache_and_table_fixed=True,
        ),
    )


def worker(options):
    from tools import benchmark_qsa_prefill_triton_stages as bench

    root = Path(options.report_dir)
    seed = next(v for v in full.manifest()["cases"] if v["case"]["name"] == CASE)
    original_make, original_geometry = (
        bench.workloads.make_inputs,
        bench.verify_geometry,
    )
    shared, identity, boundary = {}, {}, {}
    results = []
    final = dict(
        status="running",
        production_modified=False,
        acceptance_eligible=False,
        diagnostic=verify_source()["diagnostic"],
        widths=[],
    )

    def boundaries(production, helpers, torch, device):
        # Always use original saved fixtures for the existing boundary tests.
        if not boundary:
            with patch.object(bench.workloads, "make_inputs", original_make):
                checks = full.boundary_checks(production, helpers, torch, device)
            boundary.update(checks=checks, source=verify_source())
            bench.support.write_json(root / "width-boundary-proof.json", boundary)
        if boundary["source"] != verify_source():
            raise RuntimeError("Boundary source/proof identity changed")
        return [
            dict(c, reused_boundary_proof="width-boundary-proof.json")
            for c in boundary["checks"]
        ]

    try:
        for width in WIDTHS:
            folder = root / ("width-" + str(width))
            folder.mkdir()
            (folder / "qsa_indexer_vllm.py").write_bytes(
                (root / "qsa_indexer_vllm.py").read_bytes()
            )
            item = copy.deepcopy(seed)
            item["case"].update(
                query_lengths=[ROWS],
                prefix_tokens=[width * 4 - ROWS],
                batch_max_seq_len=width * 4,
                query_lengths_synthetic=True,
                origin="controlled_width_probe",
            )
            item["traces"] = [
                dict(
                    rows=ROWS,
                    offset=0,
                    width=width,
                    route="fp8",
                    constants=dict(full.fp8.probe.CONFIG),
                )
            ]
            data = dict(full.manifest(), cases=[item])

            def make_inputs(case, helpers, torch):
                if case != item["case"]:
                    return original_make(case, helpers, torch)
                if not shared:
                    original = original_make(seed["case"], helpers, torch)
                    if bench.support.fingerprint(original, torch) != seed["inputs"]:
                        raise RuntimeError("Long100k seed input identity changed")
                    values = list(original)
                    values[0] = values[0][:ROWS]
                    values[3] = torch.tensor(
                        [0, ROWS], dtype=torch.int32, device=values[0].device
                    )
                    values[7] = ROWS
                    values[8] = torch.empty(
                        (ROWS, 512), dtype=torch.int32, device=values[0].device
                    )
                    shared["args"] = values
                values = list(shared["args"])
                values[4] = torch.tensor(
                    visible_plan(width), dtype=torch.int32, device=values[0].device
                )
                values[9] = width * 4
                args = tuple(values)
                item["inputs"] = bench.support.fingerprint(args, torch)
                unchanged = bench.support.fingerprint(args[:4], torch)
                if identity and identity["q_cache_table_starts"] != unchanged:
                    raise RuntimeError("Query/cache/table/starts changed across widths")
                identity["q_cache_table_starts"] = unchanged
                return args

            def geometry(case, args, chunks):
                if case != item["case"]:
                    return original_geometry(case, args, chunks)
                if (
                    tuple(args[0].shape) != (ROWS, 4, 128)
                    or tuple(args[1].shape) != (152, 196, 1, 128)
                    or tuple(args[2].shape) != (1, 152)
                    or args[3].tolist() != [0, ROWS]
                    or args[4].tolist() != visible_plan(width)
                    or tuple(args[8].shape) != (ROWS, 512)
                    or args[7] != ROWS
                    or args[9] != width * 4
                    or bench.ab.signatures(chunks) != [(0, ROWS, width)]
                ):
                    raise RuntimeError("Controlled width probe geometry changed")
                return dict(rows=ROWS, width=width, synthetic=True)

            opts = SimpleNamespace(**dict(vars(options), report_dir=str(folder)))
            with patch.object(full.fp8, "manifest", lambda: data), patch.object(
                bench.workloads, "make_inputs", make_inputs
            ), patch.object(bench, "verify_geometry", geometry), patch.object(
                runtime, "boundary_checks", boundaries
            ):
                code = WORKER(opts)
            path = folder / (CASE + ".json")
            result = json.loads(path.read_text()) if path.is_file() else {}
            results.append(dict(width=width, exit_code=code, result=result))
            final["widths"].append(
                dict(
                    width=width,
                    exit_code=code,
                    status=result.get("status"),
                    stages=result.get("stages", {}),
                    self_controls=result.get("self_controls", {}),
                )
            )
            if code or result.get("status") != "passed":
                raise RuntimeError("Width worker failed: " + str(width))
        final.update(status="passed", invariant_inputs=identity)
        # Existing outer coordinator consumes one representative case receipt;
        # all four width results remain explicit and are not benchmark means.
        receipt = copy.deepcopy(results[-1]["result"])
        receipt.update(width_scan=final, diagnostic=True)
        bench.support.write_json(root / (CASE + ".json"), receipt)
        return 0
    except BaseException:
        import traceback

        final.update(status="failed", error=traceback.format_exc())
        bench.support.write_json(
            root / (CASE + ".json"), dict(final, stages={}, self_controls={})
        )
        print(final["error"], flush=True)
        return 1
    finally:
        bench.support.write_json(root / "fp8-dot-width-summary.json", final)
        with tarfile.open(root / "width-artifacts.tar.gz", "w:gz") as archive:
            for folder in sorted(root.glob("width-*")):
                if folder.is_dir():
                    for path in sorted(folder.iterdir()):
                        if path.is_file():
                            archive.add(
                                path,
                                arcname=str(path.relative_to(root)),
                                recursive=False,
                            )
        print("\nWidth crossover summary (production / candidate):", flush=True)
        for row in final["widths"]:
            print(
                row["width"],
                row["status"],
                {k: v.get("speedup") for k, v in row["stages"].items()},
                flush=True,
            )


def coordinate(options):
    code = COORDINATE(options)
    root = Path(options.report_dir)
    old = root / "fp8-dot-runtime-summary.json"
    path = root / "fp8-dot-width-summary.json"
    receipt = json.loads(old.read_text()) if old.is_file() else {}
    value = json.loads(path.read_text()) if path.is_file() else dict(widths=[])
    value["coordination"] = {
        k: v for k, v in receipt.items() if k not in ("aggregate", "cases")
    }
    if code or receipt.get("status") != "passed":
        value.update(status="failed", coordination_exit_code=code)
    path.write_text(json.dumps(value, indent=2) + "\n")
    if old.is_file():
        old.unlink()
    print(
        "Use fp8-dot-width-summary.json for all four points;outer case receipt is the last width only.",
        flush=True,
    )
    return code


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cases", nargs="+")
    selected, _ = parser.parse_known_args()
    if selected.cases is not None and selected.cases != [CASE]:
        parser.error("Width probe supports only the long100k seed case")
    if "--cases" not in sys.argv:
        sys.argv.extend(["--cases", CASE])
    with patch.object(full, "COORDINATE", coordinate), patch.object(
        full, "WORKER", worker
    ), patch.object(full, "verify_source", verify_source), patch.object(
        runtime, "EXTRA", runtime.EXTRA | EXTRA
    ), patch.object(
        full, "__file__", str(Path(__file__).resolve())
    ), patch.object(
        full, "__doc__", __doc__
    ):
        # full.main selects this adapter as child entry through its own __file__.
        return full.main()


if __name__ == "__main__":
    sys.exit(main())
