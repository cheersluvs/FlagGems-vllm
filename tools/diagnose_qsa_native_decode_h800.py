# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Check native QSA decode async cache policies on H800; no Top-K or timing."""

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import diagnose_qsa_native_score_h800 as shared

ROOT = shared.ROOT
BLOCK_N = 64
COLUMNS = 8192
PAGE_SIZE = 16
REQUESTS = {1: 3, 2: 128, 4: 256, 8: 1024}
KERNEL = "_qsa_mqa_paged_uniform_kernel"


def case_matrix(args):
    return {
        f"{dtype}-h{args.heads}d{args.head_dim}-q{length}-tiles{tiles}": {
            "dtype": dtype,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "decode_query_len": length,
            "tiles_per_program": tiles,
            "num_requests": REQUESTS[tiles],
            "columns": COLUMNS,
            "page_size": PAGE_SIZE,
            "seed": 20261014,
        }
        for dtype in args.dtypes
        for length in args.query_lengths
        for tiles in args.tiles
    }


def prepare(report):
    import torch

    from tests.qsa_select_prefill_utils import (
        REFERENCE_COMMIT,
        REFERENCE_SHA256,
        vllm_reference,
    )

    facts = shared.snapshot(report)
    reference = vllm_reference()
    assert importlib.metadata.version("vllm").split("+")[0] == "0.31.0"
    assert shared.sha(reference.__file__) == REFERENCE_SHA256
    shutil.copyfile(reference.__file__, report / "qsa_indexer_vllm.py")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    cases = json.loads((report / "cases.json").read_text())
    inputs = {}
    for name, config in cases.items():
        requests = config["num_requests"]
        width = config["columns"]
        page_size = config["page_size"]
        heads, dim, length = (
            config["heads"],
            config["head_dim"],
            config["decode_query_len"],
        )
        assert (
            reference._decode_tiles_per_program(requests, width)
            == config["tiles_per_program"]
        ), "shape does not select the intended upstream tile configuration"
        fp8 = config["dtype"] == "fp8"
        dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
        torch.manual_seed(config["seed"])
        q = torch.randn(
            (requests * length, heads, dim), device="cuda", dtype=torch.bfloat16
        ).to(dtype)
        # Shared physical pages model legal prefix-cache reuse, keeping inputs
        # small even for the 1024-request tile-8 dispatch. Tables are shuffled.
        pages = width // page_size
        cache = torch.randn(
            (pages * 2, page_size, 1, dim), device="cuda", dtype=torch.bfloat16
        ).to(dtype)
        table = torch.rand(pages * 2, device="cuda").argsort()[:pages]
        table = (table[None, :] + torch.arange(requests, device="cuda")[:, None]) % (
            pages * 2
        )
        table = table.to(torch.int32).contiguous()
        visible = width - torch.arange(requests * length, device="cuda") % 67
        visible = visible.to(torch.int32)
        visible[:length] = width  # Exercise every iteration of the full loop.
        if length > 1:
            visible[length] = 0  # Mixed visibility within a single request.
        visible[-length:] = 0  # A fully empty request takes the early return.
        logits = torch.empty(
            (requests, length, width), dtype=torch.float32, device="cuda"
        )
        # Independent FP32 reference, bounded temporary memory. Apply ReLU per
        # head before summing, matching decode's unscaled score contract.
        for start in range(0, requests, 16):
            end = min(start + 16, requests)
            keys = cache[table[start:end].long(), :, 0, :].flatten(1, 2).float()
            queries = q[start * length : end * length].float()
            queries = queries.reshape(end - start, length * heads, dim)
            scores = torch.bmm(keys, queries.transpose(1, 2))
            scores = scores.reshape(end - start, width, length, heads)
            logits[start:end] = scores.relu().sum(-1).transpose(1, 2)
        expected = logits.reshape(requests * length, width)
        mask = torch.arange(width, device="cuda")[None, :] >= visible[:, None]
        expected.masked_fill_(mask, -torch.inf)
        values = [q, cache, table, visible, length]
        torch.cuda.synchronize()
        path = report / (name + "-input.pt")
        shared.write_json(report / (name + "-config.json"), config)
        torch.save(
            {
                "args": [x.cpu() if isinstance(x, torch.Tensor) else x for x in values],
                "strides": [
                    list(x.stride()) if isinstance(x, torch.Tensor) else None
                    for x in values
                ],
                "expected": expected.cpu(),
                "device_index": torch.cuda.current_device(),
                "config": config,
                "score_scale": dim**-0.5 if fp8 else 1.0,
                "score_rtol": 1e-3 if fp8 else 2e-4,
                "score_atol": 1e-3 if fp8 else 2e-3,
                "check_masked_logits": True,
            },
            path,
        )
        inputs[name] = {
            "sha256": shared.sha(path),
            "bytes": path.stat().st_size,
            "query_shape": list(q.shape),
            "config": config,
        }
        print("Prepared:", name, flush=True)
    shared.write_json(
        report / "inputs.json",
        {
            "inputs": inputs,
            "reference_commit": REFERENCE_COMMIT,
            "reference_sha256": REFERENCE_SHA256,
            "stock_ptxas": facts["ptxas"]["path"],
        },
    )


def score_arguments(payload, torch, variant):
    from tests.qsa_select_prefill_utils import vllm_reference

    torch.cuda.set_device(payload["device_index"])
    values = [
        x.to("cuda") if isinstance(x, torch.Tensor) else x for x in payload["args"]
    ]
    for value, stride in zip(values, payload["strides"]):
        if stride is not None:
            assert list(value.stride()) == stride, "input stride changed"
    q, cache, table, visible, length = values
    width = table.shape[1] * cache.shape[1]
    requests = q.shape[0] // length
    tiles = vllm_reference()._decode_tiles_per_program(requests, width)
    config = payload["config"]
    assert tiles == config["tiles_per_program"]
    assert requests == config["num_requests"] and width == config["columns"]
    assert tuple(q.shape) == (requests * length, config["heads"], config["head_dim"])
    assert table.shape[0] == requests and visible.shape == (q.shape[0],)
    assert q.dtype == cache.dtype
    logits = torch.full(
        (q.shape[0], width), -torch.inf, dtype=torch.float32, device=q.device
    )
    runtime = (
        q,
        cache,
        table,
        visible,
        logits,
        *q.stride()[:-1],
        *cache.stride()[:2],
        *table.stride()[:-1],
        *logits.stride()[:-1],
    )
    constants = {
        "PAGE_SIZE": cache.shape[1],
        "PAGE_TABLE_WIDTH": table.shape[1],
        "NUM_HEADS": q.shape[1],
        "HEAD_DIM": q.shape[2],
        "DECODE_QUERY_LEN": length,
        "BLOCK_N": BLOCK_N,
        "TILES_PER_PROG": tiles,
        "STAGES": 1 if variant == "loop-stage1" else 2,
    }
    options = {"num_warps": 1 if cache.dtype == torch.float8_e4m3fn else 2}
    grid = (requests, (width + BLOCK_N * tiles - 1) // (BLOCK_N * tiles))
    return runtime, constants, options, grid, logits, visible


def artifact_evidence(folder):
    ptx_path = folder / "kernel.ptx"
    if not ptx_path.exists():
        return None
    ptx = ptx_path.read_text()
    sass_path = folder / "kernel.sass.txt"
    sass = sass_path.read_text() if sass_path.exists() else ""
    return {
        "ptx_sha256": shared.sha(ptx_path),
        "async_l2_hint_count": len(
            re.findall(r"cp\.async\.[^\s;]*\.L2::cache_hint", ptx)
        ),
        "fractional_policy_count": ptx.count("createpolicy.fractional"),
        "policy_ptx": [line for line in ptx.splitlines() if "createpolicy" in line],
        "sass_available": sass_path.exists(),
        # Keep all uniform-register lines, including writes to UR pairs.
        # No automatic claim that a descriptor is uninitialized from a regex.
        "sass_uniform_register_lines": [
            line for line in sass.splitlines() if re.search(r"\bUR\d+\b", line)
        ],
    }


def classify(probes):
    def passed(name):
        item = probes.get(name, {})
        return item.get("returncode") == 0 and (item.get("evidence") or {}).get(
            "passed", False
        )

    def illegal(name):
        item = probes.get(name, {})
        evidence = item.get("evidence") or {}
        return (
            item.get("returncode") not in (None, 0)
            and "illegal instruction"
            in (str(evidence.get("error", "")) + item.get("log_tail", "")).lower()
        )

    if not passed("stock-original"):
        return "inconclusive: stock reference did not pass"
    if passed("flagtree-original"):
        return "not reproduced for this shape and toolchain"
    left = probes.get("flagtree-no-hint", {}).get("evidence") or {}
    right = probes.get("no-hint-cubin-stock", {}).get("evidence") or {}
    if (
        illegal("flagtree-original")
        and illegal("flagtree-cubin-stock")
        and passed("flagtree-no-hint")
        and passed("no-hint-cubin-stock")
        and left.get("ptx_edits", 0) > 0
        and left.get("cubin_sha256") == right.get("cubin_sha256")
    ):
        return (
            "reproduced: removing only async L2 cache hints resolves the binary fault"
        )
    return "inconclusive: inspect failure stages and compiler artifacts"


def run_probe(
    report,
    case,
    label,
    backend,
    runtime,
    variant="original",
    replay=None,
    assembler=None,
    ptx_mode="no-hint",
):
    name = case + "-" + label
    origin = report / replay if replay else None
    if origin and not (origin / "replay.json").exists():
        return {"name": name, "unavailable": "no producer binary"}
    if (
        assembler
        and ptx_mode == "no-hint"
        and not artifact_evidence(origin)["async_l2_hint_count"]
    ):
        return {
            "name": name,
            "not_applicable": "producer PTX has no async L2 cache hint",
        }
    env = shared.child_env(report, runtime if backend == "flagtree" else None, name)
    env["QSA_VLLM_REFERENCE_FILE"] = str(report / "qsa_indexer_vllm.py")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--report",
        str(report),
        "--name",
        name,
        "--case",
        case,
        "--variant",
        variant,
    ]
    if replay:
        command += ["--replay", replay]
    if assembler:
        command += [
            "--reassemble",
            "--assembler",
            assembler,
            "--ptx-cache-hint",
            ptx_mode,
        ]
    outcome = shared.run_job(report, name, command, env)
    state_path = report / (name + ".json")
    return {
        **outcome,
        "evidence": json.loads(state_path.read_text()) if state_path.exists() else {},
        "artifact_evidence": artifact_evidence(report / name),
        "log_tail": (report / (name + ".txt")).read_text()[-8000:],
    }


def flagtree_runtime():
    runtime = os.environ.get("QSA_TRITON_RUNTIME_PATH")
    if (
        not runtime
        or not Path(runtime).is_absolute()
        or not Path(runtime, "triton").is_dir()
    ):
        raise ValueError("set QSA_TRITON_RUNTIME_PATH to the FlagTree directory")
    return str(Path(runtime).resolve())


def new_report():
    report = Path(tempfile.mkdtemp(prefix="qsa-decode-h800."))
    print("Reports:", report, flush=True)
    for command, filename in (
        (["git", "rev-parse", "HEAD"], "commit.txt"),
        (["git", "status", "--short"], "worktree-status.txt"),
    ):
        (report / filename).write_text(
            subprocess.check_output(command, cwd=ROOT, text=True)
        )
    return report


def diagnose(args):
    runtime = flagtree_runtime()
    report = new_report()
    cases = case_matrix(args)
    shared.write_json(report / "cases.json", cases)
    script = [sys.executable, str(Path(__file__).resolve())]
    prepare_env = shared.child_env(report, None, "prepare")
    # Preserve an explicitly supplied frozen source only for preparation.
    if os.environ.get("QSA_VLLM_REFERENCE_FILE"):
        prepare_env["QSA_VLLM_REFERENCE_FILE"] = os.environ["QSA_VLLM_REFERENCE_FILE"]
    preparation = shared.run_job(
        report,
        "prepare",
        script + ["--prepare", "--report", str(report)],
        prepare_env,
    )
    if preparation["returncode"]:
        shared.write_json(
            report / "decode-diagnostic-summary.json",
            {"complete": False, "preparation": preparation, "cases": cases},
        )
        return 1
    manifest = json.loads((report / "inputs.json").read_text())
    results = {}
    for case, config in cases.items():
        probes = {}

        def run(
            label,
            backend,
            variant="original",
            replay=None,
            assembler=None,
            ptx_mode="no-hint",
        ):
            probes[label] = run_probe(
                report,
                case,
                label,
                backend,
                runtime,
                variant,
                replay,
                assembler,
                ptx_mode,
            )

        run("stock-original", "stock")
        run("flagtree-original", "flagtree")
        run("flagtree-loop-stage1", "flagtree", variant="loop-stage1")
        producer = case + "-flagtree-original"
        run("flagtree-cubin-stock", "stock", replay=producer)
        run(
            "identical-ptx-stock-assembler",
            "stock",
            replay=producer,
            assembler=manifest["stock_ptxas"],
            ptx_mode="original",
        )
        # One-variable PTX intervention uses the PRODUCER's ptxas and all
        # observed flags. Replay that very same patched CUBIN under Stock.
        runtime_path = report / producer / "runtime.json"
        if runtime_path.exists():
            assembler = json.loads(runtime_path.read_text())["ptxas"]["path"]
            run("flagtree-no-hint", "flagtree", replay=producer, assembler=assembler)
        else:
            probes["flagtree-no-hint"] = {"unavailable": "no producer runtime"}
        run("no-hint-cubin-stock", "stock", replay=case + "-flagtree-no-hint")
        results[case] = {
            "config": config,
            "input_sha256": manifest["inputs"][case]["sha256"],
            "conclusion": classify(probes),
            "probes": probes,
        }
        shared.write_json(
            report / "decode-diagnostic-summary.json",
            {
                "scope": "native decode score-only cache-policy diagnosis; no Top-K or timing",
                "complete": len(results) == len(cases),
                "fresh_cache_per_job": True,
                "reference_commit": manifest["reference_commit"],
                "reference_sha256": manifest["reference_sha256"],
                "cases": results,
            },
        )
        print(case + ": " + results[case]["conclusion"], flush=True)
    print("Return all top-level *.txt and *.json, including *-artifacts.json:", report)
    # Retain failures even when an explicitly labeled intervention succeeds.
    return int(
        any(
            probe.get("returncode") not in (None, 0)
            for result in results.values()
            for probe in result["probes"].values()
        )
    )


def verify_artifacts(folder):
    info = json.loads((folder / "replay.json").read_text())
    for name, digest in info["files"].items():
        assert Path(name).name == name, "unsafe artifact filename"
        assert shared.sha(folder / name) == digest, "corrupt artifact: " + name
    return info


def diagnose_replay(source_report):
    """Finish saved decode binary controls with a normalized 3D launch grid."""
    runtime = flagtree_runtime()
    source_report = source_report.resolve()
    report = new_report()
    manifest = json.loads((source_report / "inputs.json").read_text())
    cases = json.loads((source_report / "cases.json").read_text())
    previous = json.loads(
        (source_report / "decode-diagnostic-summary.json").read_text()
    )
    assert cases and previous["complete"], "need a completed original diagnostic"
    assert (
        shared.sha(source_report / "qsa_indexer_vllm.py")
        == manifest["reference_sha256"]
    )
    for name in ("inputs.json", "cases.json", "qsa_indexer_vllm.py"):
        shutil.copyfile(source_report / name, report / name)
    shutil.copyfile(
        source_report / "decode-diagnostic-summary.json",
        report / "previous-summary.json",
    )
    results = {}
    for case, config in cases.items():
        assert re.fullmatch(r"(?:bf16|fp8)-h\d+d\d+-q\d+-tiles[1248]", case)
        input_name = case + "-input.pt"
        assert (
            shared.sha(source_report / input_name) == manifest["inputs"][case]["sha256"]
        )
        shutil.copyfile(source_report / input_name, report / input_name)
        saved = {}
        for label in (
            "flagtree-original",
            "identical-ptx-stock-assembler",
            "flagtree-no-hint",
        ):
            origin = source_report / (case + "-" + label)
            verify_artifacts(origin)
            saved[label] = "saved-" + case + "-" + label
            folder = report / saved[label]
            shutil.copytree(origin, folder)
            shared.bundle(folder, report, saved[label])
        original = report / saved["flagtree-original"]
        patched = report / saved["flagtree-no-hint"]
        old_info, new_info = verify_artifacts(original), verify_artifacts(patched)
        for field in ("arg_names", "signature", "constants", "grid"):
            assert old_info[field] == new_info[field], "patched ABI changed: " + field
        assert (original / "kernel.json").read_bytes() == (
            patched / "kernel.json"
        ).read_bytes()
        expected_ptx, edits = shared.transform_cache_hint(
            (original / "kernel.ptx").read_text(), "no-hint"
        )
        assert (
            patched / "kernel.ptx"
        ).read_text() == expected_ptx, "unexpected PTX edits"
        probes = {}
        for label in ("stock-original", "flagtree-original", "flagtree-loop-stage1"):
            probes[label] = {**previous["cases"][case]["probes"][label]}
            evidence = probes[label]["evidence"]
            assert evidence["kernel"] == KERNEL
            assert evidence["case"] == case
            assert evidence["reference_sha256"] == manifest["reference_sha256"]
            assert evidence["input_sha256"] == manifest["inputs"][case]["sha256"]
            probes[label]["reused_from"] = str(source_report)
            for suffix in (".json", ".txt"):
                name = case + "-" + label + suffix
                shutil.copyfile(source_report / name, report / name)
        for label, backend, producer in (
            ("flagtree-cubin-stock", "stock", "flagtree-original"),
            ("identical-ptx-stock-assembler", "stock", "identical-ptx-stock-assembler"),
            ("flagtree-no-hint", "flagtree", "flagtree-no-hint"),
            ("no-hint-cubin-stock", "stock", "flagtree-no-hint"),
        ):
            probes[label] = run_probe(
                report, case, label, backend, runtime, replay=saved[producer]
            )
        intervention = probes["flagtree-no-hint"]["evidence"]
        intervention.update(
            ptx_intervention="no-hint",
            ptx_edits=edits,
            original_ptx_sha256=shared.sha(original / "kernel.ptx"),
            saved_binary_replay=True,
        )
        shared.write_json(report / (case + "-flagtree-no-hint.json"), intervention)
        results[case] = {
            "config": config,
            "input_sha256": manifest["inputs"][case]["sha256"],
            "conclusion": classify(probes),
            "probes": probes,
        }
        shared.write_json(
            report / "decode-diagnostic-summary.json",
            {
                "scope": "saved decode binary controls; original baseline results reused; no compilation or timing",
                "source_report": str(source_report),
                "baselines_reused": True,
                "new_probes_per_case": 4,
                "complete": len(results) == len(cases),
                "fresh_cache_per_job": True,
                "reference_commit": manifest["reference_commit"],
                "reference_sha256": manifest["reference_sha256"],
                "cases": results,
            },
        )
        print(case + ": " + results[case]["conclusion"], flush=True)
    print("Return all top-level *.txt and *.json, including *-artifacts.json:", report)
    return int(
        any(
            probe.get("returncode") not in (None, 0)
            for result in results.values()
            for probe in result["probes"].values()
            if "reused_from" not in probe
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtypes", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"]
    )
    parser.add_argument("--query-lengths", nargs="+", type=int, default=[3])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--tiles", nargs="+", type=int, choices=(1, 2, 4, 8), default=[1, 2, 4, 8]
    )
    parser.add_argument(
        "--replay-report",
        type=Path,
        help="finish binary controls using a previous decode report's raw inputs/artifacts",
    )
    parser.add_argument("--prepare", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--report", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--name", help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument(
        "--variant",
        choices=("original", "loop-stage1"),
        default="original",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--replay", help=argparse.SUPPRESS)
    parser.add_argument("--reassemble", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--assembler", help=argparse.SUPPRESS)
    parser.add_argument(
        "--ptx-cache-hint",
        choices=("original", "no-hint"),
        default="original",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if min(args.query_lengths + [args.heads, args.head_dim]) <= 0:
        parser.error("query lengths, heads and head dimension must be positive")
    if args.replay_report:
        return diagnose_replay(args.replay_report)
    if args.prepare:
        prepare(args.report.resolve())
        return 0
    if args.name:
        return shared.worker(args, argument_builder=score_arguments, kernel_name=KERNEL)
    return diagnose(args)


if __name__ == "__main__":
    sys.exit(main())
