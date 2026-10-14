# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Collect native QSA score fault evidence; no Top-K, fixes, or timing."""

import argparse
import base64
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "h4d128": ((3,), (8192,), 4, 128, 0),
    "h1d32": ((3, 7), (1025, 1537), 1, 32, 19),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def tool_version(path):
    result = subprocess.run(
        [str(path), "--version"], capture_output=True, text=True, timeout=30
    )
    return {
        "path": str(path),
        "returncode": result.returncode,
        "text": result.stdout + result.stderr,
    }


def snapshot(folder, allow_gems=False):
    import torch
    import triton
    from triton.backends.nvidia.compiler import get_ptxas

    assert (
        "flaggems_vllm" in sys.modules
    ) == allow_gems, "unexpected FlagGems import state"
    assert torch.cuda.get_device_capability() == (9, 0), "H800/SM90 required"
    assert Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert torch.__version__ == importlib.metadata.version("torch")
    runtime = os.environ.get("QSA_TRITON_RUNTIME_PATH")
    prefix = Path(runtime) if runtime else Path(sys.prefix)
    assert Path(triton.__file__).resolve().is_relative_to(prefix.resolve())
    if not runtime:
        assert triton.__version__ == importlib.metadata.version("triton")
    ptxas = get_ptxas(90)
    ptxas = Path(getattr(ptxas, "path", ptxas)).resolve()
    facts = {
        "python": sys.executable,
        "pid": os.getpid(),
        "torch": torch.__version__,
        "torch_source": torch.__file__,
        "triton": triton.__version__,
        "triton_source": triton.__file__,
        "device": torch.cuda.get_device_name(),
        "device_index": torch.cuda.current_device(),
        "capability": torch.cuda.get_device_capability(),
        "ptxas": tool_version(ptxas),
        "environment": {
            k: os.environ.get(k)
            for k in (
                "PYTHONPATH",
                "LD_LIBRARY_PATH",
                "CUDA_VISIBLE_DEVICES",
                "CUDA_LAUNCH_BLOCKING",
                "TRITON_CACHE_DIR",
                "TRITON_OVERRIDE_ARCH",
                "TRITON_PTXAS_PATH",
                "DISABLE_PTXAS_OPT",
                "PTXAS_OPTIONS",
                "QSA_TRITON_RUNTIME_PATH",
            )
        },
        "compiler_files": {},
        "gems_imported": allow_gems,
    }
    facts["distributions"] = {}
    for package in ("torch", "triton", "flagtree", "vllm"):
        try:
            facts["distributions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            facts["distributions"][package] = None
    for name in (
        "triton.backends.nvidia.compiler",
        "triton.backends.nvidia.driver",
        "triton.knobs",
    ):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        filename = name + ".py.txt"
        shutil.copyfile(path, folder / filename)
        facts["compiler_files"][name] = {"source": str(path), "sha256": sha(path)}
    library = importlib.import_module("triton._C.libtriton")
    facts["libtriton"] = {"source": library.__file__, "sha256": sha(library.__file__)}
    write_json(folder / "runtime.json", facts)
    print(json.dumps(facts, indent=2), flush=True)
    return facts


def prepare(report):
    import torch

    from tests.qsa_select_prefill_utils import (
        REFERENCE_SHA256,
        make_inputs,
        torch_logits,
        vllm_reference,
    )

    facts = snapshot(report)
    reference = vllm_reference()
    assert sha(reference.__file__) == REFERENCE_SHA256
    shutil.copyfile(reference.__file__, report / "qsa_indexer_vllm.py")
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = {}
    for name, (lengths, contexts, heads, dim, base) in CASES.items():
        args = make_inputs(
            lengths, contexts, num_heads=heads, head_dim=dim, query_base=base
        )
        expected = torch_logits(*args[:5]).cpu()
        torch.cuda.synchronize()
        path = report / (name + "-input.pt")
        torch.save(
            {
                "args": [x.cpu() if isinstance(x, torch.Tensor) else x for x in args],
                "strides": [
                    list(x.stride()) if isinstance(x, torch.Tensor) else None
                    for x in args
                ],
                "expected": expected,
                "device_index": torch.cuda.current_device(),
            },
            path,
        )
        inputs[name] = {"sha256": sha(path), "query_shape": list(args[0].shape)}
    write_json(
        report / "inputs.json",
        {
            "inputs": inputs,
            "reference_sha256": REFERENCE_SHA256,
            "stock_ptxas": facts["ptxas"]["path"],
        },
    )


def score_arguments(payload, torch, variant):
    torch.cuda.set_device(payload["device_index"])
    args = [x.to("cuda") if isinstance(x, torch.Tensor) else x for x in payload["args"]]
    for value, stride in zip(args, payload["strides"]):
        if stride is not None:
            assert list(value.stride()) == stride, "input stride changed"
    q, cache, table, starts, visible = args[:5]
    width = table.shape[1] * cache.shape[1]
    logits = torch.full(
        (q.shape[0], width), -torch.inf, dtype=torch.float32, device=q.device
    )
    runtime = (
        q,
        cache,
        table,
        starts,
        visible,
        logits,
        *q.stride()[:-1],
        *cache.stride()[:2],
        *table.stride()[:-1],
        *logits.stride()[:-1],
        q.shape[0],
        0,
        table.shape[1],
    )
    tile = 16 if variant == "tile16" else 64
    constants = {
        "PAGE_SIZE": cache.shape[1],
        "NUM_HEADS": q.shape[1],
        "HEAD_DIM": q.shape[2],
        "TILE_R": tile,
        "BLOCK_N": 64,
        "K_TILES": 16,
        "STAGES": 1 if variant == "loop-stage1" else 2,
    }
    options = {"num_warps": 4}
    if variant == "compiler-stage1":
        options["num_stages"] = 1
    grid = (
        table.shape[0],
        (min(q.shape[0], args[7]) + tile - 1) // tile,
        (width + 1023) // 1024,
    )
    return runtime, constants, options, grid, logits, visible


def bundle(folder, report, name):
    entries = {}
    for path in sorted(folder.iterdir()):
        if path.is_file():
            data = path.read_bytes()
            entries[path.name] = {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "gzip_base64": base64.b64encode(gzip.compress(data, mtime=0)).decode(),
            }
    write_json(report / (name + "-artifacts.json"), entries)


def export_kernel(compiled, folder, fn, grid, assembler_commands):
    paths = {}
    for kind in ("ttir", "ttgir", "llir", "ptx", "cubin"):
        data = compiled.asm.get(kind)
        if data is None:
            continue
        path = folder / ("kernel." + kind)
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        paths[kind] = str(path)
    # Copy the original JSON exactly, including target, ABI and scratch fields.
    metadata = next(
        Path(p) for k, p in compiled.metadata_group.items() if k.endswith(".json")
    )
    shutil.copyfile(metadata, folder / "kernel.json")
    assert list(compiled.src.signature) == fn.arg_names
    constants = [[list(k), v] for k, v in compiled.src.constants.items()]
    write_json(
        folder / "replay.json",
        {
            "hash": compiled.hash,
            "arg_names": fn.arg_names,
            "signature": compiled.src.signature,
            "constants": constants,
            "grid": grid,
            "assembler_commands": assembler_commands,
            "files": {
                p.name: sha(p) for p in folder.iterdir() if p.name.startswith("kernel.")
            },
        },
    )
    try:
        (folder / "kernel.sass.txt").write_text(compiled.asm["sass"])
    except Exception as exc:
        (folder / "sass-unavailable.txt").write_text(str(exc) + "\n")
    ptx = compiled.asm["ptx"]
    return {
        "kernel_hash": compiled.hash,
        "ptx_sha256": sha(paths["ptx"]),
        "cubin_sha256": sha(paths["cubin"]),
        "metadata": json.loads((folder / "kernel.json").read_text()),
        "instructions": {
            op: ptx.count(op)
            for op in (
                "wgmma.",
                "mma.sync",
                "cp.async",
                "mbarrier",
                "tcgen05",
                "trap;",
            )
        },
    }


def replay_kernel(folder, fn):
    from triton.compiler.compiler import ASTSource, CompiledKernel

    info = json.loads((folder / "replay.json").read_text())
    for filename, expected in info["files"].items():
        assert sha(folder / filename) == expected, filename
    assert info["arg_names"] == fn.arg_names
    metadata = json.loads((folder / "kernel.json").read_text())
    assert metadata["num_ctas"] == 1
    assert tuple(metadata.get("cluster_dims", (1, 1, 1))) == (1, 1, 1)
    assert not metadata.get("launch_cooperative_grid", False)
    for field in ("global_scratch_size", "profile_scratch_size"):
        assert metadata.get(field, 0) == 0, "unsupported replay scratch ABI: " + field
    src = ASTSource(fn, info["signature"], {tuple(k): v for k, v in info["constants"]})
    group = {
        "kernel." + ext: str(folder / ("kernel." + ext))
        for ext in ("json", "ptx", "cubin")
    }
    return CompiledKernel(src, group, info["hash"]), info["grid"]


def reassemble(origin, folder, assembler):
    """Reuse the exact producer PTX/options, replacing only the assembler."""
    info = json.loads((origin / "replay.json").read_text())
    for filename, expected in info["files"].items():
        assert sha(origin / filename) == expected, filename
    commands = info["assembler_commands"]
    assert len(commands) == 1, "need one observed original ptxas invocation"
    command = list(commands[0])
    assert command.count("-o") == 1
    ptx_positions = [i for i, value in enumerate(command) if value.endswith(".ptx")]
    assert len(ptx_positions) == 1
    for filename in info["files"]:
        shutil.copyfile(origin / filename, folder / filename)
    command[0] = assembler
    command[ptx_positions[0]] = str(folder / "kernel.ptx")
    command[command.index("-o") + 1] = str(folder / "kernel.cubin")
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (folder / "reassembly.txt").write_text(
        repr(command) + "\n" + result.stdout + result.stderr
    )
    assert result.returncode == 0, "ptxas reassembly failed: " + result.stderr
    assert sha(folder / "kernel.ptx") == info["files"]["kernel.ptx"]
    info["files"]["kernel.cubin"] = sha(folder / "kernel.cubin")
    info["assembler_commands"] = [command]
    write_json(folder / "replay.json", info)
    return {
        "ptx_sha256": sha(folder / "kernel.ptx"),
        "cubin_sha256": sha(folder / "kernel.cubin"),
        "reassembly_command": command,
    }


def worker(args):
    report = args.report.resolve()
    folder = report / args.name
    folder.mkdir(exist_ok=True)
    state = {
        "name": args.name,
        "case": args.case,
        "variant": args.variant,
        "stage": "imports",
        "passed": False,
    }

    def mark(stage):
        state["stage"] = stage
        write_json(report / (args.name + ".json"), state)
        print("stage:", stage, flush=True)

    mark("imports")
    try:
        import torch

        if args.variant == "import-gems":
            importlib.import_module("flaggems_vllm")
        facts = snapshot(folder, allow_gems=args.variant == "import-gems")
        state["runtime"] = facts
        manifest = json.loads((report / "inputs.json").read_text())
        path = report / (args.case + "-input.pt")
        assert sha(path) == manifest["inputs"][args.case]["sha256"]
        state["input_sha256"] = sha(path)
        from tests.qsa_select_prefill_utils import vllm_reference

        reference = vllm_reference()
        assert sha(reference.__file__) == manifest["reference_sha256"]
        state["vllm_version"] = importlib.metadata.version("vllm")
        assert state["vllm_version"].split("+")[0] == "0.31.0"
        state["reference_sha256"] = sha(reference.__file__)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        mark("input preparation")
        runtime, constants, options, grid, logits, visible = score_arguments(
            payload, torch, args.variant
        )
        torch.cuda.synchronize()
        state.update(constants=constants, options=options, grid=grid)
        fn = reference._qsa_mqa_paged_prefill_kernel
        bound = [
            dict(zip(fn.arg_names, runtime), **constants)[name] for name in fn.arg_names
        ]
        if args.replay:
            mark("artifact replay construction")
            origin = report / args.replay
            if args.reassemble:
                mark("identical PTX reassembly")
                state.update(reassemble(origin, folder, manifest["stock_ptxas"]))
                origin = folder
                mark("artifact replay construction")
            compiled, original_grid = replay_kernel(origin, fn)
            assert list(grid) == original_grid
            if args.reassemble:
                try:
                    (folder / "kernel.sass.txt").write_text(compiled.asm["sass"])
                except Exception as exc:
                    (folder / "sass-unavailable.txt").write_text(str(exc) + "\n")
            state.update(
                replay_origin=args.replay,
                cubin_sha256=sha(origin / "kernel.cubin"),
                metadata=json.loads((origin / "kernel.json").read_text()),
            )
        else:
            mark("compile only")
            commands = []
            original_run = subprocess.run

            def observe(command, *positional, **keywords):
                if isinstance(command, (list, tuple)) and "-o" in command:
                    if Path(command[0]).resolve() == Path(facts["ptxas"]["path"]):
                        commands.append([str(value) for value in command])
                return original_run(command, *positional, **keywords)

            subprocess.run = observe
            try:
                compiled = fn.warmup(*runtime, **constants, **options, grid=grid)
            finally:
                subprocess.run = original_run
            assert compiled is not None
            state.update(export_kernel(compiled, folder, fn, grid, commands))
            if args.variant == "sm80":
                assert (
                    state["metadata"]["arch"] == "sm80"
                ), "architecture override ineffective"
                assert (
                    state["instructions"]["wgmma."] == 0
                ), "sm80 probe still contains WGMMA"
            if args.variant == "stock-ptxas":
                assert (
                    facts["ptxas"]["path"] == manifest["stock_ptxas"]
                ), "ptxas override ineffective"
        # Persist evidence before touching the CUDA binary or launch context.
        write_json(report / (args.name + ".json"), state)
        bundle(folder, report, args.name)
        mark("binary load / launcher initialization")
        compiled._init_handles()
        state["loaded"] = True
        mark("score launch")
        print("native score-only launch:", args.name, flush=True)
        if args.replay:
            compiled[grid](*bound)
        else:
            fn[grid](*runtime, **constants, **options)
        mark("score synchronize")
        torch.cuda.synchronize()
        mark("score validation")
        mask = (
            torch.arange(logits.shape[1], device=logits.device)[None, :]
            < visible[:, None]
        )
        expected = payload["expected"].to(logits.device)
        torch.testing.assert_close(logits[mask], expected[mask], rtol=2e-4, atol=2e-3)
        state.update(stage="passed", passed=True)
        print(
            "PASS: original score validated; no Top-K or Gems kernel executed",
            flush=True,
        )
        return 0
    except Exception as exc:
        state["error"] = repr(exc)
        traceback.print_exc()
        return 1
    finally:
        write_json(report / (args.name + ".json"), state)
        bundle(folder, report, args.name)


def child_env(report, runtime, name):
    env = os.environ.copy()
    env.update(
        PYTHONPATH=f"{ROOT / 'src'}:{ROOT}" + (":" + runtime if runtime else ""),
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        VLLM_PLUGINS="",
        DNN_VENDOR="nvidia",
        QSA_REQUIRE_VLLM_REFERENCE="1",
        CUDA_LAUNCH_BLOCKING="1",
        TRITON_CACHE_DIR=str(report / "caches" / name),
        TRITON_DUMP_DIR=str(report / "dumps" / name),
    )
    for key in (
        "PYTHONHOME",
        "QSA_NATIVE_REFERENCE_PYTHON",
        "QSA_NATIVE_REPORT_DIR",
        "QSA_TRITON_RUNTIME_PATH",
        "TRITON_OVERRIDE_ARCH",
        "TRITON_PTXAS_PATH",
        "TRITON_KERNEL_OVERRIDE",
        "TRITON_ALWAYS_COMPILE",
        "TRITON_STORE_BINARY_ONLY",
        "TRITON_CACHE_MANAGER",
        "TRITON_REMOTE_CACHE_BACKEND",
        "DISABLE_PTXAS_OPT",
        "PTXAS_OPTIONS",
    ):
        env.pop(key, None)
    if runtime:
        env["QSA_TRITON_RUNTIME_PATH"] = runtime
    return env


def run_job(report, name, command, env):
    print("Running:", name, flush=True)
    with (report / (name + ".txt")).open("w") as log:
        log.write("command: " + repr(command) + "\n")
        log.flush()
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=900,
            )
            code = result.returncode
        except subprocess.TimeoutExpired:
            log.write("\nTIMEOUT after 900 seconds\n")
            code = 124
    print(name, "returncode:", code, flush=True)
    return {"name": name, "returncode": code}


def find_tool(name):
    candidate = shutil.which(name)
    if candidate:
        return candidate
    return next(
        (
            str(p)
            for p in sorted(Path("/usr/local").glob("cuda*/bin/" + name))
            if os.access(p, os.X_OK)
        ),
        None,
    )


def diagnose():
    runtime = os.environ.get("QSA_TRITON_RUNTIME_PATH")
    if (
        not runtime
        or not Path(runtime).is_absolute()
        or not Path(runtime, "triton").is_dir()
    ):
        raise ValueError(
            "set QSA_TRITON_RUNTIME_PATH to the existing FlagTree directory"
        )
    runtime = str(Path(runtime).resolve())
    report = Path(tempfile.mkdtemp(prefix="qsa-prefill-h800."))
    print("Reports:", report, flush=True)
    for command, filename in (
        (["git", "rev-parse", "HEAD"], "commit.txt"),
        (["git", "status", "--short"], "worktree-status.txt"),
    ):
        (report / filename).write_text(
            subprocess.check_output(command, cwd=ROOT, text=True)
        )
    probe = [sys.executable, str(Path(__file__).resolve())]
    outcomes = [
        run_job(
            report,
            "prepare",
            probe + ["--prepare", "--report", str(report)],
            child_env(report, None, "prepare"),
        )
    ]
    if outcomes[0]["returncode"]:
        write_json(
            report / "score-diagnostic-summary.json",
            {"cases": outcomes, "complete": False},
        )
        return 1
    manifest = json.loads((report / "inputs.json").read_text())
    sanitizer = find_tool("compute-sanitizer")
    write_json(
        report / "sanitizer-availability.json",
        tool_version(sanitizer) if sanitizer else {"available": False},
    )
    for case in CASES:
        for backend, variant in (
            ("stock", "original"),
            ("flagtree", "original"),
            ("flagtree", "import-gems"),
            ("flagtree", "loop-stage1"),
            ("flagtree", "compiler-stage1"),
            ("flagtree", "tile16"),
            ("flagtree", "sm80"),
            ("flagtree", "stock-ptxas"),
        ):
            name = case + "-" + backend + "-" + variant
            env = child_env(report, runtime if backend == "flagtree" else None, name)
            env["QSA_VLLM_REFERENCE_FILE"] = str(report / "qsa_indexer_vllm.py")
            if variant == "sm80":
                env["TRITON_OVERRIDE_ARCH"] = "sm80"
            if variant == "stock-ptxas":
                env["TRITON_PTXAS_PATH"] = manifest["stock_ptxas"]
            command = probe + [
                "--report",
                str(report),
                "--name",
                name,
                "--case",
                case,
                "--variant",
                variant,
            ]
            outcomes.append(run_job(report, name, command, env))
        for producer, consumer in (("stock", "flagtree"), ("flagtree", "stock")):
            origin = case + "-" + producer + "-original"
            name = case + "-" + producer + "-cubin-via-" + consumer
            if not (report / origin / "replay.json").exists():
                outcomes.append(
                    {
                        "name": name,
                        "returncode": None,
                        "unavailable": "producer did not export a CUBIN",
                    }
                )
                continue
            env = child_env(report, runtime if consumer == "flagtree" else None, name)
            env["QSA_VLLM_REFERENCE_FILE"] = str(report / "qsa_indexer_vllm.py")
            command = probe + [
                "--report",
                str(report),
                "--name",
                name,
                "--case",
                case,
                "--replay",
                origin,
            ]
            outcomes.append(run_job(report, name, command, env))
        origin = case + "-flagtree-original"
        name = case + "-identical-ptx-stock-assembler"
        if (report / origin / "replay.json").exists():
            env = child_env(report, None, name)
            env["QSA_VLLM_REFERENCE_FILE"] = str(report / "qsa_indexer_vllm.py")
            command = probe + [
                "--report",
                str(report),
                "--name",
                name,
                "--case",
                case,
                "--replay",
                origin,
                "--reassemble",
            ]
            outcomes.append(run_job(report, name, command, env))
        else:
            outcomes.append(
                {
                    "name": name,
                    "returncode": None,
                    "unavailable": "producer did not export a CUBIN",
                }
            )
        if sanitizer:
            for tool in ("memcheck", "synccheck"):
                name = case + "-flagtree-" + tool
                env = child_env(report, runtime, name)
                env["QSA_VLLM_REFERENCE_FILE"] = str(report / "qsa_indexer_vllm.py")
                command = (
                    [
                        sanitizer,
                        "--tool",
                        tool,
                        "--error-exitcode",
                        "99",
                        "--show-backtrace",
                        "yes",
                    ]
                    + probe
                    + ["--report", str(report), "--name", name, "--case", case]
                )
                outcomes.append(run_job(report, name, command, env))
    states = []
    for item in outcomes[1:]:
        path = report / (item["name"] + ".json")
        states.append(
            {
                **item,
                "evidence": json.loads(path.read_text()) if path.exists() else None,
            }
        )
    write_json(
        report / "score-diagnostic-summary.json",
        {
            "scope": "native BF16 score fault diagnosis; original and explicitly labeled ablations; no Top-K/timing",
            "fresh_cache_per_job": True,
            "complete": True,
            "sanitizer_available": bool(sanitizer),
            "cases": states,
        },
    )
    print(
        "Return all top-level text/JSON files (IR/CUBIN bundles included):",
        report,
        flush=True,
    )
    return int(any(item["returncode"] != 0 for item in outcomes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--report", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--name", help=argparse.SUPPRESS)
    parser.add_argument(
        "--case", choices=tuple(CASES), default="h4d128", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--variant",
        default="original",
        choices=(
            "original",
            "import-gems",
            "loop-stage1",
            "compiler-stage1",
            "tile16",
            "sm80",
            "stock-ptxas",
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--replay", help=argparse.SUPPRESS)
    parser.add_argument("--reassemble", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.report.resolve())
        return 0
    if args.name:
        return worker(args)
    return diagnose()


if __name__ == "__main__":
    sys.exit(main())
