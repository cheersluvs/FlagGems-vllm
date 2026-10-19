# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Minimal policy-free CUBIN controls for the ptxas 13.4 loader crashes."""

import argparse
import ctypes
import faulthandler
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src"), str(ROOT)]

from tools import validate_qsa_ptxas_h800 as common  # noqa: E402

shared, support, guard = common.shared, common.support, common.guard


def ptx(entry):
    prefix = ".version 8.7\n.target sm_90a\n.address_size 64\n"
    if entry == "empty":
        return prefix + ".visible .entry loader_empty() { ret; }\n"
    if entry == "write":
        return (
            prefix
            + """.visible .entry loader_write(.param .u64 output) {
.reg .b64 %rd1;
.reg .b32 %r1;
ld.param.u64 %rd1, [output];
mov.u32 %r1, 123;
st.global.u32 [%rd1], %r1;
ret;
}
"""
        )
    raise ValueError(entry)


def driver_call(library, name, argtypes, *args):
    fn = getattr(library, name)
    fn.argtypes = argtypes
    fn.restype = ctypes.c_int
    status = fn(*args)
    if status:
        raise RuntimeError(f"{name} returned CUDA status {status}")


def load(options, report):
    # A signal in libcuda must preserve the last API boundary and Python stack.
    faulthandler.enable(all_threads=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    state = dict(
        passed=False,
        stage="imports",
        entry=options.entry,
        binary=options.binary,
        triton_loader_used=False,
    )
    path = report / (options.binary + ".json")

    def mark(stage):
        state["stage"] = stage
        shared.write_json(path, state)
        print("stage:", stage, flush=True)

    mark("imports")
    try:
        import torch

        torch.cuda.init()
        assert torch.cuda.get_device_capability() == (9, 0)
        library = ctypes.CDLL("libcuda.so.1")
        driver_version = ctypes.c_int()
        driver_call(
            library,
            "cuDriverGetVersion",
            [ctypes.POINTER(ctypes.c_int)],
            ctypes.byref(driver_version),
        )
        mappings = Path("/proc/self/maps").read_text().splitlines()
        state.update(
            driver_api_version=driver_version.value,
            libcuda_paths=sorted(
                {line.split()[-1] for line in mappings if "/libcuda.so" in line}
            ),
            ld_library_path=os.environ.get("LD_LIBRARY_PATH"),
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
            torch_version=torch.__version__,
            torch_source=torch.__file__,
            nvidia_smi=support.command(["nvidia-smi"]),
        )
        output = torch.empty((1,), dtype=torch.int32, device="cuda")
        binary = report / (options.binary + ".cubin")
        manifest = json.loads((report / "binaries.json").read_text())
        assert shared.sha(binary) == manifest[options.binary]["cubin_sha256"]
        state.update(
            binary_sha256=shared.sha(binary), elf_ident=binary.read_bytes()[:16].hex()
        )
        compat_record = report / "compat134.json"
        if compat_record.is_file():
            compat = json.loads(compat_record.read_text())
            state["requested_compat"] = compat
            assert (
                driver_version.value >= 13040
            ), "CUDA 13.4 compat override ineffective"
            assert any(
                Path(p).resolve() == Path(compat["libcuda"]).resolve()
                for p in state["libcuda_paths"]
            ), "Loaded libcuda differs from the requested isolated library"
        data = ctypes.create_string_buffer(binary.read_bytes())
        module, function = ctypes.c_void_p(), ctypes.c_void_p()
        mark("cuModuleLoadData")
        driver_call(
            library,
            "cuModuleLoadData",
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
            ctypes.byref(module),
            ctypes.cast(data, ctypes.c_void_p),
        )
        mark("cuModuleGetFunction")
        driver_call(
            library,
            "cuModuleGetFunction",
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
            ctypes.byref(function),
            module,
            ("loader_" + options.entry).encode(),
        )
        pointer = ctypes.c_uint64(output.data_ptr())
        parameters = (ctypes.c_void_p * 1)(ctypes.addressof(pointer))
        mark("cuLaunchKernel")
        driver_call(
            library,
            "cuLaunchKernel",
            [ctypes.c_void_p]
            + [ctypes.c_uint] * 7
            + [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
            ],
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            None,
            parameters if options.entry == "write" else None,
            None,
        )
        mark("cuCtxSynchronize")
        driver_call(library, "cuCtxSynchronize", [])
        if options.entry == "write":
            assert output.item() == 123, "Minimal store result mismatch"
        mark("cuModuleUnload")
        driver_call(library, "cuModuleUnload", [ctypes.c_void_p], module)
        state.update(stage="passed", passed=True)
        shared.write_json(path, state)
        return 0
    except BaseException:
        state["error"] = traceback.format_exc()
        shared.write_json(path, state)
        print(state["error"], file=sys.stderr)
        return 1


def prepare(options, report):
    facts = shared.snapshot(report)
    old = facts["ptxas"]["path"]
    binaries = {}
    for entry in ("empty", "write"):
        source = report / (entry + ".ptx")
        source.write_text(ptx(entry))
        for label, assembler in (("old", old), ("new", options.assembler)):
            name = entry + "-" + label
            target = report / (name + ".cubin")
            command = [assembler, "--gpu-name=sm_90a", str(source), "-o", str(target)]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
            (report / (name + "-assembly.txt")).write_text(
                repr(command) + "\n" + proc.stdout + proc.stderr
            )
            if proc.returncode:
                raise RuntimeError("Minimal assembly failed: " + name)
            binaries[name] = dict(
                ptx_sha256=shared.sha(source),
                cubin_sha256=shared.sha(target),
                elf_ident=target.read_bytes()[:16].hex(),
                command=command,
            )
    shared.write_json(report / "binaries.json", binaries)
    # Binary and PTX bytes are uploaded in the existing compressed bundle format.
    bundle = report / "binary-bundle"
    bundle.mkdir()
    for name in ("empty.ptx", "write.ptx", *(key + ".cubin" for key in binaries)):
        common.shutil.copyfile(report / name, bundle / name)
    shared.bundle(bundle, report, "minimal-binaries")
    return 0


def classify(probes):
    def passed(name):
        return probes[name]["returncode"] == 0 and probes[name].get("state", {}).get(
            "passed"
        )

    if not all(passed(k + "-old") for k in ("empty", "write")):
        return "inconclusive_old_control_failed"
    if all(passed(k + "-new") for k in ("empty", "write")):
        return "minimal_new_cubin_passed_qsa_specific_investigation_required"
    if any(
        probes[k + "-new"].get("state", {}).get("stage") == "cuModuleLoadData"
        for k in ("empty", "write")
    ):
        return "policy_free_new_cubin_load_failed_independently_of_triton"
    return "policy_free_new_cubin_failed_inspect_api_stage"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument(
        "--flagtree", help=argparse.SUPPRESS
    )  # shared runner compatibility
    parser.add_argument("--prepare", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--assembler", help=argparse.SUPPRESS)
    parser.add_argument("--entry", choices=("empty", "write"), help=argparse.SUPPRESS)
    parser.add_argument("--binary", help=argparse.SUPPRESS)
    options = parser.parse_args()
    report = Path(options.report_dir).resolve()
    if options.prepare:
        return prepare(options, report)
    if options.binary:
        return load(options, report)
    report.mkdir(parents=True, exist_ok=True)
    if any(p.name not in {"runner.txt", "compat134.json"} for p in report.iterdir()):
        parser.error("Use a fresh report directory")
    summary = dict(
        scope="minimal policy-free PTX; direct CUDA driver calls; no FlagTree or QSA kernels",
        status="running",
        probes={},
        code=support.command(["git", "rev-parse", "HEAD"]),
    )
    summary_path = report / "ptxas-summary.json"
    try:
        if not common.select_gpu(summary):
            summary["status"] = "environment_rejected"
            return 2
        tools = common.toolchain(Path(tempfile.mkdtemp(prefix="qsa-loader-tools.")))
        summary["toolchain"] = tools
        shared.write_json(report / "toolchain.json", tools)
        script = [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            "--report-dir",
            str(report),
        ]

        def job(name, command):
            if not support.idle_gpu_gate(summary, report, os.environ, name, guard):
                raise EnvironmentError("GPU environment rejected")
            return shared.run_job(
                report, name, command, common.environment(report, None, name)
            )

        prep = job(
            "prepare", script + ["--prepare", "--assembler", tools["ptxas"]["path"]]
        )
        summary["prepare"] = prep
        if prep["returncode"]:
            raise RuntimeError("Minimal CUBIN preparation failed; see prepare.txt")
        summary["binaries"] = json.loads((report / "binaries.json").read_text())
        for entry in ("empty", "write"):
            for label in ("old", "new"):
                name = entry + "-" + label
                result = job(name, script + ["--entry", entry, "--binary", name])
                path = report / (name + ".json")
                result["state"] = json.loads(path.read_text()) if path.is_file() else {}
                if result["returncode"] < 0:
                    result["signal"] = signal.Signals(-result["returncode"]).name
                summary["probes"][name] = result
                shared.write_json(summary_path, summary)
        summary["status"] = classify(summary["probes"])
        return 0 if all(p["returncode"] == 0 for p in summary["probes"].values()) else 1
    except BaseException:
        summary.update(
            status=(
                "environment_rejected"
                if summary.get("environment_rejected")
                else "failed"
            ),
            error=traceback.format_exc(),
        )
        print(summary["error"], file=sys.stderr)
        return 2 if summary.get("environment_rejected") else 1
    finally:
        shared.write_json(summary_path, summary)
        print("Reports:", report, flush=True)


if __name__ == "__main__":
    sys.exit(main())
