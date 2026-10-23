# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Verified FlagTree/ptxas/driver configuration for the native QSA stage A/B."""

import argparse
import ctypes
import hashlib
import importlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

PTXAS_SHA256 = "9c2c084df7eb9be48f0502b6938ce2571679b2a78cad1c564b583bd7d9ea3c50"
LIBCUDA_SHA256 = "85a8cc8cf00419ceb295ec62c481d05b4cdbdab10f25c5375d6d24bdbe00b4d9"
OVERRIDES = (
    "TRITON_OVERRIDE_ARCH",
    "TRITON_PTXAS_BLACKWELL_PATH",
    "TRITON_MOCK_PTX_VERSION",
    "TRITON_PTX_VERSION",
    "TRITON_KERNEL_OVERRIDE",
    "TRITON_ALWAYS_COMPILE",
    "TRITON_STORE_BINARY_ONLY",
    "TRITON_CACHE_MANAGER",
    "TRITON_REMOTE_CACHE_BACKEND",
    "TRITON_DUMP_DIR",
    "DISABLE_PTXAS_OPT",
    "PTXAS_OPTIONS",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(directory):
    root = Path(directory).resolve()
    if not (root / "triton/__init__.py").is_file():
        raise RuntimeError("FlagTree package missing: " + str(root))
    if any(os.environ.get(key) for key in OVERRIDES):
        raise RuntimeError("Unexpected compiler override in benchmark environment")
    os.environ["QSA_TRITON_RUNTIME_PATH"] = str(root)
    # Explicit import routing under -I; never install/replace virtualenv packages.
    sys.path.insert(0, str(root))
    os.environ.pop("FLAGGEMS_FORCE_TLE", None)


def loaded_cuda(report):
    metadata = json.loads((Path(report) / "compat134.json").read_text())
    expected = Path(metadata["libcuda"]).resolve()
    if sha(expected) != LIBCUDA_SHA256 or metadata["libcuda_sha256"] != LIBCUDA_SHA256:
        raise RuntimeError("CUDA 13.4 compatibility library identity changed")
    mappings = sorted(
        {
            str(Path(line.split()[-1]).resolve())
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/libcuda.so" in line
        }
    )
    if mappings != [str(expected)]:
        raise RuntimeError("Loaded libcuda differs from the verified CUDA 13.4 library")
    library = ctypes.CDLL("libcuda.so.1")
    version = ctypes.c_int()
    function = library.cuDriverGetVersion
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    if function(ctypes.byref(version)) or version.value < 13040:
        raise RuntimeError("CUDA 13.4 driver API override ineffective")
    return dict(
        libcuda=str(expected), libcuda_sha256=sha(expected), driver_api=version.value
    )


def compiler_runtime(torch, triton, directory, report):
    root = Path(directory).resolve() / "triton"
    if not Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Torch must come from the selected virtualenv")
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if (name == "triton" or name.startswith("triton.")) and path:
            if not Path(path).resolve().is_relative_to(root):
                raise RuntimeError("Mixed Triton compiler module: " + name)
    files = {}
    for name in (
        "triton",
        "triton.testing",
        "triton.backends.nvidia.compiler",
        "triton.backends.nvidia.driver",
        "triton.knobs",
        "triton._C.libtriton",
        "triton.experimental.tle.language",
    ):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError("Compiler/TLE outside selected FlagTree: " + name)
        files[name] = dict(path=str(path), sha256=sha(path))
    compiler = importlib.import_module("triton.backends.nvidia.compiler")
    assembler = compiler.get_ptxas(90)
    assembler = Path(getattr(assembler, "path", assembler)).resolve()
    if assembler != Path(os.environ["TRITON_PTXAS_PATH"]).resolve():
        raise RuntimeError("ptxas override ineffective")
    if sha(assembler) != PTXAS_SHA256:
        raise RuntimeError("Expected verified NVIDIA ptxas 13.4.59")
    torch.cuda.init()
    return dict(
        profile="flagtree-cuda134",
        version=triton.__version__,
        files=files,
        ptxas=dict(path=str(assembler), sha256=sha(assembler)),
        cuda=loaded_cuda(report),
    )


def cooldown_gate(
    summary,
    report,
    env,
    label,
    guard,
    timeout,
    interval,
    shared_min_free_mib=None,
    max_initial_temp_c=75,
):
    """Cool outside timing; co-residency requires an explicit shared policy."""
    from tools import benchmark_qsa_prefill_stages as support

    deadline = time.monotonic() + timeout
    attempt = 0
    while True:
        attempt += 1
        if shared_min_free_mib is None:
            check = guard.idle(
                env.get("CUDA_VISIBLE_DEVICES", ""),
                support.command,
                max_initial_temp_c=max_initial_temp_c,
            )
        else:
            from tools import qsa_prefill_shared_gpu as shared

            check = shared.inspect_gpu(
                env.get("CUDA_VISIBLE_DEVICES", ""),
                shared_min_free_mib,
                max_initial_temp_c=max_initial_temp_c,
            )
        check.update(label=label, attempt=attempt)
        summary.setdefault(
            "idle_gpu_checks" if shared_min_free_mib is None else "shared_gpu_checks",
            [],
        ).append(check)
        if check["status"] == "passed":
            support.write_json(report / "stage-summary.json", summary)
            return True
        temperature = check.get("temperature_c")
        utilization = check.get("utilization_percent")
        allowed = {
            f"Initial temperature C {temperature} exceeds limit {max_initial_temp_c}",
            f"Initial utilization percent {utilization} exceeds limit 5",
        }
        # Non-waived process/memory rejections, active thermal slowdown and
        # unreadable telemetry reject immediately; only heat/utilization wait.
        retryable = (
            (check.get("processes") == [] or shared_min_free_mib is not None)
            and bool(check["reasons"])
            and set(check["reasons"]).issubset(allowed)
        )
        remaining = deadline - time.monotonic()
        check["cooldown_retryable"] = retryable
        if not retryable or remaining <= 0:
            if check.get("processes"):
                check["process_details"] = guard.process_details(
                    [p["pid"] for p in check["processes"]], support.command
                )
            check["cooldown_timed_out"] = retryable and remaining <= 0
            summary["environment_rejected"] = True
            summary.setdefault("accuracy", dict(status="not_run_environment_rejected"))
            support.write_json(report / "stage-summary.json", summary)
            print("GPU environment rejected:", check["reasons"], flush=True)
            return False
        wait = min(interval, remaining)
        check["cooldown_wait_seconds"] = wait
        support.write_json(report / "stage-summary.json", summary)
        print(
            f"GPU cooling; temperature={temperature}C, utilization={utilization}%; "
            f"waiting {wait:g}s for idle limits (temperature <= {max_initial_temp_c}C)",
            flush=True,
        )
        time.sleep(wait)


def select_gpu(summary, options=None):
    """Prefer a cool idle H800, preserving an explicit UUID and all idle checks."""
    from tools import benchmark_qsa_prefill_stages as support
    from tools import qsa_prefill_gpu_guard as guard
    from tools import qsa_prefill_shared_gpu as shared

    requested = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if guard.UUID_PATTERN.fullmatch(requested):
        summary.update(gpu_uuid=requested, gpu_selection_policy="explicit UUID")
        return True
    if requested not in ("", "auto"):
        raise ValueError("Select one full GPU UUID or CUDA_VISIBLE_DEVICES=auto")
    query = support.command(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    summary.update(
        gpu_inventory=query,
        gpu_selection_policy=(
            "lowest inventory temperature first; shared diagnostic memory/thermal checks"
            if shared.enabled(options)
            else "lowest inventory temperature first; configured idle guard required"
        ),
    )
    candidates = []
    for row in guard.rows(query):
        if len(row) != 3:
            raise ValueError("Malformed GPU inventory row: " + repr(row))
        uuid, name, temperature = [part.strip() for part in row]
        if "H800" not in name:
            continue
        temperature = int(temperature)
        if not guard.UUID_PATTERN.fullmatch(uuid) or not 0 <= temperature <= 150:
            raise ValueError("Invalid H800 UUID/temperature in GPU inventory")
        candidates.append((temperature, uuid))
    for temperature, uuid in sorted(candidates):
        check = (
            shared.inspect_gpu(
                uuid,
                options.shared_min_free_mib,
                max_initial_temp_c=getattr(options, "max_initial_temp_c", 75),
            )
            if shared.enabled(options)
            else guard.idle(
                uuid,
                support.command,
                max_initial_temp_c=getattr(options, "max_initial_temp_c", 75),
            )
        )
        check["inventory_temperature_c"] = temperature
        summary.setdefault("gpu_selection_checks", []).append(check)
        if check["status"] == "passed":
            os.environ["CUDA_VISIBLE_DEVICES"] = uuid
            summary["gpu_uuid"] = uuid
            print(
                f"Selected H800: {uuid}, inventory temperature={temperature}C, "
                f"policy={'shared exploratory' if shared.enabled(options) else 'exclusive idle'}",
                flush=True,
            )
            return True
    summary["environment_rejected"] = True
    return False


def prepare(options, extra):
    from tools import benchmark_qsa_prefill_stages as support
    from tools import validate_qsa_ptxas_h800 as validation

    report = Path(options.report_dir).resolve()
    try:
        for key in OVERRIDES:
            os.environ.pop(key, None)
        for key in ("TRITON_PTXAS_PATH", "CUDA_LAUNCH_BLOCKING"):
            os.environ.pop(key, None)
        facts = dict(status="preparing")
        if getattr(options, "shared_gpu_exploratory", False):
            facts.update(
                measurement_context="shared_gpu_exploratory", acceptance_eligible=False
            )
        if not select_gpu(facts, options):
            facts["status"] = "environment_rejected"
            support.write_json(report / "stage-summary.json", facts)
            return 2
        folder = Path(tempfile.mkdtemp(prefix="qsa-flagtree134-tools."))
        try:
            facts["toolchain"] = validation.toolchain(folder)
        finally:
            diagnostics = folder / "toolchain-downloads.json"
            if diagnostics.is_file():
                (report / diagnostics.name).write_bytes(diagnostics.read_bytes())
        if facts["toolchain"]["ptxas"]["sha256"] != PTXAS_SHA256:
            raise RuntimeError("Unexpected ptxas binary")
        facts["status"] = "ready"
        support.write_json(report / "flagtree134-tools.json", facts)
        env = os.environ.copy()
        env["TRITON_PTXAS_PATH"] = facts["toolchain"]["ptxas"]["path"]
        if getattr(options, "shared_gpu_exploratory", False):
            extra = [
                *extra,
                "--shared-gpu-exploratory",
                "--shared-min-free-mib",
                str(options.shared_min_free_mib),
            ]
        script = ROOT / "tools/benchmark_qsa_prefill_triton_stages.py"
        os.execve(
            sys.executable,
            [
                sys.executable,
                "-I",
                str(script),
                *extra,
                "--compiler",
                "flagtree",
                "--flagtree",
                str(Path(options.flagtree).resolve()),
                "--report-dir",
                str(report),
                "--max-initial-temp-c",
                str(getattr(options, "max_initial_temp_c", 75)),
            ],
            env,
        )
        return 0
    except BaseException:
        error = traceback.format_exc()
        support.write_json(
            report / "stage-summary.json", dict(status="failed", error=error)
        )
        print(error, file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--flagtree", default="/opt/flagtree")
    parser.add_argument("--shared-gpu-exploratory", action="store_true")
    parser.add_argument("--shared-min-free-mib", type=int, default=5120)
    parser.add_argument("--max-initial-temp-c", type=int, default=80)
    options, extra = parser.parse_known_args()
    if options.shared_min_free_mib <= 0:
        parser.error("Shared diagnostic minimum free memory must be positive")
    if not 1 <= options.max_initial_temp_c <= 150:
        parser.error("Initial temperature limit must be in 1..150 C")
    return prepare(options, extra)


if __name__ == "__main__":
    sys.exit(main())
