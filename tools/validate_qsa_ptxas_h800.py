# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Unmodified native QSA/FlagTree, identical PTX, old vs NVIDIA ptxas 13.4.59.

Score-only correctness probes, not a benchmark or full vLLM acceptance suite.
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback
import types
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src"), str(ROOT)]

import diagnose_qsa_native_decode_h800 as decode  # noqa: E402
import diagnose_qsa_native_score_h800 as shared  # noqa: E402

from tools import benchmark_qsa_prefill_stages as support  # noqa: E402
from tools import qsa_prefill_gpu_guard as guard  # noqa: E402
from tools import qsa_prefill_triton_ab as reference  # noqa: E402

REDIST = "https://developer.download.nvidia.com/compute/cuda/redist/"
TOOLS = {
    "ptxas": (
        "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-13.4.59-archive.tar.xz",
        "0c08d1df80b5d0bd081778d392446ab50a6a54b047c0136b39c8dbfdf2cfed3f",
    ),
    "nvdisasm": (
        "cuda_nvdisasm/linux-x86_64/cuda_nvdisasm-linux-x86_64-13.4.49-archive.tar.xz",
        "622619236e0965fdb7a4407d5899afd1560c6f26e9d06a16cdeb2fe58af2e1f8",
    ),
}
PROBES = (
    "stock",
    "flagtree-old",
    "identical-ptx-new",
    "new-cubin-stock",
    "fresh-jit-new",
)


def download(url, path):
    request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=120) as src, path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
        return dict(
            response_url=src.geturl(),
            http_status=getattr(src, "status", None),
            content_type=src.headers.get("Content-Type"),
            content_length=src.headers.get("Content-Length"),
            content_range=src.headers.get("Content-Range"),
        )


def cached_archives(name):
    # Previous outputs of this exact wrapper only; every reuse requires the
    # original archive digest, independently of path, mtime and filename.
    return sorted(
        Path(tempfile.gettempdir()).glob(f"qsa-flagtree134-tools.*/{name}.tar.xz"),
        reverse=True,
    )


def prepare_archive(archive, name, url, expected):
    record = archive.parent / "toolchain-downloads.json"
    state = json.loads(record.read_text()) if record.exists() else {}
    item = dict(expected_sha256=expected, url=url, cached_candidates=[], attempts=[])
    state[name] = item

    def save():
        record.write_text(json.dumps(state, indent=2) + "\n")

    def inspect(path):
        return dict(bytes=path.stat().st_size, actual_sha256=shared.sha(path))

    save()
    for source in cached_archives(name):
        if source.resolve() == archive.resolve():
            continue
        entry = dict(path=str(source), status="rejected")
        item["cached_candidates"].append(entry)
        try:
            entry.update(inspect(source))
            if entry["actual_sha256"] != expected:
                continue
            shutil.copyfile(source, archive)
            if shared.sha(archive) != expected:
                raise RuntimeError("Copied NVIDIA archive SHA256 mismatch")
            entry["status"] = "verified"
            item.update(
                status="verified",
                source_kind="verified_local_cache",
                source_path=str(source),
            )
            print(
                f"Reusing SHA256-verified NVIDIA {name} archive: {source}", flush=True
            )
            return item
        except Exception as error:
            entry["error"] = str(error)
        finally:
            save()
    for attempt in range(1, 4):
        entry = dict(attempt=attempt, status="failed")
        item["attempts"].append(entry)
        try:
            archive.unlink(missing_ok=True)
            response = download(url, archive)
            if isinstance(response, dict):
                entry.update(response)
            entry.update(inspect(archive))
            if entry["actual_sha256"] != expected:
                raise RuntimeError(
                    f"NVIDIA archive SHA256 mismatch: {name}; expected={expected}, "
                    f"actual={entry['actual_sha256']}, bytes={entry['bytes']}"
                )
            entry["status"] = "verified"
            item.update(status="verified", source_kind="download", source_path=url)
            return item
        except Exception as error:
            entry["error"] = str(error)
            if archive.exists():
                entry.update(inspect(archive))
            print(
                f"NVIDIA {name} download attempt {attempt}/3 failed: {error}",
                flush=True,
            )
        finally:
            save()
    item["status"] = "failed"
    save()
    raise RuntimeError(
        f"Verified NVIDIA {name} archive unavailable; see toolchain-downloads.json"
    )


def extract_tool(archive, target, name, expected):
    if shared.sha(archive) != expected:
        raise RuntimeError("NVIDIA archive SHA256 mismatch: " + name)
    # Extract only the named regular executable; never unpack archive paths.
    with tarfile.open(archive, "r:xz") as package:
        entries = [m for m in package.getmembers() if m.name.endswith("/bin/" + name)]
        if len(entries) != 1 or not entries[0].isfile():
            raise RuntimeError("Missing unique regular executable: " + name)
        with package.extractfile(entries[0]) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    target.chmod(0o755)


def toolchain(folder):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This toolchain package requires Linux x86_64")
    facts = {}
    for name, (relative, digest) in TOOLS.items():
        archive = folder / (name + ".tar.xz")
        acquisition = prepare_archive(archive, name, REDIST + relative, digest)
        target = folder / name
        extract_tool(archive, target, name, digest)
        version = shared.tool_version(target)
        if version["returncode"] or (
            name == "ptxas" and "V13.4.59" not in version["text"]
        ):
            raise RuntimeError("NVIDIA tool could not run: " + repr(version))
        facts[name] = dict(
            version,
            sha256=shared.sha(target),
            url=REDIST + relative,
            archive_sha256=digest,
            acquisition=acquisition,
        )
    return facts


def environment(directory, runtime, name, assembler=None, disassembler=None):
    env = shared.child_env(directory, runtime, name)
    env.pop("PYTHONPATH", None)  # Children use -I and explicit source directories.
    env.pop("TRITON_PTXAS_BLACKWELL_PATH", None)
    env.pop("TRITON_MOCK_PTX_VERSION", None)
    env.pop("TRITON_PTX_VERSION", None)
    env["QSA_VLLM_REFERENCE_FILE"] = str(directory / "qsa_indexer_vllm.py")
    if assembler:
        env["TRITON_PTXAS_PATH"] = assembler
    if disassembler:
        env["QSA_PROBE_NVDISASM"] = disassembler
    return env


def child(options):
    directory = Path(options.report_dir)
    runtime = os.environ.get("QSA_TRITON_RUNTIME_PATH")
    if runtime:
        sys.path.insert(0, runtime)
    if options.prepare:
        if options.kind == "decode":
            decode.prepare(directory)
        else:
            shared.prepare(directory)
        return 0
    args = types.SimpleNamespace(
        report=directory,
        name=options.name,
        case=options.case,
        variant="original",
        replay=options.replay,
        reassemble=bool(options.assembler),
        assembler=options.assembler,
        ptx_cache_hint="original",
    )
    code = shared.worker(
        args,
        argument_builder=(
            decode.score_arguments
            if options.kind == "decode"
            else shared.score_arguments
        ),
        kernel_name=(
            decode.KERNEL
            if options.kind == "decode"
            else "_qsa_mqa_paged_prefill_kernel"
        ),
    )
    folder = directory / options.name
    binary = folder / "kernel.cubin"
    if binary.is_file():
        disassembler = os.environ["QSA_PROBE_NVDISASM"]
        proc = subprocess.run(
            [disassembler, str(binary)], capture_output=True, text=True, timeout=120
        )
        (folder / "nvdisasm.txt").write_text(proc.stderr)
        if proc.returncode == 0:
            (folder / "kernel.sass.txt").write_text(proc.stdout)
        shared.bundle(folder, directory, options.name)
    return code


def passed(probe):
    return probe.get("returncode") == 0 and probe.get("state", {}).get("passed") is True


def assess(probes):
    old = probes["flagtree-old"]
    if not passed(probes["stock"]):
        return "inconclusive_stock_failed"
    artifact = old.get("artifact") or {}
    if not artifact.get("async_l2_hint_count") or not artifact.get(
        "fractional_policy_count"
    ):
        return "inconclusive_policy_not_emitted"
    error = str(old.get("state", {}).get("error", "")) + old.get("log_tail", "")
    if passed(old) or "illegal instruction" not in error.lower():
        return "inconclusive_old_failure_not_reproduced"
    replay = probes["identical-ptx-new"]
    new_state = replay.get("state", {})
    if not (passed(replay) and passed(probes["new-cubin-stock"])):
        return "new_assembler_not_validated"
    if not (
        new_state.get("ptx_edits") == 0
        and new_state.get("ptx_intervention") == "original"
        and new_state.get("original_ptx_sha256") == artifact.get("ptx_sha256")
        and new_state.get("ptx_sha256") == artifact.get("ptx_sha256")
        and new_state.get("cubin_sha256")
        == probes["new-cubin-stock"].get("state", {}).get("cubin_sha256")
    ):
        return "inconclusive_artifact_identity"
    if passed(probes["fresh-jit-new"]):
        fresh_artifact = probes["fresh-jit-new"].get("artifact") or {}
        if not fresh_artifact.get("async_l2_hint_count") or not fresh_artifact.get(
            "fractional_policy_count"
        ):
            return "assembler_fix_validated_fresh_policy_missing"
        return "config_only_validated"
    return "assembler_fix_validated_fresh_jit_failed"


def collect(directory, name, outcome):
    path = directory / (name + ".json")
    folder = directory / name
    state = json.loads(path.read_text()) if path.is_file() else {}
    evidence = decode.artifact_evidence(folder)
    if evidence:
        ptx = (folder / "kernel.ptx").read_text()
        match = re.search(r"\.version\s+(\d+\.\d+)", ptx)
        evidence["ptx_version"] = match[1] if match else None
    return dict(
        outcome,
        state=state,
        artifact=evidence,
        log_tail=(directory / (name + ".txt")).read_text()[-5000:],
    )


def select_gpu(summary):
    requested = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if guard.UUID_PATTERN.fullmatch(requested):
        summary["gpu_uuid"] = requested
        return True
    if requested not in ("", "auto"):
        raise ValueError(
            "Select one full GPU UUID, or unset CUDA_VISIBLE_DEVICES for automatic selection"
        )
    query = support.command(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader,nounits"]
    )
    summary["gpu_inventory"] = query
    for row in guard.rows(query):
        if len(row) != 2 or "H800" not in row[1]:
            continue
        uuid = row[0].strip()
        check = guard.idle(uuid, support.command)
        summary.setdefault("gpu_selection_checks", []).append(check)
        if check["status"] == "passed":
            os.environ["CUDA_VISIBLE_DEVICES"] = uuid
            summary["gpu_uuid"] = uuid
            print("Selected idle H800:", uuid, flush=True)
            return True
    summary["environment_rejected"] = True
    return False


def check_identity(probes, assembler, expected_runtime, directory):
    stock = probes["stock"].get("state", {}).get("runtime")
    if not stock:
        return
    for label in PROBES:
        state = probes[label].get("state", {})
        facts = state.get("runtime")
        if not facts:
            continue  # Failed import remains an explicit failed/inconclusive probe.
        if (
            facts["torch_source"] != stock["torch_source"]
            or facts["torch"] != stock["torch"]
        ):
            raise RuntimeError("Torch identity changed")
        if label not in ("stock", "new-cubin-stock"):
            if (
                not Path(facts["triton_source"])
                .resolve()
                .is_relative_to(Path(expected_runtime))
            ):
                raise RuntimeError("FlagTree compiler path changed")
            old = probes["flagtree-old"].get("state", {}).get("runtime")
            if old is None:
                continue
            for field in ("triton_source", "triton", "compiler_files", "libtriton"):
                if facts[field] != old[field]:
                    raise RuntimeError(
                        "FlagTree compiler source/binary changed: " + field
                    )
        if label in ("identical-ptx-new", "fresh-jit-new"):
            # A replay assembles explicitly; a fresh JIT must select the override.
            actual = (state.get("reassembly_command") or [facts["ptxas"]["path"]])[0]
            if Path(actual).resolve() != Path(assembler).resolve():
                raise RuntimeError("New ptxas override was ineffective")
    for label in PROBES:
        if passed(probes[label]):
            assert (
                probes[label]["state"]["input_sha256"]
                == probes["stock"]["state"]["input_sha256"]
            )
            assert (
                probes[label]["state"]["reference_sha256"] == reference.REFERENCE_SHA256
            )
    a = directory / (probes["flagtree-old"]["name"])
    b = directory / (probes["identical-ptx-new"]["name"])
    if (b / "kernel.ptx").exists():
        if (a / "kernel.ptx").read_bytes() != (b / "kernel.ptx").read_bytes():
            raise RuntimeError("Identical-PTX experiment modified PTX bytes")


def orchestrate(options, directory):
    summary = dict(
        status="running",
        scope="native QSA score-only assembler/configuration validation; no Top-K or timing",
        cases=[],
        toolchain={},
        code=support.command(["git", "rev-parse", "HEAD"]),
        worktree=support.command(["git", "status", "--short"]),
        reference_sha256=reference.REFERENCE_SHA256,
        sources={
            name: shared.sha(ROOT / name)
            for name in (
                "tools/validate_qsa_ptxas_h800.py",
                "tools/diagnose_qsa_native_score_h800.py",
                "tools/diagnose_qsa_native_decode_h800.py",
                "tests/qsa_select_prefill_utils.py",
                "src/flaggems_vllm/ops/qsa_select_paged_prefill.py",
            )
        },
    )
    summary_path = directory / "ptxas-summary.json"
    shared.write_json(summary_path, summary)
    try:
        runtime = Path(options.flagtree).resolve()
        if not (runtime / "triton/__init__.py").is_file():
            raise RuntimeError("FlagTree directory not found: " + str(runtime))
        if not select_gpu(summary):
            summary["status"] = "environment_rejected"
            return 2
        tools_folder = Path(tempfile.mkdtemp(prefix="qsa-nvidia134-tools."))
        summary["toolchain"] = toolchain(tools_folder)
        assembler = summary["toolchain"]["ptxas"]["path"]
        disassembler = summary["toolchain"]["nvdisasm"]["path"]
        shared.write_json(directory / "toolchain.json", summary["toolchain"])
        native = directory / "qsa_indexer_vllm.py"
        download(reference.REFERENCE_URL, native)
        if shared.sha(native) != reference.REFERENCE_SHA256:
            raise RuntimeError("Pinned upstream native QSA SHA256 mismatch")
        (directory / "native-source.txt").write_text(native.read_text())
        source = str(Path(__file__).resolve())

        def run(
            kind,
            folder,
            name,
            *,
            case=None,
            backend=None,
            replay=None,
            reassemble=False,
            prepare=False,
        ):
            if not support.idle_gpu_gate(summary, directory, os.environ, name, guard):
                raise EnvironmentError("GPU environment rejected before " + name)
            env = environment(
                folder,
                str(runtime) if backend == "flagtree" else None,
                name,
                assembler if name.endswith("fresh-jit-new") else None,
                disassembler,
            )
            command = [
                sys.executable,
                "-I",
                source,
                "--report-dir",
                str(folder),
                "--kind",
                kind,
            ]
            if prepare:
                # Both prepare helpers copy the loaded reference into folder.
                # Load the root file here; workers load the prepared copy.
                env["QSA_VLLM_REFERENCE_FILE"] = str(native)
                command += ["--prepare"]
            else:
                command += ["--name", name, "--case", case]
                if replay:
                    command += ["--replay", replay]
                if reassemble:
                    command += ["--assembler", assembler]
            outcome = shared.run_job(folder, name, command, env)
            return outcome if prepare else collect(folder, name, outcome)

        for kind in ("prefill", "decode"):
            folder = directory / kind
            folder.mkdir()
            if kind == "decode":
                matrix = decode.case_matrix(
                    types.SimpleNamespace(
                        heads=4,
                        head_dim=128,
                        dtypes=["bf16", "fp8"],
                        query_lengths=[3],
                        tiles=[1, 2, 4, 8],
                    )
                )
                shared.write_json(folder / "cases.json", matrix)
                cases = list(matrix)
            else:
                cases = list(shared.CASES)
            prep = run(kind, folder, "prepare", prepare=True)
            if prep["returncode"]:
                raise RuntimeError("Input preparation failed: " + kind)
            for case in cases:
                record = dict(kind=kind, case=case, probes={}, status="running")
                summary["cases"].append(record)
                for label in PROBES:
                    origin = case + "-flagtree-old"
                    replay = (
                        origin
                        if label == "identical-ptx-new"
                        else (
                            case + "-identical-ptx-new"
                            if label == "new-cubin-stock"
                            else None
                        )
                    )
                    if replay and not (folder / replay / "replay.json").exists():
                        record["probes"][label] = dict(
                            name=case + "-" + label,
                            returncode=None,
                            skipped="producer did not export a binary",
                        )
                        continue
                    record["probes"][label] = run(
                        kind,
                        folder,
                        case + "-" + label,
                        case=case,
                        backend=(
                            "flagtree"
                            if label not in ("stock", "new-cubin-stock")
                            else "stock"
                        ),
                        replay=replay,
                        reassemble=label == "identical-ptx-new",
                    )
                    shared.write_json(summary_path, summary)
                check_identity(record["probes"], assembler, str(runtime), folder)
                record["status"] = assess(record["probes"])
                shared.write_json(summary_path, summary)
                print(kind, case, record["status"], flush=True)
        summary["status"] = (
            "passed"
            if all(c["status"] == "config_only_validated" for c in summary["cases"])
            else "inconclusive_or_failed"
        )
        return 0 if summary["status"] == "passed" else 1
    except EnvironmentError:
        if summary.get("environment_rejected"):
            summary["status"] = "environment_rejected"
            return 2
        summary.update(status="failed", error=traceback.format_exc())
        return 1
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        print(summary["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        shared.write_json(summary_path, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--flagtree", default="/opt/flagtree")
    parser.add_argument("--kind", choices=("prefill", "decode"))
    parser.add_argument("--prepare", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--name", help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--replay", help=argparse.SUPPRESS)
    parser.add_argument("--assembler", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.name or options.prepare:
        return child(options)
    directory = Path(options.report_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(p.name not in {"runner.txt", "compat134.json"} for p in directory.iterdir()):
        parser.error("Use a fresh report directory")
    print("Reports:", directory, flush=True)
    return orchestrate(options, directory)


if __name__ == "__main__":
    sys.exit(main())
