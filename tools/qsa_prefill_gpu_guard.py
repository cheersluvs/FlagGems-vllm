# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Read-only, fail-closed GPU environment checks for controlled QSA repeats."""

import csv
import io
import math
import os
import re
from pathlib import Path

UUID_PATTERN = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def rows(query):
    if query.get("returncode") != 0:
        raise ValueError(f"nvidia-smi query failed: {query}")
    if not isinstance(query.get("stdout"), str):
        raise ValueError("nvidia-smi query has no text output")
    return [row for row in csv.reader(io.StringIO(query["stdout"])) if row]


def thermal_reasons(query):
    if query.get("returncode") != 0:
        return [f"Cannot read thermal slowdown state: {query}"]
    if not isinstance(query.get("stdout"), str):
        return ["Missing thermal slowdown output"]
    reasons = []
    for name in ("SW Thermal Slowdown", "HW Thermal Slowdown"):
        match = re.search(
            rf"^\s*{name}\s*:\s*(Not Active|Active)\s*$", query["stdout"], re.M
        )
        if not match:
            reasons.append(f"Missing or unsupported {name} state")
        elif match[1] == "Active":
            reasons.append(f"{name}: Active")
    return reasons


def process_query(uuid, command):
    return command(
        [
            "nvidia-smi",
            "-i",
            uuid,
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )


def processes(query, uuid):
    result = []
    for row in rows(query):
        if len(row) != 3:
            raise ValueError(f"Unexpected compute-process record: {row}")
        gpu, pid, memory = [v.strip() for v in row]
        # Some driver versions return other GPUs despite the device filter.
        if gpu != uuid:
            if not UUID_PATTERN.fullmatch(gpu):
                raise ValueError(f"Invalid process GPU UUID: {gpu}")
            continue
        pid, memory = int(pid), int(memory)
        if pid <= 0 or memory < 0:
            raise ValueError(f"Invalid process PID/memory: {row}")
        result.append(dict(pid=pid, used_memory_mib=memory))
    return result


def self_pids():
    """Only positively identified self PIDs, including visible namespace IDs."""
    result = {os.getpid()}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("NSpid:"):
                result.update(int(value) for value in line.split()[1:])
    except (OSError, ValueError):
        pass
    return sorted(result)


def process_details(pids, command):
    """Read identities on rejection; unavailable /proc data never grants access."""
    details = []
    for pid in sorted(set(pids)):
        item = dict(pid=pid)
        directory = Path("/proc") / str(pid)
        try:
            argv = (
                directory.joinpath("cmdline")
                .read_bytes()
                .decode(errors="replace")
                .rstrip("\0")
                .split("\0")
            )
            item["program"] = Path(argv[0]).name
            if len(argv) > 1 and Path(argv[1]).suffix in {".py", ".sh"}:
                item["script"] = Path(argv[1]).name
            item["status"] = [
                line
                for line in directory.joinpath("status").read_text().splitlines()
                if line.split(":", 1)[0]
                in {"Name", "State", "Pid", "PPid", "NSpid", "Uid"}
            ]
        except OSError as exc:
            item["proc_error"] = str(exc)
        item["ps"] = command(
            ["ps", "-p", str(pid), "-o", "pid=,ppid=,lstart=,etime=,comm="]
        )
        details.append(item)
    return details


def process_names(uuid, command):
    response = command(
        [
            "nvidia-smi",
            "-i",
            uuid,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        data = []
        for gpu, pid, name, memory in rows(response):
            if gpu.strip() == uuid:
                # Only program identity, never other jobs' argument values.
                program = Path(name.strip().split()[0]).name
                data.append(dict(pid=int(pid), program=program, memory_mib=int(memory)))
        return dict(returncode=0, processes=data)
    except (ValueError, TypeError, KeyError, IndexError):
        return dict(error="Process-name query unavailable or unparseable")


def idle(uuid, command, max_initial_temp_c=75):
    result = dict(
        device_uuid=uuid,
        status="rejected",
        reasons=[],
        queries={},
        max_initial_temp_c=max_initial_temp_c,
    )
    if not isinstance(max_initial_temp_c, int) or not 1 <= max_initial_temp_c <= 150:
        result["reasons"].append(
            "Initial temperature limit must be an integer in 1..150"
        )
        return result
    if not UUID_PATTERN.fullmatch(uuid):
        result["reasons"].append(
            "Guard requires CUDA_VISIBLE_DEVICES to contain one complete GPU UUID"
        )
        return result
    queries = result["queries"]
    queries["gpu"] = command(
        [
            "nvidia-smi",
            "-i",
            uuid,
            "--query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    queries["performance"] = command(
        ["nvidia-smi", "-i", uuid, "-q", "-d", "PERFORMANCE"]
    )
    queries["processes"] = process_query(uuid, command)
    reasons = result["reasons"]
    reasons.extend(thermal_reasons(queries["performance"]))
    try:
        gpu_rows = rows(queries["gpu"])
        if len(gpu_rows) != 1 or len(gpu_rows[0]) != 4:
            raise ValueError(f"Expected one GPU record: {gpu_rows}")
        gpu, temperature, utilization, memory = [v.strip() for v in gpu_rows[0]]
        if gpu != uuid:
            raise ValueError(f"Requested GPU {uuid}, observed {gpu}")
        temperature, utilization, memory = map(int, (temperature, utilization, memory))
        if not (0 <= temperature <= 150 and 0 <= utilization <= 100 and memory >= 0):
            raise ValueError("Invalid GPU temperature/utilization/memory")
        result.update(
            temperature_c=temperature,
            utilization_percent=utilization,
            memory_mib=memory,
        )
        for value, limit, name in (
            (temperature, max_initial_temp_c, "temperature C"),
            (utilization, 5, "utilization percent"),
            (memory, 256, "memory MiB"),
        ):
            if value > limit:
                reasons.append(f"Initial {name} {value} exceeds limit {limit}")
        result["processes"] = processes(queries["processes"], uuid)
        if result["processes"]:
            reasons.append("GPU already has compute processes; no jobs will be stopped")
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        reasons.append(str(exc))
    result["status"] = "rejected" if reasons else "passed"
    # Utilization is a trailing sample and can outlive the previous worker.
    # No other rejection (process, memory, temperature, telemetry) is retryable.
    result["retryable_utilization"] = (
        len(reasons) == 1
        and result.get("utilization_percent", 0) > 5
        and result.get("processes") == []
    )
    return result


def timing(point, query, uuid, allowed_pids):
    result = dict(
        device_uuid=uuid,
        status="rejected",
        reasons=[],
        allowed_self_pids=list(allowed_pids),
        process_query=query,
    )
    reasons = result["reasons"]
    reasons.extend(thermal_reasons(point.get("performance", {})))
    try:
        clock_rows = rows(point.get("clocks", {}))
        if len(clock_rows) != 2 or len(clock_rows[1]) != 6:
            raise ValueError("Missing or unexpected clock telemetry")
        row = [v.strip() for v in clock_rows[1]]
        if row[1] != uuid:
            raise ValueError("Clock telemetry GPU UUID does not match selected GPU")
        temperature, sm_clock = float(row[2]), float(row[3].split()[0])
        if not (math.isfinite(temperature) and 0 <= temperature <= 150):
            raise ValueError("Invalid temperature telemetry")
        if not (math.isfinite(sm_clock) and sm_clock > 0):
            raise ValueError("Invalid SM clock telemetry")
        result.update(temperature_c=temperature, sm_clock_mhz=sm_clock)
        result["processes"] = processes(query, uuid)
        foreign = [p for p in result["processes"] if p["pid"] not in allowed_pids]
        if foreign:
            reasons.append(
                f"Unattributed compute PIDs: {foreign}; only identified self PIDs allowed. "
                "Container/host PID mapping may be unavailable; do not guess ownership"
            )
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        reasons.append(str(exc))
    result["status"] = "rejected" if reasons else "passed"
    return result
