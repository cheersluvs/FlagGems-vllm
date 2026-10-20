# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Explicit shared-GPU diagnostic policy; never performance acceptance."""

import re

from tools import benchmark_qsa_prefill_stages as support
from tools import qsa_prefill_gpu_guard as guard


def enabled(options):
    return bool(getattr(options, "shared_gpu_exploratory", False))


def inspect_gpu(uuid, min_free_mib):
    check = guard.idle(uuid, support.command)
    check["original_idle_reasons"] = list(check["reasons"])
    # Only these two exclusive-use conditions are waived, by explicit opt-in.
    # Temperature, utilization, thermal slowdown and unreadable telemetry remain.
    check["reasons"] = [
        reason
        for reason in check["reasons"]
        if reason != "GPU already has compute processes; no jobs will be stopped"
        and not re.fullmatch(r"Initial memory MiB \d+ exceeds limit 256", reason)
    ]
    query = support.command(
        [
            "nvidia-smi",
            "-i",
            uuid,
            "--query-gpu=uuid,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    check["queries"]["free_memory"] = query
    try:
        rows = guard.rows(query)
        if len(rows) != 1 or len(rows[0]) != 3:
            raise ValueError("Malformed free GPU memory query")
        observed, free, total = [v.strip() for v in rows[0]]
        free, total = int(free), int(total)
        if observed != uuid or not 0 <= free <= total:
            raise ValueError("Invalid GPU UUID/free/total memory")
        check.update(
            free_memory_mib=free, total_memory_mib=total, required_free_mib=min_free_mib
        )
        if free < min_free_mib:
            check["reasons"].append(
                f"Free memory MiB {free} below shared diagnostic minimum {min_free_mib}"
            )
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        check["reasons"].append(str(exc))
    check.update(
        status="rejected" if check["reasons"] else "passed",
        measurement_context="shared_gpu_exploratory",
        acceptance_eligible=False,
    )
    return check


def admission(summary, report, env, label, min_free_mib, timeout, interval):
    from tools import qsa_prefill_flagtree_ab as profile

    return profile.cooldown_gate(
        summary, report, env, label, guard, timeout, interval, min_free_mib
    )


def telemetry(uuid, label, result, path):
    point = support.telemetry(uuid, label)
    uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    query = guard.process_query(uuid, support.command)
    point["process_query"] = query
    result.setdefault("telemetry", []).append(point)
    support.write_json(path, result)
    # Co-residents are intentionally permitted here; this is not an ownership
    # assertion. Record every external PID and never label the timing exclusive.
    processes = guard.processes(query, uuid)
    self_pids = guard.self_pids()
    check = guard.timing(point, query, uuid, self_pids + [p["pid"] for p in processes])
    check["permitted_shared_pids"] = check.pop("allowed_self_pids")
    check["self_pids"] = self_pids
    point["gpu_guard"] = check
    point["co_resident_processes"] = [p for p in processes if p["pid"] not in self_pids]
    result["performance_guard"] = dict(
        status="shared_exploratory" if check["status"] == "passed" else "rejected",
        last_label=label,
        reasons=check["reasons"],
        acceptance_eligible=False,
    )
    support.write_json(path, result)
    if check["status"] != "passed":
        raise RuntimeError("Shared GPU diagnostic rejected: " + repr(check["reasons"]))
    return point
