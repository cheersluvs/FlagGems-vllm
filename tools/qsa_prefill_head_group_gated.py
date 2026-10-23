# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Restricted BF16 head-pair validation over frozen upstream12 configurations."""

import argparse
import contextlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_head_group_probe as base  # noqa: E402

BASE_PATH = Path(base.__file__).resolve()
MANIFEST = ROOT / "docs/qsa_prefill_head_group_upstream_configs.json"
MANIFEST_SHA = "61d3b2f1ee12fdf2cf3b8c9fbaef1c3d16d37b1af6325d11f8235c3b9c63b083"
KERNEL_SHA = "1edb354519498db5a6df79fe35c738f9623a5f65edea24c86a5e490ce0ec3a2b"
REPEAT_CASE = "perf_n4_q4096_p102400_bfloat16"
MAKE_PAIR, FREEZE_RECORDS = base.make_pair, base.freeze_records
EXTRA_SOURCES = {
    "tools/qsa_prefill_head_group_gated.py",
    "tools/test_qsa_prefill_head_group_gated.py",
    "tools/run_qsa_prefill_head_group_gated_h800.sh",
    "docs/qsa_prefill_head_group_upstream_configs.json",
    "docs/qsa_prefill_head_group_gated.md",
}


def manifest():
    from tools import qsa_prefill_observed_workloads as workloads
    from tools import qsa_prefill_triton_ab as ab

    if base.sha(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("Restricted head-group manifest changed")
    data = json.loads(MANIFEST.read_text())
    if data["production_sha256"] != ab.INTEGRATED_PRODUCTION_SHA256:
        raise RuntimeError("Production baseline changed")
    if [v["case"] for v in data["cases"]] != workloads.cases("upstream-prefill"):
        raise RuntimeError("Expected the unchanged upstream12 fixture list")
    for item in data["cases"]:
        shape = workloads.geometry(item["case"])
        expected = [
            (sum(shape["expected_chunk_rows"][:i]), n, shape["logits_width"])
            for i, n in enumerate(shape["expected_chunk_rows"])
        ]
        if [(t["offset"], t["rows"], t["width"]) for t in item["traces"]] != expected:
            raise RuntimeError("Restricted probe geometry changed")
    return data


def verify_source():
    data = manifest()
    production = ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
    kernel = ROOT / "tools/qsa_prefill_head_group_kernel.py"
    if (
        base.sha(production) != data["production_sha256"]
        or base.sha(kernel) != KERNEL_SHA
    ):
        raise RuntimeError("Frozen production/head-group kernel changed")
    return dict(
        production_sha256=base.sha(production),
        candidate_sha256=KERNEL_SHA,
        manifest_sha256=MANIFEST_SHA,
        driver_sha256=base.sha(Path(__file__)),
        base_driver_sha256=base.sha(BASE_PATH),
    )


def eligible(named):
    return (
        named.get("IS_FP8") is False
        and named.get("NUM_HEADS") == 4
        and named.get("HEAD_DIM") == 128
        and named.get("TILE_R") == 64
        and named.get("num_warps") == 4
    )


class RoutedJIT:
    def __init__(self, original, grouped, arg_names, route):
        self.original, self.grouped, self.arg_names, self.route = (
            original,
            grouped,
            arg_names,
            route,
        )
        self.decisions = {}

    def __getitem__(self, grid):
        def launch(*values, **kwargs):
            named = dict(zip(self.arg_names, values))
            named.update(kwargs)
            active = self.route == "rectangular" and eligible(named)
            key = base.fixed.signature(named)
            self.decisions[key] = dict(
                active=active,
                IS_FP8=named["IS_FP8"],
                NUM_HEADS=named["NUM_HEADS"],
                HEAD_DIM=named["HEAD_DIM"],
            )
            jit = self.grouped if active else self.original
            return jit[grid](*values, **kwargs)

        return launch


def make_pair(production, traces):
    pair = MAKE_PAIR(production, traces)
    for name, route in base.ENTRIES:
        original, entry = getattr(pair[0], name), getattr(pair[1], name)
        entry.jit = RoutedJIT(original.jit, entry.jit, entry.arg_names, route)
    return pair


def freeze_records(pair):
    records = FREEZE_RECORDS(pair)
    original = {tuple(r["chunk"]): r for r in records["baseline"]}
    decisions = {}
    for name, _ in base.ENTRIES:
        decisions.update(getattr(pair[1], name).jit.decisions)
    if set(decisions) != set(original):
        raise RuntimeError("Missing/extra actual head-group dispatch decisions")
    for record in records["candidate"]:
        key = tuple(record["chunk"])
        decision = decisions[key]
        record["head_group_gate"] = decision
        if not decision["active"]:
            for field in ("ptx_sha256", "cubin_sha256"):
                if record[field] != original[key][field]:
                    raise RuntimeError(
                        "Excluded head-group chunk changed original binary"
                    )
            record["original_binary_verified"] = True
    return records


@contextlib.contextmanager
def restricted():
    from tools import benchmark_qsa_prefill_triton_stages as bench

    with contextlib.ExitStack() as stack:
        for name, value in (
            ("manifest", manifest),
            ("verify_source", verify_source),
            ("make_pair", make_pair),
            ("freeze_records", freeze_records),
            ("__file__", str(Path(__file__).resolve())),
        ):
            stack.enter_context(patch.object(base, name, value))
        stack.enter_context(
            patch.object(bench, "SOURCES", sorted(set(bench.SOURCES) | EXTRA_SOURCES))
        )
        yield


def worker(options):
    # Case identity affects only sampling effort, never kernel routing.
    rounds = options.rounds
    if options.worker_case == REPEAT_CASE:
        options.rounds = options.promising_rounds
    try:
        print(
            "Restricted head-group case:",
            options.worker_case,
            "rounds:",
            options.rounds,
            flush=True,
        )
        code = base.worker(options)
        path = Path(options.report_dir) / (options.worker_case + ".json")
        if path.is_file():
            result = json.loads(path.read_text())
            result.update(
                algorithm="gated_bf16_h4_d128_r64_w4_head_pairs",
                policy="bf16-large",
                rounds=options.rounds,
                config_scope="original configs; exact original JIT for every excluded launch",
            )
            records = result.get("kernels_before", {}).get("candidate", [])
            result["gate_summary"] = dict(
                active_chunks=sum(
                    bool(r.get("head_group_gate", {}).get("active")) for r in records
                ),
                original_binary_fallback_chunks=sum(
                    bool(r.get("original_binary_verified")) for r in records
                ),
            )
            print("Actual restricted dispatch:", result["gate_summary"], flush=True)
            path.write_text(json.dumps(result, indent=2) + "\n")
        return code
    finally:
        options.rounds = rounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--flagtree", default="/opt/flagtree")
    parser.add_argument("--rounds", type=int, choices=(1, 2), default=2)
    parser.add_argument("--promising-rounds", type=int, choices=(2, 4), default=4)
    parser.add_argument("--kernel-warmup-ms", type=int, default=200)
    parser.add_argument("--kernel-rep-ms", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-initial-temp-c", type=int, default=80)
    parser.add_argument("--cooldown-timeout", type=int, default=120)
    parser.add_argument("--cooldown-interval", type=int, default=10)
    names = [v["case"]["name"] for v in manifest()["cases"]]
    parser.add_argument("--cases", nargs="+", choices=names, default=names)
    parser.add_argument("--worker-case", choices=names, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if (
        min(
            options.kernel_warmup_ms,
            options.kernel_rep_ms,
            options.timeout,
            options.cooldown_interval,
        )
        <= 0
        or options.cooldown_timeout < 0
        or not 1 <= options.max_initial_temp_c <= 150
    ):
        parser.error("Invalid timing/thermal limits")
    if len(set(options.cases)) != len(options.cases) or (
        options.worker_case and options.worker_case not in options.cases
    ):
        parser.error("Distinct cases and selected worker case required")
    options.report_dir = str(Path(options.report_dir).resolve())
    options.baseline_report = str(Path(options.baseline_report).resolve())
    options.compiler, options.candidate = "flagtree", "production"
    from tools.qsa_prefill_flagtree_ab import configure

    configure(options.flagtree)
    with restricted():
        return worker(options) if options.worker_case else base.coordinate(options)


if __name__ == "__main__":
    sys.exit(main())
