# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Two-case producer-first runtime accuracy and kernel-mode stage A/B."""

import argparse
import contextlib
import copy
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_async_pipeline_probe as orchestration  # noqa: E402
from tools import qsa_prefill_producer_first as producer  # noqa: E402

stage = orchestration.stage
base = stage.base
VERIFY, FIXED, CORE_SCOPE = stage.verify_source, stage.fixed_traces, stage.scoped
MAKE_PAIR = stage.make_pair
CASES = producer.trace.CASES
EXTRA_SOURCES = (
    orchestration.EXTRA_SOURCES
    | set(producer.SOURCES)
    | {
        "tools/qsa_prefill_producer_runtime.py",
        "tools/test_qsa_prefill_producer_runtime.py",
        "tools/run_qsa_prefill_producer_runtime.sh",
        "docs/qsa_prefill_producer_runtime.md",
    }
)
PROFILES = None


def program(ptx):
    """Discard DWARF/path information only; retain the entire executable program."""
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S)
    out, skipping, opened, depth = [], False, False, 0
    for line in text.splitlines():
        line = line.strip()
        if not skipping and re.match(r"\.section\s+\.debug_\w+\b", line):
            skipping, opened, depth = True, False, 0
        if skipping:
            opened |= "{" in line
            depth += line.count("{") - line.count("}")
            if opened and depth == 0:
                skipping = False
            continue
        if line and not re.match(r"\.(?:file|loc)\s", line):
            out.append(" ".join(line.split()))
    if skipping:
        raise RuntimeError("Malformed PTX debug section")
    return "\n".join(out)


def code_sha(ptx):
    return hashlib.sha256(program(ptx).encode()).hexdigest()


def profiles(*, allow_create=False):
    source = Path(os.environ["QSA_PRODUCER_REPORT"]).resolve()
    report = Path(os.environ["QSA_PRODUCER_RUNTIME_REPORT"]).resolve()
    summary = json.loads((source / "producer-first-summary.json").read_text())
    if (
        summary["status"] != "compile_feasible_pending_accuracy"
        or summary["gpu_kernel_executed"]
        or len(summary["jobs"]) != 4
    ):
        raise RuntimeError("Requires validated producer-first compile report")
    sources = json.loads((source / "sources.json").read_text())
    if set(sources) != set(producer.SOURCES):
        raise RuntimeError("Captured source set changed")
    for name, record in sources.items():
        if base.sha(ROOT / name) != record["sha256"]:
            raise RuntimeError("Captured producer probe source changed: " + name)
    result = []
    for case in CASES:
        left, right = (source / (case + suffix) for suffix in ("-control", "-producer"))
        records = [json.loads((p / "result.json").read_text()) for p in (left, right)]
        for folder, record in zip((left, right), records):
            saved = next(j for j in summary["jobs"] if j["name"] == folder.name)
            if (
                saved["exit_code"]
                or saved["result"] != record
                or record["status"] != "compiled"
            ):
                raise RuntimeError("Captured compile result mismatch")
            for file, key in (
                ("ptx.txt", "ptx_sha256"),
                ("kernel.cubin", "cubin_sha256"),
            ):
                if base.sha(folder / file) != record[key]:
                    raise RuntimeError("Captured binary changed")
        if producer.gate(*records) is not None:
            raise RuntimeError("Captured producer candidate failed feasibility")
        control_ir = (left / "scheduled.ttir").read_text()
        if (
            producer.producer_first(control_ir)[0]
            != (right / "scheduled.ttir").read_text()
        ):
            raise RuntimeError("Captured cluster-only transformation changed")
        path = report / ("producer-" + case + ".ttgir")
        ir = (right / "ttgir.txt").read_bytes()
        if path.exists() and path.read_bytes() != ir:
            raise RuntimeError("Runtime IR override changed")
        if not path.exists():
            if not allow_create:
                raise RuntimeError("Prepared runtime IR override disappeared")
            path.write_bytes(ir)
        result.append(
            dict(
                case=case,
                control_code_sha256=code_sha((left / "ptx.txt").read_text()),
                candidate_ptx_sha256=records[1]["ptx_sha256"],
                override_file=str(path),
                override_sha256=base.sha(path),
                control=records[0],
                candidate=records[1],
            )
        )
    return result


def verify_source():
    current = profiles()
    if PROFILES is not None and current != PROFILES:
        raise RuntimeError("Producer runtime artifact identity changed")
    return dict(
        VERIFY(),
        producer_profiles=current,
        runtime_sources={n: base.sha(ROOT / n) for n in sorted(EXTRA_SOURCES)},
    )


def configure_report(options):
    global PROFILES
    os.environ["QSA_PRODUCER_RUNTIME_REPORT"] = str(Path(options.report_dir).resolve())
    PROFILES = profiles(allow_create=True)


def prepared_artifacts(kernel, label):
    # Untimed warmup evidence is retained even if numerical validation fails.
    directory = os.environ.get("QSA_PRODUCER_RUNTIME_REPORT")
    if not directory:
        return
    for kind in ("ptx", "ttgir"):
        content = kernel.asm.get(kind)
        if content is None:
            continue
        digest = hashlib.sha256(content.encode()).hexdigest()
        path = Path(directory) / (
            "runtime-" + label + "-" + digest + "-" + kind + ".txt"
        )
        if path.exists() and base.sha(path) != digest:
            raise RuntimeError("Prepared evidence changed")
        if not path.exists():
            path.write_text(content)


class OverrideJIT:
    def __init__(self, original, arg_names, captures):
        self.original, self.arg_names, self.captures = original, arg_names, captures
        self.prepared = {}
        self.frozen = False

    def __getitem__(self, grid):
        def launch(*values, **kwargs):
            named = dict(zip(self.arg_names, values))
            named.update(kwargs)
            key = base.fixed.signature(named)
            if key not in self.prepared:
                if self.frozen:
                    raise RuntimeError("New candidate preparation after freeze")
                # Compilation only: never execute baseline into candidate output.
                control = self.original.warmup(*values, grid=grid, **kwargs)
                prepared_artifacts(control, "original")
                executable = code_sha(control.asm["ptx"])
                matches = [
                    c for c in self.captures if c["control_code_sha256"] == executable
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        "No exact executable profile for score signature: "
                        + str(key)
                        + " code "
                        + executable
                    )
                profile = matches[0]
                candidate = self.original.warmup(
                    *values, grid=grid, **kwargs, ir_override=profile["override_file"]
                )
                prepared_artifacts(candidate, "candidate")
                actual = hashlib.sha256(candidate.asm["ptx"].encode()).hexdigest()
                if actual != profile["candidate_ptx_sha256"]:
                    raise RuntimeError(
                        "Runtime candidate raw PTX differs from verified probe"
                    )
                self.prepared[key] = dict(
                    profile=profile["case"],
                    original_raw_ptx_sha256=hashlib.sha256(
                        control.asm["ptx"].encode()
                    ).hexdigest(),
                    original_code_sha256=executable,
                    candidate_raw_ptx_sha256=actual,
                )
            receipt = self.prepared[key]
            profile = next(c for c in self.captures if c["case"] == receipt["profile"])
            # All subsequent calls perform one normal JIT launch; no warmup,
            # artifact read, baseline launch or parameter ABI reconstruction.
            return self.original[grid](
                *values, **kwargs, ir_override=profile["override_file"]
            )

        return launch


def make_pair(production, traces):
    with patch.object(
        stage, "fixed_traces", lambda traces, candidate=False: FIXED(traces, False)
    ):
        pair = MAKE_PAIR(production, traces)
    entry = pair[1]._qsa_prefill_score_kernel
    entry.jit = OverrideJIT(entry.jit, entry.arg_names, PROFILES)
    return pair


def freeze_records(pair):
    records = stage.FREEZE(pair)
    original = {tuple(r["chunk"]): r for r in records["baseline"]}
    proxy = pair[1]._qsa_prefill_score_kernel.jit
    for r in records["candidate"]:
        key = tuple(r["chunk"])
        old = original[key]
        if r["route"] != "rectangular":
            if any(r[k] != old[k] for k in ("ptx_sha256", "cubin_sha256")):
                raise RuntimeError("Unchanged fallback binary differs")
            continue
        receipt = proxy.prepared[key]
        capture = next(v for v in PROFILES if v["case"] == receipt["profile"])
        if (
            r["constants"] != old["constants"]
            or r["ptx_sha256"] != capture["candidate_ptx_sha256"]
        ):
            raise RuntimeError("Launch config/candidate artifact mismatch")
        for side, expected in ((old, capture["control"]), (r, capture["candidate"])):
            if (
                any(
                    side[k] != expected[k]
                    for k in ("registers_per_thread", "shared_bytes")
                )
                or side["spill_count"] != 0
            ):
                raise RuntimeError(
                    "Runtime compiled resources differ from feasibility probe"
                )
        r["producer_receipt"] = receipt
    proxy.frozen = True
    return records


def boundary_plan(case):
    starts = [0, 25] if case["requests"] == 1 else [0, 6, 14, 14, 25]
    return dict(total_rows=25, query_offset=5, num_rows=17, starts=starts)


def boundary_checks(production, helpers, torch, device):
    from tools import benchmark_qsa_prefill_triton_stages as bench
    from tools import qsa_prefill_score_benchmark as scoring

    fixtures = []
    for case_name in CASES:
        saved = base.saved_record(case_name)
        args = list(bench.workloads.make_inputs(saved["case"], helpers, torch))
        plan = boundary_plan(saved["case"])
        args[0] = args[0][: plan["total_rows"]]
        args[3] = torch.tensor(
            [97 + x for x in plan["starts"]], dtype=torch.int32, device=args[0].device
        )
        fixtures.append((saved, args, plan))
    checks = []
    for scenario in ("mixed_edges", "single_tile", "zero_visible"):
        configs = []
        for saved, template, plan in fixtures:
            args = list(template)
            width = saved["traces"][0]["width"]
            values = [
                0,
                1,
                2,
                63,
                64,
                65,
                195,
                196,
                197,
                1023,
                1024,
                1025,
                width - 1,
                width,
            ]
            visible = (
                [values[i % len(values)] for i in range(plan["total_rows"])]
                if scenario == "mixed_edges"
                else [1 if scenario == "single_tile" else 0] * plan["total_rows"]
            )
            args[4] = torch.tensor(visible, dtype=torch.int32, device=args[0].device)
            args = tuple(args)
            n, offset = plan["num_rows"], plan["query_offset"]
            backing = torch.full(
                (n * width + 128,), -12345.0, dtype=torch.float32, device=args[0].device
            )
            logits = backing[64:-64].view(n, width)
            traces = [
                dict(
                    rows=n,
                    offset=offset,
                    width=width,
                    route="rectangular",
                    constants=copy.deepcopy(saved["traces"][0]["constants"]),
                )
            ]
            chunks = [
                dict(
                    score_args=(*args[:5], logits, args[7], width, offset, n),
                    score_kwargs={},
                )
            ]
            pair = make_pair(production, traces)
            before = bench.support.fingerprint(args[:5], torch)
            counts = []
            for module in pair:
                backing.fill_(-12345.0)
                counts.append(
                    scoring.validate_scores(
                        module._prefill_logits,
                        chunks,
                        args,
                        torch,
                        lambda n: list(range(n)),
                        device,
                    )
                )
                mask = (
                    torch.arange(width, device=args[0].device)[None, :]
                    < args[4][offset : offset + n, None]
                )
                if (
                    not bool((backing[:64] == -12345.0).all())
                    or not bool((backing[-64:] == -12345.0).all())
                    or not bool((logits[~mask] == -12345.0).all())
                ):
                    raise RuntimeError("Producer boundary wrote guard/masked output")
            if before != bench.support.fingerprint(args[:5], torch) or counts != [
                17,
                17,
            ]:
                raise RuntimeError("Boundary inputs changed or rows unchecked")
            compiled = freeze_records(pair)
            configs.append(
                dict(
                    case=saved["case"]["name"],
                    checked_rows=counts,
                    resources=compiled,
                    status="passed",
                )
            )
        checks.append(dict(scenario=scenario, configs=configs, status="passed"))
        print(
            "Producer boundary:",
            scenario,
            "R32/R64 all17 rows, guards, masked outputs passed",
            flush=True,
        )
    return checks


def show(result):
    from tools.qsa_prefill_report import fmt, table

    print(
        "\nProducer runtime:",
        result.get("case", {}).get("name"),
        result.get("status"),
        flush=True,
    )
    for title, rows in (
        ("kernel-mode production / candidate", result.get("stages", {})),
        ("identical-callable A/A", result.get("self_controls", {})),
    ):
        if rows:
            print(title, flush=True)
            table(
                (
                    "stage",
                    "production_us",
                    "candidate_us",
                    "speedup",
                    "paired_min",
                    "paired_max",
                ),
                [
                    (
                        k,
                        fmt(v["baseline_us"]["median"], 3),
                        fmt(v["candidate_us"]["median"], 3),
                        fmt(v["speedup"], 5),
                        fmt(v["paired_ratios"]["min"], 5),
                        fmt(v["paired_ratios"]["max"], 5),
                    )
                    for k, v in rows.items()
                    if v.get("measurement_validated")
                ],
            )
    if result.get("error"):
        print(result["error"], flush=True)


def worker(options):
    configure_report(options)
    code = base.worker(options)
    path = Path(options.report_dir) / (options.worker_case + ".json")
    if path.is_file():
        result = json.loads(path.read_text())
        result.update(
            algorithm="producer_first_verified_IR_override",
            config_scope="original configs and JIT ABI; captured producer-first final IR",
            rounds=options.rounds,
        )
        result["self_controls"] = {
            k.replace("head_group:", "producer_first:"): v
            for k, v in result.get("self_controls", {}).items()
        }
        path.write_text(json.dumps(result, indent=2) + "\n")
        show(result)
    return code


def coordinate(options):
    report = Path(options.report_dir)
    try:
        if any(n not in CASES for n in options.cases):
            raise RuntimeError("Only the two validated producer profiles are supported")
        configure_report(options)
        with patch.object(orchestration, "verify_source", verify_source), patch.object(
            orchestration, "__file__", str(Path(__file__).resolve())
        ):
            code = orchestration.coordinate(options)
        path = report / "async-pipeline-summary.json"
        if path.is_file():
            path.replace(report / "producer-runtime-summary.json")
        return code
    except BaseException:
        result = dict(
            status="failed",
            error=traceback.format_exc(),
            production_modified=False,
            acceptance_eligible=False,
        )
        (report / "producer-runtime-summary.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(result["error"], flush=True)
        return 1


@contextlib.contextmanager
def scoped():
    with contextlib.ExitStack() as stack:
        for name, value in (
            ("verify_source", verify_source),
            ("make_pair", make_pair),
            ("freeze_records", freeze_records),
            ("boundary_checks", boundary_checks),
            ("EXTRA_SOURCES", EXTRA_SOURCES),
        ):
            stack.enter_context(patch.object(stage, name, value))
        stack.enter_context(CORE_SCOPE())
        stack.enter_context(
            patch.object(base, "__file__", str(Path(__file__).resolve()))
        )
        yield


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--producer-report", default=os.environ.get("QSA_PRODUCER_REPORT")
    )
    own, remaining = parser.parse_known_args()
    if not own.producer_report and not any(x in remaining for x in ("-h", "--help")):
        parser.error("--producer-report is required")
    if own.producer_report:
        os.environ["QSA_PRODUCER_REPORT"] = str(Path(own.producer_report).resolve())
    sys.argv = [sys.argv[0], *remaining]
    for flag, values in (
        ("--cases", CASES),
        ("--kernel-warmup-ms", ("20",)),
        ("--kernel-rep-ms", ("20",)),
    ):
        if flag not in sys.argv:
            sys.argv.extend([flag, *values])
    with patch.object(stage, "__doc__", __doc__), patch.object(
        stage, "scoped", scoped
    ), patch.object(stage, "worker", worker), patch.object(
        stage, "coordinate", coordinate
    ):
        return stage.main()


if __name__ == "__main__":
    sys.exit(main())
