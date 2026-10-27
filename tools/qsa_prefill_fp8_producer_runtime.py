# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""FP8 three-buffer producer-first correctness and two-case kernel stage A/B."""

import contextlib
import copy
import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_fp8_producer as probe  # noqa: E402
from tools import qsa_prefill_producer_runtime as runtime  # noqa: E402

base, stage = runtime.base, runtime.stage
CASES = probe.CASES
EXTRA_SOURCES = (
    runtime.EXTRA_SOURCES
    | set(probe.SOURCES)
    | {
        "tools/qsa_prefill_fp8_producer_runtime.py",
        "tools/test_qsa_prefill_fp8_producer_runtime.py",
        "tools/run_qsa_prefill_fp8_producer_runtime_h800.sh",
        "docs/qsa_prefill_fp8_producer_runtime.md",
    }
)
WORKER = runtime.worker
RUNTIME_PATH = Path(runtime.__file__).resolve()


def manifest():
    data = probe.captured.manifest()
    data["cases"] = [v for v in data["cases"] if v["case"]["name"] in CASES]
    if tuple(v["case"]["name"] for v in data["cases"]) != CASES:
        raise RuntimeError("FP8 runtime fixture set changed")
    return data


def baseline_source():
    data = manifest()
    if (
        base.sha(ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py")
        != data["production_sha256"]
    ):
        raise RuntimeError("Frozen production changed")
    return dict(
        production_sha256=data["production_sha256"],
        manifest_sha256=probe.captured.MANIFEST_SHA,
        driver_sha256=base.sha(Path(__file__)),
        base_driver_sha256=base.sha(RUNTIME_PATH),
    )


def lifetime(ir, ptx):
    """Read-only checks of the captured FP8 compiler-generated K-ring protocol."""
    defs = {}
    for line in ir.splitlines():
        m = re.match(r"\s*(%[\w.$-]+)(?::\d+)? = (.*)", line)
        if m:
            defs[m[1]] = m[2]
    body = probe.body_lines(ir)
    header = next(line for line in ir.splitlines() if re.search(r"\bscf.for\b", line))
    phi = dict(
        re.findall(
            r"(%[\w.$-]+) = (%[\w.$-]+)", header.split("iter_args(")[1].split(") ->")[0]
        )
    )
    yielded = re.findall(
        r"%[\w.$-]+",
        next(
            line for line in reversed(body) if line.strip().startswith("scf.yield")
        ).split(" : ")[0],
    )
    roots = [
        v
        for v, rhs in defs.items()
        if "ttg.local_alloc" in rhs and "memdesc<3x64x128xf8E4M3FN" in rhs
    ]
    if len(roots) != 1:
        raise RuntimeError("Expected one three-slot FP8 K allocation")
    root = roots[0]
    views = {
        v: m[1]
        for v, rhs in defs.items()
        if (
            m := re.search(
                r"ttg.memdesc_index " + re.escape(root) + r"\[(%[\w.$-]+)\]", rhs
            )
        )
    }
    if len(views) != 3:
        raise RuntimeError("Unexpected K ring aliases")
    copies = [
        (v, rhs)
        for v, rhs in defs.items()
        if "ttg.async_copy_global_to_local" in rhs
        and "tensor<64x128x!tt.ptr<f8E4M3FN>" in rhs
    ]
    if len(copies) != 2:
        raise RuntimeError("Expected prologue/main K copies")
    main = next(
        (v, rhs)
        for v, rhs in copies
        if any(re.search(re.escape(v) + r" =", line) for line in body)
    )
    prologue = next((v, rhs) for v, rhs in copies if v != main[0])
    write = re.match(
        r"ttg.async_copy_global_to_local (%[\w.$-]+), (%[\w.$-]+)", main[1]
    )[2]
    initial = re.match(
        r"ttg.async_copy_global_to_local (%[\w.$-]+), (%[\w.$-]+)", prologue[1]
    )[2]
    reads = [
        re.search(r"\bttng.warp_group_dot (%[\w.$-]+),", line)[1]
        for line in body
        if re.search(r"\bttng.warp_group_dot\b", line)
    ]
    if (
        len(reads) != 2
        or len(set(reads)) != 1
        or write not in views
        or reads[0] not in views
        or initial not in views
    ):
        raise RuntimeError("Unexpected P2 K operand views")

    def constant(v):
        return int(re.match(r"arith.constant (-?\d+) : i32", defs[v])[1])

    if constant(views[initial]) != 0:
        raise RuntimeError("Prologue K slot is not zero")
    counters = []
    for view, phase in ((write, 0), (reads[0], -1)):
        select = re.match(
            r"arith.select (%[\w.$-]+), (%[\w.$-]+), (%[\w.$-]+)", defs[views[view]]
        )
        cmp = re.match(r"arith.cmpi sge, (%[\w.$-]+), (%[\w.$-]+)", defs[select[1]])
        add = re.match(r"arith.addi (%[\w.$-]+), (%[\w.$-]+)", defs[select[3]])
        if (
            constant(select[2]) != 0
            or cmp[1] != select[3]
            or constant(cmp[2]) != 3
            or constant(add[2]) != 1
            or constant(phi[add[1]]) != phase
            or yielded[list(phi).index(add[1])] != views[view]
        ):
            raise RuntimeError("K ring phase/wrap/yield changed")
        counters.append(dict(initial=phase, increment=1, wrap=3))
    commit = next(
        v
        for v, rhs in defs.items()
        if re.match(r"ttg.async_commit_group tokens " + re.escape(main[0]) + r"\b", rhs)
    )
    token = next(v for v in phi if yielded[list(phi).index(v)] == commit)
    if not re.match(
        r"ttg.async_commit_group tokens " + re.escape(prologue[0]) + r"\b",
        defs[phi[token]],
    ):
        raise RuntimeError("K prologue token changed")
    if not any(
        re.search(r"ttg.async_wait " + re.escape(token) + r" \{num = 1 : i32\}", line)
        for line in body
    ):
        raise RuntimeError("Missing current-copy token wait")
    protocol = probe.protocol.ptx_protocol(ptx)
    model = [
        dict(iterations=n, **probe.protocol.model(n, 1, slots=3)) for n in range(17)
    ]
    if not all(v["safe"] for v in model):
        raise RuntimeError("Three-slot lifetime model failed")
    return dict(
        status="checked_under_explicit_sync_assumptions",
        slots=3,
        K_tile_bytes=8192,
        warp_groups=1,
        counters=counters,
        PTX_protocol=protocol,
        model=model,
        buffer_rewrite=False,
        scope="structural and abstract lifetime review;GPU numerical validation required",
    )


def profiles(*, allow_create=False):
    source = Path(os.environ["QSA_PRODUCER_REPORT"]).resolve()
    summary = json.loads((source / "fp8-producer-summary.json").read_text())
    if (
        summary["status"] != "compile_feasible_pending_review"
        or summary["gpu_kernel_executed"]
        or len(summary["jobs"]) != 6
        or len(summary["decisions"]) != 2
        or any(v["reason"] is not None for v in summary["decisions"])
    ):
        raise RuntimeError("Requires both successful FP8 compile candidates")
    sources = json.loads((source / "sources.json").read_text())
    if set(sources) != set(probe.SOURCES) or any(
        base.sha(ROOT / n) != v["sha256"] for n, v in sources.items()
    ):
        raise RuntimeError("Captured FP8 compiler proof source changed")
    captures = []
    for case in CASES:
        left, right = (source / (case + suffix) for suffix in ("-control", "-producer"))
        records = []
        for folder in (left, right):
            record = json.loads((folder / "result.json").read_text())
            job = json.loads((folder / "spec.json").read_text())
            saved = next(v for v in summary["jobs"] if v["name"] == folder.name)
            if (
                saved["exit_code"]
                or saved["result"] != record
                or record["status"] != "compiled"
            ):
                raise RuntimeError("Invalid FP8 compile record")
            if job["original_resources"]["constants"] != probe.CONFIG:
                raise RuntimeError("FP8 launch configuration changed")
            for file, key in (
                ("ptx.txt", "ptx_sha256"),
                ("kernel.cubin", "cubin_sha256"),
            ):
                if base.sha(folder / file) != record[key]:
                    raise RuntimeError("FP8 compiled binary changed")
            if (
                base.sha(folder / "scheduled.ttir") != job["ir_sha256"]
                or base.sha(folder / "options.json") != job["metadata_sha256"]
            ):
                raise RuntimeError("Captured FP8 inputs/options changed")
            if (
                probe.inspect(
                    (folder / "ttgir.txt").read_text(), (folder / "ptx.txt").read_text()
                )
                != record["schedule"]
            ):
                raise RuntimeError("FP8 schedule evidence changed")
            records.append(record)
        if probe.gate(*records) is not None:
            raise RuntimeError("FP8 compile gate failed")
        changed, proof = probe.rewrite((left / "scheduled.ttir").read_text())
        if (
            changed != (right / "scheduled.ttir").read_text()
            or proof != json.loads((right / "spec.json").read_text())["rewrite_proof"]
        ):
            raise RuntimeError("FP8 cluster-only rewrite changed")
        ir = (right / "ttgir.txt").read_text()
        ptx = (right / "ptx.txt").read_text()
        review = lifetime(ir, ptx)
        captures.append(
            dict(
                case=case,
                control_code_sha256=runtime.code_sha((left / "ptx.txt").read_text()),
                candidate_ptx_sha256=records[1]["ptx_sha256"],
                control=records[0],
                candidate=records[1],
                ir=ir,
                options_sha256=base.sha(right / "options.json"),
                lifetime_review=review,
            )
        )
    # The two original captures are independently verified above. Canonical
    # matching must never arbitrarily choose between nonidentical candidates.
    first = captures[0]
    for c in captures[1:]:
        if any(
            c[k] != first[k]
            for k in (
                "control_code_sha256",
                "candidate_ptx_sha256",
                "control",
                "candidate",
                "ir",
                "options_sha256",
                "lifetime_review",
            )
        ):
            raise RuntimeError("Ambiguous nonidentical FP8 executable profiles")
    path = Path(os.environ["QSA_PRODUCER_RUNTIME_REPORT"]) / "fp8-producer-main.ttgir"
    if path.exists() and path.read_text() != first["ir"]:
        raise RuntimeError("Prepared FP8 IR changed")
    if not path.exists():
        if not allow_create:
            raise RuntimeError("Prepared FP8 IR disappeared")
        path.write_text(first["ir"])
    result = {k: v for k, v in first.items() if k not in ("ir", "case")}
    result.update(
        case="fp8_p2_main",
        covered_cases=list(CASES),
        override_file=str(path),
        override_sha256=base.sha(path),
    )
    return [result]


def eligible(named):
    return (
        named.get("IS_FP8") is True
        and named.get("NUM_HEADS") == 4
        and named.get("HEAD_DIM") == 128
        and all(named.get(k) == v for k, v in probe.CONFIG.items())
    )


class RoutedJIT:
    def __init__(self, original, arg_names, captures):
        self.original, self.arg_names = original, arg_names
        self.modified = runtime.OverrideJIT(original, arg_names, captures)

    def __getitem__(self, grid):
        def launch(*values, **kwargs):
            named = dict(zip(self.arg_names, values))
            named.update(kwargs)
            target = self.modified if eligible(named) else self.original
            return target[grid](*values, **kwargs)

        return launch


def make_pair(production, traces):
    with patch.object(
        stage,
        "fixed_traces",
        lambda traces, candidate=False: runtime.FIXED(traces, False),
    ):
        pair = runtime.MAKE_PAIR(production, traces)
    entry = pair[1]._qsa_prefill_fp8_store_score_kernel
    entry.jit = RoutedJIT(entry.jit, entry.arg_names, runtime.PROFILES)
    return pair


def freeze_records(pair):
    records = stage.FREEZE(pair)
    original = {tuple(v["chunk"]): v for v in records["baseline"]}
    proxy = pair[1]._qsa_prefill_fp8_store_score_kernel.jit.modified
    for record in records["candidate"]:
        old = original[tuple(record["chunk"])]
        active = record["route"] == "fp8" and eligible(
            dict(IS_FP8=True, NUM_HEADS=4, HEAD_DIM=128, **record["constants"])
        )
        if not active:
            if any(
                record[k] != old[k]
                for k in (
                    "ptx_sha256",
                    "cubin_sha256",
                    "constants",
                    "registers_per_thread",
                    "shared_bytes",
                    "spill_count",
                )
            ):
                raise RuntimeError("FP8 inactive/tail production fallback changed")
            record["original_binary_verified"] = True
            continue
        receipt = proxy.prepared[tuple(record["chunk"])]
        profile = next(v for v in runtime.PROFILES if v["case"] == receipt["profile"])
        if record["constants"] != old["constants"]:
            raise RuntimeError("FP8 launch options changed")
        if (
            record["ptx_sha256"] != profile["candidate_ptx_sha256"]
            or record["cubin_sha256"] != profile["candidate"]["cubin_sha256"]
        ):
            raise RuntimeError(
                "Runtime FP8 candidate binary differs from compile probe"
            )
        for actual, expected in (
            (old, profile["control"]),
            (record, profile["candidate"]),
        ):
            if (
                any(
                    actual[k] != expected[k]
                    for k in ("registers_per_thread", "shared_bytes")
                )
                or actual["spill_count"]
            ):
                raise RuntimeError(
                    "Runtime FP8 resources/spills differ from compile probe"
                )
        record["producer_receipt"] = receipt
    if not proxy.prepared:
        raise RuntimeError("Expected FP8 main candidate was not exercised")
    proxy.frozen = True
    return records


def boundary_page_plan(rows, num_pages, page_size, width):
    """Complete a private fixture's tested prefix;do not modify the real table."""
    if num_pages <= 0 or page_size <= 0 or width < 0 or not rows:
        raise ValueError("Invalid boundary page-table geometry")
    required = (width + page_size - 1) // page_size
    planned, filled = copy.deepcopy(rows), []
    for request, row in enumerate(planned):
        if required > len(row):
            raise ValueError("Boundary visibility exceeds table capacity")
        for logical in range(required):
            previous = row[logical]
            if not 0 <= previous < num_pages:
                row[logical] = logical % num_pages
                filled.append(
                    dict(
                        request=request,
                        logical_page=logical,
                        previous=previous,
                        physical_page=row[logical],
                    )
                )
        if any(not 0 <= value < num_pages for value in row[:required]):
            raise RuntimeError("Incomplete valid boundary page prefix")
    return planned, dict(
        policy="private valid-prefix completion;real benchmark inputs unchanged",
        required_pages=required,
        tested_visible_width=width,
        filled=filled,
    )


def boundary_page_table(table, num_pages, page_size, width, torch):
    # Host reads and Torch copies are diagnostic fixture preparation only.
    # They occur before scoring/timing and are never production fallbacks.
    planned, proof = boundary_page_plan(table.tolist(), num_pages, page_size, width)
    private = torch.empty_strided(
        table.shape, table.stride(), dtype=table.dtype, device=table.device
    )
    private.copy_(torch.tensor(planned, dtype=table.dtype, device=table.device))
    if private.shape != table.shape or private.stride() != table.stride():
        raise RuntimeError("Boundary table shape/strides changed")
    proof.update(
        shape=list(private.shape), stride=list(private.stride()), num_pages=num_pages
    )
    return private, proof


def boundary_checks(production, helpers, torch, device):
    from tools import benchmark_qsa_prefill_triton_stages as bench
    from tools import qsa_prefill_score_benchmark as scoring

    checks = []
    count, offset, total = 35, 5, 43
    progress = []
    for scenario in ("mixed_edges", "single_tile", "zero_visible"):
        configs = []
        for saved in manifest()["cases"]:
            args = list(bench.workloads.make_inputs(saved["case"], helpers, torch))
            if len(saved["case"]["query_lengths"]) != 1:
                raise RuntimeError("Boundary requires saved single-request layout")
            args[0] = args[0][:total]
            args[3] = torch.tensor(
                [97, 97 + total], dtype=torch.int32, device=args[0].device
            )
            width = saved["traces"][0]["width"]
            args[2], table_proof = boundary_page_table(
                args[2], args[1].shape[0], args[1].shape[1], width, torch
            )
            edges = [
                0,
                1,
                2,
                63,
                64,
                65,
                127,
                128,
                129,
                191,
                192,
                193,
                195,
                196,
                197,
                959,
                960,
                961,
                1023,
                1024,
                1025,
                width - 1,
                width,
            ]
            visible = (
                [edges[(i - offset) % len(edges)] for i in range(total)]
                if scenario == "mixed_edges"
                else [int(scenario == "single_tile")] * total
            )
            args[4] = torch.tensor(visible, dtype=torch.int32, device=args[0].device)
            args = tuple(args)
            backing = torch.full(
                (count * width + 128,),
                -12345.0,
                dtype=torch.float32,
                device=args[0].device,
            )
            logits = backing[64:-64].view(count, width)
            traces = [
                dict(
                    rows=count,
                    offset=offset,
                    width=width,
                    route="fp8",
                    constants=copy.deepcopy(saved["traces"][0]["constants"]),
                )
            ]
            chunks = [
                dict(
                    score_args=(*args[:5], logits, args[7], width, offset, count),
                    score_kwargs={},
                )
            ]
            pair = make_pair(production, traces)
            before = bench.support.fingerprint(args[:5], torch)
            counts = []
            for side, module in zip(("production", "candidate"), pair):
                backing.fill_(-12345.0)
                entry = dict(
                    case=saved["case"]["name"],
                    scenario=scenario,
                    side=side,
                    status="running",
                    boundary_table=table_proof,
                )
                progress.append(entry)
                progress_path = (
                    Path(os.environ["QSA_PRODUCER_RUNTIME_REPORT"])
                    / "fp8-boundary-progress.json"
                )
                probe.trace.save(progress_path, progress)
                try:
                    checked = scoring.validate_scores(
                        module._prefill_logits,
                        chunks,
                        args,
                        torch,
                        lambda n: list(range(n)),
                        device,
                    )
                except BaseException as exc:
                    entry.update(status="failed", error=repr(exc))
                    probe.trace.save(progress_path, progress)
                    raise
                counts.append(checked)
                entry.update(status="scores_passed", checked_rows=checked)
                probe.trace.save(progress_path, progress)
                mask = (
                    torch.arange(width, device=args[0].device)[None, :]
                    < args[4][offset : offset + count, None]
                )
                if (
                    not bool((backing[:64] == -12345.0).all())
                    or not bool((backing[-64:] == -12345.0).all())
                    or not bool((logits[~mask] == -12345.0).all())
                ):
                    raise RuntimeError("FP8 boundary changed masked output/canaries")
            if counts != [count, count] or before != bench.support.fingerprint(
                args[:5], torch
            ):
                raise RuntimeError("FP8 boundary inputs changed/rows missing")
            configs.append(
                dict(
                    case=saved["case"]["name"],
                    checked_rows=counts,
                    resources=freeze_records(pair),
                    boundary_table=table_proof,
                    status="passed",
                )
            )
        checks.append(dict(scenario=scenario, configs=configs, status="passed"))
        print(
            "FP8 boundary:",
            scenario,
            "all35 offset rows,left32/right3,guards/masks passed",
            flush=True,
        )
    return checks


def worker(options):
    code = WORKER(options)
    path = Path(options.report_dir) / (options.worker_case + ".json")
    if path.is_file():
        value = json.loads(path.read_text())
        value.update(
            algorithm="fp8_three_buffer_producer_first",
            config_scope="original fixed FP8 main configs;TR8/P1 tail exact original",
        )
        path.write_text(json.dumps(value, indent=2) + "\n")
    return code


@contextlib.contextmanager
def scoped_runtime():
    with contextlib.ExitStack() as stack:
        for name, value in (
            ("CASES", CASES),
            ("profiles", profiles),
            ("VERIFY", baseline_source),
            ("make_pair", make_pair),
            ("freeze_records", freeze_records),
            ("boundary_checks", boundary_checks),
            ("worker", worker),
            ("__file__", str(Path(__file__).resolve())),
            ("__doc__", __doc__),
            ("EXTRA_SOURCES", EXTRA_SOURCES),
        ):
            stack.enter_context(patch.object(runtime, name, value))
        stack.enter_context(patch.object(stage, "manifest", manifest))
        stack.enter_context(patch.object(runtime.orchestration, "manifest", manifest))
        yield


def main():
    with scoped_runtime():
        code = runtime.main()
    if "--worker-case" not in sys.argv and "--report-dir" in sys.argv:
        report = Path(sys.argv[sys.argv.index("--report-dir") + 1])
        old = report / "producer-runtime-summary.json"
        if old.is_file():
            old.replace(report / "fp8-producer-runtime-summary.json")
    return code


if __name__ == "__main__":
    sys.exit(main())
