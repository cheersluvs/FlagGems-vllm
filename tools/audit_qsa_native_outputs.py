# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Audit saved native/Gems index differences without rerunning either kernel."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def compare_scores(actual, native, *, rtol, atol):
    a, b = sorted(actual), sorted(native)
    assert len(a) == len(b), "different selected-score counts"
    assert all(math.isfinite(v) for v in a + b), "nonfinite selected score"
    delta = [abs(x - y) for x, y in zip(a, b)]
    return {
        "exact_scores_equal": a == b,
        "within_tolerance": all(d <= atol + rtol * abs(y) for d, y in zip(delta, b)),
        "max_abs_delta": max(delta, default=0.0),
        "max_relative_delta": max(
            (d / max(abs(y), 1e-30) for d, y in zip(delta, b)), default=0.0
        ),
    }


def audit(report):
    import torch

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from tests.qsa_select_prefill_utils import torch_logits

    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = report.resolve()
    summary = json.loads((report / "native-summary.json").read_text())
    results = []
    for item in summary["cases"]:
        if not item["native_execution_completed"]:
            results.append(
                {
                    "case": item["case"],
                    "status": "native unavailable",
                    "test": item["manifest"]["pytest_test"],
                }
            )
            continue
        if item.get("alignment", {}).get("sorted_indices_equal") is True:
            results.append({"case": item["case"], "status": "indices equal"})
            continue
        case = (report / item["case"]).resolve()
        assert case.is_relative_to(report)
        hashes = {
            "input.pt": item["manifest"]["input_sha256"],
            "output.pt": item["native"]["output_sha256"],
            "gems-output.pt": item["alignment"]["gems_output_sha256"],
        }
        for name, expected in hashes.items():
            path = case / name
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}; use the original card Reports directory including native/"
                )
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        payload = torch.load(case / "input.pt", map_location="cpu", weights_only=True)
        actual = torch.load(
            case / "gems-output.pt", map_location="cpu", weights_only=True
        )
        native = torch.load(case / "output.pt", map_location="cpu", weights_only=True)
        args = payload["args"]
        torch.cuda.set_device(payload["device_index"])
        fp8 = args[0].dtype == torch.float8_e4m3fn
        logits = torch_logits(*(value.to("cuda") for value in args[:5])).cpu()
        if fp8:
            logits /= args[0].shape[2] ** 0.5
        rtol, atol = (1e-3, 1e-3) if fp8 else (2e-4, 2e-3)
        rows = []
        for row, length in enumerate(args[4].tolist()):
            a = actual[row][actual[row] >= 0].long()
            b = native[row][native[row] >= 0].long()
            assert a.numel() == b.numel() == min(length, actual.shape[1])
            assert (actual[row] >= -1).all() and (native[row] >= -1).all()
            assert a.unique().numel() == a.numel() and b.unique().numel() == b.numel()
            assert (a < length).all() and (b < length).all()
            sa, sb = set(a.tolist()), set(b.tolist())
            if sa == sb:
                continue
            only_a, only_b = sorted(sa - sb), sorted(sb - sa)
            changed = compare_scores(
                logits[row, only_a].tolist(),
                logits[row, only_b].tolist(),
                rtol=rtol,
                atol=atol,
            )
            whole = compare_scores(
                logits[row, a].tolist(), logits[row, b].tolist(), rtol=rtol, atol=atol
            )
            rows.append(
                {
                    "row": row,
                    "changed_indices": len(only_a),
                    "whole_selection": whole,
                    "changed_selection": changed,
                    "gems_only_index_sample": only_a[:8],
                    "native_only_index_sample": only_b[:8],
                    "gems_only_score_sample": sorted(logits[row, only_a].tolist())[:8],
                    "native_only_score_sample": sorted(logits[row, only_b].tolist())[
                        :8
                    ],
                }
            )
        assert rows, "summary reported different index sets but saved outputs agree"
        if all(r["changed_selection"]["exact_scores_equal"] for r in rows):
            status = "exact score ties"
        elif all(r["changed_selection"]["within_tolerance"] for r in rows):
            status = "within tolerance; not exact ties"
        else:
            status = "changed scores outside tolerance"
        results.append(
            {
                "case": item["case"],
                "test": item["manifest"]["pytest_test"],
                "dtype": str(args[0].dtype),
                "query_shape": list(args[0].shape),
                "status": status,
                "rows": rows,
            }
        )
        print(
            item["case"],
            str(args[0].dtype),
            status,
            "changed rows:",
            len(rows),
            flush=True,
        )
    # BF16 differing sets should be attributable to exact ties before claiming
    # exact functional interchangeability; tolerated numeric differences stay
    # explicit and require follow-up. FP8 retains the established tolerances.
    needs_followup = any(
        r["status"] == "changed scores outside tolerance"
        or (r.get("dtype") == "torch.bfloat16" and r["status"] != "exact score ties")
        for r in results
    )
    output = report / "native-output-audit.json"
    output.write_text(
        json.dumps(
            {
                "scope": "saved outputs; FP32 Torch score audit, no native/Gems rerun",
                "torch": torch.__version__,
                "torch_source": torch.__file__,
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "needs_followup": needs_followup,
                "cases": results,
            },
            indent=2,
        )
        + "\n"
    )
    print("Audit report:", output, flush=True)
    return int(needs_followup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    sys.exit(audit(parser.parse_args().report))
