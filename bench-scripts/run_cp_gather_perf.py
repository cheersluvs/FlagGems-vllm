# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collect vendor-native vLLM versus FlagGems through the repository harness.

Run from a FlagGems-vllm checkout, using the same working GPU environment as
previous successful runs. Does not modify repository source files.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--vendor", required=True, choices=["metax", "hygon", "mthreads", "thead", "ascend"]
)
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()
if args.runs < 1:
    raise SystemExit("--runs must be positive")
root = Path.cwd().resolve()
bench_path = root / "benchmark/test_cp_gather_indexer_k_quant_cache.py"
if not bench_path.is_file():
    raise SystemExit("Run this command from the FlagGems-vllm repository root.")

# Retain the vendor platform plugin, exclude the FL replacement plugin.
entry_points = importlib.metadata.entry_points()
platform_plugins = (
    list(entry_points.select(group="vllm.platform_plugins"))
    if hasattr(entry_points, "select")
    else list(entry_points.get("vllm.platform_plugins", []))
)
selected = sorted({e.name for e in platform_plugins if e.name != "fl"})
if len(selected) > 1:
    chosen = os.environ.get("VLLM_PLUGINS", "").split(",")
    selected = [name for name in selected if name in chosen]
    if len(selected) != 1:
        raise SystemExit(
            "Multiple vendor platform plugins installed. Set VLLM_PLUGINS to the correct vendor name first: "
            + str([e.name for e in platform_plugins])
        )
os.environ["VLLM_PLUGINS"] = ",".join(selected)
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
os.environ["PYTHONPATH"] = str(root / "src") + (
    os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
)
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

import torch  # noqa: E402
import triton  # noqa: E402
import vllm  # noqa: E402
import vllm._custom_ops as vllm_ops  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402

# Save the vendor-native wrapper before importing FlagGems-vllm.
native = getattr(vllm_ops, "cp_gather_indexer_k_quant_cache", None)
if native is None:
    raise SystemExit("Native vLLM cp_gather_indexer_k_quant_cache is unavailable.")
native_source = inspect.getsourcefile(native)
if native_source and (
    "flaggems_vllm" in native_source or "/flag_gems/" in native_source
):
    raise SystemExit("vLLM entry already points to Gems: " + native_source)
fp8_dtype = current_platform.fp8_dtype()

import pytest  # noqa: E402

import flaggems_vllm as fg  # noqa: E402

if fg.vendor_name != args.vendor:
    raise SystemExit(
        f"Vendor mismatch: expected {args.vendor}, detected {fg.vendor_name}"
    )
device_api = getattr(torch, fg.device)
if not device_api.is_available():
    raise SystemExit("Device unavailable: " + fg.device)
gems = fg.cp_gather_indexer_k_quant_cache
if args.vendor in ("metax", "ascend") and f"backend/_{args.vendor}/" not in (
    inspect.getsourcefile(gems) or ""
):
    raise SystemExit(
        "Required vendor override is not active: " + str(inspect.getsourcefile(gems))
    )
SHAPES = [
    (4, 256, 16, 128, 128),
    (8, 512, 16, 128, 128),
    (16, 1024, 16, 512, 128),
    (32, 1024, 16, 512, 128),
    (1, 8192, 64, 128, 128),
    (8, 4096, 64, 128, 128),
    (64, 1024, 64, 128, 128),
]


def git(*parts):
    return subprocess.check_output(["git", *parts], text=True).strip()


def file_info(path):
    if not path:
        return None
    p = Path(path).resolve()
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = root / "perf-results" / f"cp_gather_{args.vendor}_{stamp}"
out_dir.mkdir(parents=True, exist_ok=False)
meta = {
    "operator": "cp_gather_indexer_k_quant_cache",
    "vendor": fg.vendor_name,
    "device_backend": fg.device,
    "device_name": device_api.get_device_name(0),
    "python": sys.executable,
    "torch": torch.__version__,
    "triton": triton.__version__,
    "vllm": vllm.__version__,
    "vllm_path": vllm.__file__,
    "cuda": getattr(torch.version, "cuda", None),
    "hip": getattr(torch.version, "hip", None),
    "platform_plugins_available": [
        {"name": e.name, "value": e.value} for e in platform_plugins
    ],
    "VLLM_PLUGINS": os.environ["VLLM_PLUGINS"],
    "git_head": git("rev-parse", "HEAD"),
    "git_branch": git("branch", "--show-current"),
    "git_status": git("status", "--short", "--", "benchmark", "src/flaggems_vllm"),
    "baseline": "vendor native vLLM",
    "native_entry": native.__module__ + "." + native.__name__,
    "native_source": file_info(native_source),
    "gems_source": file_info(inspect.getsourcefile(gems)),
    "generic_source": file_info(
        root / "src/flaggems_vllm/ops/cp_gather_indexer_k_quant_cache.py"
    ),
    "benchmark_source": file_info(bench_path),
    "harness_source": file_info(root / "benchmark/base.py"),
    "fp8_dtype": str(fp8_dtype),
    "mode": "kernel",
    "level": "comprehensive",
    "warmup_ms_arg": 1000,
    "rep_ms_arg": 100,
    "timing_note": "Ascend uses the repository do_bench_npu defaults; inspect benchmark/base.py for timer details.",
    "shapes": SHAPES,
    "speedup_definition": "native_vllm_ms / gems_ms",
}
(out_dir / "environment.json").write_text(
    json.dumps(meta, indent=2, ensure_ascii=False)
)
print("\n=== ENVIRONMENT ===", flush=True)
print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


class NativePlugin:
    def __init__(self, check_accuracy):
        self.check_accuracy = check_accuracy

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, items):
        targets = [
            item
            for item in items
            if item.module.__name__.endswith("test_cp_gather_indexer_k_quant_cache")
        ]
        if len(targets) != 1:
            raise RuntimeError("Expected exactly one operator benchmark test.")
        module = targets[0].module
        from benchmark import conftest as conf

        conf.TEST_RESULTS.clear()
        module.base.Config = conf.Config
        if not hasattr(module, "load_vllm_op_and_fp8_dtype"):
            raise RuntimeError(
                "Benchmark is an older gated version. "
                "Update to the PR #849 benchmark with seven shapes first."
            )
        module.load_vllm_op_and_fp8_dtype = lambda: (native, fp8_dtype)
        module.cp_gather_indexer_k_quant_cache = gems

        def forbidden_torch_baseline(*a, **kw):
            raise RuntimeError("Torch baseline is forbidden for this collection.")

        module.torch_gather = forbidden_torch_baseline
        benchmark = module.CpGatherIndexerKQuantCacheBenchmark(native, fp8_dtype)
        benchmark.set_shapes()
        if benchmark.shapes != SHAPES:
            raise RuntimeError(
                "Benchmark shapes differ from the required seven: "
                + str(benchmark.shapes)
            )
        print("\nBASELINE:", native, "GEMS:", inspect.getsourcefile(gems), flush=True)
        if self.check_accuracy:
            torch.manual_seed(0)
            for shape, tensors in zip(SHAPES, benchmark.get_input_iter(torch.float16)):
                native(*tensors)
                expected_k = tensors[1].view(torch.uint8).clone()
                expected_scale = tensors[2].view(torch.uint8).clone()
                gems(*tensors)
                device_api.synchronize()
                if not torch.equal(
                    expected_k, tensors[1].view(torch.uint8)
                ) or not torch.equal(expected_scale, tensors[2].view(torch.uint8)):
                    raise RuntimeError("Native/Gems byte mismatch at " + str(shape))
                print("BYTE_CHECK_PASS:", shape, flush=True)


runs = []
for repeat in range(1, args.runs + 1):
    output = out_dir / f"run_{repeat}.json"
    print(f"\n=== RUN {repeat}/{args.runs} ===", flush=True)
    code = pytest.main(
        [
            "-s",
            "-v",
            "--tb=short",
            "-rs",
            str(bench_path),
            "--mode",
            "kernel",
            "--level",
            "comprehensive",
            "--warmup",
            "1000",
            "--iter",
            "100",
            "--record",
            "json",
            "--output",
            str(output),
        ],
        plugins=[NativePlugin(repeat == 1)],
    )
    if code != 0:
        raise SystemExit(
            f"Benchmark failed (exit={code}); send the complete log. No successful performance report will be produced."
        )
    report = json.loads(output.read_text())
    result = report.get("cp_gather_indexer_k_quant_cache", {})
    if result.get("result") != "passed":
        raise SystemExit("Benchmark did not pass: " + str(result))
    details = result.get("details", [])
    if len(details) != 1 or len(details[0].get("result", [])) != 7:
        raise SystemExit("Expected one dtype and seven measured rows: " + str(result))
    rows = []
    for shape, metric in zip(SHAPES, details[0]["result"]):
        a, b = metric["latency_base"], metric["latency"]
        if (
            not isinstance(a, (int, float))
            or not isinstance(b, (int, float))
            or not math.isfinite(a)
            or not math.isfinite(b)
            or a <= 0
            or b <= 0
            or metric.get("error_msg")
        ):
            raise SystemExit("Invalid measurement: " + str(metric))
        rows.append(
            {
                "shape": shape,
                "native_vllm_ms": a,
                "gems_ms": b,
                "speedup": a / b,
                "native_gbps": metric.get("gbps_base"),
                "gems_gbps": metric.get("gbps"),
            }
        )
    runs.append({"run": repeat, "rows": rows})
    avg_speedup = sum(row["speedup"] for row in rows) / 7
    min_speedup = min(row["speedup"] for row in rows)
    print(
        f"RUN_SUMMARY {repeat}: avg={avg_speedup:.4f} min={min_speedup:.4f}",
        flush=True,
    )

bundle = out_dir / "collection.json"
bundle.write_text(
    json.dumps({"environment": meta, "runs": runs}, ensure_ascii=False, indent=2)
)
print("\nSEND_BACK_JSON:", bundle, flush=True)
print("Send this collection.json together with the complete tee log.", flush=True)
