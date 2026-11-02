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

"""Use the repository Ascend kernel timer, with an explicit invocation-sum check."""

import hashlib
import inspect
import math
import statistics
from pathlib import Path


def setup(args, context, torch, result):
    from triton.backends.ascend.testing import do_bench_npu

    parameters = inspect.signature(do_bench_npu).parameters
    if not {"warmup", "active"} <= parameters.keys():
        raise RuntimeError(
            "Installed do_bench_npu lacks explicit warmup/active support"
        )
    source = Path(inspect.getfile(do_bench_npu)).resolve()
    options = dict(warmup=args.kernel_warmup, active=args.kernel_active)
    if "clear_l2_cache" in parameters:
        options["clear_l2_cache"] = False
    identity = dict(
        backend="triton.backends.ascend.testing.do_bench_npu",
        source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        signature=str(inspect.signature(do_bench_npu)),
        warmup=args.kernel_warmup,
        active=args.kernel_active,
        unit="microseconds per invocation, sum of device kernel durations",
        host_gaps_included=False,
        clear_l2_cache=(
            False if "clear_l2_cache" in parameters else "backend_default_not_exposed"
        ),
        scope="Kernel mode; separate stages are not additive; native score/TopK split unavailable",
    )
    (Path(args.report) / "installed-ascend-testing.txt").write_bytes(
        source.read_bytes()
    )
    result["kernel_timer"] = identity

    def measure(function):
        # One callable per timer call: implementations may contain multiple kernels.
        torch.npu.synchronize()
        milliseconds = float(do_bench_npu(function, **options))
        torch.npu.synchronize()
        if not math.isfinite(milliseconds) or milliseconds <= 0:
            raise RuntimeError(f"Invalid kernel timer result: {milliseconds}")
        return dict(kernel_us=milliseconds * 1000)

    first, count = context["chunks"][0]

    def one():
        context["score_into"](first, count, context["logits"][:count])

    def triple():
        for _ in range(3):
            one()

    single = measure(one)["kernel_us"]
    repeated = measure(triple)["kernel_us"]
    ratio = repeated / single
    identity["invocation_sum_check"] = dict(
        single_us=single, triple_us=repeated, ratio=ratio
    )
    print("KERNEL_TIMER_CHECK", identity, flush=True)
    if not 2.25 <= ratio <= 3.75:
        raise RuntimeError(
            "Kernel timer failed 3-call scaling check; do not use these benchmark results"
        )
    identity["invocation_sum_check"]["status"] = "passed"
    return measure


def aggregate(samples):
    values = [v["kernel_us"] for v in samples]
    median = statistics.median(values)
    return dict(
        kernel_us=dict(
            median=median,
            min=min(values),
            max=max(values),
            range_over_median=(max(values) - min(values)) / median,
        )
    )
