# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Score-only benchmark plumbing; the kernel timer is the shared repo timer."""

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def make_timer(warmup_ms, rep_ms):
    import triton.testing as testing

    from benchmark import base, consts

    instance = object.__new__(base.Benchmark)
    instance.is_backward = False
    config = SimpleNamespace(
        mode=consts.BenchMode.KERNEL, warm_up=warmup_ms, repetition=rep_ms
    )
    sources = {}
    for name, fn in (
        ("get_latency", base.Benchmark.get_latency),
        ("do_bench", testing.do_bench),
    ):
        path = Path(inspect.getsourcefile(fn)).resolve()
        sources[name] = dict(
            path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )

    def measure(fn):
        with patch.object(base, "Config", config):
            return base.Benchmark.get_latency(instance, fn) * 1000.0

    return measure, dict(
        mode="kernel",
        unit="us",
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
        return_mode="median",
        cache_policy="shared do_bench default, unchanged",
        includes_topk=False,
        sources=sources,
    )


def score_function(launcher, chunks):
    def run():
        for chunk in chunks:
            launcher(*chunk["score_args"], **chunk.get("score_kwargs", {}))

    return run


def validate_scores(launcher, chunks, args, torch, sample_rows, device):
    q, cache, table, starts, visible = args[:5]
    offsets = (starts - starts[0]).cpu().tolist()
    lengths = visible.cpu().tolist()
    fp8 = q.dtype == torch.float8_e4m3fn
    checked = 0
    for chunk in chunks:
        call = chunk["score_args"]
        logits = launcher(*call, **chunk.get("score_kwargs", {}))
        offset, count = call[-2:]
        for row in sample_rows(count):
            query = offset + row
            request = next(
                i
                for i in range(len(offsets) - 1)
                if offsets[i] <= query < offsets[i + 1]
            )
            length = min(lengths[query], logits.shape[1])
            columns = torch.arange(length, device=q.device)
            pages = table[request, columns // cache.shape[1]].long()
            keys = cache[pages, columns % cache.shape[1], 0].float()
            expected = (q[query].float() @ keys.T).clamp_min(0).sum(0)
            torch.testing.assert_close(
                logits[row, :length],
                expected,
                rtol=1e-3 if fp8 else 2e-4,
                atol=1e-3 * q.shape[2] ** 0.5 if fp8 else 2e-3,
            )
            checked += 1
    device.synchronize()
    return checked


def measure_scores(timer, baseline, candidate, rounds, monitor, stats):
    functions = dict(baseline=baseline, candidate=candidate)
    samples = {name: [] for name in functions}
    observations = [monitor("before_timing")]
    order = []
    for repeat in range(rounds):
        names = (
            ["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]
        )
        for name in names:
            samples[name].append(timer(functions[name]))
        order.append(names)
        observations.append(monitor(f"after_round_{repeat + 1}"))
    baseline_stats, candidate_stats = stats(samples["baseline"]), stats(
        samples["candidate"]
    )
    return (
        dict(
            status="passed",
            baseline_us=baseline_stats,
            candidate_us=candidate_stats,
            speedup=baseline_stats["median"] / candidate_stats["median"],
            paired_ratios=stats(
                [a / b for a, b in zip(samples["baseline"], samples["candidate"])]
            ),
        ),
        observations,
        order,
    )
