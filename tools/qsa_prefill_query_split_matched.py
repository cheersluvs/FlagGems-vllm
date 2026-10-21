# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Untuned score comparison with exact real baseline config per public chunk."""

import types


def signature(named):
    return tuple(named[k] for k in ("num_rows", "query_offset", "logits_width"))


class FixedEntry:
    def __init__(self, entry, traces, route):
        self.arg_names = entry.arg_names
        self.jit = entry.fn.fn
        self.configs = {
            (t["rows"], t["offset"], t["width"]): dict(t["constants"])
            for t in traces
            if t["route"] == route
        }
        self.compiled = {}
        self.frozen = False

    def __getitem__(self, grid):
        def launch(*values, **kwargs):
            named = dict(zip(self.arg_names, values))
            named.update(kwargs)
            key = signature(named)
            if key not in self.configs:
                raise RuntimeError("Fixed-config score chunk was not captured")
            if self.frozen and key not in self.compiled:
                raise RuntimeError("New fixed-config compilation during timing")
            constants = self.configs[key]
            kernel = self.jit[grid(constants) if callable(grid) else grid](
                *values, **kwargs, **constants
            )
            self.compiled[key] = (kernel, dict(constants))
            return kernel, dict(constants)

        return launch


def modules(production, candidate, traces):
    keys = [(t["rows"], t["offset"], t["width"]) for t in traces]
    if len(set(keys)) != len(keys) or any(
        t["route"] not in ("rectangular", "compact") for t in traces
    ):
        raise RuntimeError("Ambiguous fixed-config chunk traces")
    result = []
    for source in (production, candidate):
        module = types.ModuleType(source.__name__ + ".fixed_query_split")
        module.__dict__.update(vars(source))
        module.__name__ = source.__name__ + ".fixed_query_split"
        for name, route in (
            ("_qsa_prefill_score_kernel", "rectangular"),
            ("_qsa_prefill_compact_score_kernel", "compact"),
        ):
            setattr(module, name, FixedEntry(getattr(source, name), traces, route))
        for name in ("_prefill_logits", "qsa_select_paged_prefill"):
            original = getattr(source, name)
            cloned = types.FunctionType(
                original.__code__,
                vars(module),
                name,
                original.__defaults__,
                original.__closure__,
            )
            cloned.__kwdefaults__ = original.__kwdefaults__
            setattr(module, name, cloned)
        result.append(module)
    return tuple(result)


def freeze_and_records(pair):
    from tools import qsa_prefill_kernel_resources as resources

    records = []
    for module in pair:
        kernels = []
        for name in ("_qsa_prefill_score_kernel", "_qsa_prefill_compact_score_kernel"):
            entry = getattr(module, name)
            if set(entry.compiled) != set(entry.configs):
                raise RuntimeError("Fixed-config preparation omitted a chunk")
            entry.frozen = True
            kernels.extend(
                dict(
                    chunk=list(key),
                    constants=dict(constants),
                    **resources.compiled_record(kernel),
                )
                for key, (kernel, constants) in entry.compiled.items()
            )
        records.append(kernels)
    left = {tuple(r["chunk"]): r["constants"] for r in records[0]}
    right = {tuple(r["chunk"]): r["constants"] for r in records[1]}
    if left != right:
        raise RuntimeError("Fixed-config baseline and candidate differ")
    return dict(baseline=records[0], candidate=records[1], configs_match=True)
