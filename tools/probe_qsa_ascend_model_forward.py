#!/usr/bin/env python3
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

"""Trace QSA page tables during an eager model forward in the Qwen Ascend image."""

import argparse
import hashlib
import inspect
import json
from functools import wraps
from pathlib import Path
from time import monotonic_ns


BLOCK_TABLE_SHA256 = "22e07cb7a5f4ac47ab4f19d54533b9857a381e4ebff2d294f91733c95977e699"
SELECTORS = {
    "triton": "qsa_select_paged_tokens_triton",
    "lightning": "qsa_select_paged_tokens_lightning",
    "reference": "qsa_select_paged_tokens_reference",
}


def tensor_key(tensor):
    return (str(tensor.device), tensor.data_ptr(), tuple(tensor.stride()))


class QSAForwardTrace:
    def start_qsa_trace(self, max_steps):
        from vllm_ascend.models.qwen4_exp import qsa
        from vllm_ascend.worker import block_table

        source = Path(inspect.getfile(block_table)).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != BLOCK_TABLE_SHA256:
            raise RuntimeError(f"Unexpected BlockTable source {source}: {digest}")

        runner = self.model_runner
        if hasattr(self, "_qsa_trace"):
            raise RuntimeError("QSA trace already installed")
        state = {
            "step": 0,
            "commit": 0,
            "snapshot_ns": None,
            "commit_ns": None,
            "groups": {},
            "records": [],
            "seen": set(),
            "prefill_rows": (),
            "active": False,
            "layer": None,
        }
        self._qsa_trace = state
        table_type = block_table.MultiGroupBlockTable
        commit = table_type.commit_block_table
        prepare = runner._prepare_inputs
        select = qsa.AscendQSAIndexer._select

        @wraps(prepare)
        def trace_prepare(scheduler_output, num_scheduled_tokens):
            state["step"] += 1
            state["groups"].clear()
            state["seen"].clear()
            state["active"] = state["step"] <= max_steps
            batch = runner.input_batch
            state["prefill_rows"] = tuple(
                int(batch.num_computed_tokens_cpu_tensor[row])
                < int(batch.num_prompt_tokens_cpu_tensor[row])
                for row in range(batch.num_reqs)
            )
            return prepare(scheduler_output, num_scheduled_tokens)

        @wraps(commit)
        def trace_commit(tables, num_reqs):
            if not state["active"] or tables is not runner.input_batch.block_table:
                return commit(tables, num_reqs)
            state["commit"] += 1
            state["seen"].clear()
            state["snapshot_ns"] = monotonic_ns()
            groups = {}
            for gid, table in enumerate(tables.block_tables):
                if table.is_mamba_group:
                    continue
                device_table = table.get_device_tensor()
                groups.setdefault(tensor_key(device_table), []).append(
                    (gid, table.block_size, table.get_cpu_tensor()[:num_reqs].clone())
                )
            result = commit(tables, num_reqs)
            state["commit_ns"] = monotonic_ns()
            state["groups"] = groups
            return result

        @wraps(select)
        def trace_select(indexer, query, metadata, out):
            state["layer"] = indexer.compressed_key_cache.prefix
            try:
                return select(indexer, query, metadata, out)
            finally:
                state["layer"] = None

        def wrap_selector(path, function):
            @wraps(function)
            def trace_selector(query, cache, page_table, token_to_req, *args, **kwargs):
                if state["active"] and state["layer"] and query.shape[0]:
                    layer = state["layer"]
                    key = tensor_key(page_table)
                    sample_key = (layer, path, key, tuple(page_table.shape))
                    if sample_key not in state["seen"]:
                        state["seen"].add(sample_key)
                        matches = state["groups"].get(key, ())
                        record = {
                            "step": state["step"],
                            "commit": state["commit"],
                            "selector": path,
                            "layer": layer,
                            "table_ptr": hex(page_table.data_ptr()),
                            "table_shape": list(page_table.shape),
                            "query_shape": list(query.shape),
                            "cache_shape": list(cache.shape),
                            "query_dtype": str(query.dtype),
                            "cache_dtype": str(cache.dtype),
                            "snapshot_ns": state["snapshot_ns"],
                            "commit_ns": state["commit_ns"],
                            "consume_ns": monotonic_ns(),
                            "matching_groups": [gid for gid, _, _ in matches],
                        }
                        if len(matches) == 1:
                            gid, block_size, snapshot = matches[0]
                            rows = sorted(set(token_to_req[:query.shape[0]].cpu().tolist()))
                            columns = page_table.shape[1]
                            valid = (
                                bool(rows)
                                and rows[0] >= 0
                                and rows[-1] < min(snapshot.shape[0], page_table.shape[0])
                                and rows[-1] < len(state["prefill_rows"])
                                and columns <= snapshot.shape[1]
                            )
                            record.update(
                                kv_cache_gid=gid,
                                logical_block_size=block_size,
                                request_rows=rows,
                                matches_snapshot=False,
                            )
                            if valid:
                                record["prefill_rows"] = [
                                    row for row in rows if state["prefill_rows"][row]
                                ]
                                record["matches_snapshot"] = (
                                    page_table[: rows[-1] + 1].cpu()[rows].equal(
                                        snapshot[rows, :columns]
                                    )
                                )
                        state["records"].append(record)
                return function(query, cache, page_table, token_to_req, *args, **kwargs)

            return trace_selector

        runner._prepare_inputs = trace_prepare
        table_type.commit_block_table = trace_commit
        qsa.AscendQSAIndexer._select = trace_select
        for path, name in SELECTORS.items():
            setattr(qsa, name, wrap_selector(path, getattr(qsa, name)))
        return {"source": str(source), "source_sha256": digest}

    def get_qsa_trace(self):
        return self._qsa_trace["records"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-args", default="{}", help="JSON object of vLLM LLM arguments")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")

    from vllm import LLM, SamplingParams

    options = json.loads(args.engine_args)
    if options.get("speculative_config"):
        parser.error("Disable speculative decoding for this initial trace")
    options.update(
        model=args.model,
        enforce_eager=True,
        worker_extension_cls="probe_qsa_ascend_model_forward.QSAForwardTrace",
    )
    llm = LLM(**options)
    sources = llm.collective_rpc("start_qsa_trace", args=(args.max_steps,))
    llm.generate(
        ["Explain how a page table maps logical addresses.", "Summarize: " + "cache " * 256],
        SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
        use_tqdm=False,
    )
    workers = llm.collective_rpc("get_qsa_trace")
    passed = bool(workers) and all(
        any(record.get("prefill_rows") for record in records)
        and all(record.get("matches_snapshot") for record in records)
        for records in workers
    )
    print(json.dumps({"passed": passed, "sources": sources, "workers": workers}, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
