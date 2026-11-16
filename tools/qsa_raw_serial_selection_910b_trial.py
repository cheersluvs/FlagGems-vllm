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

"""Serial integration of verified raw score closures and unchanged TopK."""


class SerialRawSelection:
    def __init__(self, context):
        self.adapter = context["adapter"]
        self.jobs = context["jobs"]
        self.scores = context["scores"]

    def score_chunk(self, pieces):
        for item in pieces:
            item["execute"]("empty1024")

    def score_all(self):
        for _, pieces in self.jobs:
            self.score_chunk(pieces)

    def __call__(self):
        scratch = self.adapter._new_topk_scratch()
        for entry, pieces in self.jobs:
            self.score_chunk(pieces)
            self.adapter.select_chunk(
                entry["first"], self.scores[: entry["count"]], scratch
            )
        return self.adapter.output


def check_rank(torch, indices, scores, lengths, *, rtol, atol):
    """Vectorized CPU check of every row, including cutoff ties and padding."""
    from probe_qsa_int8_ten_shapes_910b import verify_structure

    verify_structure(indices, lengths)
    selected = scores.gather(1, indices.clamp_min(0).long())
    selected[indices < 0] = -float("inf")
    selected = selected.sort(dim=1, descending=True).values
    expected = scores.topk(512, dim=1).values
    torch.testing.assert_close(selected, expected, rtol=rtol, atol=atol)


def verify_pipeline(torch, context, raw, native, phase, extra=None):
    """Recompute chunk logits only for checks, outside all timed entry points."""
    adapter = context["adapter"]
    cpu = context["cpu"]
    extra = extra or {}
    outputs = {}
    for name, call in [("current", adapter), ("raw", raw)] + list(extra.items()):
        print("RAW_SERIAL_SELECTION_START", phase, name, flush=True)
        target = getattr(call, "output", adapter.output)
        target.fill_(-3)
        if hasattr(call, "reset_guards"):
            call.reset_guards()
        returned = call()
        torch.npu.synchronize()
        if hasattr(call, "check_guards"):
            call.check_guards()
        outputs[name] = returned.cpu()
    print("RAW_SERIAL_SELECTION_START", phase, "native", flush=True)
    native_output = native()
    torch.npu.synchronize()
    if (
        native_output.shape != (adapter.rows, 1, 512)
        or native_output.dtype != torch.int32
    ):
        raise AssertionError("Unexpected CANN V1 output contract")
    outputs["native"] = native_output.cpu().squeeze(1)
    changed = int(
        (outputs["current"].sort(1).values != outputs["raw"].sort(1).values)
        .any(1)
        .sum()
    )
    for chunk_index, (entry, pieces) in enumerate(context["jobs"]):
        first, count = entry["first"], entry["count"]
        adapter.score_into(first, count, context["reference_scores"][:count])
        raw.score_chunk(pieces)
        torch.npu.synchronize()
        original = context["reference_scores"][:count].cpu()
        candidate = context["scores"][:count].cpu()
        lengths = cpu["visible"][first : first + count]
        for name in outputs:
            indices = outputs[name][first : first + count]
            check_rank(torch, indices, original, lengths, rtol=1e-3, atol=1e-3)
            if name == "raw" or name in extra:
                check_rank(torch, indices, candidate, lengths, rtol=0, atol=0)
        for call in extra.values():
            call.check_chunk_scores(torch, entry, candidate)
        if chunk_index % 16 == 0 or chunk_index + 1 == len(context["jobs"]):
            print(
                "RAW_SERIAL_RANK_CHECK",
                phase,
                chunk_index + 1,
                len(context["jobs"]),
                flush=True,
            )
    result = dict(
        phase=phase,
        all_rows=adapter.rows,
        passed=True,
        raw_current_changed_index_sets=changed,
        extra_candidates=list(extra),
        extra_score_trace_rows=adapter.rows if extra else 0,
        extra_current_changed_index_sets={
            name: int(
                (outputs["current"].sort(1).values != outputs[name].sort(1).values)
                .any(1)
                .sum()
            )
            for name in extra
        },
        semantics="Tie-aware ranked score checks; raw exact own scores, current FP32 tolerance1e-3",
    )
    print("RAW_SERIAL_SELECTION_CHECK", result, flush=True)
    return result
