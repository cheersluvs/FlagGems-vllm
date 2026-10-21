# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Benchmark-only Top-K substitution and stage binding; no production edits."""

import base64
import hashlib
import importlib.metadata
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

PRODUCTION_SHA256 = "d70faeb6928c2c7a56636ded833f3359368871e353ecf4c66c6dfd35cae93e73"
INTEGRATED_PRODUCTION_SHA256 = (
    "d345e5e0eeec5de84958f0ffe81aa2ce286ce982448f6d167f94d87e02f7c849"
)
FROZEN_PRODUCTION_PATH = "tools/qsa_prefill_frozen_production.py"

REFERENCE_COMMIT = "7a877ae34dcf4a7d46de08eb16db4699e1e0047c"
REFERENCE_SHA256 = "4d09eb2f409e7cd71c7849c4f3bdd71d722bce6feaccb558f7ae67ac2d489a2b"
REFERENCE_URL = (
    f"https://raw.githubusercontent.com/vllm-project/vllm/{REFERENCE_COMMIT}/"
    "vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py"
)


def frozen_production():
    """Exact historical production for private trials; never a production import."""
    path = Path(__file__).resolve().parents[1] / FROZEN_PRODUCTION_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != PRODUCTION_SHA256:
        raise RuntimeError("Frozen production source SHA256 mismatch")
    name = "tools.qsa_prefill_frozen_production"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


def stock_runtime(torch, triton, prefix, distribution=None):
    """Reject shadowed/mixed packages and verify stock wheel RECORD identities."""
    prefix = Path(prefix).resolve()
    for name, module in (("torch", torch), ("triton", triton)):
        if not Path(module.__file__).resolve().is_relative_to(prefix):
            raise RuntimeError(
                f"{name} is outside selected virtualenv: {module.__file__}"
            )
    distribution = distribution or importlib.metadata.distribution("triton")
    if distribution.metadata["Name"].lower() != "triton":
        raise RuntimeError("The selected compiler distribution must be upstream triton")
    if distribution.version != triton.__version__:
        raise RuntimeError("Triton module and installed distribution versions differ")
    description = " ".join(str(v) for _, v in distribution.metadata.items()).lower()
    direct = distribution.read_text("direct_url.json") or ""
    if "flagtree" in description or "flagtree" in direct.lower():
        raise RuntimeError("FlagTree distribution cannot be used for this experiment")
    package = Path(triton.__file__).resolve().parent
    if (package / "experimental/tle").exists():
        raise RuntimeError(
            "TLE package found; select a clean upstream Triton environment"
        )
    wanted = {
        "triton/__init__.py",
        "triton/testing.py",
        "triton/backends/nvidia/compiler.py",
    }
    records = {}
    for entry in distribution.files or ():
        name = str(entry)
        if name not in wanted and not (
            name.startswith("triton/_C/libtriton") and name.endswith(".so")
        ):
            continue
        path = Path(distribution.locate_file(entry)).resolve()
        if (
            not path.is_relative_to(package)
            or entry.hash is None
            or entry.hash.mode != "sha256"
        ):
            raise RuntimeError(f"Missing stock wheel SHA256 identity: {name}")
        digest = hashlib.sha256(path.read_bytes()).digest()
        encoded = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        if encoded != entry.hash.value:
            raise RuntimeError(
                f"Triton file differs from installed wheel RECORD: {name}"
            )
        records[name] = dict(path=str(path), sha256=digest.hex())
    if not wanted.issubset(records) or not any(
        k.startswith("triton/_C/libtriton") for k in records
    ):
        raise RuntimeError("Stock Triton wheel RECORD is incomplete")
    if (package / "__init__.py").resolve() != Path(triton.__file__).resolve():
        raise RuntimeError("Triton import path does not match selected package")
    return dict(
        distribution=distribution.metadata["Name"],
        version=distribution.version,
        files=records,
    )


class PrefillTopK:
    """Match persistent_topk's call shape using the existing non-TLE operator."""

    def __init__(self, topk, torch):
        self.topk, self.torch = topk, torch
        self.row_starts = {}
        self.frozen = False

    def prepare(self, logits):
        key = (logits.shape[0], str(logits.device))
        if key not in self.row_starts:
            if self.frozen:
                raise RuntimeError(
                    "Unexpected row count/device after Top-K preparation"
                )
            self.row_starts[key] = self.torch.zeros(
                (logits.shape[0],), dtype=self.torch.int32, device=logits.device
            )
        return self.row_starts[key]

    def __call__(self, logits, visible, output, workspace, *, k, max_seq_len):
        del workspace
        if max_seq_len != logits.shape[1]:
            raise ValueError("Top-K width must match score rows")
        starts = self.prepare(logits)
        return self.topk(
            logits,
            starts,
            visible,
            output,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            k,
        )


def candidate_module(production, adapter):
    path = Path(production.__file__).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != INTEGRATED_PRODUCTION_SHA256:
        raise RuntimeError("Production source changed; review and repin the experiment")
    module = types.ModuleType(production.__name__ + ".prefill_topk_ab")
    namespace = module.__dict__
    namespace.update(production.__dict__)
    namespace["__name__"] = production.__name__ + ".prefill_topk_ab"
    namespace["persistent_topk"] = adapter
    original = production.qsa_select_paged_prefill
    public = types.FunctionType(
        original.__code__,
        namespace,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    public.__kwdefaults__ = original.__kwdefaults__
    namespace["qsa_select_paged_prefill"] = public
    for name in (
        "_prefill_logits",
        "_qsa_prefill_score_kernel",
        "_validate_inputs",
        "_prefill_score_metadata",
    ):
        if getattr(module, name) is not getattr(production, name):
            raise RuntimeError(f"Production scoring/validation must be shared: {name}")
    return module


def observe(module, topk_name, args):
    """Capture the real public pipeline outside timing, keeping native source intact."""
    chunks = []
    score, topk = module._prefill_logits, getattr(module, topk_name)

    def scores(*a, **kw):
        # Invalid columns are not read by either Top-K. Initialize once outside
        # timing so common-logits fingerprints never hash uninitialized memory.
        a[5].fill_(-float("inf"))
        logits = score(*a, **kw)
        offset = kw.get("query_offset", a[8] if len(a) > 8 else None)
        count = kw.get("num_queries", a[9] if len(a) > 9 else None)
        if offset is None or count is None:
            raise RuntimeError("Missing public score chunk offsets")
        canonical = (*a[:8], offset, count)
        extra = {
            k: v for k, v in kw.items() if k not in ("query_offset", "num_queries")
        }
        chunks.append(
            dict(
                score_args=canonical,
                score_kwargs=extra,
                offset=offset,
                rows=count,
                width=a[7],
                common_logits=logits.clone(),
            )
        )
        return logits

    def selection(*a, **kw):
        if not chunks or "topk_args" in chunks[-1]:
            raise RuntimeError("Expected alternating score and Top-K calls")
        chunks[-1].update(topk_args=a, topk_kwargs=kw)
        return topk(*a, **kw)

    with patch.object(module, "_prefill_logits", scores), patch.object(
        module, topk_name, selection
    ):
        module.qsa_select_paged_prefill(*args)
    if not chunks or any("topk_args" not in c for c in chunks):
        raise RuntimeError("This benchmark requires a nonempty scoring pipeline")
    return chunks


def signatures(chunks):
    return [(c["offset"], c["rows"], c["width"]) for c in chunks]


def select_all_stage(module, args):
    """Capture the actual public fast-path launch; time only its device kernel."""
    launches = []
    original = module._select_all_kernel

    class Capture:
        def __getitem__(self, grid):
            launch = original[grid]

            def invoke(*a, **kw):
                launches.append((launch, a, kw))
                return launch(*a, **kw)

            return invoke

    def unexpected(*a, **kw):
        raise RuntimeError("Expected select-all without score or TopK")

    with patch.object(module, "_select_all_kernel", Capture()), patch.object(
        module, "_prefill_logits", unexpected
    ), patch.object(module, "persistent_topk", unexpected):
        module.qsa_select_paged_prefill(*args)
    if len(launches) != 1:
        raise RuntimeError("Expected exactly one select-all kernel launch")

    def pipeline():
        for launch, a, kw in launches:
            launch(*a, **kw)

    return pipeline


def bind_stages(native, candidate, native_chunks, candidate_chunks):
    if signatures(native_chunks) != signatures(candidate_chunks):
        raise RuntimeError("Native and candidate chunk geometry differs")
    stages = {name: {} for name in ("score", "topk", "pipeline")}
    for side, module, chunks, name in (
        ("baseline", native, native_chunks, "_topk"),
        ("candidate", candidate, candidate_chunks, "persistent_topk"),
    ):
        score, topk = module._prefill_logits, getattr(module, name)

        def scores(score=score, chunks=chunks):
            for c in chunks:
                score(*c["score_args"], **c["score_kwargs"])

        def selection(topk=topk, chunks=chunks):
            for common, c in zip(native_chunks, chunks):
                topk(common["common_logits"], *c["topk_args"][1:], **c["topk_kwargs"])

        def pipeline(score=score, topk=topk, chunks=chunks):
            for c in chunks:
                logits = score(*c["score_args"], **c["score_kwargs"])
                topk(logits, *c["topk_args"][1:], **c["topk_kwargs"])

        stages["score"][side] = scores
        stages["topk"][side] = selection
        stages["pipeline"][side] = pipeline
    return stages
