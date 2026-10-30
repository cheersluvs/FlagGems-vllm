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

"""Test-only overrides around the unchanged public Ascend entry."""

from contextlib import contextmanager
from functools import wraps
from unittest.mock import patch


@contextmanager
def combined():
    from qsa_prefill_production_910b import load_production

    _, module, topk, _ = load_production()
    original = module._score

    @wraps(original)
    def score(*args, **kwargs):
        kwargs["programs"] = 20
        return original(*args, **kwargs)

    with patch.object(module, "_score", score), patch.multiple(
        topk, _G=40, _BLOCK=256
    ), patch.object(module, "_qsa_probe_original_score", original, create=True):
        yield


@contextmanager
def original(module=None, topk=None):
    from qsa_prefill_production_910b import load_production

    if module is None:
        _, module, topk, _ = load_production()
    source = getattr(module, "_qsa_probe_original_score", module._score)

    @wraps(source)
    def legacy_score(*args, **kwargs):
        kwargs["programs"] = 8
        return source(*args, **kwargs)

    with patch.object(module, "_score", legacy_score), patch.multiple(
        topk, _G=40, _BLOCK=1024
    ), patch.object(module, "_QSA_TOPK_BLOCK", 1024, create=True):
        yield
