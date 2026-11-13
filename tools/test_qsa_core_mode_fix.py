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

"""CPU tests for source patching and native task-mode reconciliation."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from prepare_qsa_core_mode_fix_910b import (
    COMPILER,
    ROUTES,
    _reconcile_mix_mode_from_task_type,
    patch_source,
    prepare,
    sha256,
)


def fixture_source():
    source = """def _parse_linalg_metadata(linalg, metadata):
    metadata["mix_mode"] = linalg

def __get_metadata_attr_by_callback(lib, name, metadata, key):
    metadata[key] = lib[name]

"""
    for name in sorted(ROUTES):
        source += f"""def {name}(task, metadata, callback=True):
    if callback:
        lib = {{"_infer_task_type_function": task, "_infer_workspace_shape_function": 4096}}
        __get_metadata_attr_by_callback(
            lib, "_infer_task_type_function", metadata, "bs_task_type"
        )
        __get_metadata_attr_by_callback(
            lib, "_infer_workspace_shape_function", metadata, "workspace_size"
        )
    return metadata

"""
    return source


class CoreModeCandidateTests(unittest.TestCase):
    def test_known_task_kinds_preserve_ratios_and_other_fields(self):
        for task, mode in (
            (10, "aiv"),
            (20, "aic"),
            (31, "mix"),
            (32, "mix"),
            (41, "mix"),
            (42, "mix"),
        ):
            with self.subTest(task=task):
                code = ["same.bc"]
                before = dict(
                    mix_mode="wrong",
                    bs_task_type=task,
                    workspace_size=4096,
                    lock_num=3,
                    bitcodes=code,
                )
                after = dict(before)
                _reconcile_mix_mode_from_task_type(after)
                self.assertEqual(after, {**before, "mix_mode": mode})
                self.assertIs(after["bitcodes"], code)

    def test_unknown_and_missing_callbacks_preserve_behavior(self):
        for task in (None, 0, 50, -1):
            metadata = dict(mix_mode="existing", bs_task_type=task)
            before = dict(metadata)
            _reconcile_mix_mode_from_task_type(metadata)
            self.assertEqual(metadata, before)
        metadata = dict(mix_mode="aiv")
        _reconcile_mix_mode_from_task_type(metadata)
        self.assertEqual(metadata, dict(mix_mode="aiv"))

    def test_both_binary_routes_reconcile_after_callback(self):
        patched, routes = patch_source(fixture_source())
        self.assertEqual(set(routes), ROUTES)
        namespace = {}
        exec(compile(patched, "fixture-compiler.py", "exec"), namespace)
        for name in routes:
            for task, mode in ((10, "aiv"), (20, "aic"), (32, "mix"), (42, "mix")):
                metadata = dict(mix_mode="aiv", bitcodes=["native.bc"])
                result = namespace[name](task, metadata)
                self.assertIs(result, metadata)
                self.assertEqual(result["mix_mode"], mode)
                self.assertEqual(result["bs_task_type"], task)
                self.assertEqual(result["workspace_size"], 4096)
            self.assertEqual(
                namespace[name](20, dict(mix_mode="aiv"), False), dict(mix_mode="aiv")
            )

    def test_already_patched_and_unknown_routes_are_rejected(self):
        source = fixture_source()
        patched, _ = patch_source(source)
        with self.assertRaises(ValueError):
            patch_source(patched)
        with self.assertRaises(ValueError):
            patch_source(
                source.replace(
                    "linalg_to_bin_enable_npu_compile_A2_A3", "unknown_route"
                )
            )

    def test_copy_materializes_links_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination, report = (
                root / "original",
                root / "candidate",
                root / "report",
            )
            compiler = source / COMPILER
            compiler.parent.mkdir(parents=True)
            before = fixture_source().encode()
            compiler.write_bytes(before)
            compiler.chmod(0o444)
            alias = compiler.with_name("alias.py")
            alias.symlink_to(compiler)
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = prepare(source, destination, report, sha256(before))
            self.assertEqual(compiler.read_bytes(), before)
            self.assertFalse((destination / COMPILER).is_symlink())
            self.assertFalse(
                (destination / COMPILER).with_name("alias.py").is_symlink()
            )
            self.assertNotEqual((destination / COMPILER).read_bytes(), before)
            self.assertTrue(manifest["original_unchanged"])
            self.assertTrue((report / "flagtree-core-mode.patch.txt").is_file())
            with self.assertRaises(ValueError):
                prepare(source, source / "bad", report, sha256(before))


if __name__ == "__main__":
    unittest.main()
