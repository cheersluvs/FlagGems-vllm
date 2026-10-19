# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU checks for identical policy-free PTX and direct driver API diagnostics."""

import ctypes
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import diagnose_qsa_cubin_loader_h800 as probe


class LoaderTests(unittest.TestCase):
    def test_identical_policy_free_ptx_is_assembled_with_both_tools(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calls = []

            def assemble(command, **kw):
                calls.append(command)
                Path(command[-1]).write_bytes(b"\x7fELF" + command[0].encode())
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch.object(
                probe.shared, "snapshot", return_value={"ptxas": {"path": "/old/ptxas"}}
            ), patch.object(probe.subprocess, "run", side_effect=assemble):
                self.assertEqual(
                    probe.prepare(types.SimpleNamespace(assembler="/new/ptxas"), root),
                    0,
                )
            self.assertEqual(len(calls), 4)
            manifest = json.loads((root / "binaries.json").read_text())
            for entry in ("empty", "write"):
                self.assertEqual(
                    manifest[entry + "-old"]["ptx_sha256"],
                    manifest[entry + "-new"]["ptx_sha256"],
                )
                ptx = (root / (entry + ".ptx")).read_text()
                self.assertNotIn("createpolicy", ptx)
                self.assertNotIn("cp.async", ptx)
                self.assertIn(".version 8.7", ptx)
            self.assertTrue((root / "minimal-binaries-artifacts.json").is_file())

    def test_direct_api_boundaries_and_failure_are_persisted(self):
        for entry, fail, compat_requested in [
            ("empty", None, False),
            ("write", None, False),
            ("empty", "cuModuleLoadData", False),
            ("write", None, True),
        ]:
            with self.subTest(
                entry=entry, fail=fail
            ), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                name = entry + "-new"
                binary = root / (name + ".cubin")
                binary.write_bytes(b"\x7fELFtest")
                probe.shared.write_json(
                    root / "binaries.json",
                    {name: {"cubin_sha256": probe.shared.sha(binary)}},
                )
                calls = []
                driver_version = 13040 if compat_requested else 12090
                library_path = (
                    "/cuda-13.4/compat/libcuda.so.1"
                    if compat_requested
                    else "/cuda-12.9/compat/libcuda.so.1"
                )
                if compat_requested:
                    probe.shared.write_json(
                        root / "compat134.json", {"libcuda": library_path}
                    )

                class Function:
                    def __init__(self, name):
                        self.name = name

                    def __call__(self, *args):
                        calls.append(self.name)
                        if self.name == fail:
                            return 200
                        if self.name == "cuDriverGetVersion":
                            ctypes.cast(args[0], ctypes.POINTER(ctypes.c_int))[
                                0
                            ] = driver_version
                        if self.name == "cuModuleLoadData":
                            ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = (
                                ctypes.c_void_p(1)
                            )
                        if self.name == "cuModuleGetFunction":
                            ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = (
                                ctypes.c_void_p(2)
                            )
                        if self.name == "cuLaunchKernel":
                            assert args[1:8] == (1, 1, 1, 1, 1, 1, 0)
                            assert len(args) == 11
                            if entry == "write":
                                assert (
                                    ctypes.cast(
                                        args[9][0], ctypes.POINTER(ctypes.c_uint64)
                                    )[0]
                                    == 1234
                                )
                            else:
                                assert args[9] is None
                        return 0

                library = types.SimpleNamespace(
                    **{
                        name: Function(name)
                        for name in (
                            "cuDriverGetVersion",
                            "cuModuleLoadData",
                            "cuModuleGetFunction",
                            "cuLaunchKernel",
                            "cuCtxSynchronize",
                            "cuModuleUnload",
                        )
                    }
                )
                torch = types.ModuleType("torch")
                torch.__version__ = "test"
                torch.__file__ = "/venv/torch/__init__.py"
                torch.int32 = "i32"
                torch.cuda = types.SimpleNamespace(
                    init=lambda: None,
                    get_device_capability=lambda: (9, 0),
                    get_device_properties=lambda _: types.SimpleNamespace(uuid="gpu"),
                )
                torch.empty = lambda *a, **kw: types.SimpleNamespace(
                    data_ptr=lambda: 1234, item=lambda: 123
                )
                original = Path.read_text

                def read(path, *a, **kw):
                    return (
                        f"1 2 3 4 {library_path}\n"
                        if str(path) == "/proc/self/maps"
                        else original(path, *a, **kw)
                    )

                with patch.dict(sys.modules, torch=torch), patch.object(
                    probe.ctypes, "CDLL", return_value=library
                ), patch.object(probe.resource, "setrlimit"), patch.object(
                    probe.faulthandler, "enable"
                ), patch.object(
                    Path, "read_text", read
                ), patch.object(
                    probe.support, "command", return_value={}
                ):
                    code = probe.load(
                        types.SimpleNamespace(entry=entry, binary=name), root
                    )
                state = json.loads((root / (name + ".json")).read_text())
                self.assertEqual(code, int(bool(fail)))
                self.assertEqual(state["libcuda_paths"], [library_path])
                self.assertEqual(state["driver_api_version"], driver_version)
                self.assertFalse(state["triton_loader_used"])
                self.assertEqual(state["stage"], fail or "passed")
                self.assertEqual("cuLaunchKernel" in calls, not fail)

    def test_exit_signal_after_execution_does_not_count_as_passing(self):
        cases = {
            entry
            + "-"
            + backend: dict(returncode=0, state=dict(passed=True, stage="passed"))
            for entry in ("empty", "write")
            for backend in ("old", "new")
        }
        self.assertEqual(
            probe.classify(cases),
            "minimal_new_cubin_passed_qsa_specific_investigation_required",
        )
        cases["empty-new"] = dict(
            returncode=-11, state=dict(passed=False, stage="cuModuleLoadData")
        )
        self.assertEqual(
            probe.classify(cases),
            "policy_free_new_cubin_load_failed_independently_of_triton",
        )
        cases["empty-old"]["returncode"] = -11
        self.assertEqual(probe.classify(cases), "inconclusive_old_control_failed")


if __name__ == "__main__":
    unittest.main()
