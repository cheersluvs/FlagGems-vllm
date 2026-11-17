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

"""Retain fallback evidence without changing outer timer backend selection."""

import functools
import gzip
import inspect
import json
import os
import runpy
import sys
import traceback
from pathlib import Path


def install(timer, report, case):
    report.mkdir(parents=True, exist_ok=True)
    events = report / (case + "-timer-events.txt")
    captures = []

    def log(data):
        with events.open("a") as stream:
            stream.write(json.dumps(data, default=repr) + "\n")
        print("QSA_TIMER_DIAGNOSTIC", json.dumps(data, default=repr), flush=True)

    for name in ("do_bench_npu_mspti", "do_bench_npu_profiler"):
        original = getattr(timer, name)
        signature = inspect.signature(original)

        def decorate(original, signature, name):
            @functools.wraps(original)
            def wrapper(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                parameters = {k: v for k, v in bound.arguments.items() if k != "funcs"}
                if name == "do_bench_npu_profiler":
                    dest = report / (case + "-profile-" + str(len(captures)))
                    captures.append(dest)
                    bound.arguments["prof_dir"] = str(dest)
                    bound.arguments["keep_res"] = True
                log(
                    dict(
                        event="start",
                        backend=name,
                        original_parameters=parameters,
                        retained_path=(
                            str(captures[-1]) if name.endswith("profiler") else None
                        ),
                    )
                )
                try:
                    result = original(*bound.args, **bound.kwargs)
                except Exception:
                    log(
                        dict(
                            event="exception",
                            backend=name,
                            traceback=traceback.format_exc(),
                        )
                    )
                    raise
                log(dict(event="return", backend=name, result=repr(result)))
                return result

            return wrapper

        setattr(timer, name, decorate(original, signature, name))

    def collect():
        records = []
        budget = 8 * 1024 * 1024
        for dest in captures:
            for path in sorted(dest.rglob("*")):
                if not path.is_file():
                    continue
                size = path.stat().st_size
                item = dict(path=str(path), bytes=size)
                if "logs" in path.parts or path.suffix in (".log", ".csv"):
                    take = min(size, 128 * 1024, budget)
                    if take:
                        with path.open("rb") as stream:
                            item["head"] = stream.read(take).decode(errors="replace")
                        item["truncated"] = take < size
                        budget -= take
                records.append(item)
        with gzip.open(report / (case + "-profiler-evidence.json.gz"), "wt") as stream:
            json.dump(
                dict(files=records, raw_directories=[str(p) for p in captures]), stream
            )

    return collect


def main():
    from triton.backends.ascend import testing

    report = Path(os.environ["QSA_RAW_SERIAL_REPORT"])
    case = os.environ["QSA_TIMER_CASE"]
    source = Path(inspect.getfile(testing))
    (report / (case + "-timer-source.txt")).write_bytes(source.read_bytes())
    collect = install(testing, report, case)
    sys.argv = ["pytest", *sys.argv[1:]]
    try:
        runpy.run_module("pytest", run_name="__main__")
    finally:
        collect()


if __name__ == "__main__":
    main()
