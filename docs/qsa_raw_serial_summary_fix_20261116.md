# Serial raw harness summary recovery

Report commit be802a04826534d7e1651213db66e9d3468561f7 contains ten pytest
workers with exit0, successful stock kernel metrics and passing full pre/post
score/selection checks. The master exit1 was an aggregation error, not a
kernel correctness or timing failure.

The fixture had incorrectly placed result=passed under qsa_raw_serial_int8_910b.
Real benchmark/conftest.py records the test outcome under the pytest marker
qsa_select_paged_prefill_int8_910b and latency details under the benchmark op
qsa_raw_serial_int8_910b. The summary now validates that marker's outcome and
exact test identity, the timing op's mode/dtype/case/error/ratio, worker exit
and every pre/post check. No missing outcome is assumed successful.

All ten actual report files replay successfully with corrected parsing:
stock CANN V1/raw serial whole-call arithmetic mean0.5052244089238668x.
This is not a production acceptance pass or a claim of matching index identity;
the original tests use documented score-cutoff tolerances and record index-set
changes from the FP16 raw intermediate.

--summarize-only --source-report copies top-level original artifacts to a fresh
report, preserves old summary/exit/code/console records with original- prefixes,
revalidates all ten and uploads through the existing shell runner. It invokes
no pytest worker and no device runtime. Missing or failed checks still prevent
a mean. No source kernel, timer, production or TopK changes.

Local validation: seven summary/serial/worker CPU tests, including real-schema
missing-result regression and no-worker recovery; replay of ten actual report
JSON pairs, CLI recovery without NPU imports, Python syntax and focused
pre-commit. No NPU rerun is required.
