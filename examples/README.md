# CPU quickstart

This example creates a tiny **synthetic** SQLite database and evaluates two sets
of handwritten SQL predictions. It needs no model, GPU, network endpoint or BIRD
download. It demonstrates execution scoring, duplicate-row semantics and paired
gains/losses. **Its scores are not SFT or OPD results.**

From the repository root, using Python 3.10 or newer:

```sh
python -m pip install -e .
python scripts/demo.py
```

The demo writes `work/demo/` and refuses to overwrite an existing directory. For
another run, choose a fresh location:

```sh
python scripts/demo.py --output-dir work/demo-second-run
```

Expected output:

| Handwritten fixture | Set-equality correct | Row-multiset correct |
|---|---:|---:|
| reference | 2/3 | 1/3 |
| candidate | 3/3 | 3/3 |

The reference deliberately uses `MAX(amount)` for a total question and
`DISTINCT amount` for a question that requires duplicate rows. Set equality
ignores duplicates; the stricter row-multiset metric retains duplicate counts.
Both metrics ignore row order. Successful execution alone does not establish
answer correctness.

The generated directory contains the toy database, reference records, both
prediction files and both full evaluation reports. Re-evaluate a prediction file
through the same CLI used for external predictions:

```sh
python -m opd_sql.bird_evaluation \
  --records work/demo/records.jsonl \
  --predictions work/demo/candidate-predictions.jsonl \
  --output work/demo/cli-report.json
```

On PowerShell, put that command on one line or replace line continuations with
PowerShell backticks. An existing output report is refused unless `--overwrite`
is explicitly supplied. No overwrite option is provided for the demo directory.

## Lightweight tests

These tests exercise prompts, bounded SQLite execution, data preparation and
report auditing without installing PyTorch or downloading models:

```sh
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -p test_inference.py -v
python -m unittest discover -s tests -p test_evaluation.py -v
python -m unittest discover -s tests -p test_data_preparation.py -v
python -m pytest -q tests/test_experiment_preparation.py tests/test_effect_summary.py
```

The entire test directory also includes PyTorch loss/gradient tests and
server-runtime tests; the selected lightweight commands above are intentionally
usable with the core package and the `dev` extra alone.
