# Experiment v1: portable evidence bundle

This directory publishes the recorded result evidence for a **300-update OPD model** on a frozen **BIRD dev300 / 11-database subset**. It includes aggregate scores, complete paired question IDs, SQL-free per-question correctness flags, data membership/length metadata and provenance hashes. It contains no model weights, dataset question text, gold or predicted SQL, schema text, SQLite databases, credentials or SSH host files.

Read the [full results report](../../docs/results.md) for the protocol, five-arm comparison and limitations. The [result chart](../../assets/results.svg) is generated from the public summary.

## Contents

| File | Purpose |
|---|---|
| [effect-summary.json](effect-summary.json) | All original aggregate scores, three paired comparisons and IDs, per-database changes, bootstrap intervals, teacher-gap accounting, protocol metadata and source hashes. |
| [final-arm-metrics.json](final-arm-metrics.json) | Five original-report summaries and breakdowns, with all 1500 per-arm question records reduced to `id`, `db_id`, `difficulty`, `correct` and `bird_correct`. Supports arithmetic/statistical verification without SQL or data downloads. |
| [frozen-selection.json](frozen-selection.json) | Checkpoint selection frozen before final testing, including final-test record SHA and portable checkpoint identifiers. |
| [dataset-manifest.json](dataset-manifest.json) | Train/validation/test membership IDs, database IDs, counts, isolation checks and source/dataset hashes. Database entries are metadata only, not database contents. |
| [eligible-pool-manifest.json](eligible-pool-manifest.json) | Shared 6216-question pool, length constraints, excluded question IDs/lengths and pool hashes. |
| [600-internal-comparison.json](600-internal-comparison.json) | The retained 600-update negative internal result, including full gained/lost IDs at updates 200/400/600 and the independently selected internal bests. |
| [publication-manifest.json](publication-manifest.json) | Original-source hashes and separate checksums/byte sizes for the published derivatives, report, chart and helpers. Excludes its own checksum. |
| [verify_results.py](verify_results.py) | CPU-only verification of published counts, pairings, bootstrap intervals, frozen identities, budgets, internal arithmetic and public-file checksums. |
| [plot_results.py](plot_results.py) | Matplotlib rendering of the result chart from `effect-summary.json`. |

## Verify without a model, GPU or BIRD download

Run from the repository root:

```bash
python artifacts/experiment-v1/verify_results.py
```

Only the Python standard library is required. The helper uses the repository's [original paired-comparison implementation](../../scripts/analysis/summarize_effect.py), with 2000 database-cluster bootstrap draws and seed `20261004`.

Expected main results:

```text
Base / Teacher / S0 / Continued-SFT / OPD:
150 / 193 / 163 / 162 / 182 correct out of 300

OPD vs Base:          gained 46 / lost 14 / +10.67 pp
OPD vs S0:            gained 32 / lost 13 /  +6.33 pp
OPD vs Continued-SFT: gained 33 / lost 13 /  +6.67 pp
```

The verifier compares all paired IDs and per-database changes, not only the headline totals. Floating-point comparisons allow an absolute tolerance of `1e-12` for Python-version summation rounding. This does **not** regenerate model outputs, execute SQL, re-audit original model weights or prove that the recorded correctness decisions are independently reproducible.

Regenerate the SVG and PNG with Matplotlib installed:

```bash
python artifacts/experiment-v1/plot_results.py
```

Rendering is deterministic within the same Matplotlib environment. Published byte checksums identify the exact release files; different rendering-library versions can legitimately produce different image bytes.

## Public derivatives and provenance

The five source derivatives preserve every original statistic, membership/paired ID and SHA field. They replace absolute deployment paths with logical portable identifiers:

- `experiment:` identifies an external run or checkpoint relative to its experiment namespace.
- `data:` identifies external dataset files; obtain BIRD separately under upstream terms.
- `project:` identifies project-relative sources.
- `source:` identifies another external source by basename.

These identifiers are metadata, not local paths, URLs or downloadable weights. Each file's `_publication_provenance` records its original filename, original byte size, original SHA-256, transformation description and the exact JSON pointers rewritten. Original absolute paths are not republished. Original source files remain unchanged outside this bundle.

`final-arm-metrics.json` is a selected-field projection of the five original evaluation reports. Its report hashes match the original summary's `input_sha256` registry. It retains both metrics and complete per-question flags while deliberately omitting original SQL, input text, execution columns/errors, paths and model identities.

An **original-source SHA** identifies the unmodified source file. A **published-file SHA** identifies the transformed public file. They differ by design; the public derivatives must not be presented as byte-identical originals. The data manifest records the BIRD source website and `CC BY-SA 4.0`; upstream data and model terms remain applicable to separately acquired inputs.

## Results must be read with their limits

- Final OPD accuracy is **60.67% for the 300-update model**. The 600-update lower-LR branch was evaluated only internally; at update 600, SFT scores 72/120 and OPD 67/120. No 600-update model received another dev300 final test.
- These are local SQLite result-set and row-multiset metrics, **not the official BIRD harness, full dev set or leaderboard**.
- The fixed final denominator is 300, with 299 valid gold queries and one shared gold timeout counted wrong in every arm. The internal denominator is 120, with 118 valid gold queries and two invalid queries.
- OPD and continued SFT share S0, pool, seed, example order and update budget. Learning rates, output-token counts, wall time and total search compute are not identical. This is evidence for the recorded configuration, not a fully isolated algorithmic causal claim or a training-speed gain.
- One training seed and eleven database clusters limit generalization. Cluster bootstrap does not capture training-seed or checkpoint/LR-selection uncertainty.
- The raw-to-OPD improvement includes SFT warm-up. The 74.42% teacher-gap closure describes the complete SFT-to-OPD pipeline, not OPD alone.

The evidence retains both improvements and regressions. Checkpoint selection was frozen before final testing and was not changed in response to the final result.
