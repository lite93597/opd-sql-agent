# Experiment v1: results and audit trail

On a **frozen BIRD dev subset of 300 questions across 11 databases**, the 300-update OPD student achieved **60.67%** result-set execution accuracy. Its continued-SFT control achieved **54.00%**, a paired difference of **+6.67 percentage points**. These are local evaluator results from one training seed, not an official BIRD leaderboard submission.

![Frozen five-arm execution comparison](../assets/results.svg)

## Final five-arm comparison

The student is Qwen3.5-9B and the fixed teacher is Qwen3.8-27B. Every arm uses the same frozen questions, prompt token IDs, generation settings and query-execution limits. No final-test feedback was used to select checkpoints.

| Model / branch | Result-set correct | Result-set accuracy | Strict row-multiset correct |
|---|---:|---:|---:|
| Base student | 150/300 | 50.00% | 135/300 |
| Fixed teacher | 193/300 | 64.33% | 177/300 |
| SFT starting point S0 (300 updates) | 163/300 | 54.33% | 147/300 |
| Continued SFT (300 updates) | 162/300 | 54.00% | 148/300 |
| OPD, LR 1e-5 (300 updates) | 182/300 | 60.67% | 165/300 |

**Metric distinction.** The main metric compares sets of result rows, ignoring row order and duplicate counts. The strict metric compares row multisets, preserving duplicate counts while still ignoring row order. Neither is SQL string exact match. Column order is preserved. Both are local SQLite execution metrics; `official_harness` is `false`.

Every score uses the fixed denominator of 300. Gold queries were executable for 299 questions; the gold query for `dev:518` timed out under the 30-second limit and counts as incorrect for every arm. All non-timing gold diagnostics matched across arms. All arms had zero missing/duplicate predictions, generation failures and repair attempts. The teacher had one generation at the 512-token length limit.

[Public final summary](../artifacts/experiment-v1/effect-summary.json) retains all aggregate statistics, paired IDs, per-database changes and source hashes. [SQL-free final-arm metrics](../artifacts/experiment-v1/final-arm-metrics.json) contains per-question correctness flags and aggregate breakdowns for both metrics, without question text, schemas, SQL or database contents.

## Paired gains and uncertainty

A gained question is wrong for the reference and correct for OPD; a lost question is correct for the reference and wrong for OPD.

| Comparison | Gained | Lost | Net correct | Accuracy difference | 95% database-cluster bootstrap CI |
|---|---:|---:|---:|---:|---|
| OPD vs Base student | 46 | 14 | +32 | +10.67 pp | [6.73, 15.02] pp |
| OPD vs S0 | 32 | 13 | +19 | +6.33 pp | [0.96, 12.25] pp |
| OPD vs Continued SFT | 33 | 13 | +20 | +6.67 pp | [2.29, 11.54] pp |

The bootstrap resamples 11 observed database clusters with replacement, preserves all within-database paired questions, and weights each draw by its question count. It uses 2000 draws, seed `20261004`, and percentile intervals. Unequal database sizes mean the question denominator can vary between draws. These intervals describe uncertainty across the observed database sample; they do not include training-seed, checkpoint/LR-selection or backend numerical uncertainty.

The SFT starting point improves over the raw student by **13 questions / +4.33 pp**. OPD adds **19 questions / +6.33 pp** over S0 and **20 questions / +6.67 pp** over continued SFT. The **+10.67 pp** raw-to-OPD improvement includes SFT warm-up and must not be attributed entirely to OPD.

The fixed teacher answers 193 questions correctly. The raw student's gap is 43 questions; OPD's gap is 11. The complete SFT-to-OPD pipeline closes **32/43 = 74.42%** of that observed gap. SFT contributes 13 questions and OPD adds another 19 relative to S0. This is a gap statistic, not a claim that OPD alone contributed 74.42% or that the teacher is an absolute capability ceiling.

## Data and frozen selection

- Official train: 9428 questions / 69 databases. The database-level split assigns 7231 questions / 55 databases to training and 2197 questions / 14 databases to internal validation.
- A fixed internal-validation sample of 120 questions selects checkpoints. Train difficulty annotations are missing, so internal sampling should not be described as validated difficulty stratification.
- The shared training pool contains 6216 eligible questions out of 7231. All 1015 excluded questions exceeded the 7680-token prompt budget. Complete schemas are retained; they are not truncated.
- The training budget reserves at most 512 completion tokens within an 8192-token context. Each 300-update branch uses four examples per optimizer update, covering 1200 distinct questions rather than a full pass over all 6216 candidates.
- Final test: 300 questions / 11 databases, with 181 simple, 89 moderate and 30 challenging questions. The subset excludes the 120 dev question IDs used in the earlier engineering baseline. This does not establish that all dev databases or the public dataset were unseen throughout the models' history.
- Final generation is greedy (`temperature=0`), at most 512 new tokens, no SQL repair. Evaluation uses a 16384-token context and full schemas.
- Query execution is read-only and process-bounded: 30 seconds, at most 100000 rows and 16 MiB of results. Limit violations count as incorrect rather than truncated successful answers.

The [dataset manifest](../artifacts/experiment-v1/dataset-manifest.json) preserves split membership IDs, database IDs, counts and input hashes. The [eligible-pool manifest](../artifacts/experiment-v1/eligible-pool-manifest.json) preserves length rules and filtered IDs. The [frozen selection](../artifacts/experiment-v1/frozen-selection.json) records the final 300-update checkpoints and final-test record SHA. Model weights and BIRD data are not distributed in this evidence bundle; obtain them separately under their source terms.

## The 600-update branch: retained negative result

A predeclared lower-LR branch ran to 600 updates from the same S0, with a corresponding 600-update SFT control. It was assessed **only on internal validation**, using the fixed 120-question denominator (118 valid gold queries and two invalid gold queries).

| Internal checkpoint | SFT correct | OPD LR 5e-6 correct | OPD gained / lost | OPD difference |
|---|---:|---:|---:|---:|
| 200 | 76/120 | 63/120 | 8 / 21 | -10.83 pp |
| 400 | 74/120 | 67/120 | 10 / 17 | -5.83 pp |
| 600 | 72/120 | 67/120 | 9 / 14 | -4.17 pp |

The independently selected internal bests are SFT at update 200 (**76/120**) and OPD at update 400 (**67/120**): OPD gains 8 questions and loses 17, net -9 / -7.50 pp. At update 600, OPD gains 9 and loses 14 relative to the SFT checkpoint, net -5 / -4.17 pp.

The lower-LR OPD best (67) did not strictly exceed the original OPD best (68). The frozen final comparison therefore retains the original 300-update pair. **No 600-update model received an additional final dev300 test. The 60.67% final score belongs to the 300-update OPD model.**

[Full 600-update internal comparison, including paired IDs](../artifacts/experiment-v1/600-internal-comparison.json).

## Scope and limitations

- This is one selected training seed and a 300-question subset, not full BIRD dev or independent leaderboard evaluation. Eleven clusters provide limited evidence for generalization.
- OPD and continued SFT share S0, the eligible pool, seed 43, example order, rank 8, accumulation 4 and 300 updates. Their LRs differ (1e-5 vs 2e-5); generated-token counts, wall time and hyperparameter-search compute are not matched. A documented first-update scheduler offset also remains. The comparison supports this OPD configuration, not isolated causality for every algorithmic ingredient.
- Internal validation favored continued SFT, while the frozen final subset favored OPD. Different database distributions and sampling variability are possible explanations, not established causes. Both outcomes are retained.
- The lower-LR branch changes LR and training horizon together; it cannot establish a pure LR effect. There is no forward-KL or off-policy soft-distillation ablation.
- The 8192-token probe completed backward, optimizer update and parameter synchronization. Actual OPD training reached a longest sequence of 7137 tokens; the probe is not evidence that sustained training ran at exactly 8192 tokens.
- An execution-equivalent result on one database is not a proof of SQL semantic equivalence on every database state. Neither local metric fully checks row-order semantics.
- Public artifacts preserve recorded decisions and hashes; they are not the original checkpoint weights or a rerun of generation and SQL execution. The CPU verifier below checks arithmetic, pairing and recorded evidence only.

## Verify the published statistics without a GPU

From the repository root, run:

```bash
python artifacts/experiment-v1/verify_results.py
```

The verifier uses the Python standard library and the repository's original paired-comparison implementation. It recomputes all three paired changes and the exact bootstrap intervals from the SQL-free correctness flags. It also checks final counts, frozen-test identity, source report hashes, data budgets and internal comparison arithmetic. It does not load models, download data, generate SQL or execute queries.

To regenerate the SVG/PNG chart after installing Matplotlib:

```bash
python artifacts/experiment-v1/plot_results.py
```

## Publication provenance and original hashes

The files below are **public derivatives**, not byte-identical originals. All source statistics, paired question IDs and SHA values are retained. Absolute deployment paths are replaced by portable logical identifiers (`experiment:`, `data:`, `project:`, or `source:`). Each derivative logs the exact JSON pointers changed, the original file byte count and its SHA-256 in `_publication_provenance`; no original deployment paths are published.

These identifiers identify external experiment/data sources and are not downloadable checkpoints or filesystem paths. The original files remain outside the public bundle. No question text, gold/predicted SQL, schema text, SQLite database, credentials or SSH host files are included. Dataset provenance records BIRD's source website and `CC BY-SA 4.0`; use the dataset's upstream terms when obtaining it.

| Original source (link opens its public derivative) | Original SHA-256 |
|---|---|
| [effect-summary.json](../artifacts/experiment-v1/effect-summary.json) | `7e58e0228f2099731e16d61640c1a3976653f140813b807c6959554844f1a0a7` |
| [effect-600-internal-comparison.json](../artifacts/experiment-v1/600-internal-comparison.json) | `12291fbe87fbadad67a93061fdcfbfb82ae1d052928c7c549c232452ae98b7ed` |
| [effect-frozen-selection.json](../artifacts/experiment-v1/frozen-selection.json) | `3f8f883c66b06b768aa8fa2c482be71247b84c2e58a1b43343bddd3bf5b9ab37` |
| [effect-dataset-manifest.json](../artifacts/experiment-v1/dataset-manifest.json) | `8a3e48be148439198a8308829ebfffd4830ae8827a005fbd167418770817af25` |
| [effect-eligible-pool-manifest.json](../artifacts/experiment-v1/eligible-pool-manifest.json) | `9bf45e5464f4e26d612f2e078c22bb539fc488609ec6704ed2f7bae504d19c15` |

The [SQL-free final-arm metrics](../artifacts/experiment-v1/final-arm-metrics.json) is a selected-field projection of the five original reports. Their hashes agree with the original summary's `input_sha256` registry:

| Arm | Original report SHA-256 |
|---|---|
| base | `cc3550eb6f7b7453fff66cef36e373c652f6f199e39e77981a290c90e8cbd40b` |
| teacher | `ece2f08b64955d4cb2c2856e8abd6c4949abc72e0e3aebf89c0cd7452a10774e` |
| warm | `2829685103b2fec305f84f1e20fde432629fbfa12d7ec74f103f972bfa97ca7a` |
| continued-sft | `bac29f898bdd7f825d36fe66b3a9fd3237cf1fabf833e2bcb74d962b4b9b9500` |
| opd | `58e7c7f13e04739c52c8ccb91fa1aebca0dff520cc847cbbbe37285ad46f0c92` |

Checksums for the published derivative bytes, this report and the chart are recorded separately in [publication-manifest.json](../artifacts/experiment-v1/publication-manifest.json). Original-source hashes identify the private originals; published-file hashes identify the portable public derivatives. They are intentionally different.
