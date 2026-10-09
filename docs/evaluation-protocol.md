# BIRD baseline execution protocol

Official source was located from https://bird-bench.github.io/, whose code link
points to AlibabaResearch/DAMO-ConvAI. The inspected, fixed source is:

https://github.com/AlibabaResearch/DAMO-ConvAI/blob/188835d4f9948563a6b9c8ac50cd0f3ae4021ed6/bird/llm/src/evaluation.py

Commit was resolved using `git ls-remote` on 2026-10-04. `execute_sql` fetches
prediction rows, then gold rows, and awards 1 exactly when
`set(predicted_res) == set(ground_truth_res)`. Exceptions and timeouts award 0.
No official benchmark score has yet been obtained by these unit tests.
The inspected raw source SHA256 is
`2f591e559dc2d97e5b35d5b656e80b0c2edf968f0bb5a78ddfd1d88b4bbbc472`.

## Metrics and fixed denominator

`summary.accuracy` retains the original `local_sqlite_execution_row_multiset`
metric: row order / column aliases are ignored, column order and duplicate counts
are preserved, and returned column counts must agree even for empty results.

`summary.bird_execution_accuracy` is the added `bird_execution_set_equality`
metric. It faithfully reproduces the official successful-query comparison:
duplicate rows and row order are ignored; cell / column order is preserved for
non-empty results; two empty result sets compare equal even if the numbers of
columns differ. Python numeric equality applies (e.g. 1 equals 1.0), without
float tolerance or string coercion. JSON blob / non-finite wrappers are reversed
semantically by tagged immutable values; SQLite does not return dictionary
values that could collide with these tags.

Both metrics use **every reference record** as the denominator. Missing,
duplicate, malformed or generation-error predictions cannot score correct.
Unknown prediction IDs are reported but cannot change membership. Invalid gold
(including timeout, missing database and result caps) always scores 0 even when
prediction and gold fail identically. Valid-gold-only accuracy is secondary and
never replaces the fixed denominator. Input membership / predictions are
identified by file SHA256 and a reference-ID SHA256 in CLI output. Breakdown
tables report the same metrics by difficulty and database; paired teacher /
student comparison uses the identical reference IDs and reports teacher-only,
student-only, both-correct and both-incorrect counts.

## Execution harness differences from official code

The added metric is **not an unmodified official-harness run**. Our queries
execute separately in disposable subprocesses, with read-only SQLite URI,
query-only mode, disabled extension loading, and an authorizer rejecting writes,
ATTACH, PRAGMA and transactions. Prediction execution cannot change gold results.
Official code opens an ordinary connection, executes prediction before gold on
the same connection, fetches all rows without result caps, and uses func_timeout
around the combined pair; it matches prediction / gold by list position. We
match by explicit IDs and never silently zip away missing predictions.

Our hard deadline is **per query** and includes process startup, execution and
result transfer. A timed-out worker is killed and reaped. A SQLite VM progress
handler also interrupts long VM work. CLI defaults: 30 seconds, 100000 rows,
16 MiB JSON row / column data per query. Library defaults: 5 seconds, 10000 rows, 8 MiB.
Caps return `rows=null` and score 0, never an answer truncated to look correct.
These subprocess / cap changes can differ from official scores near execution
limits. We record all limits and separate gold / prediction failure statuses.
The process guard is not a native memory limit or an OS security sandbox.

## Data / prediction contract and relocation

Reference JSONL: `id` (unique nonempty string), `db_id`, `question`, `evidence`,
`gold_sql`, `schema`, `db_path`, `difficulty`, `source_split`. Prompt construction
must never use `gold_sql`. Dataset splitting must keep databases separated.

Prediction JSONL: `id`, `sql`, `raw_output`, `input_tokens`, `output_tokens`,
`generation_seconds`, `finish_reason`, `error`, `model` plus generation metadata.
`error` must be null / empty on successful generation; a nonempty error scores 0.
SQL is expected to have already been extracted from raw model output.
`finish_reason=error` or a final nested attempt with `status=generation_error`
also scores 0, even if stale SQL was retained. `finish_reason=length` remains
eligible for SQL execution scoring: complete SQL can end exactly at a token
budget. `summary.generation_length_limits` reports such budget hits separately;
`summary.generation_failures` reports failed generations independently of gold.

`--db-root` overrides host-specific reference paths using the official layout
`<db-root>/<db_id>/<db_id>.sqlite`. Without it, `db_path` is used as supplied.

```text
python -m opd_sql.bird_evaluation --records baseline-records.jsonl \
  --predictions student-predictions.jsonl \
  --comparison-predictions teacher-predictions.jsonl \
  --db-root /root/autodl-tmp/datasets/BIRD/dev/dev_databases \
  --output paired-evaluation.json
```

Report generation does not print per-query SQL or result rows. Generation timing
is retained separately from SQL subprocess time; subprocess startup is not part
of the model generation latency.
