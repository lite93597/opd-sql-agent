"""Check the public result evidence without models, GPUs or BIRD data."""

import hashlib
import json
import math
from pathlib import Path
import sys


def main():
    evidence_dir = Path(__file__).resolve().parent
    project_dir = evidence_dir.parents[1]
    sys.path.insert(0, str(project_dir))
    from scripts.analysis.summarize_effect import paired_comparison

    def read(name):
        return json.loads((evidence_dir / name).read_text("utf-8"))

    def equivalent(left, right):
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(equivalent(left[key], right[key]) for key in left)
        if isinstance(left, list):
            return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
        if isinstance(left, float):
            # Python versions can differ in sum() rounding of the bootstrap mean.
            return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
        return left == right

    summary = read("effect-summary.json")
    arms = read("final-arm-metrics.json")["arms"]
    frozen = read("frozen-selection.json")
    data = read("dataset-manifest.json")
    pool = read("eligible-pool-manifest.json")
    internal = read("600-internal-comparison.json")
    assert summary["protocol_verified"] and not summary["official_harness"]
    assert summary["total"] == 300 and summary["database_count"] == 11
    assert summary["gold_diagnostics"]["valid_gold"] == 299
    assert frozen["test_records_sha256"] == summary["evaluation_protocol"]["records_sha256"]
    assert internal["frozen_selection_sha256"] == frozen["_publication_provenance"]["original_sha256"]
    assert pool["accepted"] == 6216 and len(pool["filtered"]) == 1015
    assert pool["same_pool_for_sft_and_opd"] and not pool["schema_truncated"]
    assert data["files"]["internal_train.jsonl"]["count"] == 7231
    assert data["files"]["internal_validation.jsonl"]["count"] == 2197
    assert data["files"]["dev-heldout-test-300.jsonl"]["count"] == 300

    for arm, report in arms.items():
        rows = report["results"]
        assert len(rows) == len({row["id"] for row in rows}) == 300
        assert all(isinstance(row["correct"], bool) and isinstance(row["bird_correct"], bool)
                   for row in rows)
        assert sum(row["bird_correct"] for row in rows) == summary["scores"][arm]["correct"]
        assert sum(row["correct"] for row in rows) == report["summary"]["correct"]
        assert report["original_report_sha256"] == summary["input_sha256"][arm]["report.json"]

    comparisons = [("opd_vs_base", "base"), ("opd_vs_warm", "warm"),
                   ("opd_vs_continued_sft", "continued-sft")]
    for name, reference in comparisons:
        recorded = summary["paired_comparisons"][name]
        calculated = paired_comparison(
            arms["opd"]["results"], arms[reference]["results"],
            seed=recorded["bootstrap"]["seed"],
            replicates=recorded["bootstrap"]["replicates"],
        )
        assert equivalent(calculated, recorded), name
        print(f"{name}: gained={calculated['gained']}, lost={calculated['lost']}, "
              f"delta={calculated['delta_pp']:.2f} pp, CI={calculated['bootstrap']['ci95_pp']}")

    for entry in internal["same_checkpoint_step_comparisons"].values():
        for comparison in entry.values():
            assert len(comparison["gained_ids"]) == comparison["gained"]
            assert len(comparison["lost_ids"]) == comparison["lost"]
            assert comparison["gained"] - comparison["lost"] == comparison["net"]
            assert comparison["opd_correct"] - comparison["sft_correct"] == comparison["net"]
            assert math.isclose(comparison["delta_pp"], 100 * comparison["net"] / 120)

    manifest_path = evidence_dir / "publication-manifest.json"
    if manifest_path.exists():
        for name, metadata in read("publication-manifest.json")["published_files"].items():
            path = (project_dir / name).resolve()
            assert path.is_relative_to(project_dir)
            raw = path.read_bytes()
            assert len(raw) == metadata["bytes"] and hashlib.sha256(raw).hexdigest() == metadata["sha256"], name

    print("PASS: five-arm counts, paired IDs, database bootstrap, frozen selection, "
          "data budgets and 600-step internal comparisons agree. No SQL was executed.")


if __name__ == "__main__":
    main()
