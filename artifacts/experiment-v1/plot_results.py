"""Render the published experiment-v1 result chart from its public JSON."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    evidence_dir = Path(__file__).resolve().parent
    project_dir = evidence_dir.parents[1]
    summary = json.loads((evidence_dir / "effect-summary.json").read_text("utf-8"))
    assert summary["total"] == 300 and not summary["official_harness"]

    arms = ["base", "warm", "continued-sft", "opd", "teacher"]
    labels = ["Base student", "SFT starting point", "Continued SFT", "OPD", "Fixed teacher"]
    values = [summary["scores"][arm]["accuracy_percent"] for arm in arms]
    counts = [summary["scores"][arm]["correct"] for arm in arms]
    control = summary["paired_comparisons"]["opd_vs_continued_sft"]
    assert counts == [150, 163, 162, 182, 193]
    assert control["gained"] - control["lost"] == counts[3] - counts[2]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "svg.fonttype": "none",
        "svg.hashsalt": "opd-sql-agent-experiment-v1",
    })
    fig = plt.figure(figsize=(10.8, 6.0), facecolor="#ffffff")
    ax = fig.add_axes([0.23, 0.35, 0.69, 0.48])
    colors = ["#cad2de", "#cad2de", "#cad2de", "#087f74", "#567398"]
    ax.barh(range(5), values, height=0.58, color=colors)
    ax.set_yticks(range(5), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 80)
    ax.set_xticks(range(0, 81, 10))
    ax.set_xticklabels([f"{value}%" for value in range(0, 81, 10)])
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#edf0f4", linewidth=0.8)
    ax.tick_params(axis="both", length=0, labelcolor="#43546a", pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for index, (value, count) in enumerate(zip(values, counts)):
        ax.text(
            value + 1.0, index, f"{value:.2f}%  ({count}/300)",
            va="center", color=colors[index] if index >= 3 else "#43546a",
            fontsize=11, fontweight="bold" if index >= 3 else "normal",
        )
    ax.get_yticklabels()[3].set_fontweight("bold")
    ax.get_yticklabels()[3].set_color("#087f74")

    fig.text(0.065, 0.94, "On-policy distillation for Text-to-SQL", fontsize=20,
             fontweight="bold", color="#18334a")
    fig.text(0.065, 0.885, "Frozen five-arm comparison / result-set execution accuracy",
             fontsize=11.5, color="#53657a")
    fig.text(0.065, 0.25, f"OPD vs continued SFT: +{control['delta_pp']:.2f} percentage points",
             fontsize=15, fontweight="bold", color="#087f74")
    lo, hi = control["bootstrap"]["ci95_pp"]
    fig.text(0.065, 0.20,
             f"Paired: {control['gained']} gained / {control['lost']} lost  |  "
             f"95% database-cluster bootstrap CI: [{lo:.2f}, {hi:.2f}] pp",
             fontsize=10.5, color="#53657a")
    fig.text(0.065, 0.105,
             "BIRD dev: frozen 300-question subset / 11 databases / one training seed",
             fontsize=10, color="#53657a")
    fig.text(0.065, 0.065,
             "Local SQLite evaluator, not the official BIRD harness. Final OPD: 300 updates.",
             fontsize=10, color="#53657a")

    assets = project_dir / "assets"
    assets.mkdir(exist_ok=True)
    for suffix in ("svg", "png"):
        metadata = {"Date": None} if suffix == "svg" else {"Software": "OPD SQL Agent / Matplotlib"}
        fig.savefig(assets / f"results.{suffix}", dpi=160, facecolor=fig.get_facecolor(), metadata=metadata)
    plt.close(fig)


if __name__ == "__main__":
    main()
