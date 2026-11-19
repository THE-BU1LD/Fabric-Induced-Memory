"""Generate tables and the paired figure from retained development artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    results = summary["results"]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "paired_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "selected_backbone",
                "comparator",
                "memory_mse",
                "comparator_mse",
                "paired_difference",
                "global_comparator",
                "global_comparator_mse",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in writer.fieldnames})
    with (args.output / "all_test_cells.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "cell", "alpha", "mse", "owners", "forecasts"])
        for result in results:
            for name, cell in result["test_cells"].items():
                metric = cell["metric"]
                writer.writerow(
                    [
                        result["seed"],
                        name,
                        cell["selected_alpha"],
                        metric["mse"],
                        metric["owners"],
                        metric["forecasts"],
                    ]
                )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.7), constrained_layout=True)
    positions = list(range(len(results)))
    axes[0].plot(
        positions,
        [r["memory_mse"] * 1e4 for r in results],
        "o-",
        color="#176b87",
        label="FIM retrieval",
    )
    axes[0].plot(
        positions,
        [r["comparator_mse"] * 1e4 for r in results],
        "s-",
        color="#5e6472",
        label="Validation-selected conventional control",
    )
    axes[0].set_ylabel(r"Test MSE ($10^{-4}$; lower is better)")
    axes[0].set_title("Paired outcomes on the selected backbone", loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    differences = [r["paired_difference"] * 1e4 for r in results]
    colors = ["#a33b38" if value > 0 else "#176b87" for value in differences]
    axes[1].bar(positions, differences, width=0.5, color=colors)
    axes[1].axhline(0, color="#333333", lw=0.9)
    axes[1].axhline(
        summary["mean_paired_difference"] * 1e4,
        color="#a33b38",
        ls="--",
        label="Mean difference",
    )
    axes[1].set_ylabel(r"FIM minus control MSE ($10^{-4}$)")
    axes[1].set_title("Two wins; worse average error", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xticks(positions, ["101*", "202", "303"])
        axis.set_xlabel("Generated development repetition (seed)")
        axis.grid(axis="y", alpha=0.16)
        axis.spines[["right", "top"]].set_visible(False)
    figure.suptitle("FIM baseline diagnostic: mixed evidence", fontweight="bold")
    figure.supxlabel(
        "3 small synthetic repetitions; 2 test owners each. "
        "*Seed 101 was previously exposed. No confirmatory inference.",
        fontsize=9,
    )
    figure.savefig(args.output / "paired_outcomes.png", dpi=180)
    figure.savefig(args.output / "paired_outcomes.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
