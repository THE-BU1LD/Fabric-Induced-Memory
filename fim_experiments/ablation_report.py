from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _is_scalar_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _collect_scalar_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    return {
        str(key): float(value)
        for key, value in summary.items()
        if _is_scalar_number(value)
    }


def aggregate_manifest(manifest_path: str | Path) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    runs: List[Dict[str, Any]] = []

    for run in manifest.get("runs", []):
        summary_path = Path(run["summary"])
        if not summary_path.is_absolute():
            candidate = manifest_path.parent / summary_path
            if candidate.exists():
                summary_path = candidate
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        scalar_metrics = _collect_scalar_metrics(summary)
        variant = str(run["variant"])
        for metric, value in scalar_metrics.items():
            values[variant][metric].append(value)
        runs.append(
            {
                "variant": variant,
                "seed": int(run["seed"]),
                "summary": str(summary_path),
                "metrics": scalar_metrics,
            }
        )

    aggregates: Dict[str, Dict[str, Dict[str, float | int]]] = {}
    for variant, metric_map in values.items():
        aggregates[variant] = {}
        for metric, metric_values in sorted(metric_map.items()):
            n = len(metric_values)
            aggregates[variant][metric] = {
                "n": n,
                "mean": statistics.fmean(metric_values),
                "std": statistics.stdev(metric_values) if n > 1 else 0.0,
                "min": min(metric_values),
                "max": max(metric_values),
            }

    return {
        "source_manifest": str(manifest_path),
        "benchmark": manifest.get("benchmark"),
        "variants": manifest.get("variants", []),
        "seeds": manifest.get("seeds", []),
        "runs": runs,
        "aggregates": aggregates,
        "interpretation_rule": (
            "These are descriptive across-seed summaries only. Do not infer statistical significance "
            "or mechanism benefit without a predeclared comparison and appropriate statistical test."
        ),
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# FIM Ablation Summary",
        "",
        f"Benchmark: `{report.get('benchmark')}`",
        "",
        "Descriptive across-seed statistics only. This file does not establish statistical significance.",
        "",
    ]

    aggregates = report.get("aggregates", {})
    if not aggregates:
        lines.append("No scalar top-level metrics were found in the referenced summaries.")
        return "\n".join(lines) + "\n"

    for variant in report.get("variants", []):
        metric_map = aggregates.get(variant, {})
        if not metric_map:
            continue
        lines.extend(
            [
                f"## {variant}",
                "",
                "| Metric | n | Mean | Std | Min | Max |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for metric, stats in sorted(metric_map.items()):
            lines.append(
                f"| `{metric}` | {stats['n']} | {stats['mean']:.6g} | {stats['std']:.6g} | "
                f"{stats['min']:.6g} | {stats['max']:.6g} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            report["interpretation_rule"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate completed FIM ablation runs across seeds.")
    parser.add_argument("manifest", nargs="?", default="ablation_results/ablation_manifest.json")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    report = aggregate_manifest(manifest_path)

    json_out = Path(args.json_out) if args.json_out else manifest_path.parent / "ablation_summary.json"
    markdown_out = Path(args.markdown_out) if args.markdown_out else manifest_path.parent / "ablation_summary.md"

    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_out.write_text(render_markdown(report), encoding="utf-8")

    print(f"Wrote {json_out}")
    print(f"Wrote {markdown_out}")


if __name__ == "__main__":
    main()
