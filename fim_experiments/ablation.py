from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

from fim_experiments import main as experiment_main
from fim_experiments.systems import FIMSystem


VARIANT_CONTROLS: Dict[str, Tuple[bool, bool]] = {
    "full": (True, True),
    "no_memory": (False, False),
    "no_retrieval": (True, False),
}


def variant_controls(variant: str) -> Tuple[bool, bool]:
    try:
        return VARIANT_CONTROLS[variant]
    except KeyError as exc:
        supported = ", ".join(sorted(VARIANT_CONTROLS))
        raise ValueError(f"Unknown ablation variant '{variant}'. Supported: {supported}") from exc


def normalize_variants(variants: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for variant in variants:
        key = variant.strip().lower().replace("-", "_")
        variant_controls(key)
        if key not in seen:
            normalized.append(key)
            seen.add(key)
    if not normalized:
        raise ValueError("At least one ablation variant is required.")
    return normalized


class ControlledFIMSystem(FIMSystem):
    """FIM with explicit, immutable controls for mechanism ablations.

    The shared trainer passes its own ``store_traces``/``retrieve`` flags to
    ``step``. Ablation semantics must therefore be enforced in ``step`` itself,
    not only in ``forward``. Caller-provided values are intentionally ignored.
    """

    def __init__(self, *args: Any, store_traces: bool, retrieve: bool, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ablation_store_traces = bool(store_traces)
        self.ablation_retrieve = bool(retrieve)

    def step(
        self,
        x: torch.Tensor,
        store_traces: bool = True,
        retrieve: bool = True,
    ):
        del store_traces, retrieve
        return super().step(
            x,
            store_traces=self.ablation_store_traces,
            retrieve=self.ablation_retrieve,
        )

    def forward(self, x: torch.Tensor):
        return self.step(x)


def build_controlled_system(
    cfg: Dict[str, Any],
    benchmark: Any,
    device: torch.device,
    variant: str,
) -> torch.nn.Module:
    model_cfg = cfg.get("model", {}) or {}
    model_name = experiment_main._benchmark_name(str(model_cfg.get("name", "fim")))
    if model_name not in {"fim", "fim_plus"}:
        raise ValueError(
            "Mechanism ablations are defined only for FIM/FIM+; "
            f"received model '{model_name}'."
        )

    state_shape = experiment_main._state_shape_from_benchmark(benchmark)
    in_channels = state_shape[0] if len(state_shape) > 1 else 1
    store_traces, retrieve = variant_controls(variant)

    model = ControlledFIMSystem(
        in_channels=in_channels,
        hidden=int(model_cfg.get("hidden", 64)),
        trace_dim=int(model_cfg.get("trace_dim", 32)),
        memory_capacity=int(model_cfg.get("memory_capacity", 2048)),
        retrieval_topk=int(model_cfg.get("retrieval_topk", 32)),
        memory_decay=float(model_cfg.get("memory_decay", 0.02)),
        salience_threshold=float(model_cfg.get("salience_threshold", 0.45)),
        store_traces=store_traces,
        retrieve=retrieve,
    ).to(device)

    if bool(model_cfg.get("compile", False)) and device.type == "cuda":
        try:
            model = torch.compile(model)
        except Exception:
            pass

    return model


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return proc.stdout.strip() or None
    except Exception:
        return None


def _provenance(variant: str, seed: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    store_traces, retrieve = variant_controls(variant)
    return {
        "variant": variant,
        "controls": {
            "store_traces": store_traces,
            "retrieve": retrieve,
        },
        "seed": seed,
        "git_sha": _git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config": experiment_main._to_jsonable(cfg),
    }


def run_variant(
    *,
    variant: str,
    seed: int,
    benchmark: str,
    device: str,
    epochs: int,
    batch_size: int,
    dataset_size: int,
    rollout_steps: int,
    results_root: str,
) -> Dict[str, Any]:
    variant = normalize_variants([variant])[0]
    cfg = experiment_main.default_config()
    cfg["experiment"]["name"] = f"ablation_{variant}_seed_{seed}"
    cfg["runtime"]["seed"] = int(seed)
    cfg["runtime"]["device"] = device
    cfg["runtime"]["deterministic"] = True
    cfg["runtime"]["results_root"] = results_root
    cfg["benchmark"]["name"] = benchmark
    cfg["model"]["name"] = "fim_plus"
    cfg["train"]["epochs"] = int(epochs)
    cfg["train"]["batch_size"] = int(batch_size)
    cfg["train"]["dataset_size"] = int(dataset_size)
    cfg["train"]["workers"] = 0
    cfg["train"]["rollout_steps"] = int(rollout_steps)
    cfg["eval"]["rollout_steps"] = int(rollout_steps)

    original_builder = experiment_main.build_system
    experiment_main.build_system = (
        lambda active_cfg, active_benchmark, active_device: build_controlled_system(
            active_cfg,
            active_benchmark,
            active_device,
            variant,
        )
    )
    try:
        metrics = experiment_main.run_experiment(cfg)
    finally:
        experiment_main.build_system = original_builder

    dirs = experiment_main.build_experiment_dirs(cfg)
    provenance_path = dirs["metrics"] / "provenance.json"
    with provenance_path.open("w", encoding="utf-8") as handle:
        json.dump(_provenance(variant, seed, cfg), handle, indent=2)

    return {
        "variant": variant,
        "seed": int(seed),
        "summary": str(dirs["metrics"] / "summary.json"),
        "provenance": str(provenance_path),
        "metrics": experiment_main._to_jsonable(metrics),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evidence-bounded FIM mechanism ablations through the current experiment path."
    )
    parser.add_argument("--variants", nargs="+", default=["full", "no_memory", "no_retrieval"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--benchmark", default="lorenz96")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dataset-size", type=int, default=4096)
    parser.add_argument("--rollout-steps", type=int, default=50)
    parser.add_argument("--results-root", default="ablation_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = normalize_variants(args.variants)
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    if not seeds:
        raise ValueError("At least one seed is required.")

    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    for variant in variants:
        for seed in seeds:
            print(f"\n=== variant={variant} seed={seed} ===", flush=True)
            runs.append(
                run_variant(
                    variant=variant,
                    seed=seed,
                    benchmark=args.benchmark,
                    device=args.device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    dataset_size=args.dataset_size,
                    rollout_steps=args.rollout_steps,
                    results_root=str(root),
                )
            )

    manifest = {
        "benchmark": args.benchmark,
        "variants": variants,
        "seeds": seeds,
        "runs": [
            {
                "variant": run["variant"],
                "seed": run["seed"],
                "summary": run["summary"],
                "provenance": run["provenance"],
            }
            for run in runs
        ],
    }
    manifest_path = root / "ablation_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nAblation manifest: {manifest_path}")


if __name__ == "__main__":
    main()
