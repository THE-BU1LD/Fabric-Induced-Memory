"""Bounded development campaign. Called only by the guarded run.py worker."""

import json
import time
from pathlib import Path

import torch

from core import admission, decide, exact_sign_flip, sha, write_new
from data import causal_features, generate
from models import Base, SequenceModel, retrieve


class Budget:
    def __init__(self, protocol, output):
        self.deadline = time.monotonic() + protocol["budget"]["max_seconds"]
        self.limit = protocol["budget"]["max_output_gib"] * 1024**3
        self.output = Path(output)
        self.last_check = 0

    def check(self, force=False):
        if time.monotonic() > self.deadline:
            raise TimeoutError("campaign wall-time budget exhausted")
        if force or time.monotonic() - self.last_check >= 15:
            report = admission()
            if report["status"] != "PASS":
                raise RuntimeError("resource admission lost: " + json.dumps(report))
            if (
                sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file())
                > self.limit
            ):
                raise RuntimeError("output budget exceeded")
            self.last_check = time.monotonic()


def score(prediction, data, variance, protocol):
    errors = (prediction - data["target"]).square().mean(-1) / variance
    if not torch.isfinite(errors).all():
        raise ValueError("nonfinite predictions/errors")
    mask = data["primary"]
    counts = mask.sum(1)
    if (counts < protocol["endpoint"]["minimum_rows_per_owner"]).any():
        raise ValueError("insufficient primary observations; do not exclude owners")
    nonprimary = (~mask) & (
        torch.arange(mask.shape[1])[None, :] >= protocol["data"]["warmup"]
    )
    if (nonprimary.sum(1) == 0).any():
        raise ValueError("nonprimary endpoint empty")
    owner = (errors * mask).sum(1) / counts
    other = (errors * nonprimary).sum(1) / nonprimary.sum(1)
    return {
        "primary": float(owner.mean()),
        "nonprimary": float(other.mean()),
        "per_owner": owner.tolist(),
        "primary_counts": counts.tolist(),
    }


def select_alpha(base, correction, data, variance, protocol):
    candidates = []
    for alpha in protocol["memory"]["alphas"]:
        candidates.append(
            (
                score(base + alpha * correction, data, variance, protocol)["primary"],
                alpha,
            )
        )
    value, alpha = min(candidates)
    return alpha, value, candidates


def fit(model, features, targets, protocol, budget, seed, epochs, log_path):
    torch.manual_seed(seed)
    model.train()
    cfg = protocol["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    history = []
    updates = 0
    start = time.monotonic()
    with log_path.open("x") as handle:
        handle.write(
            json.dumps({"status": "STARTED", "seed": seed, "epochs": epochs}) + "\n"
        )
    for epoch in range(epochs):
        budget.check()
        order = torch.randperm(len(features))
        losses = []
        for indices in order.split(cfg["batch_size"]):
            budget.check()
            optimizer.zero_grad(set_to_none=True)
            predicted = model(features[indices])
            loss = (
                (
                    predicted[:, protocol["data"]["warmup"] :]
                    - targets[indices, protocol["data"]["warmup"] :]
                )
                .square()
                .mean()
            )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach()))
            updates += 1
        history.append(sum(losses) / len(losses))
        with log_path.open("a") as handle:
            handle.write(
                json.dumps({"epoch": epoch, "loss": history[-1], "updates": updates})
                + "\n"
            )
    model.eval()
    return {
        "epoch_loss": history,
        "updates": updates,
        "fit_seconds": time.monotonic() - start,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def memory_options(protocol, mode, temperature):
    cfg = protocol["memory"]
    return {
        "capacity": cfg["capacity"],
        "topk": 1 if mode in ("knn1", "last_value", "oracle") else cfg["topk"],
        "temperature": temperature,
        "minimum_age": cfg["minimum_age"],
        "maximum_age": cfg["recent_max_age"] if mode == "recent" else None,
        "mode": mode,
    }


def predict(model, features, batch_size):
    with torch.no_grad():
        return torch.cat([model(x) for x in features.split(batch_size)])


def cell(protocol, condition, seed, output, budget):
    output.mkdir(exist_ok=False)
    train = generate(protocol, condition, seed, "train")
    val = generate(protocol, condition, seed, "validation")
    if not set(train["owners"]).isdisjoint(val["owners"]):
        raise ValueError("training/validation owner overlap")
    variance = float(train["target"].var(unbiased=False))
    if variance <= 0:
        raise ValueError("invalid training normalizer")
    torch.save({"train": train, "validation": val}, output / "development_data.pt")
    torch.manual_seed(seed)
    base_model = Base(protocol["data"]["cue_dim"], protocol["data"]["value_dim"])
    base_training = fit(
        base_model,
        train["cue"],
        train["target"],
        protocol,
        budget,
        seed,
        protocol["training"]["base_epochs"],
        output / "base-training.jsonl",
    )
    torch.save(base_model.state_dict(), output / "base.pt")
    train_base = predict(base_model, train["cue"], protocol["training"]["batch_size"])
    val_base = predict(base_model, val["cue"], protocol["training"]["batch_size"])
    train_residual = train["target"] - train_base
    val_residual = val["target"] - val_base
    train_features = causal_features(train["cue"], train_residual)
    val_features = causal_features(val["cue"], val_residual)
    selected = {
        "base": {
            "validation": score(val_base, val, variance, protocol)["primary"],
            "alpha": 0.0,
        }
    }
    candidates = {"base": base_training}
    fitted = {}
    for kind in ("finite_history", "gru", "attention"):
        alternatives = []
        for width in protocol["training"]["widths"]:
            budget.check(force=True)
            torch.manual_seed(seed + width)
            history = (
                protocol["training"]["attention_history"]
                if kind == "attention"
                else protocol["training"]["history"]
            )
            model = SequenceModel(
                kind,
                train_features.shape[-1],
                protocol["data"]["value_dim"],
                width,
                history,
            )
            training = fit(
                model,
                train_features,
                train_residual,
                protocol,
                budget,
                seed,
                protocol["training"]["epochs"],
                output / f"{kind}-{width}-training.jsonl",
            )
            correction = predict(
                model, val_features, protocol["training"]["batch_size"]
            )
            alpha, value, grid = select_alpha(
                val_base, correction, val, variance, protocol
            )
            name = f"{kind}-{width}"
            torch.save(model.state_dict(), output / f"{name}.pt")
            candidates[name] = {
                **training,
                "alpha_grid": grid,
                "validation": value,
                "alpha": alpha,
                "width": width,
                "history": history,
            }
            alternatives.append((value, width, alpha, model))
        value, width, alpha, model = min(alternatives, key=lambda row: (row[0], row[1]))
        fitted[kind] = model
        selected[kind] = {"width": width, "alpha": alpha, "validation": value}
    for mode in ("tensor", "knn1", "knn4", "recent", "last_value"):
        alternatives = []
        # All retrieval families receive the same declared temperature/alpha search budget.
        # Temperature is redundant for top1/uniform kNN; duplicated results stay visible.
        for temperature in protocol["memory"]["temperatures"]:
            budget.check()
            result = retrieve(
                val["cue"], val_residual, **memory_options(protocol, mode, temperature)
            )
            alpha, value, grid = select_alpha(
                val_base, result["correction"], val, variance, protocol
            )
            candidates[f"{mode}-{temperature}"] = {
                "validation": value,
                "alpha_grid": grid,
                "parameters": 0,
            }
            alternatives.append((value, temperature, alpha))
        value, temperature, alpha = min(alternatives)
        selected[mode] = {
            "temperature": temperature,
            "alpha": alpha,
            "validation": value,
        }
    strongest = min(
        protocol["baselines"],
        key=lambda name: (
            selected[name]["validation"],
            protocol["baselines"].index(name),
        ),
    )
    # Written BEFORE generation/access of development evaluation outcomes.
    write_new(
        output / "SELECTION.json",
        {
            "selected": selected,
            "strongest_baseline": strongest,
            "candidates": candidates,
            "normalization_variance": variance,
            "evaluation_accessed": False,
        },
    )
    budget.check(force=True)
    evaluation = generate(protocol, condition, seed, "evaluation")
    if not set(evaluation["owners"]).isdisjoint(train["owners"] + val["owners"]):
        raise ValueError("evaluation owner overlap")
    base = predict(base_model, evaluation["cue"], protocol["training"]["batch_size"])
    residual = evaluation["target"] - base
    features = causal_features(evaluation["cue"], residual)
    predictions = {"base": base}
    retrievals = {}
    timings = {}
    for kind, model in fitted.items():
        start = time.monotonic()
        predictions[kind] = base + selected[kind]["alpha"] * predict(
            model, features, protocol["training"]["batch_size"]
        )
        timings[kind] = time.monotonic() - start
    for mode in (
        "tensor",
        "knn1",
        "knn4",
        "recent",
        "last_value",
        "oracle",
        *protocol["interventions"],
    ):
        budget.check()
        config = selected.get(mode, selected["tensor"])
        options = memory_options(protocol, mode, config["temperature"])
        start = time.monotonic()
        result = retrieve(
            evaluation["cue"],
            residual,
            **options,
            labels=evaluation["anchor"] if mode == "oracle" else None,
        )
        retrievals[mode] = result
        predictions[mode] = (
            base + (1.0 if mode == "oracle" else config["alpha"]) * result["correction"]
        )
        timings[mode] = time.monotonic() - start
    raw = {
        "data": evaluation,
        "base": base,
        "predictions": predictions,
        "retrievals": retrievals,
        "variance": variance,
    }
    torch.save(raw, output / "raw.pt")
    scores = {
        name: score(pred, evaluation, variance, protocol)
        for name, pred in predictions.items()
    }
    metrics = {name: scores[name]["primary"] for name in scores}
    metrics["strongest"] = metrics[strongest]
    metrics["nonprimary_delta"] = (
        scores["tensor"]["nonprimary"] - scores["base"]["nonprimary"]
    )
    metrics["permutation_fraction"] = float(
        retrievals["value_permutation"]["permuted"][evaluation["primary"]]
        .float()
        .mean()
    )
    result = {
        "seed": seed,
        "condition": condition,
        "strongest_validation_baseline": strongest,
        "metrics": metrics,
        "scores": scores,
        "decision": decide(metrics, protocol["decision"]),
        "timings_seconds": timings,
        "information_budget": {
            "memory_float_slots_per_owner": protocol["memory"]["capacity"]
            * (protocol["data"]["cue_dim"] + protocol["data"]["value_dim"]),
            "gru_hidden_slots": selected["gru"]["width"],
            "attention_window_events": protocol["training"]["attention_history"],
            "finite_history_events": protocol["training"]["history"],
            "fairness": "Same streams/targets and learned frozen base; two widths per trained family and equal update budgets. State, FLOPs and parameters are reported, not claimed identical.",
        },
        "artifacts": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    }
    write_new(output / "RESULT.json", result)
    return result


def run(protocol, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    budget = Budget(protocol, output)
    results = []
    for condition in protocol["conditions"]:
        for seed in protocol["seeds"]:
            budget.check(force=True)
            result = cell(
                protocol,
                condition,
                seed,
                output / f"{condition['name']}-{seed}",
                budget,
            )
            results.append(result)
    comparisons = {}
    for condition in protocol["conditions"]:
        rows = [r for r in results if r["condition"]["name"] == condition["name"]]
        differences = [r["metrics"]["strongest"] - r["metrics"]["tensor"] for r in rows]
        comparisons[condition["name"]] = {
            "paired_seed_differences": differences,
            "mean": sum(differences) / len(differences),
            "exact_two_sided_sign_flip_p": exact_sign_flip(differences),
            "multiplicity": "Descriptive development analysis; no significance claim. Five seeds cannot yield two-sided p below 0.0625.",
        }
    all_pass = all(
        r["decision"]["status"] == "SUPPORTED_BOUNDED_DEVELOPMENT"
        for r in results
        if r["condition"]["name"] != "clean"
    )
    summary = {
        "status": "SUPPORTED_BOUNDED_DEVELOPMENT" if all_pass else "NOT_SUPPORTED",
        "completed_cells": len(results),
        "expected_cells": len(protocol["seeds"]) * len(protocol["conditions"]),
        "comparisons": comparisons,
        "boundary": "Constructed synthetic development; no natural-domain, public benchmark, novelty or external reproduction claim.",
    }
    write_new(output / "SUMMARY.json", summary)
    return summary
