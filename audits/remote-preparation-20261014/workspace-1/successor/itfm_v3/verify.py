"""Recompute endpoints and retrieval from stored tensors, without production retrieval.

This is a separate calculation by the same operator, not external reproduction.
"""

import csv
import json
import math

import torch

from core import decide, sha, write_new
from data import causal_features
from experiment import Budget
from models import Base, SequenceModel


def reference(cue, values, labels, t, config, protocol, mode):
    begin = max(0, t - protocol["memory"]["capacity"])
    if mode == "recent":
        begin = max(begin, t - protocol["memory"]["recent_max_age"])
    times = list(range(begin, t))
    if mode == "oracle":
        times = [s for s in times if int(labels[s]) == int(labels[t])]
    if not times:
        return [0.0] * values.shape[-1], []
    if mode in ("oracle", "last_value"):
        chosen = [max(times)]
        weights = [1.0]
    else:
        distances = {s: float((cue[t] - cue[s]).square().mean()) for s in times}
        count = 1 if mode == "knn1" else protocol["memory"]["topk"]
        chosen = sorted(times, key=lambda s: (distances[s], s))[:count]
        offset = min(distances[s] for s in chosen)
        weights = [
            1.0
            if mode == "knn4"
            else math.exp(-(distances[s] - offset) / config["temperature"])
            for s in chosen
        ]
        total = sum(weights)
        weights = [w / total for w in weights]
    source = chosen
    if mode == "value_permutation":
        source = [times[(times.index(s) + 1) % len(times)] for s in chosen]
    value = [
        sum(w * float(values[s, d]) for w, s in zip(weights, source))
        for d in range(values.shape[-1])
    ]
    if mode == "erase":
        value = [0.0] * len(value)
    if mode == "sign_flip":
        value = [-v for v in value]
    if mode == "spatial_mean":
        value = [sum(value) / len(value)] * len(value)
    return value, chosen


def verify(run_root, output, protocol):
    torch.set_num_threads(1)
    budget = Budget(protocol, output)
    checks = {}
    table = []
    for condition in protocol["conditions"]:
        for seed in protocol["seeds"]:
            budget.check(force=True)
            name = f"{condition['name']}-{seed}"
            folder = run_root / name
            result = json.loads((folder / "RESULT.json").read_text())
            selection = json.loads((folder / "SELECTION.json").read_text())
            checks[name + ":hashes"] = all(
                sha(folder / file) == digest
                for file, digest in result["artifacts"].items()
            )
            raw = torch.load(folder / "raw.pt", map_location="cpu", weights_only=True)
            data = raw["data"]
            base = raw["base"]
            variance = raw["variance"]
            reconstructed = torch.zeros_like(data["primary"])
            for owner in range(len(data["owners"])):
                seen = {}
                for t, label in enumerate(data["anchor"][owner].tolist()):
                    prior = seen.get(label)
                    reconstructed[owner, t] = (
                        prior is not None
                        and t - prior > protocol["training"]["history"]
                        and t >= protocol["data"]["warmup"]
                    )
                    seen[label] = t
            checks[name + ":primary_mask"] = torch.equal(
                reconstructed, data["primary"]
            ) and bool(
                (
                    reconstructed.sum(1)
                    >= protocol["endpoint"]["minimum_rows_per_owner"]
                ).all()
            )
            development = torch.load(
                folder / "development_data.pt", map_location="cpu", weights_only=True
            )
            checks[name + ":split"] = all(
                set(data["owners"]).isdisjoint(development[s]["owners"])
                for s in ("train", "validation")
            )
            checks[name + ":variance"] = math.isclose(
                variance,
                float(development["train"]["target"].var(unbiased=False)),
                rel_tol=1e-7,
            )
            metrics = {}
            nonprimary = {}
            for mode, predictions in raw["predictions"].items():
                owners = []
                others = []
                for owner in range(len(data["owners"])):
                    squared = (
                        predictions[owner].double() - data["target"][owner].double()
                    ).square().mean(-1) / variance
                    mask = data["primary"][owner]
                    eligible = torch.arange(len(mask)) >= protocol["data"]["warmup"]
                    owners.append(float(squared[mask].mean()))
                    others.append(float(squared[eligible & ~mask].mean()))
                metrics[mode] = sum(owners) / len(owners)
                nonprimary[mode] = sum(others) / len(others)
                expected = result["scores"][mode]
                checks[name + ":score:" + mode] = math.isclose(
                    metrics[mode], expected["primary"], rel_tol=2e-5, abs_tol=2e-6
                ) and math.isclose(
                    nonprimary[mode], expected["nonprimary"], rel_tol=2e-5, abs_tol=2e-6
                )
                table.append(
                    {
                        "condition": condition["name"],
                        "seed": seed,
                        "model": mode,
                        "primary_nmse": metrics[mode],
                        "nonprimary_nmse": nonprimary[mode],
                    }
                )
            residual = data["target"] - base
            # Reexecute the selected saved weights. This reuses model definitions;
            # it is distinct from the independent arithmetic/retrieval checks below.
            base_model = Base(
                protocol["data"]["cue_dim"], protocol["data"]["value_dim"]
            ).eval()
            base_model.load_state_dict(
                torch.load(folder / "base.pt", map_location="cpu", weights_only=True)
            )
            with torch.no_grad():
                regenerated_base = torch.cat(
                    [
                        base_model(x)
                        for x in data["cue"].split(protocol["training"]["batch_size"])
                    ]
                )
            checks[name + ":base_checkpoint"] = torch.allclose(
                regenerated_base, base, atol=1e-6, rtol=1e-6
            )
            features = causal_features(data["cue"], residual)
            for kind in ("finite_history", "gru", "attention"):
                config = selection["selected"][kind]
                history = (
                    protocol["training"]["attention_history"]
                    if kind == "attention"
                    else protocol["training"]["history"]
                )
                model = SequenceModel(
                    kind,
                    features.shape[-1],
                    protocol["data"]["value_dim"],
                    config["width"],
                    history,
                ).eval()
                model.load_state_dict(
                    torch.load(
                        folder / f"{kind}-{config['width']}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                with torch.no_grad():
                    prediction = base + config["alpha"] * torch.cat(
                        [
                            model(x)
                            for x in features.split(protocol["training"]["batch_size"])
                        ]
                    )
                checks[name + ":checkpoint:" + kind] = torch.allclose(
                    prediction, raw["predictions"][kind], atol=1e-6, rtol=1e-6
                )
            strongest = min(
                protocol["baselines"],
                key=lambda n: (
                    selection["selected"][n]["validation"],
                    protocol["baselines"].index(n),
                ),
            )
            checks[name + ":baseline_selection"] = (
                strongest == selection["strongest_baseline"]
            )
            for mode, retrieval in raw["retrievals"].items():
                config = selection["selected"].get(
                    mode, selection["selected"]["tensor"]
                )
                ok = True
                for owner in range(len(data["owners"])):
                    budget.check()
                    for t in range(data["cue"].shape[1]):
                        value, sources = reference(
                            data["cue"][owner],
                            residual[owner],
                            data["anchor"][owner],
                            t,
                            config,
                            protocol,
                            mode,
                        )
                        actual = retrieval["sources"][owner, t]
                        ok &= sources == actual[actual >= 0].tolist()
                        correction = torch.tensor(value)
                        ok &= torch.allclose(
                            correction,
                            retrieval["correction"][owner, t],
                            atol=3e-5,
                            rtol=3e-5,
                        )
                        alpha = 1.0 if mode == "oracle" else config["alpha"]
                        ok &= torch.allclose(
                            base[owner, t] + alpha * correction,
                            raw["predictions"][mode][owner, t],
                            atol=3e-5,
                            rtol=3e-5,
                        )
                checks[name + ":reference:" + mode] = bool(ok)
            strongest = selection["strongest_baseline"]
            metrics["strongest"] = metrics[strongest]
            metrics["nonprimary_delta"] = nonprimary["tensor"] - nonprimary["base"]
            metrics["permutation_fraction"] = float(
                raw["retrievals"]["value_permutation"]["permuted"][data["primary"]]
                .float()
                .mean()
            )
            checks[name + ":decision"] = (
                decide(metrics, protocol["decision"]) == result["decision"]
            )
    with (output / "metrics.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    receipt = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "boundary": "Independent arithmetic/retrieval implementation on retained tensors; same host/operator and reused generator. Not external reproduction.",
    }
    write_new(output / "VERIFICATION.json", receipt)
    if not all(checks.values()):
        raise RuntimeError("raw evidence discrepancies retained")
