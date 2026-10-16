"""Prospective raw-evidence integrity checks; no model or scientific-claim authority."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

MAX_INPUT_BYTES = 8 * 1024 * 1024
CLAIM_BOUNDARY = "NONE_REQUIRES_FROZEN_SCIENTIFIC_PROTOCOL_AND_EXTERNAL_REVIEW"


class EvidenceError(ValueError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.details = details


def reject(code, message, details=None):
    raise EvidenceError(code, message, details)


def object_fields(value, required, optional=(), where="object"):
    if not isinstance(value, dict) or not set(required) <= value.keys():
        reject("SCHEMA", f"{where}: missing required object fields")
    if set(value) - set(required) - set(optional):
        reject("SCHEMA", f"{where}: unexpected fields")
    return value


def array(value, where):
    if not isinstance(value, list) or not value:
        reject("SCHEMA", f"{where}: nonempty list required")
    return value


def identifier(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        reject("IDENTITY", "Nonempty identifier of at most 256 characters required")
    return value


def integer(value, where, minimum=None, maximum=None):
    if type(value) is not int:
        reject("TYPE", f"{where}: integer required")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        reject("RANGE", f"{where}: integer outside declared bounds")
    return value


def finite(value, where="number", minimum=None, maximum=None):
    if type(value) not in (int, float):
        reject("TYPE", f"{where}: real JSON number required")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        reject("NONFINITE", f"{where}: number cannot be represented finitely")
    if not math.isfinite(result):
        reject("NONFINITE", f"{where}: nonfinite number")
    if (minimum is not None and result < minimum) or (maximum is not None and result > maximum):
        reject("RANGE", f"{where}: number outside declared bounds")
    return result


def digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        reject("IDENTITY", "Lowercase SHA-256 identity required")
    return value


def mean(values):
    if not values:
        reject("EMPTY", "Cannot average an empty collection")
    try:
        return finite(math.fsum(x / len(values) for x in values), "mean")
    except OverflowError:
        reject("NONFINITE", "Mean overflow")


def vector(value):
    values = array(value, "vector")
    if len(values) > 100_000:
        reject("LIMIT", "Vector exceeds reference implementation limit")
    return tuple(finite(x, "vector element") for x in values)


def sample_metrics(prediction, target, epsilon):
    p, y = vector(prediction), vector(target)
    epsilon = finite(epsilon, "epsilon", minimum=0)
    if epsilon == 0 or len(p) != len(y):
        reject("METRIC", "Positive epsilon and equal nonempty vector shapes required")
    try:
        squared = [finite((a - b) ** 2, "squared error") for a, b in zip(p, y, strict=True)]
        energy = finite(math.fsum(finite(b * b, "target energy") for b in y), "target energy sum")
        numerator = finite(math.fsum(squared), "error energy sum")
        l2 = finite(math.sqrt(finite(numerator / max(energy, epsilon), "relative energy")), "L2")
        return {"mse": mean(squared), "relative_l2": l2}
    except OverflowError:
        reject("NONFINITE", "Metric arithmetic overflow")


def verify_metric_claims(claimed, recomputed, tolerance=1e-10):
    tolerance = finite(tolerance, "tolerance", minimum=0)
    if not isinstance(claimed, dict) or not isinstance(recomputed, dict) or not claimed or claimed.keys() != recomputed.keys():
        reject("METRIC_KEYS", "Claimed and recomputed metric names must match exactly")
    for name in claimed:
        a, b = finite(claimed[name], name), finite(recomputed[name], name)
        if abs(a - b) > tolerance:
            reject("METRIC_MISMATCH", f"Metric differs: {name}")


def sample_key(row):
    return (identifier(row["cell_id"]), integer(row["seed"], "seed"),
            identifier(row["trajectory_id"]), integer(row["origin"], "origin", minimum=0))


def validate_plan(plan):
    object_fields(plan, ("schema_version", "study_id", "source_sha256", "data_sha256",
                        "methods", "cells", "seeds", "samples", "metric", "gates", "bootstrap"), where="plan")
    integer(plan["schema_version"], "schema version", 1, 1)
    identifier(plan["study_id"])
    digest(plan["source_sha256"])
    digest(plan["data_sha256"])
    methods = object_fields(plan["methods"], ("candidate", "baseline"), where="methods")
    names = [identifier(methods[k]) for k in ("candidate", "baseline")]
    if len(set(names)) != 2:
        reject("IDENTITY", "Candidate and baseline must be distinct")
    cells = {}
    for cell in array(plan["cells"], "cells"):
        object_fields(cell, ("cell_id", "family_id"), where="cell")
        name, family = identifier(cell["cell_id"]), identifier(cell["family_id"])
        if name in cells:
            reject("DUPLICATE", "Duplicate planned cell")
        cells[name] = family
    seeds = [integer(x, "seed") for x in array(plan["seeds"], "seeds")]
    if len(seeds) != len(set(seeds)):
        reject("DUPLICATE", "Duplicate planned seed")
    expected = set()
    for sample in array(plan["samples"], "samples"):
        object_fields(sample, ("cell_id", "seed", "trajectory_id", "origin"), where="sample")
        key = sample_key(sample)
        if key in expected:
            reject("DUPLICATE", "Duplicate planned sample")
        if key[0] not in cells or key[1] not in seeds:
            reject("IDENTITY", "Sample references unplanned cell or seed")
        expected.add(key)
    if {(k[0], k[1]) for k in expected} != {(c, s) for c in cells for s in seeds}:
        reject("COVERAGE", "Every cell/seed block needs prospective samples")
    metric = object_fields(plan["metric"], ("name", "epsilon"), where="metric")
    if metric["name"] != "mean_sample_relative_l2" or finite(metric["epsilon"], "epsilon", minimum=0) == 0:
        reject("METRIC", "Declared mean-sample L2 and positive energy floor required")
    gates = object_fields(plan["gates"], ("minimum_macro_gain", "minimum_cells_won",
        "maximum_cell_regression", "minimum_seed_win_fraction", "minimum_families_won",
        "maximum_family_regression", "maximum_candidate_error"), where="gates")
    finite(gates["minimum_macro_gain"], minimum=0, maximum=1)
    integer(gates["minimum_cells_won"], "cell wins", 0, len(cells))
    finite(gates["maximum_cell_regression"], minimum=0)
    finite(gates["minimum_seed_win_fraction"], minimum=0, maximum=1)
    integer(gates["minimum_families_won"], "family wins", 0, len(set(cells.values())))
    finite(gates["maximum_family_regression"], minimum=0)
    if gates["maximum_candidate_error"] is not None:
        finite(gates["maximum_candidate_error"], minimum=0)
    bootstrap = object_fields(plan["bootstrap"], ("repetitions", "confidence", "seed"), where="bootstrap")
    integer(bootstrap["repetitions"], "bootstrap repetitions", 100, 10_000)
    confidence = finite(bootstrap["confidence"], "confidence")
    if not 0.5 < confidence < 1:
        reject("RANGE", "Confidence must lie strictly between .5 and 1")
    integer(bootstrap["seed"], "bootstrap seed")
    return cells, tuple(seeds), expected


def paired_rows(plan, evidence, expected):
    object_fields(evidence, ("schema_version", "source_sha256", "data_sha256", "rows"),
                  ("claimed_metrics",), where="evidence")
    integer(evidence["schema_version"], "schema version", 1, 1)
    for field in ("source_sha256", "data_sha256"):
        if digest(evidence[field]) != plan[field]:
            reject("PROVENANCE", f"Evidence does not match planned {field}")
    methods = set(plan["methods"].values())
    required = {(method, *key) for method in methods for key in expected}
    indexed, failures = {}, []
    for row in array(evidence["rows"], "rows"):
        object_fields(row, ("method", "cell_id", "seed", "trajectory_id", "origin", "status"),
                      ("prediction", "target", "error"), where="row")
        key = (identifier(row["method"]), *sample_key(row))
        if key in indexed:
            reject("DUPLICATE", "Duplicate evidence identity", list(key))
        if key not in required:
            reject("UNEXPECTED", "Unplanned evidence identity", list(key))
        indexed[key] = row
        status = identifier(row["status"])
        if "error" in row and not isinstance(row["error"], str):
            reject("SCHEMA", "Failure detail must be a string")
        if status != "PASS":
            failures.append({"key": list(key), "status": status, "error": row.get("error")})
    missing = sorted(required - indexed.keys())
    if failures or missing:
        reject("INCOMPLETE", "Failed or absent planned evidence prohibits aggregation",
               {"failed_rows": failures, "missing_rows": [list(k) for k in missing]})
    validated = {}
    for key, row in indexed.items():
        if "prediction" not in row or "target" not in row or "error" in row:
            reject("SCHEMA", "Successful row requires prediction/target and no error")
        prediction, target = vector(row["prediction"]), vector(row["target"])
        if len(prediction) != len(target):
            reject("SHAPE", "Paired vectors have different lengths")
        validated[key] = (prediction, target)
    candidate, baseline = (plan["methods"][k] for k in ("candidate", "baseline"))
    for key in expected:
        if validated[(candidate, *key)][1] != validated[(baseline, *key)][1]:
            reject("TARGET_MISMATCH", "Methods have different target vectors", list(key))
    return validated


def relative_gain(candidate, baseline):
    candidate = finite(candidate, "candidate error", minimum=0)
    baseline = finite(baseline, "baseline error", minimum=0)
    if baseline == 0:
        if candidate == 0:
            return 0.0
        reject("UNDEFINED_GAIN", "Positive candidate error against zero baseline error")
    return finite((baseline - candidate) / baseline, "relative gain")


def cell_macro_bootstrap(cell_gains, cell_families, *, repetitions, confidence, seed):
    if not isinstance(cell_gains, dict) or not isinstance(cell_families, dict) or not cell_gains or cell_gains.keys() != cell_families.keys():
        reject("COVERAGE", "Bootstrap needs matching nonempty cell identities")
    for cell in cell_gains:
        identifier(cell)
    integer(repetitions, "repetitions", 100, 10_000)
    confidence = finite(confidence, "confidence")
    if not 0.5 < confidence < 1:
        reject("RANGE", "Invalid confidence")
    integer(seed, "seed")
    groups = defaultdict(list)
    for cell in sorted(cell_gains):
        groups[identifier(cell_families[cell])].append(finite(cell_gains[cell], "gain"))
    result = {"estimate": mean(list(cell_gains.values())), "clusters": len(groups),
              "independent_unit_assumption": "family; not empirically established by this tool",
              "confidence": confidence, "repetitions": repetitions, "seed": seed,
              "interval_status": "UNAVAILABLE_SINGLE_CLUSTER", "lower": None, "upper": None,
              "confirmatory_authorized": False}
    if len(groups) < 2:
        return result
    families = sorted(groups)
    rng = random.Random(seed)
    distribution = []
    for _ in range(repetitions):
        sampled = [rng.choice(families) for _ in families]
        distribution.append(mean([g for family in sampled for g in groups[family]]))
    distribution.sort()
    tail = (1 - confidence) / 2
    lower = max(0, math.ceil(tail * repetitions) - 1)
    upper = min(repetitions - 1, math.ceil((1 - tail) * repetitions) - 1)
    result.update(interval_status="EXPLORATORY_NOT_CONFIRMATORY", lower=distribution[lower], upper=distribution[upper])
    return result


def analyze(plan, evidence):
    cells, seeds, expected = validate_plan(plan)
    raw = paired_rows(plan, evidence, expected)
    trajectory = defaultdict(lambda: defaultdict(list))
    for key in sorted(raw):
        prediction, target = raw[key]
        metrics = sample_metrics(list(prediction), list(target), plan["metric"]["epsilon"])
        for metric, value in metrics.items():
            trajectory[key[:-1]][metric].append(value)
    seed_values = defaultdict(lambda: defaultdict(list))
    for key in sorted(trajectory):
        for metric, values in trajectory[key].items():
            seed_values[key[:-1]][metric].append(mean(values))
    reduced_seeds = {key: {metric: mean(values) for metric, values in metrics.items()}
                     for key, metrics in seed_values.items()}
    methods = plan["methods"]
    cell_values = {}
    for method in methods.values():
        for cell in sorted(cells):
            cell_values[(method, cell)] = {metric: mean([reduced_seeds[(method, cell, seed)][metric] for seed in seeds])
                                          for metric in ("mse", "relative_l2")}
    summaries = {method: {metric: mean([cell_values[(method, cell)][metric] for cell in sorted(cells)])
                          for metric in ("mse", "relative_l2")} for method in methods.values()}
    if "claimed_metrics" in evidence:
        claims = object_fields(evidence["claimed_metrics"], methods.values(), where="claimed metrics")
        for method in methods.values():
            verify_metric_claims(claims[method], summaries[method])
    candidate, baseline = methods["candidate"], methods["baseline"]
    gain, seed_wins = {}, {}
    for cell in sorted(cells):
        gain[cell] = relative_gain(cell_values[(candidate, cell)]["relative_l2"], cell_values[(baseline, cell)]["relative_l2"])
        seed_wins[cell] = sum(reduced_seeds[(candidate, cell, seed)]["relative_l2"] < reduced_seeds[(baseline, cell, seed)]["relative_l2"]
                              for seed in seeds) / len(seeds)
    family_gains = {family: mean([gain[cell] for cell in sorted(cells) if cells[cell] == family])
                    for family in sorted(set(cells.values()))}
    gates = plan["gates"]
    checks = {
        "minimum_macro_gain": mean(list(gain.values())) >= gates["minimum_macro_gain"],
        "minimum_cells_won": sum(x > 0 for x in gain.values()) >= gates["minimum_cells_won"],
        "maximum_cell_regression": max(max(0., -x) for x in gain.values()) <= gates["maximum_cell_regression"],
        "minimum_seed_win_fraction": min(seed_wins.values()) >= gates["minimum_seed_win_fraction"],
        "minimum_families_won": sum(x > 0 for x in family_gains.values()) >= gates["minimum_families_won"],
        "maximum_family_regression": max(max(0., -x) for x in family_gains.values()) <= gates["maximum_family_regression"],
        "maximum_candidate_error": gates["maximum_candidate_error"] is None or summaries[candidate]["relative_l2"] <= gates["maximum_candidate_error"],
    }
    return {"status": "PASS_INTEGRITY", "study_id": plan["study_id"],
            "planned_samples": len(expected), "accepted_rows": len(raw),
            "metrics": summaries, "cell_gains": gain, "cell_seed_win_fractions": seed_wins,
            "family_gains": family_gains, "numerical_gates": checks,
            "numerical_gates_pass": all(checks.values()),
            "bootstrap": cell_macro_bootstrap(gain, cells, **plan["bootstrap"]),
            "claim_authorization": CLAIM_BOUNDARY}


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                reject("DUPLICATE_JSON_KEY", "Duplicate JSON object key")
            result[key] = value
        return result

    def constant(value):
        reject("NONFINITE", "Nonstandard JSON numerical constant: " + value)

    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=constant)
    except EvidenceError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        reject("JSON", str(error))


def read_bounded(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        reject("LIMIT", "Input exceeds 8 MiB reference CLI limit")
    return raw


def write_receipt(path, result):
    """Publish complete JSON atomically and refuse to replace any existing path."""
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".evidence-receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.receipt.exists():
        print("Refusing to overwrite receipt", file=sys.stderr)
        return 3
    provenance = {"plan_path": str(args.plan.resolve()), "evidence_path": str(args.evidence.resolve()),
                  "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "python": sys.version.split()[0], "claim_authorization": CLAIM_BOUNDARY}
    try:
        plan_bytes, evidence_bytes = read_bounded(args.plan), read_bounded(args.evidence)
        provenance["plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
        provenance["evidence_sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
        if provenance["plan_sha256"] != digest(args.plan_sha256):
            reject("PLAN_HASH", "Plan does not match separately supplied expected identity")
        result = analyze(strict_json(plan_bytes), strict_json(evidence_bytes))
        code = 0
    except (EvidenceError, OSError) as error:
        result = {"status": "FAIL_INTEGRITY", "error_code": getattr(error, "code", "IO"),
                  "reason": str(error), "details": getattr(error, "details", None)}
        code = 2
    result["provenance"] = provenance
    result["claim_authorization"] = CLAIM_BOUNDARY
    try:
        write_receipt(args.receipt, result)
    except OSError as error:
        print(f"Could not create new receipt: {error}", file=sys.stderr)
        return 3
    print(json.dumps({"status": result["status"], "receipt": str(args.receipt), "claim_authorization": CLAIM_BOUNDARY}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
