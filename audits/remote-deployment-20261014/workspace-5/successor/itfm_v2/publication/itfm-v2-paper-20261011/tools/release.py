"""Read-only evidence verification and non-overwriting paper release gates.

Uses the standard library only. Never executes the archived runner in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import runpy
import shutil
import signal
import subprocess
import time
from pathlib import Path

PUB = Path(__file__).resolve().parents[1]
ROOT = PUB.parents[1]
SEEDS = (101, 211, 307)
PAPER_FILES = ("main.tex", "references.bib", "generated_results.tex")


def bounded_command(command, *, cwd, env=None, timeout):
    """Bound the complete child process group, including nested experiment jobs."""
    started = time.time()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    return {
        "command": command,
        "exit_code": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "seconds": time.time() - started,
    }


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path escapes evidence root")
    return path


def write_new(path, payload):
    path = Path(path).resolve()
    protected = (PUB / "frozen_source", PUB / "before")
    if (path.is_relative_to(ROOT) and not path.is_relative_to(PUB)) or any(
        path.is_relative_to(p.resolve()) for p in protected
    ):
        raise ValueError("refusing to write in frozen evidence/source")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def resource_admission():
    def sample():
        vm = subprocess.check_output(["vm_stat"], text=True)
        page = int(re.search(r"page size of (\d+)", vm)[1])

        def value(name):
            return int(re.search(re.escape(name) + r":\s+(\d+)", vm)[1])

        swap = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
        return {
            "free_plus_speculative_bytes": page
            * (value("Pages free") + value("Pages speculative")),
            "pageouts": value("Pageouts"),
            "swap_used_mib": float(re.search(r"used = ([0-9.]+)M", swap)[1]),
        }

    try:
        a = sample()
        time.sleep(0.5)
        b = sample()
        passed = (
            min(a["free_plus_speculative_bytes"], b["free_plus_speculative_bytes"])
            >= 536870912
            and a["pageouts"] == b["pageouts"]
            and b["swap_used_mib"] <= a["swap_used_mib"]
        )
        return {
            "status": "PASS" if passed else "BLOCKED",
            "samples": [a, b],
            "threshold_bytes": 536870912,
        }
    except (OSError, subprocess.CalledProcessError, TypeError, ValueError) as exc:
        return {"status": "BLOCKED", "reason": str(exc)}


def table_text(root=ROOT):
    results = {
        seed: read(root / f"results/development-v2/seed-{seed}/result.json")
        for seed in SEEDS
    }
    modes = (
        ("no_memory", "Base output"),
        ("tensor_memory", "Tensor memory"),
        ("erase", "Erase"),
        ("sign_flip", "Sign flip"),
        ("vector_broadcast", "Spatial mean"),
        ("recent_only", "Recent restriction (empty)"),
        ("last_value", "Last residual"),
        ("paired_swap", "Paired swap"),
        ("oracle_anchor", "Oracle anchor"),
    )
    lines = [
        r"% Generated from unchanged retained records by publication tools/release.py.",
        r"\begin{table}[ht]\centering\small",
        (
            "\\caption{Query NMSE for every declared condition and seed. E"
            "ach cell uses 108 query rows. Paired-swap error is measured "
            "against the recipient target, while its diagnostic success r"
            "ate below concerns the donor.}"
        ),
        r"\label{tab:results}\begin{tabular}{lrrr}\toprule",
        r"Condition & 101 & 211 & 307 \\\midrule",
    ]
    for mode, label in modes:
        cells = [f"{results[s]['summaries'][mode]['query_nmse']:.6f}" for s in SEEDS]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule\end{tabular}\end{table}",
        r"\begin{table}[ht]\centering\small",
        (
            "\\caption{Additional retained diagnostics. Non-query metrics "
            "use 486 rows per seed. Precision and donor success use the r"
            "elevant valid query retrievals.}"
        ),
        r"\begin{tabular}{lrrr}\toprule",
        r"Metric & 101 & 211 & 307 \\\midrule",
    ]
    for mode, key, label in (
        ("no_memory", "nonquery_nmse", "Base non-query NMSE"),
        ("tensor_memory", "nonquery_nmse", "Tensor non-query NMSE"),
        ("tensor_memory", "retrieval_anchor_precision", "Matching-anchor precision"),
        ("paired_swap", "swap_donor_direction", "Donor improvement rate"),
    ):
        lines.append(
            label
            + " & "
            + " & ".join(f"{results[s]['summaries'][mode][key]:.6f}" for s in SEEDS)
            + r" \\"
        )
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(lines) + "\n"


def verify(root=ROOT, pub=PUB):
    checks = {}
    baseline = read(pub / "PRESERVATION_BASELINE.json")
    for relative, expected in baseline.items():
        path = safe_path(root, relative)
        checks["preserved:" + relative] = path.is_file() and sha(path) == expected
    manifest = read(root / "results/development-v2/manifest.json")
    h = hashlib.sha256()
    for item in manifest["source"]["files"]:
        path = safe_path(pub / "frozen_source", item["path"])
        checks["executed_source:" + item["path"]] = (
            path.is_file() and sha(path) == item["sha256"]
        )
        h.update(item["path"].encode())
        h.update(item["sha256"].encode())
    checks["source_archive_file_set"] = {
        p.relative_to(pub / "frozen_source").as_posix()
        for p in (pub / "frozen_source").rglob("*")
        if p.is_file()
    } == {item["path"] for item in manifest["source"]["files"]}
    checks["source_bundle"] = h.hexdigest() == manifest["source"]["sha256"]
    protocol = read(pub / "frozen_source/protocols/development_v2.json")
    checks["seed_identity"] = protocol["dataset"]["seeds"] == list(SEEDS) and [
        r["seed"] for r in manifest["runs"]
    ] == list(SEEDS)
    checks["manifest_complete"] = (
        manifest["completed_runs"] == manifest["expected_runs"] == 3
        and not manifest["failures"]
    )
    # These functions reaggregate recorded scalars, not raw prediction tensors.
    verifier = runpy.run_path(str(pub / "frozen_source/scripts/verify_results.py"))
    for seed in SEEDS:
        result = read(root / f"results/development-v2/seed-{seed}/result.json")
        summaries = {}
        for relative, expected in result["prediction_artifacts"].items():
            path = safe_path(root / "results/development-v2", relative)
            checks["record_hash:" + relative] = sha(path) == expected
            rows = read(path)
            mode = path.stem.removeprefix("predictions-")
            summaries[mode] = verifier["summarize_rows"](
                rows, result["normalization_variance"]
            )
            checks[f"record_metrics:{seed}:{mode}"] = all(
                math.isclose(
                    value, result["summaries"][mode][key], rel_tol=1e-10, abs_tol=1e-10
                )
                for key, value in summaries[mode].items()
            )
            checks[f"row_structure:{seed}:{mode}"] = (
                len(rows) == 594 and sum(bool(row["query"]) for row in rows) == 108
            )
            checks[f"finite:{seed}:{mode}"] = all(
                math.isfinite(row["squared_error"])
                and row["squared_error"] >= 0
                and row["element_count"] == 256
                for row in rows
            )
            if mode == "recent_only":
                checks[f"empty_recent:{seed}"] = all(
                    not row["retrieval_valid"] and row["retrieval_eligible_count"] == 0
                    for row in rows
                )
            checks[f"past_times:{seed}:{mode}"] = all(
                not row["retrieval_valid"]
                or row["event_time"] - row["retrieval_source_time"] >= 4
                for row in rows
            )
        gates = verifier["independently_adjudicate"](summaries, protocol["decision"])
        checks[f"record_decisions:{seed}"] = gates == result["decision"][
            "gates"
        ] and all(gates.values())
        checks[f"condition_completeness:{seed}"] = set(summaries) == set(
            protocol["controls"]
        )
    checks["generated_table"] = (
        root / "paper/generated_results.tex"
    ).read_text() == table_text(root)
    tex = (root / "paper/main.tex").read_text()
    bib = (root / "paper/references.bib").read_text()
    cited = {
        k.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", tex)
        for k in group.split(",")
    }
    defined = set(re.findall(r"@\w+\{([^,]+)", bib))
    checks["citations_resolve"] = cited == defined and bool(cited)
    checks["paper_no_placeholders"] = not any(
        x in tex for x in ("TODO", "FIXME", "TBD", "INSERT RESULT")
    )
    manuscript = {name: sha(root / "paper" / name) for name in PAPER_FILES}
    return {
        "status": "PASS_RECORD_SOURCE_CHECKS"
        if all(checks.values())
        else "FAIL_RECORD_SOURCE_CHECKS",
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "failed": [key for key, value in checks.items() if not value],
        "checks": checks,
        "manuscript": manuscript,
        "boundary": (
            "Local source/hash/scalar-record verification. Not raw-tensor"
            " replay, external reproduction, PDF inspection, or publicati"
            "on readiness."
        ),
    }


def substantive_receipt(name, evidence):
    """Reject generic PASS labels that carry no gate-specific evidence."""
    common = bool(evidence.get("checked_at"))
    if name == "author_metadata":
        return common and all(
            evidence.get(key)
            for key in (
                "authors",
                "manuscript_license",
                "code_distribution_terms",
                "destination",
                "author_confirmation_reference",
            )
        )
    if name == "clean_environment_replay":
        return (
            common
            and evidence.get("isolated_environment") is True
            and (
                evidence.get("exit_code") == 0
                and bool(evidence.get("environment"))
                and bool(evidence.get("command"))
                and bool(evidence.get("artifact_hashes"))
            )
        )
    if name == "raw_output_verification":
        return (
            common
            and bool(evidence.get("raw_files"))
            and bool(evidence.get("checks"))
            and all(evidence["checks"].values())
        )
    if name == "independent_reproduction":
        return (
            common
            and evidence.get("independent_of_original_operator") is True
            and all(
                evidence.get(key)
                for key in ("operator", "environment", "report", "artifact_hashes")
            )
        )
    if name == "scientific_review":
        return (
            common
            and evidence.get("independent_reviewer") is True
            and (
                evidence.get("unresolved_critical_findings") == 0
                and bool(evidence.get("reviewer"))
                and bool(evidence.get("review_report"))
            )
        )
    if name == "pdf_build":
        return (
            common
            and evidence.get("exit_code") == 0
            and all(evidence.get(key) for key in ("pdf_sha256", "build_log", "command"))
        )
    if name == "visual_inspection":
        pages = evidence.get("pages_reviewed", [])
        count = evidence.get("page_count", 0)
        return (
            common
            and isinstance(count, int)
            and count > 0
            and pages == list(range(1, count + 1))
            and (
                evidence.get("unresolved_defects") == 0
                and bool(evidence.get("pdf_sha256"))
                and bool(evidence.get("reviewer"))
            )
        )
    if name == "public_archive":
        return (
            common
            and str(evidence.get("url", "")).startswith("https://")
            and all(
                evidence.get(key)
                for key in (
                    "immutable_identifier",
                    "download_verification",
                    "archive_sha256",
                )
            )
        )
    if name == "venue_requirements":
        return common and all(
            evidence.get(key)
            for key in (
                "destination",
                "official_requirements_url",
                "completed_requirements",
            )
        )
    return False


def release_gate(root=ROOT, pub=PUB):
    report = verify(root, pub)
    ledger = read(pub / "RELEASE_GATES.json")
    missing = []
    for name in (
        "author_metadata",
        "clean_environment_replay",
        "raw_output_verification",
        "independent_reproduction",
        "scientific_review",
        "pdf_build",
        "visual_inspection",
        "public_archive",
        "venue_requirements",
    ):
        entry = ledger.get(name, {})
        if entry.get("status") != "PASS":
            missing.append(name)
            continue
        relative = entry.get("receipt")
        if not isinstance(relative, str):
            missing.append(name + ":missing_receipt")
            continue
        receipt = safe_path(pub, relative)
        if not receipt.is_file() or sha(receipt) != entry.get("sha256"):
            missing.append(name + ":receipt_hash")
            continue
        evidence = read(receipt)
        if (
            evidence.get("status") != "PASS"
            or evidence.get("manuscript") != report["manuscript"]
            or not substantive_receipt(name, evidence)
        ):
            missing.append(name + ":stale_or_failed")
    if report["failed"]:
        missing.append("source_or_records")
    return {
        "status": "READY_FOR_AUTHOR_SUBMISSION_REVIEW" if not missing else "BLOCKED",
        "missing_gates": missing,
        "verification": report,
        "boundary": (
            "Receipt validation is not authentication of external attesta"
            "tions. Final author review and actual submission remain sepa"
            "rate."
        ),
    }


def stage(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite output")
    if (output.is_relative_to(ROOT) and not output.is_relative_to(PUB)) or any(
        output.is_relative_to(PUB / name) for name in ("before", "frozen_source")
    ):
        raise ValueError("refusing frozen destination")
    gate = release_gate()
    if gate["status"] == "BLOCKED":
        raise RuntimeError("release blocked: " + ", ".join(gate["missing_gates"]))
    admission = resource_admission()
    if admission["status"] != "PASS":
        raise RuntimeError("resource admission blocked")
    output.mkdir(parents=True, exist_ok=False)
    for name in PAPER_FILES:
        shutil.copyfile(ROOT / "paper" / name, output / name)
    write_new(
        output / "SOURCE_MANIFEST.json",
        {name: sha(output / name) for name in PAPER_FILES},
    )
    return gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("verify", "gate", "table", "admission", "stage")
    )
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "table":
        print(table_text(), end="")
        return
    if args.command == "stage" and args.output is None:
        parser.error("stage requires --output")
    result = {
        "verify": verify,
        "gate": release_gate,
        "admission": resource_admission,
        "stage": lambda: stage(args.output),
    }[args.command]()
    if args.receipt:
        write_new(args.receipt, result)
    print(json.dumps(result, indent=2))
    if result["status"].startswith("FAIL") or result["status"] == "BLOCKED":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
