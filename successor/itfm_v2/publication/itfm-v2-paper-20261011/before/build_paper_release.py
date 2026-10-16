from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

EXPECTED_HASHES = {
    "results/development-v2/manifest.json": (
        "76c2bda7436f9b4798c423b02c99a3d65df3170e08629b1f847c0b6a3e4606bf"
    ),
    "results/result_verification.json": (
        "fe7ab41ec8d9634e82c08b292cf4c1d0f4d2f7a7d91d9dbe2463fce23969c6ce"
    ),
    "results/verification_receipt.json": (
        "4b91f3db87dedb80e2851b1b28427a33b9846c0ca3fd47c7f53de4cd64f30f79"
    ),
    "results/development-v2/seed-101/result.json": (
        "8f03e8449a20fd05ea34c95a600d9e53ca195d145d1b543bbe1a948ab82f92b0"
    ),
    "results/development-v2/seed-211/result.json": (
        "948bb206d63721a260e1742f3ce7e39d2771d07b5a3b9840535d3ceffc4f7867"
    ),
    "results/development-v2/seed-307/result.json": (
        "f147987396d95af850d400bd6a7490cf3b11877aa91175a72085a270d942d165"
    ),
}

PAPER_FILES = (
    "main.tex",
    "references.bib",
    "generated_results.tex",
    "README.md",
    "CLAIM_EVIDENCE_MAP.md",
    "SOURCE_VERIFICATION_2026-10-11.json",
)

EXPECTED_SEEDS = (101, 211, 307)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


class ReleaseVerificationError(RuntimeError):
    """Raised when the paper package is not evidence-bound."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReleaseVerificationError(f"expected JSON object: {path}")
    return payload


def rounded_row(payload: dict[str, Any]) -> str:
    summaries = payload["summaries"]
    cells: list[str] = [str(int(payload["seed"]))]
    for condition in (
        "no_memory",
        "tensor_memory",
        "erase",
        "sign_flip",
        "vector_broadcast",
        "recent_only",
        "last_value",
        "oracle_anchor",
    ):
        cells.append(f"{float(summaries[condition]['query_nmse']):.6f}")
    cells.extend(
        [
            f"{float(summaries['tensor_memory']['nonquery_nmse']):.6f}",
            f"{float(summaries['tensor_memory']['retrieval_anchor_precision']):.3f}",
            f"{float(summaries['paired_swap']['swap_donor_direction']):.3f}",
        ]
    )
    return " & ".join(cells) + r" \\"


def citation_keys(tex: str) -> set[str]:
    found: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", tex, flags=re.DOTALL):
        found.update(key.strip() for key in group.split(",") if key.strip())
    return found


def bibliography_keys(bib: str) -> set[str]:
    return set(re.findall(r"@[A-Za-z]+\{([^,]+),", bib))


def verify() -> dict[str, Any]:
    checks: list[Check] = []

    for relative, expected in EXPECTED_HASHES.items():
        path = ROOT / relative
        actual = sha256_file(path) if path.is_file() else "MISSING"
        checks.append(Check(f"hash:{relative}", actual == expected, actual))

    manifest = read_json(ROOT / "results/development-v2/manifest.json")
    verification = read_json(ROOT / "results/result_verification.json")
    state = read_json(ROOT / "RESEARCH_STATE.json")
    campaign = read_json(ROOT / "SOTA_CAMPAIGN_STATE.json")

    checks.extend(
        [
            Check(
                "manifest_complete",
                manifest.get("status") == "PASS_BOUNDED_DEVELOPMENT"
                and manifest.get("completed_runs") == len(EXPECTED_SEEDS)
                and manifest.get("expected_runs") == len(EXPECTED_SEEDS)
                and manifest.get("failures") == [],
                str(manifest.get("status")),
            ),
            Check(
                "source_identity",
                manifest.get("source", {}).get("sha256")
                == "5d01a2917fe78719e40fecbc702131e76a3b7f842986d79d47e5660c7221631a",
                str(manifest.get("source", {}).get("sha256")),
            ),
            Check(
                "protocol_identity",
                manifest.get("protocol_sha256")
                == "8cb1ea0493da40ece0663f1777dccf5bf01f367050b771434a9574b0ef52330b",
                str(manifest.get("protocol_sha256")),
            ),
            Check(
                "independent_script_verification",
                verification.get("status") == "pass"
                and len(verification.get("checks", {})) == 96
                and all(verification.get("checks", {}).values()),
                f"{verification.get('status')}:{len(verification.get('checks', {}))}",
            ),
            Check(
                "public_claims_still_closed",
                campaign.get("claims", {}).get("public_data_advantage")
                == "NOT_EVALUATED"
                and campaign.get("claims", {}).get("state_of_the_art")
                == "NOT_EVALUATED"
                and campaign.get("claims", {}).get("publication_ready") == "NO"
                and state.get("checkpoints", {}).get("10_research_complete")
                == "NOT_REACHED",
                json.dumps(campaign.get("claims", {}), sort_keys=True),
            ),
        ]
    )

    result_rows: list[str] = []
    for seed in EXPECTED_SEEDS:
        result = read_json(ROOT / f"results/development-v2/seed-{seed}/result.json")
        summaries = result.get("summaries", {})
        decision = result.get("decision", {})
        gates = decision.get("gates", {})
        exact_mechanism = (
            result.get("seed") == seed
            and result.get("validation_alpha") == 1.0
            and summaries.get("tensor_memory", {}).get("query_nmse") == 0.0
            and summaries.get("oracle_anchor", {}).get("query_nmse") == 0.0
            and summaries.get("tensor_memory", {}).get("retrieval_anchor_precision")
            == 1.0
            and summaries.get("paired_swap", {}).get("swap_donor_direction") == 1.0
            and decision.get("status") == "PASS_BOUNDED_DEVELOPMENT"
            and len(gates) == 8
            and all(gates.values())
            and all(result.get("identifiability", {}).get("gates", {}).values())
        )
        checks.append(Check(f"seed:{seed}", exact_mechanism, str(decision)))
        result_rows.append(rounded_row(result))

    table = (PAPER / "generated_results.tex").read_text(encoding="utf-8")
    for seed, row in zip(EXPECTED_SEEDS, result_rows, strict=True):
        checks.append(Check(f"table_row:{seed}", row in table, row))

    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    cited = citation_keys(tex)
    defined = bibliography_keys(bib)
    checks.append(
        Check(
            "citations_resolved",
            cited <= defined and bool(cited),
            f"cited={sorted(cited)} missing={sorted(cited - defined)}",
        )
    )
    forbidden = ("TODO", "FIXME", "TBD", "INSERT RESULT", "PLACEHOLDER")
    combined = "\n".join(
        (PAPER / name).read_text(encoding="utf-8") for name in PAPER_FILES
    )
    checks.append(
        Check(
            "no_placeholder_markers",
            not any(marker in combined.upper() for marker in forbidden),
            ",".join(marker for marker in forbidden if marker in combined.upper()),
        )
    )

    failed = [check for check in checks if not check.passed]
    report = {
        "schema_version": 1,
        "status": "PASS_SOURCE_EVIDENCE_BOUND"
        if not failed
        else "FAIL_SOURCE_EVIDENCE_BOUND",
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "checks": [check.__dict__ for check in checks],
        "release_boundary": (
            "Source verification only. This is not compilation, rendered visual QA, "
            "public confirmation, independent reproduction, or arXiv submission."
        ),
    }
    if failed:
        names = ", ".join(check.name for check in failed)
        raise ReleaseVerificationError(f"paper verification failed: {names}")
    return report


def stage(output: Path) -> dict[str, Any]:
    verification = verify()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    for name in PAPER_FILES:
        shutil.copy2(PAPER / name, output / name)
    shutil.copy2(
        ROOT / "LICENSE" if (ROOT / "LICENSE").is_file() else PAPER / "README.md",
        output
        / (
            "LICENSE"
            if (ROOT / "LICENSE").is_file()
            else "SOURCE_LICENSE_UNRESOLVED.md"
        ),
    )

    sums = []
    for path in sorted(item for item in output.iterdir() if item.is_file()):
        sums.append(f"{sha256_file(path)}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "STAGED_SOURCE_NOT_RELEASE_CERTIFIED",
        "paper_verification": verification,
        "files": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output.iterdir())
            if path.is_file()
        ],
        "open_gates": [
            "successful clean LaTeX compilation",
            "rendered inspection of every PDF page",
            "authorship and source-license confirmation",
            "public confirmation disposition",
            "independent reproduction",
            "explicit submission authorization",
        ],
    }
    (output / "ARXIV_SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Verify and stage the evidence-bound ITFM v2 paper source."
    )
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    stage_parser = commands.add_parser("stage")
    stage_parser.add_argument("--output", required=True, type=Path)
    return root


def main() -> None:
    arguments = parser().parse_args()
    result = verify() if arguments.command == "verify" else stage(arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
