from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from fim_v2.resources import resource_admission

from .config import load_config

STAGE_COMMANDS = {
    "reproduce-lano": "S1_LANO",
    "reproduce-hsfno": "S1_HSFNO",
    "run-oracle-gap": "S2_ORACLE_GAP",
    "train-adapter": "S3_TRAIN_ADAPTER",
    "run-ablation": "S3_ABLATION",
    "run-pobench-final": "S4_POBENCH_FINAL",
    "run-hsfno-final": "S4_HSFNO_FINAL",
}


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def print_config(arguments: argparse.Namespace) -> None:
    config = load_config(arguments.config)
    print(config.resolved_json(), end="")


def fetch_data(arguments: argparse.Namespace) -> None:
    from .orchestration import fetch_asset

    registry = json.loads(Path(arguments.registry).read_text(encoding="utf-8"))
    assets = registry.get("assets", [])
    if not assets:
        raise SystemExit("registry has no assets")
    if not arguments.execute:
        _print(
            {
                "status": "DRY_RUN_VALIDATED",
                "assets": len(assets),
                "note": "No network request executed.",
            }
        )
        return
    admission = resource_admission()
    if admission.get("status") != "PASS":
        _print({"status": "BLOCKED_RESOURCE", "resource": admission})
        raise SystemExit(3)
    receipts = [fetch_asset(**asset) for asset in assets]
    _print({"status": "FETCHED_VERIFIED", "assets": receipts})


def verify_data(arguments: argparse.Namespace) -> None:
    from .datasets import load_manifest, verify_source_files

    manifest = load_manifest(arguments.manifest)
    verified = verify_source_files(manifest, arguments.root)
    _print(
        {
            "status": "PASS_SCHEMA_ONLY_FIXTURE"
            if manifest.schema_only_fixture
            else "PASS_DATA_VERIFIED",
            "manifest": dataclasses.asdict(manifest),
            "verified_files": verified,
            "results_admissible": not manifest.schema_only_fixture,
        }
    )


def run_stage(arguments: argparse.Namespace) -> None:
    from .orchestration import StageRunner, load_stage_spec

    spec = load_stage_spec(arguments.stage_spec)
    expected = STAGE_COMMANDS[arguments.command]
    if spec.stage_id != expected:
        raise SystemExit(f"stage id mismatch: expected {expected}, got {spec.stage_id}")
    if arguments.command in {"run-pobench-final", "run-hsfno-final"} and not (
        spec.protected_test
    ):
        raise SystemExit("final evaluation stage must set protected_test=true")
    result = StageRunner(arguments.run_root).run(
        spec, execute=arguments.execute, run_id=arguments.run_id
    )
    _print(result)
    if result["status"] not in {"DRY_RUN_VALIDATED", "PASS_EXECUTED_UNVERIFIED"}:
        raise SystemExit(3 if result["status"] == "BLOCKED_RESOURCE" else 1)


def verify_results(arguments: argparse.Namespace) -> None:
    from .verification import verify_run

    result = verify_run(arguments.manifest, arguments.receipt)
    _print(result)
    if result["status"] != "PASS_VERIFIED":
        raise SystemExit(1)


def build_report(arguments: argparse.Namespace) -> None:
    from .verification import build_verified_report

    build_verified_report(arguments.verification_receipt, arguments.output)
    _print(
        {"status": "REPORT_BUILT_FROM_VERIFIED_ARTIFACTS", "output": arguments.output}
    )


def build_archive(arguments: argparse.Namespace) -> None:
    from .verification import archival_manifest

    _print(archival_manifest(arguments.root, arguments.output))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="fim-sota")
    subcommands = root.add_subparsers(dest="command", required=True)
    from .engineering import register
    register(subcommands)
    resolved = subcommands.add_parser("print-config")
    resolved.add_argument("--config", required=True)
    resolved.set_defaults(handler=print_config)

    fetch = subcommands.add_parser("fetch-data")
    fetch.add_argument("--registry", required=True)
    fetch.add_argument("--execute", action="store_true")
    fetch.set_defaults(handler=fetch_data)

    verify = subcommands.add_parser("verify-data")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--root", required=True)
    verify.set_defaults(handler=verify_data)

    for name in STAGE_COMMANDS:
        stage = subcommands.add_parser(name)
        stage.add_argument("--stage-spec", required=True)
        stage.add_argument("--run-root", default="results/sota/runs")
        stage.add_argument("--run-id")
        stage.add_argument("--execute", action="store_true")
        stage.set_defaults(handler=run_stage)

    verify_run_parser = subcommands.add_parser("verify-results")
    verify_run_parser.add_argument("--manifest", required=True)
    verify_run_parser.add_argument("--receipt", required=True)
    verify_run_parser.set_defaults(handler=verify_results)

    report = subcommands.add_parser("build-report")
    report.add_argument("--verification-receipt", required=True)
    report.add_argument("--output", required=True)
    report.set_defaults(handler=build_report)

    archive = subcommands.add_parser("build-archive-manifest")
    archive.add_argument("--root", required=True)
    archive.add_argument("--output", required=True)
    archive.set_defaults(handler=build_archive)
    return root


def main() -> None:
    arguments = parser().parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
