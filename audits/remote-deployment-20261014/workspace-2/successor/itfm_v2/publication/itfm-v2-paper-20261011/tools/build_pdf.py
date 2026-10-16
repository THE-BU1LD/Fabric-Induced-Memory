"""Resource-guarded, fresh-directory paper compilation; not visual certification."""

import argparse
import datetime
import json
import shutil
from pathlib import Path

import release


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("every build requires a new output directory")
    if output.is_relative_to(release.ROOT) and not output.is_relative_to(release.PUB):
        raise ValueError(
            "build output must not modify the research source/evidence tree"
        )
    if any(
        output.is_relative_to(release.PUB / name)
        for name in ("before", "frozen_source")
    ):
        raise ValueError("refusing a protected build destination")
    verified = release.verify()
    if verified["failed"]:
        raise RuntimeError("source or evidence verification failed")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "BLOCKED_RESOURCE",
        "checked_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "manuscript": verified["manuscript"],
        "budget_seconds": 120,
        "admission": release.resource_admission(),
        "boundary": (
            "Local compilation only. Every rendered page still requires inspection."
        ),
    }
    if receipt["admission"]["status"] != "PASS":
        release.write_new(output / "BUILD_RECEIPT.json", receipt)
        return receipt
    source = output / "source"
    source.mkdir()
    for name in release.PAPER_FILES:
        shutil.copyfile(release.ROOT / "paper" / name, source / name)
    command = [
        "tectonic",
        "--untrusted",
        "--keep-logs",
        "--keep-intermediates",
        "--outdir",
        str(output),
        str(source / "main.tex"),
    ]
    try:
        step = release.bounded_command(command, cwd=source, timeout=120)
        receipt.update(step)
        log = output / "build-console.log"
        with log.open("x") as handle:
            handle.write(step["stdout"] + "\n" + step["stderr"])
        receipt["build_log"] = log.name
        receipt["build_log_sha256"] = release.sha(log)
        pdf = output / "main.pdf"
        passed = step["exit_code"] == 0 and pdf.is_file()
        receipt["status"] = "PASS" if passed else "FAILED_BUILD"
        if passed:
            receipt["pdf_sha256"] = release.sha(pdf)
    except OSError as exc:
        receipt.update(status="FAILED_BUILD", error=str(exc))
    release.write_new(output / "BUILD_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(3)
