"""Additional orchestration only: never patch frozen code or relax its guards."""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from prepare import digest, safe

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "successor/itfm_v2"
PUB = V2 / "publication/itfm-v2-paper-20261011"
V3 = ROOT / "successor/itfm_v3"


def write(path, data):
    with path.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def verify_manifest(root, expected):
    manifest = root / "REMOTE_MANIFEST.json"
    if digest(manifest) != expected:
        raise RuntimeError("manifest does not match reviewed identity")
    data = json.loads(manifest.read_text())
    if not data["files"]:
        raise RuntimeError("empty source manifest")
    for name, checksum in data["files"].items():
        path = safe(root, name)
        if not path.is_file() or digest(path) != checksum:
            raise RuntimeError("source mismatch: " + name)
    return data


def expect(path, status):
    result = json.loads(path.read_text())
    if result["status"] != status:
        raise RuntimeError(str(path) + ": " + result["status"])
    return result


def stage(name, command, cwd, output, timeout):
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
    }
    env.pop("PYTHONPATH", None)
    with (output / (name + ".log")).open("x") as log:
        p = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            code = -9
    write(
        output / (name + ".json"),
        {"command": command, "exit_code": code, "timeout_seconds": timeout},
    )
    if code:
        raise RuntimeError(name + " failed; receipt/log retained")


def preflight(expected, output):
    verify_manifest(ROOT, expected)
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("macOS ARM64 required; no platform/guard substitution")
    if sys.version_info[:3] != (3, 11, 15):
        raise RuntimeError("Python 3.11.15 required")
    if shutil.disk_usage(ROOT).free < 3 * 1024**3:
        raise RuntimeError("less than 3 GiB scratch")
    if (V3 / "RUNNER.lock").exists() or (V3 / "runs/CAMPAIGN_CONSUMED.json").exists():
        raise RuntimeError("existing v3 owner or consumed campaign")
    sys.path.insert(0, str(PUB / "tools"))
    import release

    checks = release.verify()
    write(output / "SOURCE_CHECKS.json", checks)
    if checks["failed"]:
        raise RuntimeError("v2 source/scalar verification failed")
    admission = release.resource_admission()
    write(output / "ADMISSION.json", admission)
    if admission["status"] != "PASS":
        raise RuntimeError("BLOCKED_RESOURCE")


def execute(expected, mode):
    output = ROOT / ("remote-" + mode)
    output.mkdir(exist_ok=False)
    summary = {
        "status": "STARTED",
        "mode": mode,
        "manifest_sha256": expected,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "boundary": "Remote execution by same operator, not independent replication",
    }
    try:
        preflight(expected, output)
        if mode == "install":
            stage(
                "install",
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--only-binary=:all:",
                    "-r",
                    str(PUB / "REPLAY_REQUIREMENTS.txt"),
                ],
                ROOT,
                output,
                300,
            )
            stage(
                "dependency_check",
                [sys.executable, "-m", "pip", "check"],
                ROOT,
                output,
                60,
            )
            stage(
                "resolved_environment",
                [sys.executable, "-m", "pip", "freeze", "--all"],
                ROOT,
                output,
                60,
            )
            summary["status"] = "ENVIRONMENT_PREPARED_NOT_EXECUTED"
        elif mode == "run":
            for name, version in {
                "torch": "2.12.1",
                "numpy": "2.4.6",
                "pytest": "9.1.1",
                "setuptools": "81.0.0",
            }.items():
                if importlib.metadata.version(name) != version:
                    raise RuntimeError("dependency mismatch: " + name)
            write(
                ROOT / "REMOTE_EXECUTION_STARTED.json",
                {
                    "manifest": expected,
                    "run_id": summary["run_id"],
                    "note": "No automatic retry or duplicate deployment",
                },
            )
            replay = PUB / "replays/remote-1"
            stage(
                "v2",
                [
                    sys.executable,
                    "-B",
                    str(PUB / "tools/reproduce.py"),
                    "execute",
                    "--python",
                    sys.executable,
                    "--output",
                    str(replay),
                ],
                V2,
                output,
                330,
            )
            expect(replay / "REPLAY_RECEIPT.json", "PASS_LOCAL_REPLAY")
            raw = expect(replay / "RAW_OUTPUT_RECEIPT.json", "PASS")
            if not raw["checks"] or not all(raw["checks"].values()):
                raise RuntimeError("v2 raw checks missing/failed")
            for name, checksum in raw["raw_files"].items():
                if digest(safe(replay / "raw", name)) != checksum:
                    raise RuntimeError("v2 raw hash mismatch")
            freeze = V3 / "freezes/development-2.json"
            tests, campaign, verification = [
                V3 / "runs" / n
                for n in (
                    "remote-tests-1",
                    "remote-campaign-1",
                    "remote-verification-1",
                )
            ]
            for name, destination, extra, status, timeout in [
                ("test", tests, [], "PASS_RUNTIME_TESTS", 150),
                (
                    "run",
                    campaign,
                    ["--test-receipt", str(tests / "RECEIPT.json")],
                    "COMPLETED_DEVELOPMENT_EXECUTION",
                    1830,
                ),
                (
                    "verify",
                    verification,
                    ["--input", str(campaign)],
                    "PASS_RAW_VERIFICATION",
                    1830,
                ),
            ]:
                verify_manifest(ROOT, expected)
                stage(
                    "v3_" + name,
                    [
                        sys.executable,
                        "-B",
                        str(V3 / "run.py"),
                        name,
                        "--python",
                        sys.executable,
                        "--freeze",
                        str(freeze),
                        "--output",
                        str(destination),
                        *extra,
                    ],
                    V3,
                    output,
                    timeout,
                )
                receipt = expect(destination / "RECEIPT.json", status)
                if receipt["source"] != json.loads(freeze.read_text()):
                    raise RuntimeError("v3 receipt source mismatch")
            verified = expect(verification / "VERIFICATION.json", "PASS")
            if not verified["checks"] or not all(verified["checks"].values()):
                raise RuntimeError("v3 raw checks missing/failed")
            summary["status"] = "COMPLETED_REMOTE_EXECUTION_AND_VERIFICATION"
        else:
            summary["status"] = "PASS_PREFLIGHT_ONLY"
    except Exception as exc:
        summary["status"] = "BLOCKED_OR_FAILED_RETAINED"
        summary["error"] = str(exc)
    finally:
        try:
            verify_manifest(ROOT, expected)
            summary["source_preserved"] = True
        except Exception as exc:
            summary.update(
                status="INVALID_SOURCE_CHANGED",
                source_preserved=False,
                preservation_error=str(exc),
            )
        write(output / "SUMMARY.json", summary)
        print(json.dumps(summary))
    return (
        0
        if summary["status"].startswith(("PASS_", "ENVIRONMENT_", "COMPLETED_"))
        else 3
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["preflight", "install", "run"])
    p.add_argument("--manifest-sha256", required=True)
    a = p.parse_args()
    raise SystemExit(execute(a.manifest_sha256, a.mode))
