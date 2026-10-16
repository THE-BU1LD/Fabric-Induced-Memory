"""Fail-closed admission and exclusive output orchestration; no eager Torch import."""

import argparse
import datetime
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from core import ROOT, admission, identity, sha, validate, write_new


def checked_source(freeze):
    manifest = json.loads(Path(freeze).read_text())
    if manifest["files"] != identity():
        raise RuntimeError(
            "source differs from prospective freeze; preserve previous freeze and revise before outcome access"
        )
    return manifest


def output_path(value):
    output = Path(value).resolve()
    if not output.is_relative_to(ROOT / "runs") or output == ROOT / "runs":
        raise ValueError("output must be a fresh subdirectory of itfm_v3/runs")
    if output.exists():
        raise FileExistsError("output exists; no overwriting or silent retry")
    return output


def execute(args, protocol):
    frozen = checked_source(args.freeze)
    output = output_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "STARTED",
        "mode": args.command,
        "at": datetime.datetime.now(datetime.UTC).isoformat(),
        "source": frozen,
        "admission": admission(),
        "budget": protocol["budget"],
        "boundary": "Local operator; no independent reproduction; tests are not scientific results",
    }
    try:
        if receipt["admission"]["status"] != "PASS":
            receipt["status"] = "BLOCKED_RESOURCE"
            return receipt
        if shutil.disk_usage(output).free < 3 * 1024**3:
            receipt["status"] = "BLOCKED_DISK"
            return receipt
        # Lock is local to this new study; never remove another study's owner/lock.
        lock = ROOT / "RUNNER.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            receipt["status"] = "BLOCKED_OWNERSHIP"
            return receipt
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        try:
            if args.command == "run":
                test = json.loads(Path(args.test_receipt).read_text())
                if test["status"] != "PASS_RUNTIME_TESTS" or test["source"] != frozen:
                    raise RuntimeError(
                        "passing runtime test receipt for exact frozen source required"
                    )
                # Consumed only when a campaign is actually admitted; partial failure is retained.
                write_new(
                    ROOT / "runs/CAMPAIGN_CONSUMED.json",
                    {"output": str(output), "freeze": frozen, "at": receipt["at"]},
                )
            command = [
                args.python,
                str(ROOT / "run.py"),
                "worker",
                "--output",
                str(output),
                "--mode",
                args.command,
                "--freeze",
                str(Path(args.freeze).resolve()),
            ]
            if args.command == "verify":
                command += ["--input", str(Path(args.input).resolve())]
            env = {
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "ITFM_V3_PARENT_PID": str(os.getpid()),
            }
            snapshot = output / "source"
            snapshot.mkdir()
            for name in frozen["files"]:
                shutil.copyfile(ROOT / name, snapshot / name)
            timeout = (
                protocol["budget"]["max_runtime_tests_seconds"]
                if args.command == "test"
                else protocol["budget"]["max_seconds"]
            )
            start = time.monotonic()
            with (
                (output / "stdout.log").open("x") as stdout,
                (output / "stderr.log").open("x") as stderr,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = -9
            receipt.update(
                command=command,
                exit_code=code,
                elapsed_seconds=time.monotonic() - start,
            )
            receipt["status"] = (
                {
                    "test": "PASS_RUNTIME_TESTS",
                    "run": "COMPLETED_DEVELOPMENT_EXECUTION",
                    "verify": "PASS_RAW_VERIFICATION",
                }[args.command]
                if code == 0
                else "FAILED_RETAINED"
            )
            receipt["logs"] = {
                name: sha(output / name) for name in ("stdout.log", "stderr.log")
            }
            if identity() != frozen["files"]:
                receipt["status"] = "INVALID_SOURCE_CHANGED_DURING_EXECUTION"
        finally:
            lock.unlink()
    except Exception as exc:
        receipt["status"] = "FAILED_RETAINED"
        receipt["error"] = repr(exc)
    finally:
        write_new(output / "RECEIPT.json", receipt)
    return receipt


def worker(args, protocol):
    if (
        os.environ.get("ITFM_V3_PARENT_PID") != str(os.getppid())
        or not (ROOT / "RUNNER.lock").exists()
    ):
        raise RuntimeError("workers require a live guarded parent; use test/run/verify")
    checked_source(args.freeze)
    report = admission()
    write_new(Path(args.output) / "WORKER_ADMISSION.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("blocked before numerical imports")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Python 3.11 required; no environment substitutions")
    version = importlib.metadata.version("torch")
    if version != "2.12.1":
        raise RuntimeError(
            "Torch 2.12.1 required; no automatic install or version substitution"
        )
    write_new(
        Path(args.output) / "ENVIRONMENT.json",
        {
            "python": sys.version,
            "torch": version,
            "platform": sys.platform,
            "threads": 1,
        },
    )
    if args.mode == "test":
        import unittest

        suite = unittest.defaultTestLoader.discover(
            str(ROOT), pattern="test_runtime.py"
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise RuntimeError("runtime tests failed")
    elif args.mode == "run":
        from experiment import run

        run(protocol, Path(args.output))
    else:
        from verify import verify

        verify(Path(args.input), Path(args.output), protocol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["check", "freeze", "test", "run", "verify", "worker"]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--test-receipt", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--mode", choices=["test", "run", "verify"])
    parser.add_argument(
        "--python", default="/opt/homebrew/opt/python@3.11/bin/python3.11"
    )
    args = parser.parse_args()
    protocol = validate(json.loads((ROOT / "protocol.json").read_text()))
    if args.command == "check":
        print(json.dumps({"status": "STATIC_PROTOCOL_VALID", "files": identity()}))
        return
    if args.command == "freeze":
        if (ROOT / "runs/CAMPAIGN_CONSUMED.json").exists():
            raise RuntimeError(
                "cannot revise this development version after campaign access"
            )
        if args.output is None or not args.output.resolve().is_relative_to(
            ROOT / "freezes"
        ):
            raise ValueError("freeze output must be under freezes/")
        write_new(
            args.output,
            {
                "at": datetime.datetime.now(datetime.UTC).isoformat(),
                "files": identity(),
                "scope": "Prospective development source freeze, not external preregistration",
            },
        )
        return
    if args.freeze is None or args.output is None:
        parser.error("--freeze and --output are required")
    if args.command == "run" and args.test_receipt is None:
        parser.error("--test-receipt required for run")
    if (args.command == "verify" or args.mode == "verify") and args.input is None:
        parser.error("--input required for verification")
    if args.command == "worker":
        try:
            worker(args, protocol)
        except Exception:
            write_new(
                args.output / "FAILURE.json",
                {"traceback": traceback.format_exc(), "status": "FAILED_RETAINED"},
            )
            raise
    else:
        receipt = execute(args, protocol)
        print(json.dumps(receipt))
        if receipt["status"].startswith(("BLOCKED", "FAILED", "INVALID")):
            raise SystemExit(3)


if __name__ == "__main__":
    main()
