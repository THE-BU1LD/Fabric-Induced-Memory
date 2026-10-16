"""Exact-inventory tensor verification, bounded by a separate supervising process.

No scientific campaign runs here. Both preflight and worker enforce admission.
"""

import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import resource
import signal
import shutil
import subprocess
import sys
import time
import unittest

from integrity import ev
from journal import read_bounded
from run_tensor_tests import sample

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
MANIFEST = ROOT / "SOURCE_MANIFEST.json"
WORKER_SECONDS = 30


def verify_sources(path, expected_sha256):
    raw = read_bounded(path)
    if hashlib.sha256(raw).hexdigest() != ev.digest(expected_sha256):
        raise ValueError("Source manifest differs from externally retained hash")
    manifest = ev.strict_json(raw)
    ev.object_fields(manifest, ("schema_version", "files", "expected_tests"))
    ev.integer(manifest["schema_version"], "manifest version", 1, 1)
    if not isinstance(manifest["files"], dict) or not manifest["files"]:
        raise ValueError("Nonempty file manifest required")
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
            raise ValueError("Unsafe manifest path")
        target = path.parent / name
        cursor = path.parent
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("Symlinked source path")
        if hashlib.sha256(read_bounded(target)).hexdigest() != ev.digest(expected):
            raise ValueError("Changed source: " + name)
    expected = manifest["expected_tests"]
    if (
        not isinstance(expected, list)
        or not expected
        or any(not isinstance(x, str) or not x for x in expected)
        or len(expected) != len(set(expected))
    ):
        raise ValueError("Unique nonempty test inventory required")
    return manifest


def stable(previous, current):
    return (
        previous["free_mib"] >= 512
        and current["free_mib"] >= 512
        and current["pageouts"] <= previous["pageouts"]
        and current["swap_mib"] <= previous["swap_mib"]
    )


def admission(receipt):
    receipt["status"] = "BLOCKED_ADMISSION"
    if platform.system() != "Darwin":
        raise ValueError("This verification contract requires macOS")
    receipt["samples"] = [sample()]
    time.sleep(2)
    receipt["samples"].append(sample())
    if not stable(*receipt["samples"]):
        raise ValueError("Memory/pageout/swap admission failed")
    receipt["disk_free_bytes"] = shutil.disk_usage(ROOT).free
    if receipt["disk_free_bytes"] < 1024**3:
        raise ValueError("At least 1 GiB scratch required")
    if (BASE / "tensor-tests.lock").exists():
        raise ValueError("Original tensor runner owns this source tree")
    recheck = sample()
    receipt["pre_import_sample"] = recheck
    if not stable(receipt["samples"][-1], recheck):
        raise ValueError("Pre-import resource recheck failed")
    receipt["status"] = "READY_TO_VERIFY"


def test_ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from test_ids(test)
        else:
            yield test.id()


class InventoryResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed = []

    def startTest(self, test):
        self.executed.append(test.id())
        super().startTest(test)


def assess(result, expected):
    return bool(expected) and (
        result.wasSuccessful()
        and result.testsRun == len(expected)
        and sorted(result.executed) == sorted(expected)
        and not result.skipped
        and not result.expectedFailures
        and not result.unexpectedSuccesses
    )


def bounded_process(command, log_path, seconds=WORKER_SECONDS):
    """Kill only this owned process group on timeout, including its descendants."""
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return process.wait(timeout=seconds), False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return process.returncode, True


def worker(args, manifest, receipt):
    lock = BASE / "tensor-runtime.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        admission(receipt)
        verify_sources(MANIFEST, args.manifest_sha256)
        # No tensor import, module discovery or model construction before admission.
        sys.path.insert(0, str(ROOT / "successor/itfm_v2/src"))
        import torch
        import numpy

        torch.set_num_threads(1)
        receipt["environment"] = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
        }
        suite = unittest.TestSuite(
            unittest.defaultTestLoader.discover(str(BASE), pattern=pattern)
            for pattern in ("test_tensor.py", "test_durable_tensor.py")
        )
        discovered = list(test_ids(suite))
        if sorted(discovered) != sorted(manifest["expected_tests"]):
            raise ValueError("Discovered tests differ from frozen inventory")
        output = io.StringIO()
        result = unittest.TextTestRunner(
            stream=output, verbosity=2, resultclass=InventoryResult
        ).run(suite)
        verify_sources(MANIFEST, args.manifest_sha256)
        receipt.update(
            tests_run=result.testsRun,
            executed=result.executed,
            test_log=output.getvalue(),
            skipped=len(result.skipped),
            failures=len(result.failures),
            errors=len(result.errors),
            expected_failures=len(result.expectedFailures),
        )
        receipt["status"] = (
            "PASS_TENSOR_SOFTWARE_ONLY"
            if assess(result, manifest["expected_tests"])
            else "FAIL_TENSOR_TESTS"
        )
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run", "worker"))
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError("Refusing existing receipt")
    receipt = {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "FAIL_RUNTIME_VERIFICATION",
        "tests_run": 0,
        "manifest_sha256": args.manifest_sha256,
        "model_modules_imported": [],
        "scope": "Tensor software verification only; zero scientific trials.",
    }
    started = time.monotonic()
    try:
        manifest = verify_sources(MANIFEST, args.manifest_sha256)
        if args.mode == "worker":
            worker(args, manifest, receipt)
        else:
            if (BASE / "tensor-runtime.lock").exists():
                raise ValueError(
                    "Tensor worker lock exists; prior failure or live owner requires audit"
                )
            admission(receipt)
            if args.mode == "preflight":
                receipt["status"] = "PASS_PREFLIGHT_ONLY"
            else:
                child = args.receipt.with_name(args.receipt.name + ".worker.json")
                log = args.receipt.with_name(args.receipt.name + ".worker.log")
                if child.exists() or log.exists():
                    raise FileExistsError("Refusing previous worker outputs")
                command = [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "worker",
                    "--manifest-sha256",
                    args.manifest_sha256,
                    "--receipt",
                    str(child),
                ]
                code, timed_out = bounded_process(command, log)
                receipt.update(worker_exit_code=code, timed_out=timed_out)
                if timed_out or code != 0 or not child.is_file():
                    raise ValueError(
                        "Worker failed or exhausted its 30-second total budget"
                    )
                raw = read_bounded(child)
                detail = ev.strict_json(raw)
                if (
                    detail["status"] != "PASS_TENSOR_SOFTWARE_ONLY"
                    or detail["tests_run"] != len(manifest["expected_tests"])
                    or sorted(detail["executed"]) != sorted(manifest["expected_tests"])
                    or detail["manifest_sha256"] != args.manifest_sha256
                ):
                    raise ValueError(
                        "Worker receipt lacks complete exact-source acceptance"
                    )
                verify_sources(MANIFEST, args.manifest_sha256)
                receipt.update(
                    status=detail["status"],
                    tests_run=detail["tests_run"],
                    worker_receipt_sha256=hashlib.sha256(raw).hexdigest(),
                )
    except BaseException as error:
        receipt["reason"] = repr(error)
        if receipt["status"] != "BLOCKED_ADMISSION" or "torch" in sys.modules:
            receipt["status"] = "FAIL_TENSOR_TESTS"
    receipt.update(
        elapsed_seconds=time.monotonic() - started,
        peak_rss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        model_modules_imported=[x for x in ("torch", "numpy") if x in sys.modules],
    )
    ev.write_receipt(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
