"""Run bounded software checks with a retained receipt, including failures."""

import argparse
import datetime
import hashlib
import io
import json
import resource
import signal
import sys
import time
import unittest
from pathlib import Path

from runtime_verification import (
    MANIFEST,
    InventoryResult,
    assess,
    test_ids,
    verify_sources,
)


class BudgetExceeded(KeyboardInterrupt):
    pass


def deadline(*_):
    raise BudgetExceeded("30 second test budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError("Refusing to overwrite test receipt")
    base = Path(__file__).resolve().parent
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in base.iterdir()
        if p.suffix in (".py", ".md")
    }
    output = io.StringIO()
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(30)
    error = None
    result = None
    try:
        manifest = verify_sources(MANIFEST, args.manifest_sha256)
        suite = unittest.TestSuite(
            unittest.defaultTestLoader.discover(str(base), pattern=pattern)
            for pattern in (
                "test_pipeline.py",
                "test_recovery.py",
                "test_runtime_gate.py",
            )
        )
        expected = manifest["expected_scalar_tests"]
        if sorted(test_ids(suite)) != sorted(expected):
            raise ValueError("Scalar discovery differs from frozen inventory")
        result = unittest.TextTestRunner(
            stream=output, verbosity=2, resultclass=InventoryResult
        ).run(suite)
        verify_sources(MANIFEST, args.manifest_sha256)
    except BaseException as exception:
        error = repr(exception)
    finally:
        signal.alarm(0)
    changed = [
        name
        for name, digest in hashes.items()
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != digest
    ]
    model_modules = [
        name for name in ("torch", "numpy", "tensorflow") if name in sys.modules
    ]
    ok = (
        result is not None
        and assess(result, expected)
        and error is None
        and not changed
        and not model_modules
    )
    receipt = {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "PASS_SOFTWARE_TESTS_ONLY" if ok else "FAIL_SOFTWARE_TESTS",
        "tests_run": result.testsRun if result else 0,
        "manifest_sha256": args.manifest_sha256,
        "executed": result.executed if result else [],
        "skipped": len(result.skipped) if result else None,
        "expected_failures": len(result.expectedFailures) if result else None,
        "unexpected_successes": len(result.unexpectedSuccesses) if result else None,
        "failures": len(result.failures) if result else None,
        "errors": len(result.errors) if result else None,
        "exception": error,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_rss_units": "bytes on macOS; KiB on Linux",
        "python": sys.version,
        "source_sha256": hashes,
        "files_changed_during_tests": changed,
        "model_modules_imported": model_modules,
        "boundary": "Adversarial software fixtures only; no scientific data, model run, performance result or independent review.",
        "test_log": output.getvalue(),
    }
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "status",
                    "tests_run",
                    "failures",
                    "errors",
                    "elapsed_seconds",
                    "model_modules_imported",
                )
            }
        )
    )
    if not ok:
        print(output.getvalue())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
