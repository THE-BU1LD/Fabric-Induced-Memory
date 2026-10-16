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


class BudgetExceeded(KeyboardInterrupt):
    pass


def deadline(*_):
    raise BudgetExceeded("30 second test budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
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
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(
            unittest.TestSuite(
                [
                    unittest.defaultTestLoader.discover(str(base), pattern=p)
                    for p in ("test_pipeline.py", "test_recovery.py")
                ]
            )
        )
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
        and result.wasSuccessful()
        and error is None
        and not changed
        and not model_modules
    )
    receipt = {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "PASS_SOFTWARE_TESTS_ONLY" if ok else "FAIL_SOFTWARE_TESTS",
        "tests_run": result.testsRun if result else 0,
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
