"""Guard actual tensor integration tests; preserve refusals before torch import."""

import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
import unittest

from integrity import ev


def sample():
    raw = subprocess.check_output(["vm_stat"], text=True)
    page_size = int(re.search(r"page size of (\d+)", raw)[1])
    fields = dict((k, int(v)) for k, v in re.findall(r"^([^:]+):\s+(\d+)\.", raw, re.M))
    swap = subprocess.check_output(
        ["sysctl", "vm.swapusage"], text=True, stderr=subprocess.STDOUT
    )
    used = re.search(r"used = ([\d.]+)([MG])", swap)
    if used is None:
        raise ValueError("Unable to establish swap usage")
    return {
        "free_mib": (fields["Pages free"] + fields["Pages speculative"])
        * page_size
        / 2**20,
        "pageouts": fields["Pageouts"],
        "swap_mib": float(used[1]) * (1024 if used[2] == "G" else 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError("Refusing existing receipt")
    base = Path(__file__).resolve().parent
    repo = base.parents[3]
    receipt = {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Actual tensor software integration, no scientific trial",
        "status": "BLOCKED_ADMISSION",
        "samples": [],
        "floor_mib": 512,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in base.glob("*.py")
        },
        "tests_run": 0,
        "model_modules_imported": [],
        "errors": [],
    }
    try:
        for i in range(2):
            if i:
                time.sleep(2)
            receipt["samples"].append(sample())
        a, b = receipt["samples"]
        if (
            min(a["free_mib"], b["free_mib"]) < 512
            or b["pageouts"] > a["pageouts"]
            or b["swap_mib"] > a["swap_mib"]
        ):
            raise ValueError("Memory/pageout/swap admission failed")
        receipt["disk_free_bytes"] = shutil.disk_usage(base).free
        if receipt["disk_free_bytes"] < 1024**3:
            raise ValueError("Less than 1 GiB scratch space")
        processes = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
        conflicts = [
            line
            for line in processes.splitlines()
            if line.strip().split(maxsplit=1)[0] != str(os.getpid())
            and re.search(
                r"python\S* .*?(itfm|fim_sota|run_tensor_tests|test_tensor)", line
            )
        ]
        receipt["conflicting_processes"] = conflicts
        if conflicts or any(base.glob("*.lock")):
            raise ValueError("Conflicting owner or lock")
        launch = sample()
        receipt["pre_import_recheck"] = launch
        if (
            launch["free_mib"] < 512
            or launch["pageouts"] > b["pageouts"]
            or launch["swap_mib"] > b["swap_mib"]
        ):
            raise ValueError("Mandatory pre-import resource recheck failed")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        receipt["errors"].append(str(error))
        ev.write_receipt(args.receipt, receipt)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "errors": receipt["errors"],
                    "model_modules_imported": [],
                }
            )
        )
        return 2
    # Exclusive lock plus bounded worker lifetime, only after complete admission.
    lock = base / "tensor-tests.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    output = io.StringIO()
    started = time.perf_counter()
    result = None

    def deadline(*_):
        raise TimeoutError("Tensor correctness test 30 second budget exhausted")

    try:
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(30)
        sys.path.insert(0, str(repo / "successor/itfm_v2/src"))
        import torch

        torch.set_num_threads(1)
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(
            unittest.defaultTestLoader.discover(str(base), pattern="test_tensor.py")
        )
        receipt["status"] = (
            "PASS_TENSOR_SOFTWARE_ONLY"
            if result.wasSuccessful()
            else "FAIL_TENSOR_TESTS"
        )
    except BaseException as error:
        receipt["status"] = "FAIL_TENSOR_TESTS"
        receipt["errors"].append(repr(error))
    finally:
        signal.alarm(0)
        lock.unlink()
    receipt.update(
        tests_run=result.testsRun if result else 0,
        test_log=output.getvalue(),
        elapsed_seconds=time.perf_counter() - started,
        peak_rss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        model_modules_imported=[m for m in ("torch", "numpy") if m in sys.modules],
    )
    ev.write_receipt(args.receipt, receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "tests_run", "errors")}))
    return 0 if receipt["status"] == "PASS_TENSOR_SOFTWARE_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
