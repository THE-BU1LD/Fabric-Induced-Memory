"""The trajectory runner must not reuse an unrelated cached ``main`` module."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("cached_main", ["root_wrapper", "foreign_with_run"])
def test_trajectory_runner_uses_the_maintained_experiment_module(cached_main, tmp_path):
    program = r"""
import importlib
import importlib.util
from pathlib import Path
import sys
import types

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
if sys.argv[2] == "root_wrapper":
    cached = importlib.import_module("main")
    assert Path(cached.__file__).resolve() == root / "main.py"
else:
    cached = types.ModuleType("main")
    cached.run = lambda *args, **kwargs: {"unrelated": True}
    sys.modules["main"] = cached

spec = importlib.util.spec_from_file_location(
    "trajectory_import_regression", root / "scripts/run_trajectory_isolated_ablations.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
canonical = importlib.import_module("fim_experiments.main")
assert runner.experiment_main is canonical
assert runner.CANONICAL_EVAL_RUN is canonical.run
assert sys.modules["main"] is cached
"""
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MPLCONFIGDIR=str(tmp_path),
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(ROOT), cached_main],
        check=False,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_direct_runner_help_works_outside_repository(tmp_path):
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MPLCONFIGDIR=str(tmp_path),
    )
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_trajectory_isolated_ablations.py"),
            "--help",
        ],
        check=False,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout
