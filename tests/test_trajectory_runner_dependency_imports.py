"""Cached foreign modules cannot replace trajectory-runner dependencies."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "cached_name",
    [
        "train",
        "ablation_systems",
        "systems",
        "models",
        "benchmark",
        "results.run_results",
    ],
)
def test_trajectory_dependencies_keep_package_identity(cached_name, tmp_path):
    program = r"""
import importlib
import importlib.util
from pathlib import Path
import sys
import types

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
name = sys.argv[2]
foreign = types.ModuleType(name)
if name == "train":
    foreign.train = lambda *args, **kwargs: {"foreign": True}
    foreign._rollout_loss = lambda *args, **kwargs: {"foreign": True}
elif name == "ablation_systems":
    foreign.AblatedFIMSystem = type("ForeignSystem", (), {})
    foreign.variant_switches = lambda *args, **kwargs: {"foreign": True}
elif name == "results.run_results":
    foreign.run = lambda *args, **kwargs: {"foreign": True}
    parent = types.ModuleType("results")
    parent.__path__ = []
    sys.modules["results"] = parent
sys.modules[name] = foreign

spec = importlib.util.spec_from_file_location(
    "trajectory_dependency_regression",
    root / "scripts/run_trajectory_isolated_ablations.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
main = importlib.import_module("fim_experiments.main")
train = importlib.import_module("fim_experiments.train")
ablations = importlib.import_module("fim_experiments.ablation_systems")
systems = importlib.import_module("fim_experiments.systems")
models = importlib.import_module("fim_experiments.models")
benchmark = importlib.import_module("fim_experiments.benchmark")
results = importlib.import_module("fim_experiments.results.run_results")
assert runner.train_module is train
assert main.train is train.train
assert runner.AblatedFIMSystem is ablations.AblatedFIMSystem
assert ablations.FIMSystem is systems.FIMSystem
assert main.FIMSystem is systems.FIMSystem
assert systems.Decoder is models.Decoder
assert main.Lorenz96Benchmark is benchmark.Lorenz96Benchmark
assert runner.CANONICAL_EVAL_RUN is results.run
assert sys.modules[name] is foreign
if name == "results.run_results":
    assert sys.modules["results"] is parent
"""
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MPLCONFIGDIR=str(tmp_path),
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(ROOT), cached_name],
        check=False,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
