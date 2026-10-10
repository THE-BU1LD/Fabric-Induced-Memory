from pathlib import Path
import importlib, sys, tempfile, types, time

start = time.perf_counter()
with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    package = root / "fim_experiments"
    package.mkdir()
    for name in ["benchmark.py", "runtime_benchmarks.py", "episode_bank_v1.py"]:
        (package / name).write_bytes((Path("fim_experiments") / name).read_bytes())
    alias = types.ModuleType("bank_source_review")
    alias.__path__ = [str(package)]
    sys.modules[alias.__name__] = alias
    bank = importlib.import_module("bank_source_review.episode_bank_v1")
    runtime = importlib.import_module("bank_source_review.runtime_benchmarks")
    source = package / "runtime_benchmarks.py"
    original = source.read_text()
    needle = "visible = gate * mem + self.config.distractor_scale * noise"
    assert original.count(needle) == 1
    source.write_text(original.replace(needle, needle + " + 0.125"))
    config = bank.DelayedRecallConfig(dimension=2, memory_dim=1, delay=2, steps=3)
    first = bank.create_development_bank(root / "stale", seed=37, config=config, horizon=3)
    loaded = bank.load_development_bank(root / "stale", expected_bank_sha256=first["bank_sha256"])
    print("STALE_PROCESS_ADMITTED:", len(loaded.dataset("train")))
    importlib.reload(runtime)
    importlib.reload(bank)
    second = bank.create_development_bank(root / "fresh", seed=37, config=config, horizon=3)
    import json
    a = json.loads((root / "stale" / "manifest.json").read_text())
    b = json.loads((root / "fresh" / "manifest.json").read_text())
    print("EXACT_SAME_SOURCE_HASHES:", a["source_sha256"] == b["source_sha256"])
    print("DIFFERENT_OBSERVATION_BYTES:", first["observations_sha256"] != second["observations_sha256"])
    print("SECONDS:", round(time.perf_counter() - start, 3))

