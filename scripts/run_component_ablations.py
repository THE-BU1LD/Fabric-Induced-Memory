#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'fim_experiments'
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import main as experiment_main
from ablation_systems import AblatedFIMSystem, variant_switches


DEFAULT_VARIANTS = ['full', 'no_memory', 'no_retrieval', 'no_salience_gating']


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return 'UNKNOWN'


def model_factory(switches: dict[str, bool]):
    def factory(**kwargs):
        return AblatedFIMSystem(**kwargs, **switches)
    return factory


def metric_scalar(metrics: dict[str, Any], key: str):
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def run_one(args, benchmark: str, seed: int, variant: str) -> dict[str, Any]:
    switches = variant_switches(variant)
    experiment_main.FIMSystem = model_factory(switches)

    cfg = experiment_main.default_config()
    cfg['experiment']['name'] = f'current_ablation_{benchmark}_{variant}_s{seed}'
    cfg['runtime']['seed'] = seed
    cfg['runtime']['device'] = args.device
    cfg['runtime']['results_root'] = str(args.results_root)
    cfg['benchmark']['name'] = benchmark
    cfg['model']['name'] = 'fim_plus'
    cfg['train']['epochs'] = args.epochs
    cfg['train']['batch_size'] = args.batch_size
    cfg['train']['dataset_size'] = args.dataset_size
    cfg['train']['rollout_steps'] = args.rollout_steps
    cfg['train']['workers'] = 0
    cfg['eval']['rollout_steps'] = args.eval_steps

    started = datetime.now(timezone.utc).isoformat()
    metrics = experiment_main.run_experiment(cfg)
    ended = datetime.now(timezone.utc).isoformat()

    row = {
        'benchmark': benchmark,
        'seed': seed,
        'variant': variant,
        **switches,
        'started_utc': started,
        'ended_utc': ended,
        'git_commit': git_commit(),
        'experiment_name': cfg['experiment']['name'],
        'results_root': str(args.results_root),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'dataset_size': args.dataset_size,
        'train_rollout_steps': args.rollout_steps,
        'eval_rollout_steps': args.eval_steps,
    }
    for key in ['rollout_mse', 'rollout_mae', 'mse', 'mae']:
        value = metric_scalar(metrics, key)
        if value is not None:
            row[key] = value
    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Run explicit current-code FIM component ablations. Historical unsupported labels fail closed.'
    )
    ap.add_argument('--benchmarks', nargs='+', default=['lorenz96', 'delayed_recall'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[11, 23, 37])
    ap.add_argument('--variants', nargs='+', default=DEFAULT_VARIANTS)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--dataset-size', type=int, default=2048)
    ap.add_argument('--rollout-steps', type=int, default=4)
    ap.add_argument('--eval-steps', type=int, default=30)
    ap.add_argument('--results-root', type=Path, default=ROOT / 'results' / 'current_component_ablations')
    ap.add_argument('--manifest', type=Path, default=ROOT / 'results' / 'current_component_ablations' / 'manifest.json')
    args = ap.parse_args()

    # Resolve every label before any training so unsupported historical labels
    # fail before compute and cannot silently map to a different mechanism.
    for variant in args.variants:
        variant_switches(variant)

    args.results_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for benchmark in args.benchmarks:
        for seed in args.seeds:
            for variant in args.variants:
                print(f'RUN benchmark={benchmark} seed={seed} variant={variant}', flush=True)
                rows.append(run_one(args, benchmark, seed, variant))

    payload = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_commit(),
        'scientific_boundary': (
            'Fresh current-code component ablations only. These results do not reproduce or replace historical '
            'paper-reference labels unless a separate equivalence audit establishes matching semantics.'
        ),
        'benchmarks': args.benchmarks,
        'seeds': args.seeds,
        'variants': args.variants,
        'runs': rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f'WROTE {args.manifest} runs={len(rows)}', flush=True)


if __name__ == '__main__':
    main()
