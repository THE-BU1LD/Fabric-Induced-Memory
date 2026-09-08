#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
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


def common_row(args, benchmark: str, seed: int, variant: str) -> dict[str, Any]:
    switches = variant_switches(variant)
    return {
        'benchmark': benchmark,
        'seed': seed,
        'variant': variant,
        **switches,
        'git_commit': git_commit(),
        'experiment_name': f'current_ablation_{benchmark}_{variant}_s{seed}',
        'results_root': str(args.results_root),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'dataset_size': args.dataset_size,
        'train_rollout_steps': args.rollout_steps,
        'eval_rollout_steps': args.eval_steps,
    }


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
        **common_row(args, benchmark, seed, variant),
        'status': 'success',
        'started_utc': started,
        'ended_utc': ended,
    }
    for key in ['rollout_mse', 'rollout_mae', 'mse', 'mae']:
        value = metric_scalar(metrics, key)
        if value is not None:
            row[key] = value
    return row


def failed_row(args, benchmark: str, seed: int, variant: str, exc: BaseException, started: str) -> dict[str, Any]:
    return {
        **common_row(args, benchmark, seed, variant),
        'status': 'failed',
        'started_utc': started,
        'ended_utc': datetime.now(timezone.utc).isoformat(),
        'error_type': type(exc).__name__,
        'error_message': str(exc),
        'traceback': traceback.format_exc(),
    }


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
    ap.add_argument(
        '--continue-on-error', action='store_true',
        help='Record failed cells in the manifest and continue so failure evidence is retained. The caller must still fail closed on any failed cell.'
    )
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
                started = datetime.now(timezone.utc).isoformat()
                try:
                    rows.append(run_one(args, benchmark, seed, variant))
                except Exception as exc:
                    failure = failed_row(args, benchmark, seed, variant, exc, started)
                    rows.append(failure)
                    print(
                        f'FAILED benchmark={benchmark} seed={seed} variant={variant}: '
                        f'{failure["error_type"]}: {failure["error_message"]}',
                        file=sys.stderr,
                        flush=True,
                    )
                    if not args.continue_on_error:
                        # Write the partial manifest before propagating so completed and
                        # failed evidence is not lost when a canonical fail-fast run stops.
                        payload = build_payload(args, rows)
                        args.manifest.parent.mkdir(parents=True, exist_ok=True)
                        args.manifest.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
                        raise

    payload = build_payload(args, rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    failures = sum(r.get('status') == 'failed' for r in rows)
    print(f'WROTE {args.manifest} runs={len(rows)} failures={failures}', flush=True)


def build_payload(args, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'schema_version': 2,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_commit(),
        'scientific_boundary': (
            'Fresh current-code component ablations only. These results do not reproduce or replace historical '
            'paper-reference labels unless a separate equivalence audit establishes matching semantics. Failed '
            'cells are retained and must not be silently excluded from paper-facing aggregation.'
        ),
        'benchmarks': args.benchmarks,
        'seeds': args.seeds,
        'variants': args.variants,
        'runs': rows,
    }


if __name__ == '__main__':
    main()
