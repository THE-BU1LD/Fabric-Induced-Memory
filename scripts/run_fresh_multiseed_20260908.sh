#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="results/fresh_multiseed_20260908"
mkdir -p "$OUT"

printf 'protocol=research/FRESH_MULTI_SEED_REPLICATION_PROTOCOL_20260908.md\n' > "$OUT/execution_receipt.txt"
printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf UNKNOWN)" >> "$OUT/execution_receipt.txt"
printf 'git_status=%s\n' "$(git status --porcelain 2>/dev/null | tr '\n' ';' || true)" >> "$OUT/execution_receipt.txt"
printf 'python=%s\n' "$(python --version 2>&1)" >> "$OUT/execution_receipt.txt"
printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/execution_receipt.txt"

# Fail before outcome-bearing compute if canonical implementation semantics are not sound.
python -m compileall -q fim fim_experiments scripts tests
pytest -q

# Frozen 40-cell mechanism matrix.
python scripts/run_component_ablations.py \
  --benchmarks lorenz96 delayed_recall \
  --seeds 101 202 303 404 505 \
  --variants full no_memory no_retrieval no_salience_gating \
  --device cpu \
  --epochs 6 \
  --batch-size 64 \
  --dataset-size 1024 \
  --rollout-steps 4 \
  --eval-steps 30 \
  --results-root "$OUT/ablations" \
  --manifest "$OUT/ablations/manifest.json"

# Frozen 50-cell maintained-baseline matrix. Each seed gets its own experiment namespace
# so raw configs/logs/checkpoints are never silently overwritten across seeds.
for seed in 101 202 303 404 505; do
  python scripts/run_full_suite.py \
    --benchmarks lorenz96,delayed_recall \
    --models fim_plus,transformer,ssm,deeponet,mlp \
    --results_root "$OUT/baselines" \
    --exp_name "baseline_seed_${seed}" \
    --seed "$seed" \
    --device cpu \
    --epochs 6 \
    --batch_size 64 \
    --dataset_size 1024 \
    --workers 0 \
    --eval_steps 30 \
    --rollout_steps 4
done

python scripts/summarize_fresh_multiseed_20260908.py
printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/execution_receipt.txt"
