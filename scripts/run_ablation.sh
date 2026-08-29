#!/usr/bin/env bash

set -euo pipefail

DEVICE=${DEVICE:-auto}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-128}
DATASET_SIZE=${DATASET_SIZE:-4096}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-50}
BENCHMARK=${BENCHMARK:-lorenz96}
RESULTS_ROOT=${RESULTS_ROOT:-ablation_results}
SEEDS=${SEEDS:-"41 42 43"}
VARIANTS=${VARIANTS:-"full no_memory no_retrieval"}

read -r -a SEED_ARGS <<< "$SEEDS"
read -r -a VARIANT_ARGS <<< "$VARIANTS"

printf '%s\n' "==============================" "FIM mechanism ablation suite" "=============================="
printf 'benchmark: %s\n' "$BENCHMARK"
printf 'variants:  %s\n' "$VARIANTS"
printf 'seeds:     %s\n' "$SEEDS"
printf 'outputs:   %s\n' "$RESULTS_ROOT"

python -m fim_experiments.ablation \
  --benchmark "$BENCHMARK" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --dataset-size "$DATASET_SIZE" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --results-root "$RESULTS_ROOT" \
  --variants "${VARIANT_ARGS[@]}" \
  --seeds "${SEED_ARGS[@]}"

python -m fim_experiments.ablation_report "$RESULTS_ROOT/ablation_manifest.json"

printf '\nDone. Inspect the manifest, per-run summary/provenance files, and ablation_summary.md before making paper claims.\n'
