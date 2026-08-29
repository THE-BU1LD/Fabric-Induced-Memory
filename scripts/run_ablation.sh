#!/usr/bin/env bash

set -euo pipefail

# Fresh current-code component ablations. These switches map only to mechanisms
# that exist in the maintained FIMSystem implementation. Historical labels such
# as `no_spectral_mixing` intentionally fail closed and are not silently reused.

DEVICE=${DEVICE:-auto}
EPOCHS=${EPOCHS:-12}
BATCH_SIZE=${BATCH_SIZE:-64}
DATASET_SIZE=${DATASET_SIZE:-2048}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-4}
EVAL_STEPS=${EVAL_STEPS:-30}
BENCHMARKS=${BENCHMARKS:-"lorenz96 delayed_recall"}
SEEDS=${SEEDS:-"11 23 37"}
VARIANTS=${VARIANTS:-"full no_memory no_retrieval no_salience_gating"}
RESULTS_ROOT=${RESULTS_ROOT:-results/current_component_ablations}
MANIFEST=${MANIFEST:-$RESULTS_ROOT/manifest.json}

# Intentional word splitting turns the documented space-separated environment
# variables into argparse lists.
# shellcheck disable=SC2086
python scripts/run_component_ablations.py \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --dataset-size "$DATASET_SIZE" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --eval-steps "$EVAL_STEPS" \
  --results-root "$RESULTS_ROOT" \
  --manifest "$MANIFEST" \
  --benchmarks $BENCHMARKS \
  --seeds $SEEDS \
  --variants $VARIANTS

echo "Fresh current-code FIM ablation matrix completed."
echo "Manifest: $MANIFEST"
