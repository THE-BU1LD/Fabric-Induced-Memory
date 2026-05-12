#!/bin/bash

set -e

# ----------------------------
# CONFIG
# ----------------------------
DEVICE=${DEVICE:-cuda}
MODEL_PATH=${MODEL_PATH:-runs/latest/model.pt}
STEPS=${STEPS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}

EXP_NAME=${EXP_NAME:-eval_$(date +%s)}
SAVE_DIR=${SAVE_DIR:-eval_runs/$EXP_NAME}
LOG_FILE=$SAVE_DIR/eval.log

mkdir -p $SAVE_DIR

echo "=============================="
echo "Running Evaluation: $EXP_NAME"
echo "=============================="

python run_eval.py \
    --device $DEVICE \
    --model_path $MODEL_PATH \
    --steps $STEPS \
    --batch_size $BATCH_SIZE \
    --save_dir $SAVE_DIR \
    2>&1 | tee $LOG_FILE

# ----------------------------
# OPTIONAL PLOTS
# ----------------------------
if [ -f plot_rollouts.py ]; then
    echo "Generating rollout plots..."
    python plot_rollouts.py --input $SAVE_DIR/eval_outputs.pt --out $SAVE_DIR
fi

echo "Evaluation complete."
echo "Saved to $SAVE_DIR"