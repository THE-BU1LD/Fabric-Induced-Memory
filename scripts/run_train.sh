#!/bin/bash

set -e

# ----------------------------
# CONFIG
# ----------------------------
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-3e-4}
STEPS=${STEPS:-100}
SEED=${SEED:-42}
NUM_GPUS=${NUM_GPUS:-1}

EXP_NAME=${EXP_NAME:-exp_$(date +%s)}
SAVE_DIR=${SAVE_DIR:-runs/$EXP_NAME}
LOG_FILE=$SAVE_DIR/train.log
CSV_FILE=$SAVE_DIR/metrics.csv

mkdir -p $SAVE_DIR

echo "=============================="
echo "Starting Training: $EXP_NAME"
echo "=============================="

# ----------------------------
# REPRODUCIBILITY
# ----------------------------
export PYTHONHASHSEED=$SEED

# ----------------------------
# TRAIN COMMAND
# ----------------------------
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCH_CMD="torchrun --nproc_per_node=$NUM_GPUS run_train.py"
else
    LAUNCH_CMD="python run_train.py"
fi

$LAUNCH_CMD \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --steps $STEPS \
    --device $DEVICE \
    --seed $SEED \
    --log_csv $CSV_FILE \
    2>&1 | tee $LOG_FILE

# ----------------------------
# SAVE MODEL
# ----------------------------
mv model.pt $SAVE_DIR/model.pt

# ----------------------------
# AUTO PLOT (if script exists)
# ----------------------------
if [ -f plot_metrics.py ]; then
    echo "Generating plots..."
    python plot_metrics.py --csv $CSV_FILE --out $SAVE_DIR
fi

echo "Training complete."
echo "Saved to $SAVE_DIR"