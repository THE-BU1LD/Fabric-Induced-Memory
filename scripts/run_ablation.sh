#!/bin/bash

set -e

# ----------------------------
# CONFIG
# ----------------------------
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-10}
LR=${LR:-3e-4}
STEPS=${STEPS:-100}
SEED=${SEED:-42}

BASE_DIR=${BASE_DIR:-ablations_$(date +%s)}
mkdir -p $BASE_DIR

echo "=============================="
echo "Running Ablation Suite"
echo "=============================="

run_exp () {
    NAME=$1
    EXTRA_ARGS=$2

    EXP_DIR=$BASE_DIR/$NAME
    mkdir -p $EXP_DIR

    echo "---- Running: $NAME ----"

    python run_train.py \
        --epochs $EPOCHS \
        --lr $LR \
        --steps $STEPS \
        --device $DEVICE \
        --seed $SEED \
        --log_csv $EXP_DIR/metrics.csv \
        $EXTRA_ARGS \
        2>&1 | tee $EXP_DIR/train.log

    mv model.pt $EXP_DIR/model.pt

    if [ -f plot_metrics.py ]; then
        python plot_metrics.py \
            --csv $EXP_DIR/metrics.csv \
            --out $EXP_DIR
    fi
}

# ----------------------------
# RUNS
# ----------------------------
run_exp "baseline" ""
run_exp "no_memory" "--no_memory"
run_exp "no_retrieval" "--no_retrieval"
run_exp "low_capacity" "--hidden_dim 32"
run_exp "high_noise" "--noise 0.5"

echo "All ablations complete."
echo "Saved in $BASE_DIR"