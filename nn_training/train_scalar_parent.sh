#!/bin/bash
# Reproduce nn_models/tire_force_static_parent, the smaller static tire-force
# checkpoint that supplies the scalar comparison arm in Table 2. It is trained
# on the controlled single-tire Chrono SCM rig corpus in data/tire_rig_static.
# Restore that dataset with data_sync/data_sync.sh, or re-collect it with
# data_collection/collect_static_data.cpp.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-${ROOT}/data/tire_rig_static/train.csv}"
MODELS="${MODELS:-${ROOT}/nn_models}"
TRAINER="${TRAINER:-${ROOT}/nn_training/train_variant.py}"

if [ ! -f "$DATA" ]; then
    echo "ERROR: dataset not found at $DATA"
    echo "Either re-collect via data_collection/collect_static_data.cpp"
    echo "or export DATA=/path/to/train.csv before running."
    exit 1
fi

EPOCHS=300
LR=0.01
PATIENCE=50
BATCH=256
SEED=42

echo "=== Training tire_force_static_parent ==="
echo "Data: $DATA"
echo "Rows: $(wc -l < "$DATA")"
echo ""

python "$TRAINER" \
    --data "$DATA" \
    --output-dir "${MODELS}/tire_force_static_parent" \
    --arch mlp --mode static --hidden 32 16 \
    --epochs $EPOCHS --lr $LR --patience $PATIENCE --batch-size $BATCH --seed $SEED

echo "=== tire_force_static_parent trained ==="
