#!/usr/bin/env bash
# Reproduce the two deployed tire-force surrogates from the tracked rig corpus.
#
#   nn_models/tire_force_static  -- the NMPC and safety-filter internal model
#                                   (Table 1's neural arm), static mode.
#   nn_models/tire_force_rate    -- the force surrogate GRIT evaluates over the
#                                   (n, phi) grid (Table 2's joint row), rate mode.
#
# Both are trained on data/tire_rig_commanded/train.csv, a DERIVED corpus
# collected end to end by the tracked data_collection/build_rig_cmd binaries
# from the controlled single-tire Chrono SCM rig at the commanded operating
# point, and scored against an independent holdout. Hyperparameters and seed
# are those recorded in each checkpoint's TRAINING_METADATA.md, so re-running
# this script reproduces the published metrics.
#
#   ./nn_training/train_deployed_surrogates.sh              # both, in place
#   MODELS=/tmp/check ./nn_training/train_deployed_surrogates.sh   # side-by-side
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-${ROOT}/data/tire_rig_commanded/train.csv}"
HOLDOUT="${HOLDOUT:-${ROOT}/data/tire_rig_commanded/holdout.csv}"
MODELS="${MODELS:-${ROOT}/nn_models}"
TRAINER="${TRAINER:-${ROOT}/nn_training/train_variant.py}"

for f in "$DATA" "$HOLDOUT"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: corpus not found at $f" >&2
        echo "data/tire_rig_commanded is tracked in-repo; restore it with git checkout." >&2
        exit 1
    fi
done

EPOCHS=300
LR=0.01
PATIENCE=50
BATCH=256
SEED=42

for mode in static rate; do
    out="${MODELS}/tire_force_${mode}"
    echo "=== Training tire_force_${mode} -> ${out} ==="
    python "$TRAINER" \
        --data "$DATA" \
        --holdout "$HOLDOUT" \
        --output-dir "$out" \
        --arch mlp --mode "$mode" --hidden 64 32 \
        --operating-point commanded \
        --epochs $EPOCHS --lr $LR --patience $PATIENCE \
        --batch-size $BATCH --seed $SEED
done

echo "=== deployed surrogates trained ==="
