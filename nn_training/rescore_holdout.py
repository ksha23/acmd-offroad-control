#!/usr/bin/env python3
"""Re-score a trained tire surrogate on an independent holdout, without retraining.

The checkpoint records which operating-point and load columns it was trained on, and the holdout rows are
mapped the same way before scoring (train_variant.apply_feature_choices). With --write, the holdout block of
the checkpoint's test_metrics.json is replaced by the new scores.

    python nn_training/rescore_holdout.py nn_models/tire_force_static data/tire_rig_commanded/holdout.csv
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_variant import MLP, OUT_COLS, apply_feature_choices, r2, sha256, validate_rows


def load_checkpoint(model_dir: Path):
    checkpoint = torch.load(model_dir / "best_terrain_nn.pt", map_location="cpu", weights_only=False)
    with (model_dir / "scalers.pkl").open("rb") as stream:
        scalers = pickle.load(stream)
    model = MLP(checkpoint["input_size"], list(checkpoint["hidden_sizes"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, scalers, model


def score(model_dir: Path, holdout_csv: Path) -> dict:
    checkpoint, scalers, model = load_checkpoint(model_dir)
    mode = "rate" if checkpoint.get("rate_augmented") else "static"
    choices = argparse.Namespace(operating_point=checkpoint["operating_point"],
                                 load_feature=checkpoint["load_feature"])

    frame = validate_rows(pd.read_csv(holdout_csv), mode)
    frame = apply_feature_choices(frame, choices)
    inputs = frame[checkpoint["feature_cols"]].to_numpy(np.float32)
    targets = frame[OUT_COLS].to_numpy(float)

    scaled = torch.tensor(scalers["X"].transform(inputs), dtype=torch.float32)
    with torch.no_grad():
        predicted = scalers["y"].inverse_transform(model(scaled).numpy()).astype(float)
    error = targets - predicted

    return {
        "r2_fx": r2(targets[:, 0], predicted[:, 0]),
        "r2_fy": r2(targets[:, 1], predicted[:, 1]),
        "rmse_fx": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "rmse_fy": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "mae_fx": float(np.mean(np.abs(error[:, 0]))),
        "mae_fy": float(np.mean(np.abs(error[:, 1]))),
        "holdout_csv": str(holdout_csv.relative_to(Path(__file__).resolve().parents[1])),
        "holdout_csv_sha256": sha256(holdout_csv),
        "holdout_rows": int(len(frame)),
        "operating_point": checkpoint["operating_point"],
        "load_feature": checkpoint["load_feature"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("holdout_csv", type=Path)
    parser.add_argument("--write", action="store_true",
                        help="Replace the holdout block of the checkpoint's test_metrics.json.")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    result = score(model_dir, args.holdout_csv.resolve())
    print(json.dumps(result, indent=2))

    if args.write:
        metrics_path = model_dir / "test_metrics.json"
        payload = json.loads(metrics_path.read_text())
        payload["holdout"] = result
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
