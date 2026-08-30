#!/usr/bin/env python3
"""Train a static or rate-augmented tire-force map from single-tire rig data.

Every learned tire model in this repository is supervised by the controlled
single-tire Chrono SCM rig, and this trainer enforces that at its input. It
accepts the schemas produced by ``data_collection/collect_static_data.cpp``
and ``data_collection/collect_rate_data.cpp``, and rejects any corpus carrying
whole-vehicle or axle-labelled columns. A model fitted to vehicle traces could
absorb the closed-loop behaviour it is later used to predict, so the check is
a refusal rather than a warning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)

OP_COLS = ["slip_ratio", "slip_angle", "velocity", "vertical_load", "steering_rate"]
RATE_COLS = ["d_slip_ratio", "d_slip_angle", "d_velocity"]
TERRAIN_COLS = [
    "bekker_Kphi", "bekker_Kc", "bekker_n", "mohr_cohesion",
    "mohr_friction", "janosi_shear",
]
OUT_COLS = ["Fx", "Fy"]
CHECKPOINT_FORMATS = {
    "static": "tire_force_static_mlp",
    "rate": "tire_force_rate_mlp",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rows(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    required = OP_COLS + TERRAIN_COLS + OUT_COLS
    if mode == "rate":
        required += RATE_COLS
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Rig {mode} CSV is missing columns: {missing}")
    forbidden = {
        "axle_id", "u_body", "v_body", "yaw_rate", "ax_imu", "ay_imu",
        "throttle_cmd", "brake_cmd",
    }
    present = sorted(forbidden.intersection(frame.columns))
    if present:
        raise ValueError(
            "Vehicle-trace columns are not accepted by the single-tire rig trainer: "
            + ", ".join(present)
        )

    mask = (
        frame["slip_ratio"].between(-1.2, 1.2)
        & frame["slip_angle"].between(-0.7, 0.7)
        & frame["velocity"].between(0.25, 20.0)
        & frame["vertical_load"].between(1000.0, 12000.0)
        & frame["steering_rate"].between(-2.0, 2.0)
        & frame["Fx"].between(-5.0e4, 5.0e4)
        & frame["Fy"].between(-5.0e4, 5.0e4)
    )
    if mode == "rate":
        mask &= (
            frame["d_slip_ratio"].between(-5.0, 5.0)
            & frame["d_slip_angle"].between(-2.0, 2.0)
            & frame["d_velocity"].between(-10.0, 10.0)
        )
        duplicate = np.isclose(
            frame["steering_rate"].to_numpy(float),
            frame["d_slip_angle"].to_numpy(float),
            atol=1e-8,
        ).mean()
        if duplicate >= 0.995:
            raise ValueError(
                "steering_rate duplicates d_slip_angle in at least 99.5% of rows; "
                "recollect the rig CSV because the rate channels are not independent"
            )
    dropped = int((~mask).sum())
    if dropped:
        LOG.warning("Dropping %d physically invalid rows", dropped)
    extra = [c for c in ("Fz", "kappa_cmd", "alpha_cmd", "velocity_cmd")
             if c in frame.columns and c not in required]
    keep = required + extra
    return frame.loc[mask, keep].reset_index(drop=True)


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden: list[int]):
        super().__init__()
        sizes = [input_size, *hidden, 2]
        self.layers = nn.ModuleList(
            nn.Linear(sizes[index], sizes[index + 1])
            for index in range(len(sizes) - 1)
        )
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, values):
        for layer in self.layers[:-1]:
            values = torch.tanh(layer(values))
        return self.layers[-1](values)


@dataclass
class Metrics:
    r2_fx: float
    r2_fy: float
    rmse_fx: float
    rmse_fy: float
    mae_fx: float
    mae_fy: float


def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    numerator = np.sum((actual - predicted) ** 2)
    denominator = np.sum((actual - np.mean(actual)) ** 2) + 1e-12
    return float(1.0 - numerator / denominator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arch", choices=["mlp"], required=True)
    parser.add_argument("--mode", choices=["static", "rate"], required=True)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--operating-point", choices=["measured", "commanded"],
                        default="measured",
                        help="Which operating point the network is conditioned on: the "
                             "tire model's internal slip state, or the value the rig "
                             "commanded. The commanded point matches what the "
                             "controller can supply at runtime.")
    parser.add_argument("--drop-features", default="",
                        help="Comma-separated feature columns to exclude. Excluding one "
                             "channel at a time isolates its contribution, which "
                             "identifies which input carries a difference between two "
                             "corpora that record the same physical quantities from "
                             "different sources.")
    parser.add_argument(
        "--load-feature", choices=["commanded", "measured"], default="commanded",
        help="Which normal load the network is conditioned on. 'commanded' is the "
             "LHS setpoint written to vertical_load; 'measured' is the load the tire "
             "actually carried, logged as Fz. At runtime the controller supplies a "
             "weight-transfer estimate of the actual load, so 'measured' matches "
             "deployment, while 'commanded' mislabels any row where the rig missed "
             "its setpoint.",
    )
    parser.add_argument(
        "--holdout", default="",
        help="Optional CSV from an independent sampling design. The internal test "
             "split is a random slice of the training design and inherits its "
             "stratification, so it overstates generalisation; a holdout drawn "
             "from a separate design does not.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    data_path = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    frame = validate_rows(pd.read_csv(data_path), args.mode)
    feature_cols = OP_COLS + (RATE_COLS if args.mode == "rate" else []) + TERRAIN_COLS
    if args.drop_features:
        drop = {c.strip() for c in args.drop_features.split(",") if c.strip()}
        unknown = drop - set(feature_cols)
        if unknown:
            raise SystemExit(f"--drop-features names unknown columns: {sorted(unknown)}")
        feature_cols = [c for c in feature_cols if c not in drop]
        LOG.info("dropped %d feature(s): %s", len(drop), ", ".join(sorted(drop)))
    if args.operating_point == "commanded":
        # The rig commands slip angle over +/-0.40 rad but the tire's internal
        # accessor realises only about a quarter of that, so a surrogate trained on
        # the tire-computed column sees a domain four times too narrow. The
        # commanded columns restore the intended coverage, and match what the
        # controller supplies at runtime: an estimate from vehicle state, not the
        # tire model's internal slip state.
        missing = [c for c in ("kappa_cmd", "alpha_cmd", "velocity_cmd") if c not in frame]
        if missing:
            raise SystemExit(f"--operating-point commanded needs columns {missing}; "
                             "recollect with a collector that records them")
        frame = frame.assign(slip_ratio=frame["kappa_cmd"],
                             slip_angle=frame["alpha_cmd"],
                             velocity=frame["velocity_cmd"])
    if args.load_feature == "measured":
        # Substitute in place so the runtime feature order is unchanged; the
        # controller keeps passing its weight-transfer load into the same slot.
        frame = frame.assign(vertical_load=frame["Fz"])
    values = frame[feature_cols].to_numpy(np.float32)
    labels = frame[OUT_COLS].to_numpy(np.float32)
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        values, labels, test_size=0.2, random_state=args.seed
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.5, random_state=args.seed
    )

    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = [
        torch.tensor(x_scaler.transform(array), dtype=torch.float32, device=device)
        for array in (x_train, x_val, x_test)
    ]
    y_train_t = torch.tensor(y_scaler.transform(y_train), dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_scaler.transform(y_val), dtype=torch.float32, device=device)

    model = MLP(len(feature_cols), list(args.hidden)).to(device)
    architecture = {"architecture_type": "mlp", "hidden_sizes": list(args.hidden)}

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(tensors[0], y_train_t),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15
    )
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(tensors[1]), y_val_t))
        scheduler.step(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            LOG.info("epoch %d/%d val=%.6f best=%.6f", epoch + 1, args.epochs,
                     validation_loss, best_loss)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = y_scaler.inverse_transform(model(tensors[2]).cpu().numpy())
    error = y_test.astype(float) - predicted.astype(float)
    metrics = Metrics(
        r2_fx=r2(y_test[:, 0], predicted[:, 0]),
        r2_fy=r2(y_test[:, 1], predicted[:, 1]),
        rmse_fx=float(np.sqrt(np.mean(error[:, 0] ** 2))),
        rmse_fy=float(np.sqrt(np.mean(error[:, 1] ** 2))),
        mae_fx=float(np.mean(np.abs(error[:, 0]))),
        mae_fy=float(np.mean(np.abs(error[:, 1]))),
    )

    holdout_metrics = None
    holdout_info = {}
    if args.holdout:
        holdout_path = Path(args.holdout).expanduser().resolve()
        holdout_frame = validate_rows(pd.read_csv(holdout_path), args.mode)
        h_x = holdout_frame[feature_cols].to_numpy(np.float32)
        h_y = holdout_frame[OUT_COLS].to_numpy(np.float32)
        h_t = torch.tensor(x_scaler.transform(h_x), dtype=torch.float32, device=device)
        with torch.no_grad():
            h_pred = y_scaler.inverse_transform(model(h_t).cpu().numpy())
        h_err = h_y.astype(float) - h_pred.astype(float)
        holdout_metrics = Metrics(
            r2_fx=r2(h_y[:, 0], h_pred[:, 0]),
            r2_fy=r2(h_y[:, 1], h_pred[:, 1]),
            rmse_fx=float(np.sqrt(np.mean(h_err[:, 0] ** 2))),
            rmse_fy=float(np.sqrt(np.mean(h_err[:, 1] ** 2))),
            mae_fx=float(np.mean(np.abs(h_err[:, 0]))),
            mae_fy=float(np.mean(np.abs(h_err[:, 1]))),
        )
        holdout_info = {
            "holdout_csv": str(holdout_path),
            "holdout_csv_sha256": sha256(holdout_path),
            "holdout_rows": int(len(holdout_frame)),
        }
        LOG.info("holdout R2 Fx=%.4f Fy=%.4f (internal test %.4f / %.4f)",
                 holdout_metrics.r2_fx, holdout_metrics.r2_fy,
                 metrics.r2_fx, metrics.r2_fy)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scalers.pkl").open("wb") as stream:
        pickle.dump({"X": x_scaler, "y": y_scaler}, stream)
    checkpoint = {
        "checkpoint_format": CHECKPOINT_FORMATS[args.mode],
        "model_state_dict": best_state,
        **architecture,
        "temporal_K": 1,
        "rate_augmented": args.mode == "rate",
        "feature_cols": feature_cols,
        "input_size": len(feature_cols),
        "output_size": 2,
        "val_loss": best_loss,
        "training_source": "chrono_scm_single_tire_rig",
        "training_csv_sha256": sha256(data_path),
        "load_feature": args.load_feature,
        "operating_point": args.operating_point,
        "seed": args.seed,
    }
    torch.save(checkpoint, output_dir / "best_terrain_nn.pt")
    payload = {
        "architecture": {
            **{key: value for key, value in checkpoint.items()
               if key not in {"model_state_dict"}},
            "n_params": int(sum(parameter.numel() for parameter in model.parameters())),
        },
        "test": asdict(metrics),
    }
    if holdout_metrics is not None:
        payload["holdout"] = {**asdict(holdout_metrics), **holdout_info}
    (output_dir / "test_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")

    # Human-readable provenance beside the machine-readable metrics. The joint
    # replay refuses to bind a force model that lacks it, so the trainer emits
    # one for every checkpoint it writes and no model can reach a benchmark
    # without a record of the corpus and seed that produced it.
    frame_cols = ", ".join(f"`{c}`" for c in feature_cols)
    lines = [
        f"# {output_dir.name} — training metadata", "",
        "| | |", "| --- | --- |",
        f"| Architecture | {args.mode}-mode MLP, hidden = {list(args.hidden)}, "
        f"{payload['architecture']['n_params']} params, tanh |",
        f"| Inputs ({len(feature_cols)}) | {frame_cols} |",
        "| Outputs | per-wheel tire-frame `(Fx, Fy)` |",
        f"| Operating point | {args.operating_point} "
        f"({'rig command at the measurement instant' if args.operating_point == 'commanded' else 'tire-model internal state'}) |",
        f"| Normal load | {args.load_feature} |",
        f"| Random-row test | R2 Fx = {metrics.r2_fx:.4f}, R2 Fy = {metrics.r2_fy:.4f}, "
        f"MAE Fx = {metrics.mae_fx:.1f} N, MAE Fy = {metrics.mae_fy:.1f} N |",
        "", "## Training data", "",
        f"- **Source**: Chrono SCM single-tire rig, `data_collection/collect_{args.mode}_data.cpp`.",
        f"- **CSV**: `{data_path.relative_to(Path.cwd()) if data_path.is_relative_to(Path.cwd()) else data_path}` "
        f"({len(frame):,} rows after validation).",
        f"- **CSV SHA-256**: `{checkpoint['training_csv_sha256']}`.",
        f"- **Seed**: {args.seed}.",
    ]
    if holdout_metrics is not None:
        lines += [
            "", "## Independent holdout", "",
            f"- **CSV**: `{holdout_info['holdout_csv']}` ({holdout_info['holdout_rows']:,} rows).",
            f"- **SHA-256**: `{holdout_info['holdout_csv_sha256']}`.",
            f"- **Scores**: R2 Fx = {holdout_metrics.r2_fx:.4f}, R2 Fy = {holdout_metrics.r2_fy:.4f}, "
            f"MAE Fx = {holdout_metrics.mae_fx:.1f} N, MAE Fy = {holdout_metrics.mae_fy:.1f} N.",
        ]
    (output_dir / "TRAINING_METADATA.md").write_text("\n".join(lines) + "\n")
    LOG.info("saved rig-trained checkpoint to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
