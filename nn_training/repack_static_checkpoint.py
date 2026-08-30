#!/usr/bin/env python3
"""Attach verifiable rig provenance to a static tire-force checkpoint.

A checkpoint that records no training corpus cannot be verified against the
data it was fitted to, and the runtime loader refuses to bind one. This
utility supplies the missing provenance fields without altering the model: it
is a metadata operation, not retraining.

Preserving the model exactly is what makes the operation safe, so the utility
validates the static architecture, hashes the source CSV, carries over every
state tensor unchanged, leaves the scaler file untouched, and evaluates
deterministic probe rows before and after serialization to confirm the
predictions are identical.

The output is also reproducible. PyTorch names the root of its zip container
after the temporary output file, so this utility writes through one fixed
temporary basename; together with a canonical mapping order, identical inputs
then produce an identical checkpoint hash at any destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_FORMAT = "tire_force_static_mlp"
TRAINING_SOURCE = "chrono_scm_single_tire_rig"
TRAINING_CSV_SHA256 = (
    "926938c0f44e6c3914a4e1e99f05d304ca710a8690188ada1c014aa7fad5923c"
)
TRAINING_SEED = 42
REPACK_CONTRACT = "tire_force_metadata_repack"
STATIC_FEATURES = [
    "slip_ratio", "slip_angle", "velocity", "vertical_load", "steering_rate",
    "bekker_Kphi", "bekker_Kc", "bekker_n", "mohr_cohesion",
    "mohr_friction", "janosi_shear",
]
PROVENANCE = {
    "checkpoint_format": CHECKPOINT_FORMAT,
    "training_source": TRAINING_SOURCE,
    "training_csv_sha256": TRAINING_CSV_SHA256,
    "seed": TRAINING_SEED,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and exact contiguous bytes."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"model_state_dict entry {name!r} is not a tensor")
        values = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(values.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(values.shape)).encode("ascii") + b"\0")
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _validate_static_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    required = {
        "model_state_dict", "architecture_type", "hidden_sizes", "temporal_K",
        "rate_augmented", "feature_cols", "input_size", "output_size",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError("source checkpoint is missing: " + ", ".join(missing))
    if checkpoint["architecture_type"] != "mlp":
        raise ValueError("the source must be a static MLP checkpoint")
    if int(checkpoint["temporal_K"]) != 1 or checkpoint["rate_augmented"] is not False:
        raise ValueError("checkpoint is not an instantaneous static MLP")
    if list(checkpoint["feature_cols"]) != STATIC_FEATURES:
        raise ValueError("checkpoint static feature schema/order is invalid")
    if int(checkpoint["input_size"]) != len(STATIC_FEATURES):
        raise ValueError("checkpoint input width is not 11")
    if int(checkpoint["output_size"]) != 2:
        raise ValueError("checkpoint output width is not two forces")
    if "torch_seed" in checkpoint and int(checkpoint["torch_seed"]) != TRAINING_SEED:
        raise ValueError("source torch_seed does not match the documented seed")
    for key, expected in PROVENANCE.items():
        if key in checkpoint and checkpoint[key] != expected:
            raise ValueError(
                f"checkpoint already contains conflicting {key}: "
                f"{checkpoint[key]!r}"
            )
    state_dict_sha256(checkpoint["model_state_dict"])


def _load_scalers(path: Path) -> tuple[dict[str, Any], str]:
    before = sha256_file(path)
    with path.open("rb") as stream:
        scalers = pickle.load(stream)
    if not isinstance(scalers, Mapping) or set(scalers) != {"X", "y"}:
        raise ValueError("static scaler artifact must contain exactly X and y")
    x_mean = np.asarray(scalers["X"].mean_, dtype=float)
    x_scale = np.asarray(scalers["X"].scale_, dtype=float)
    y_mean = np.asarray(scalers["y"].mean_, dtype=float)
    y_scale = np.asarray(scalers["y"].scale_, dtype=float)
    if (
        x_mean.shape != (len(STATIC_FEATURES),)
        or x_scale.shape != x_mean.shape
        or y_mean.shape != (2,)
        or y_scale.shape != y_mean.shape
        or not np.isfinite(np.concatenate((x_mean, x_scale, y_mean, y_scale))).all()
        or np.any(x_scale <= 0.0)
        or np.any(y_scale <= 0.0)
    ):
        raise ValueError("static scaler dimensions or values are invalid")
    return dict(scalers), before


def _probe_rows(scalers: Mapping[str, Any]) -> np.ndarray:
    """Deterministic full-width rows centered on the fitted scaler domain."""

    mean = np.asarray(scalers["X"].mean_, dtype=float)
    scale = np.asarray(scalers["X"].scale_, dtype=float)
    standardized = np.asarray([
        [0.0] * 11,
        [-0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, -0.60, 0.40, -0.20, 0.10],
        [0.80, 0.55, 0.30, 0.05, -0.20, -0.45, -0.70, 0.65, -0.35, 0.15, -0.05],
        [-0.10, 0.20, -0.30, 0.40, -0.50, 0.60, -0.70, 0.80, -0.40, 0.30, -0.20],
    ], dtype=float)
    return mean[None, :] + standardized * scale[None, :]


def _predict(
    checkpoint: Mapping[str, Any], scalers: Mapping[str, Any], rows: np.ndarray
) -> np.ndarray:
    hidden = (
        np.asarray(rows, dtype=float) - np.asarray(scalers["X"].mean_, dtype=float)
    ) / np.asarray(scalers["X"].scale_, dtype=float)
    state = checkpoint["model_state_dict"]
    indices = sorted(
        int(name.split(".")[1])
        for name in state
        if name.startswith("layers.") and name.endswith(".weight")
    )
    if not indices:
        raise ValueError("checkpoint has no layers.<i>.weight tensors")
    for index in indices:
        weight = state[f"layers.{index}.weight"].detach().cpu().numpy()
        bias = state[f"layers.{index}.bias"].detach().cpu().numpy()
        hidden = hidden @ weight.T + bias
        if index != indices[-1]:
            hidden = np.tanh(hidden)
    return (
        hidden * np.asarray(scalers["y"].scale_, dtype=float)
        + np.asarray(scalers["y"].mean_, dtype=float)
    )


def _assert_tensors_identical(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> None:
    if set(before) != set(after):
        raise RuntimeError("repack changed state-dict keys")
    for name in before:
        lhs, rhs = before[name], after[name]
        if lhs.dtype != rhs.dtype or tuple(lhs.shape) != tuple(rhs.shape):
            raise RuntimeError(f"repack changed tensor schema for {name}")
        if not torch.equal(lhs.detach().cpu(), rhs.detach().cpu()):
            raise RuntimeError(f"repack changed tensor values for {name}")


def _validated_existing_manifest(
    manifest_path: Path,
    *,
    checkpoint_sha: str,
    scaler_sha: str,
    state_sha: str,
    require_training_csv_verification: bool,
) -> dict[str, Any]:
    """Return an intact prior proof for an idempotent in-place invocation."""

    if not manifest_path.is_file():
        raise ValueError(
            "checkpoint is already provenance-repacked but its identity "
            "manifest is missing"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "checkpoint is already provenance-repacked but its identity "
            "manifest is unreadable"
        ) from error
    expected = {
        "repack_contract": REPACK_CONTRACT,
        "repacked_checkpoint_sha256": checkpoint_sha,
        "scalers_sha256": scaler_sha,
        "state_dict_sha256": state_sha,
        "metadata_injected": PROVENANCE,
        "tensor_values_identical": True,
        "scaler_file_unchanged": True,
    }
    invalid = [key for key, value in expected.items() if manifest.get(key) != value]
    if invalid or float(
        manifest.get("max_abs_prediction_difference", float("inf"))
    ) != 0.0:
        details = ", ".join(invalid) if invalid else "prediction identity"
        raise ValueError(
            "checkpoint is already provenance-repacked but its identity "
            f"manifest is invalid: {details}"
        )
    if require_training_csv_verification and manifest.get(
        "training_csv_verified"
    ) is not True:
        raise ValueError(
            "existing repack manifest does not prove the requested training CSV"
        )
    return manifest


def repack_checkpoint(
    source_checkpoint: Path,
    scaler_path: Path,
    output_checkpoint: Path,
    manifest_path: Path,
    *,
    training_csv: Path | None = None,
) -> dict[str, Any]:
    """Repack and return a deterministic prediction/tensor identity manifest."""

    source_checkpoint = source_checkpoint.resolve()
    scaler_path = scaler_path.resolve()
    output_checkpoint = output_checkpoint.resolve()
    manifest_path = manifest_path.resolve()
    source_sha = sha256_file(source_checkpoint)
    if training_csv is not None:
        observed = sha256_file(training_csv.resolve())
        if observed != TRAINING_CSV_SHA256:
            raise ValueError(
                "training CSV hash mismatch: "
                f"expected {TRAINING_CSV_SHA256}, got {observed}"
            )
    checkpoint = torch.load(
        source_checkpoint, map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    _validate_static_checkpoint(checkpoint)
    scalers, scaler_sha = _load_scalers(scaler_path)
    state_sha = state_dict_sha256(checkpoint["model_state_dict"])
    already_repacked = all(
        checkpoint.get(key) == expected for key, expected in PROVENANCE.items()
    )
    if already_repacked:
        if source_checkpoint != output_checkpoint:
            raise ValueError(
                "source checkpoint already carries provenance; supply the "
                "unrepacked checkpoint as the source instead"
            )
        return _validated_existing_manifest(
            manifest_path,
            checkpoint_sha=source_sha,
            scaler_sha=scaler_sha,
            state_sha=state_sha,
            require_training_csv_verification=training_csv is not None,
        )
    rows = _probe_rows(scalers)
    predictions_before = _predict(checkpoint, scalers, rows)

    repacked = dict(checkpoint)
    repacked.update(PROVENANCE)
    # Canonical top-level ordering plus a fixed temporary basename makes the
    # PyTorch zip bytes independent of the requested destination filename.
    repacked = {key: repacked[key] for key in sorted(repacked)}
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_checkpoint.parent / "tire_force_static_mlp_repack.pt"
    if temporary.exists():
        temporary.unlink()
    try:
        torch.save(repacked, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=True)
        _validate_static_checkpoint(reloaded)
        for key, expected in PROVENANCE.items():
            if reloaded.get(key) != expected:
                raise RuntimeError(f"serialized checkpoint lost {key}")
        _assert_tensors_identical(
            checkpoint["model_state_dict"], reloaded["model_state_dict"]
        )
        predictions_after = _predict(reloaded, scalers, rows)
        if not np.array_equal(predictions_before, predictions_after):
            raise RuntimeError("metadata repack changed deterministic probe predictions")
        if sha256_file(scaler_path) != scaler_sha:
            raise RuntimeError("metadata repack changed the scaler artifact")
        repacked_sha = sha256_file(temporary)
        os.replace(temporary, output_checkpoint)
    finally:
        if temporary.exists():
            temporary.unlink()

    prediction_sha = hashlib.sha256(
        np.asarray(predictions_after, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    manifest = {
        "repack_contract": REPACK_CONTRACT,
        "source_checkpoint_sha256": source_sha,
        "repacked_checkpoint_sha256": repacked_sha,
        "scalers_sha256": scaler_sha,
        "state_dict_sha256": state_dict_sha256(reloaded["model_state_dict"]),
        "prediction_probe_sha256": prediction_sha,
        "prediction_probe_rows": int(len(rows)),
        "max_abs_prediction_difference": float(
            np.max(np.abs(predictions_after - predictions_before))
        ),
        "tensor_values_identical": True,
        "scaler_file_unchanged": True,
        "training_csv_verified": training_csv is not None,
        "metadata_injected": PROVENANCE,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir", type=Path,
        help="Directory containing best_terrain_nn.pt and scalers.pkl.",
    )
    parser.add_argument(
        "--source-checkpoint", type=Path,
        help="Input checkpoint to repack (default: MODEL_DIR/best_terrain_nn.pt).",
    )
    parser.add_argument(
        "--output-checkpoint", type=Path,
        help="Repacked output (default: replace MODEL_DIR/best_terrain_nn.pt).",
    )
    parser.add_argument(
        "--manifest", type=Path,
        help="Identity proof (default: MODEL_DIR/repack_manifest.json).",
    )
    parser.add_argument(
        "--training-csv", type=Path,
        help="Optional source CSV to independently verify against the embedded hash.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    source = (
        args.source_checkpoint.expanduser().resolve()
        if args.source_checkpoint else model_dir / "best_terrain_nn.pt"
    )
    output = (
        args.output_checkpoint.expanduser().resolve()
        if args.output_checkpoint else model_dir / "best_terrain_nn.pt"
    )
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest else model_dir / "repack_manifest.json"
    )
    manifest = repack_checkpoint(
        source,
        model_dir / "scalers.pkl",
        output,
        manifest_path,
        training_csv=args.training_csv,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
