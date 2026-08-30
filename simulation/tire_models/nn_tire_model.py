#!/usr/bin/env python3
"""Loader for the neural tire-force surrogate embedded in the NMPC.

The loader admits static and rate-augmented multilayer-perceptron checkpoints
supervised by the controlled Chrono SCM single-tire rig, and rejects
vehicle-level or axle-level feature schemas and temporal observers before any
CasADi expression is constructed.  Admissibility is decided from metadata
recorded inside the checkpoint, so a directory name can neither grant nor
revoke it.
"""

from __future__ import annotations

import os as _os
import pickle
import re
import sys as _sys
from collections.abc import Mapping
from pathlib import Path

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import flatpath  # noqa: E402,F401

import casadi as ca
import numpy as np
import torch


STATIC_FEATURES = [
    "slip_ratio", "slip_angle", "velocity", "vertical_load", "steering_rate",
    "bekker_Kphi", "bekker_Kc", "bekker_n", "mohr_cohesion",
    "mohr_friction", "janosi_shear",
]
RATE_FEATURES = STATIC_FEATURES[:5] + [
    "d_slip_ratio", "d_slip_angle", "d_velocity",
] + STATIC_FEATURES[5:]

STATIC_CHECKPOINT_FORMAT = "tire_force_static_mlp"
RATE_CHECKPOINT_FORMAT = "tire_force_rate_mlp"
RIG_TRAINING_SOURCE = "chrono_scm_single_tire_rig"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _validate_checkpoint_provenance(checkpoint: Mapping, model_path: Path) -> None:
    """Reject force maps without embedded, machine-checkable rig provenance."""

    required = {
        "checkpoint_format", "model_state_dict", "architecture_type",
        "temporal_K", "rate_augmented", "feature_cols", "input_size",
        "output_size", "training_source", "training_csv_sha256", "seed",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(
            f"{model_path} is missing required rig checkpoint metadata: "
            + ", ".join(missing)
        )
    if not isinstance(checkpoint["rate_augmented"], bool):
        raise ValueError("rate_augmented must be a boolean")
    expected_format = (
        RATE_CHECKPOINT_FORMAT
        if checkpoint["rate_augmented"] else STATIC_CHECKPOINT_FORMAT
    )
    if checkpoint["checkpoint_format"] != expected_format:
        raise ValueError(
            "force checkpoint format does not match its static/rate schema: "
            f"expected {expected_format!r}, got "
            f"{checkpoint['checkpoint_format']!r}"
        )
    if checkpoint["training_source"] != RIG_TRAINING_SOURCE:
        raise ValueError(
            "force checkpoint is not provenance-bound to the controlled "
            "Chrono SCM single-tire rig"
        )
    if (
        isinstance(checkpoint["output_size"], bool)
        or int(checkpoint["output_size"]) != 2
    ):
        raise ValueError("force checkpoint output_size must be two tire forces")
    training_digest = checkpoint["training_csv_sha256"]
    if (
        not isinstance(training_digest, str)
        or _SHA256_PATTERN.fullmatch(training_digest) is None
    ):
        raise ValueError(
            "force checkpoint training_csv_sha256 must be 64 lowercase hex digits"
        )
    seed = checkpoint["seed"]
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("force checkpoint seed must be a nonnegative integer")


def _layer_indices(weights: dict[str, np.ndarray]) -> list[int]:
    indices = sorted({
        int(key.split(".")[1])
        for key in weights
        if key.startswith("layers.") and key.endswith(".weight")
    })
    if not indices:
        raise ValueError("Rig checkpoint does not contain layers.<i>.weight tensors")
    return indices


def _forward(weights: dict[str, np.ndarray], values, indices: list[int], columns: int = 1):
    hidden = values
    for index in indices:
        matrix = ca.DM(weights[f"layers.{index}.weight"])
        bias = ca.DM(weights[f"layers.{index}.bias"]).reshape((-1, 1))
        hidden = ca.mtimes(matrix, hidden)
        hidden += ca.repmat(bias, 1, columns) if columns > 1 else bias
        if index != indices[-1]:
            hidden = ca.tanh(hidden)
    return hidden


class NNTireModel:
    """Tire force map, exposed both symbolically and numerically.

    The same checkpoint serves the NMPC and the safety filter, which embed the
    CasADi expressions, and the terrain estimator, which evaluates the numeric
    forward pass over many soil hypotheses.
    """

    temporal_K = 1
    # Width of the batched evaluation the NMPC issues once per right-hand-side
    # call.  The lateral-load-transfer path reads rows 0..5 (four cornering
    # samples plus two traction samples) and the mean-load path reads rows
    # 0..3, so six columns cover every consumer.  Columns are independent --
    # _forward is a matrix product, a per-column bias, and an elementwise tanh,
    # with no cross-column term -- hence the width affects cost alone and never
    # the values of the rows that are read.
    _BATCH = 6

    def __init__(self, model_dir: str | Path, terrain_params: dict):
        self.model_dir = Path(model_dir).expanduser().resolve()
        # Provenance is enforced from the checkpoint's embedded metadata
        # (``training_source``), not from the directory name: the recorded
        # field cannot be satisfied by renaming a folder.
        model_path = self.model_dir / "best_terrain_nn.pt"
        scaler_path = self.model_dir / "scalers.pkl"
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"{model_path} lacks the rig checkpoint metadata contract")
        _validate_checkpoint_provenance(checkpoint, model_path)
        if checkpoint.get("offline_only", False):
            raise ValueError(f"{self.model_dir.name} is marked offline_only")
        if checkpoint["architecture_type"] != "mlp":
            raise ValueError("Force checkpoints must declare a multilayer-perceptron architecture")
        if int(checkpoint["temporal_K"]) != 1:
            raise ValueError("Temporal observer checkpoints are outside the runtime force contract")

        self.rate_augmented = checkpoint["rate_augmented"]
        expected_features = RATE_FEATURES if self.rate_augmented else STATIC_FEATURES
        feature_cols = list(checkpoint["feature_cols"] or [])
        input_size = int(checkpoint["input_size"])
        if feature_cols != expected_features or input_size != len(expected_features):
            raise ValueError(
                f"{self.model_dir.name} does not match the single-tire-rig feature "
                f"schema: input_size={input_size}, features={feature_cols}"
            )

        state_dict = checkpoint["model_state_dict"]
        if any(key.startswith("layer") and not key.startswith("layers") for key in state_dict):
            remapped = {}
            layer = 1
            while f"layer{layer}.weight" in state_dict:
                remapped[f"layer{layer}.weight"] = f"layers.{layer - 1}.weight"
                remapped[f"layer{layer}.bias"] = f"layers.{layer - 1}.bias"
                layer += 1
            state_dict = {remapped.get(key, key): value for key, value in state_dict.items()}
        self._weights = {key: value.detach().numpy() for key, value in state_dict.items()}
        self.n_params = int(sum(value.size for value in self._weights.values()))
        self._indices = _layer_indices(self._weights)

        with scaler_path.open("rb") as stream:
            scalers = pickle.load(stream)
        self._X_mean = np.asarray(scalers["X"].mean_, dtype=float)
        self._X_scale = np.asarray(scalers["X"].scale_, dtype=float)
        self._y_mean = np.asarray(scalers["y"].mean_, dtype=float)
        self._y_scale = np.asarray(scalers["y"].scale_, dtype=float)
        if len(self._X_mean) != len(expected_features):
            raise ValueError("Checkpoint scaler width does not match the declared feature schema")

        self.input_dim = len(expected_features)
        self.model_type = "rate_mlp" if self.rate_augmented else "static_mlp"
        self.model_format = str(checkpoint["checkpoint_format"])
        self.n_nominal = float(terrain_params.get("n", 1.1))
        self._terrain_nominals = {
            "Kphi": float(terrain_params["Kphi"]),
            "Kc": float(terrain_params["Kc"]),
            "c": float(terrain_params["c"]),
            "phi": float(terrain_params["phi"]),
            "k": float(terrain_params["k"]),
        }
        self._build()

    def phi_feature_value(self, phi: float) -> float:
        """Convert friction angle to the convention recorded by the rig scaler.

        The decision is by magnitude: any |phi| <= 2*pi is taken as radians.
        That is unambiguous for the deployed paths (both deployed checkpoints
        record radians, the estimator passes radians, and control enforces
        phi >= 10 deg before any degrees-valued call), but a *degrees*-valued
        phi below 6.28 deg -- possible, since the estimator grid bottoms at
        6.0 deg -- would be silently misread as radians. Callers passing
        degrees must therefore stay above the 2*pi ambiguity band; audit
        finding 9 (2026-08-26) documents this as a latent trap rather than a
        deployed defect.
        """
        value = float(phi)
        expects_degrees = abs(float(self._X_mean[-2])) > 2.0 * np.pi
        if expects_degrees:
            return float(np.degrees(value)) if abs(value) <= 2.0 * np.pi else value
        return float(np.radians(value)) if abs(value) > 2.0 * np.pi else value

    def _build(self) -> None:
        alpha = ca.SX.sym("alpha")
        vertical_load = ca.SX.sym("Fz")
        velocity = ca.SX.sym("u")
        slip_ratio = ca.SX.sym("kappa")
        sinkage_exponent = ca.SX.sym("n_terrain")
        steering_rate = ca.SX.sym("sr")
        rates = ca.SX.sym("rates", 3)
        Kphi = ca.SX.sym("Kphi")
        Kc = ca.SX.sym("Kc")
        cohesion = ca.SX.sym("c")
        friction = ca.SX.sym("phi")
        janosi = ca.SX.sym("k")

        components = [slip_ratio, alpha, velocity, vertical_load, steering_rate]
        if self.rate_augmented:
            components.append(rates)
        components += [Kphi, Kc, sinkage_exponent, cohesion, friction, janosi]
        raw = ca.vertcat(*components)
        scaled = (raw - self._X_mean.reshape(-1, 1)) / self._X_scale.reshape(-1, 1)
        standardized = _forward(self._weights, scaled, self._indices)
        output = standardized * self._y_scale.reshape(-1, 1) + self._y_mean.reshape(-1, 1)

        arguments = [
            alpha, vertical_load, velocity, slip_ratio, sinkage_exponent,
            steering_rate,
        ]
        names = ["alpha", "Fz", "u", "kappa", "n_terrain", "sr"]
        if self.rate_augmented:
            arguments.append(rates)
            names.append("rates")
        arguments += [Kphi, Kc, cohesion, friction, janosi]
        names += ["Kphi", "Kc", "c", "phi", "k"]
        function_name = f"tire_{'rate' if self.rate_augmented else 'static'}"
        self.predict_tire_force = ca.Function(
            function_name,
            arguments,
            [output[0], output[1]],
            names,
            ["Fx", "Fy"],
        )
        self._build_batch()

    def _build_batch(self) -> None:
        count = self._BATCH
        alphas = ca.SX.sym("alphas", count)
        loads = ca.SX.sym("Fzs", count)
        velocities = ca.SX.sym("us", count)
        slips = ca.SX.sym("kappas", count)
        exponents = ca.SX.sym("n_ts", count)
        steering_rates = ca.SX.sym("srs", count)
        d_slip = ca.SX.sym("dk", count)
        d_alpha = ca.SX.sym("da", count)
        d_velocity = ca.SX.sym("du", count)
        Kphi = ca.SX.sym("Kphi")
        Kc = ca.SX.sym("Kc")
        cohesion = ca.SX.sym("c")
        friction = ca.SX.sym("phi")
        janosi = ca.SX.sym("k")

        rows = [slips.T, alphas.T, velocities.T, loads.T, steering_rates.T]
        if self.rate_augmented:
            rows += [d_slip.T, d_alpha.T, d_velocity.T]
        rows += [
            ca.repmat(Kphi, 1, count),
            ca.repmat(Kc, 1, count),
            exponents.T,
            ca.repmat(cohesion, 1, count),
            ca.repmat(friction, 1, count),
            ca.repmat(janosi, 1, count),
        ]
        raw = ca.vertcat(*rows)
        mean = ca.DM(self._X_mean.reshape(-1, 1))
        scale = ca.DM(self._X_scale.reshape(-1, 1))
        hidden = (raw - ca.repmat(mean, 1, count)) / ca.repmat(scale, 1, count)
        standardized = _forward(self._weights, hidden, self._indices, columns=count)
        out_mean = ca.DM(self._y_mean.reshape(-1, 1))
        out_scale = ca.DM(self._y_scale.reshape(-1, 1))
        output = standardized * ca.repmat(out_scale, 1, count) + ca.repmat(out_mean, 1, count)

        common_args = [
            alphas, loads, velocities, slips, exponents, steering_rates,
        ]
        common_names = ["alphas", "Fzs", "us", "kappas", "n_ts", "srs"]
        if self.rate_augmented:
            self.predict_batch_rate = ca.Function(
                "tire_batch_rate",
                common_args + [d_slip, d_alpha, d_velocity, Kphi, Kc, cohesion, friction, janosi],
                [output[0, :].T, output[1, :].T],
                common_names + ["dk", "da", "du", "Kphi", "Kc", "c", "phi", "k"],
                ["Fxs", "Fys"],
            )
            self.predict_batch = None
        else:
            self.predict_batch = ca.Function(
                "tire_batch_static",
                common_args + [Kphi, Kc, cohesion, friction, janosi],
                [output[0, :].T, output[1, :].T],
                common_names + ["Kphi", "Kc", "c", "phi", "k"],
                ["Fxs", "Fys"],
            )

    def predict(
        self,
        alpha,
        Fz,
        u,
        kappa=0.0,
        n_terrain=None,
        steering_rate=0.0,
        terrain_params=None,
        hist=None,
        rates=None,
    ):
        del hist
        exponent = self.n_nominal if n_terrain is None else n_terrain
        terrain = self._terrain_nominals if terrain_params is None else terrain_params
        arguments = [alpha, Fz, u, kappa, exponent, steering_rate]
        if self.rate_augmented:
            arguments.append(np.zeros(3) if rates is None else rates)
        arguments += [
            terrain["Kphi"], terrain["Kc"], terrain["c"],
            self.phi_feature_value(terrain["phi"]), terrain["k"],
        ]
        return self.predict_tire_force(*arguments)

    def predict_numeric(self, *args, **kwargs) -> tuple[float, float]:
        longitudinal, lateral = self.predict(*args, **kwargs)
        return float(longitudinal), float(lateral)

    def predict_feature_rows(self, features: np.ndarray) -> np.ndarray:
        """Evaluate an arbitrary batch of already assembled feature rows.

        The terrain estimator evaluates one operating point against many soil
        hypotheses at once.  This method exposes the checkpoint's numeric
        forward pass directly, under the same feature ordering and scaler
        contract as the CasADi path, so a hypothesis sweep costs one array
        operation per layer instead of one CasADi call per wheel and
        hypothesis.
        """

        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(
                f"feature rows must have shape (N, {self.input_dim}), "
                f"got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("feature rows must be finite")
        hidden = (values - self._X_mean) / self._X_scale
        for index in self._indices:
            hidden = hidden @ self._weights[f"layers.{index}.weight"].T
            hidden += self._weights[f"layers.{index}.bias"]
            if index != self._indices[-1]:
                hidden = np.tanh(hidden)
        return hidden * self._y_scale + self._y_mean


def load_nn_tire_model(model_dir: str | Path, terrain_params: dict) -> NNTireModel:
    """Load and validate one single-tire-rig force checkpoint."""
    model = NNTireModel(model_dir, terrain_params)
    print(
        f"✓ Loaded {model.model_type}: {model.model_dir.name} "
        f"({model.n_params} params, source=single-tire-rig)"
    )
    return model


if __name__ == "__main__":
    import argparse

    from param_consistency import TERRAIN_PRESETS, terrain_preset_to_internal

    parser = argparse.ArgumentParser(
        description="Validate a single-tire-rig force checkpoint"
    )
    parser.add_argument("model_dir")
    parser.add_argument("--terrain", choices=sorted(TERRAIN_PRESETS), default="dirt")
    arguments = parser.parse_args()
    parameters = terrain_preset_to_internal(TERRAIN_PRESETS[arguments.terrain])
    loaded = load_nn_tire_model(arguments.model_dir, parameters)
    fx, fy = loaded.predict_numeric(0.1, 5000.0, 5.0, kappa=0.05, rates=np.zeros(3))
    print(f"Fx={fx:.3f} N, Fy={fy:.3f} N")
