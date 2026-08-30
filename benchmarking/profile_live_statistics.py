#!/usr/bin/env python3
"""Deterministic paired statistics for the online terrain-conditioning study.

This module produces ``profile_live_statistics.json``, the artifact behind the
manuscript's comparison of terrain-parameter sources: joint online estimation
against the matched-terrain baseline, the scalar estimator, and the fixed
low-grip fallback, on paired cells of the conditioning matrix.

Every quantity is recomputed from the publisher-selected result directories
rather than carried forward from a previous run, and each input is recorded
with its SHA-256 digest, so the published artifact can be reproduced exactly
and any drift in its inputs is detectable. Seeding and resample counts are
fixed here, which makes the bootstrap intervals a deterministic function of
the inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


SCHEMA_VERSION = 2
STATISTICS_CONTRACT = "profile_live_paired_statistics"
BOOTSTRAP_SEED = 20260722
BOOTSTRAP_RESAMPLES = {
    "conditioning": 100_000,
}


# The acquisition design the conditioning study publishes: four conditioning
# arms of the same neural tire model, isolated per ROS domain.
CONDITIONING_DESIGN = "conditioning_joint_parent_fallback_ros_isolated"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise RuntimeError(f"statistics input is outside repository root: {path}") from exc


def _source_record(directory: Path, filenames: Iterable[str], root: Path) -> dict:
    directory = Path(directory)
    files = []
    for filename in filenames:
        path = directory / filename
        if not path.is_file():
            raise RuntimeError(f"statistics source is missing: {path}")
        files.append({"path": _relative(path, root), "sha256": _digest(path)})
    return {
        "directory": _relative(directory, root),
        "files": files,
    }


def source_input_hashes(payload: dict) -> dict[str, str]:
    """Flatten the source records for the canonical publication manifest."""

    hashes: dict[str, str] = {}
    studies = payload.get("studies", {})
    for study in studies.values():
        for item in study.get("source", {}).get("files", []):
            path = str(item.get("path", ""))
            sha256 = str(item.get("sha256", ""))
            if not path or len(sha256) != 64 or path in hashes:
                raise RuntimeError("invalid or duplicate statistics source record")
            hashes[path] = sha256
    return dict(sorted(hashes.items()))


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"cannot read statistics source {path}: {exc}") from exc


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read statistics source {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"statistics source must contain an object: {path}")
    return value


def _read_key_value_manifest(path: Path) -> dict[str, str]:
    frame = _read_csv(path)
    if list(frame.columns) != ["key", "value"] or frame["key"].duplicated().any():
        raise RuntimeError(f"invalid key/value manifest: {path}")
    return {
        str(key): str(value)
        for key, value in frame[["key", "value"]].itertuples(index=False, name=None)
    }


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], study: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{study} statistics columns are missing: {missing}")


def _numeric(frame: pd.DataFrame, columns: Iterable[str], study: str) -> None:
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{study} contains non-finite numeric endpoints")


def _strict_truth(values: pd.Series, label: str) -> pd.Series:
    def parse(value: object) -> bool | None:
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "1.0"}:
            return True
        if normalized in {"false", "0", "0.0"}:
            return False
        return None

    parsed = values.map(parse)
    if parsed.isna().any():
        raise RuntimeError(f"{label} contains a non-Boolean value")
    return parsed.astype(bool)


def _summary(values: pd.Series, *, p95: bool = False) -> dict:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric) and not np.isfinite(numeric).all():
        raise RuntimeError("summary contains a non-finite value")
    result = {
        "n": int(len(numeric)),
        "mean": float(np.mean(numeric)) if len(numeric) else None,
        "median": float(np.median(numeric)) if len(numeric) else None,
        "minimum": float(np.min(numeric)) if len(numeric) else None,
        "maximum": float(np.max(numeric)) if len(numeric) else None,
    }
    if p95:
        result["p95"] = (
            float(np.percentile(numeric, 95.0, method="linear"))
            if len(numeric) else None
        )
    return result


def _readiness_summary(frame: pd.DataFrame, prefix: str = "") -> dict:
    names = {
        "applicable": f"{prefix}profile_estimator_diagnostics_applicable",
        "complete": f"{prefix}profile_estimator_diagnostics_complete",
        "ready": f"{prefix}profile_estimator_publication_ready",
        "applied": f"{prefix}profile_estimator_publication_applied",
        "abstained": f"{prefix}profile_estimator_abstained",
        "update_rows": f"{prefix}profile_estimator_update_rows",
        "ready_time": f"{prefix}profile_estimator_time_to_first_ready_s",
        "update_time": f"{prefix}profile_estimator_time_to_first_update_s",
        "windows": f"{prefix}profile_estimator_max_concurrent_windows",
        "accepted": f"{prefix}profile_estimator_lifetime_accepted_windows",
        "rejected": f"{prefix}profile_estimator_lifetime_rejected_windows",
    }
    _require_columns(frame, names.values(), "profile-readiness")
    flags = {
        key: _strict_truth(frame[column], column)
        for key, column in names.items()
        if key in {"applicable", "complete", "ready", "applied", "abstained"}
    }
    applicable = flags["applicable"]
    selected = frame.loc[applicable]
    if (
        not flags["complete"].loc[applicable].all()
        or (flags["applied"] & ~flags["ready"]).any()
        or (flags["abstained"] & flags["applied"]).any()
        or (flags["ready"] | flags["applied"] | flags["abstained"])
        .loc[~applicable]
        .any()
    ):
        raise RuntimeError("profile-readiness flags violate the live-estimator contract")
    numeric_columns = [
        names["update_rows"], names["windows"], names["accepted"], names["rejected"]
    ]
    if len(selected):
        _numeric(selected, numeric_columns, "profile-readiness")
    ready_times = frame.loc[flags["ready"], names["ready_time"]]
    update_times = frame.loc[flags["applied"], names["update_time"]]
    return {
        "rows": int(len(frame)),
        "applicable_rows": int(applicable.sum()),
        "diagnostics_complete_rows": int(flags["complete"].loc[applicable].sum()),
        "publication_ready_rows": int(flags["ready"].sum()),
        "publication_applied_rows": int(flags["applied"].sum()),
        "abstained_rows": int(flags["abstained"].sum()),
        "update_rows": _summary(selected[names["update_rows"]]),
        "time_to_first_ready_s": _summary(ready_times, p95=True),
        "time_to_first_update_s": _summary(update_times, p95=True),
        "max_concurrent_windows": _summary(selected[names["windows"]]),
        "lifetime_accepted_windows": _summary(selected[names["accepted"]]),
        "lifetime_rejected_windows": _summary(selected[names["rejected"]]),
    }


def _readiness_by(frame: pd.DataFrame, group: str, prefix: str = "") -> dict:
    return {
        "overall": _readiness_summary(frame, prefix),
        f"by_{group}": {
            str(value): _readiness_summary(part, prefix)
            for value, part in frame.groupby(group, sort=True)
        },
    }


def _paired_wide(
    frame: pd.DataFrame,
    *,
    keys: list[str],
    arm_column: str,
    metric: str,
    arms: set[str],
    study: str,
) -> pd.DataFrame:
    if frame[keys + [arm_column]].duplicated().any():
        raise RuntimeError(f"{study} contains duplicate paired cells")
    wide = frame.pivot(index=keys, columns=arm_column, values=metric).sort_index()
    if set(wide.columns.astype(str)) != arms or wide.isna().any().any():
        raise RuntimeError(f"{study} paired matrix is incomplete")
    return wide


def _bootstrap_ci(differences: np.ndarray, resamples: int) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    distribution = np.empty(int(resamples), dtype=float)
    count = len(differences)
    offset = 0
    while offset < resamples:
        block = min(8192, resamples - offset)
        indices = rng.integers(0, count, size=(block, count))
        distribution[offset:offset + block] = differences[indices].mean(axis=1)
        offset += block
    low, high = np.percentile(
        distribution, [2.5, 97.5], method="linear"
    )
    return float(low), float(high)


def _paired_comparison(
    wide: pd.DataFrame,
    *,
    reference: str,
    treatment: str,
    resamples: int,
) -> dict:
    reference_values = pd.to_numeric(wide[reference], errors="coerce").to_numpy(float)
    treatment_values = pd.to_numeric(wide[treatment], errors="coerce").to_numpy(float)
    if (
        len(reference_values) == 0
        or not np.isfinite(reference_values).all()
        or not np.isfinite(treatment_values).all()
    ):
        raise RuntimeError("paired comparison has missing or non-finite endpoints")
    reduction = reference_values - treatment_values
    low, high = _bootstrap_ci(reduction, resamples)
    method = "exact" if len(reduction) <= 50 and not (reduction == 0.0).any() else "approx"
    test = wilcoxon(
        reduction,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method=method,
    )
    reference_mean = float(reference_values.mean())
    return {
        "reference": reference,
        "treatment": treatment,
        "paired_units": int(len(reduction)),
        "reduction_definition": "reference endpoint minus treatment endpoint; positive favors treatment",
        "reference_mean_m": reference_mean,
        "treatment_mean_m": float(treatment_values.mean()),
        "mean_reduction_m": float(reduction.mean()),
        "mean_reduction_percent_of_reference": float(
            100.0 * reduction.mean() / reference_mean
        ),
        "mean_reduction_ci95_m": {"lower": low, "upper": high},
        "median_reduction_m": float(np.median(reduction)),
        "treatment_wins": int((reduction > 0.0).sum()),
        "reference_wins": int((reduction < 0.0).sum()),
        "ties": int((reduction == 0.0).sum()),
        "wilcoxon": {
            "test": "paired signed-rank",
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "method": method,
            "continuity_correction": False,
            "statistic": float(test.statistic),
            "p_value": float(test.pvalue),
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": int(resamples),
            "unit": "complete paired cell",
            "statistic": "mean paired reduction",
            "interval": "2.5th and 97.5th percentiles",
            "percentile_method": "linear",
            "rng": "numpy.random.default_rng (PCG64), reinitialized per contrast",
        },
    }


def _conditioning_statistics(directory: Path, root: Path) -> dict:
    source = _source_record(
        directory,
        ("results.csv", "manifest.csv", "reference_profile_contract.json"),
        root,
    )
    frame = _read_csv(directory / "results.csv")
    manifest = _read_key_value_manifest(directory / "manifest.csv")
    contract = _read_json(directory / "reference_profile_contract.json")
    if (
        manifest.get("design_version", "").strip("'\"")
        != CONDITIONING_DESIGN
    ):
        raise RuntimeError(
            "conditioning acquisition manifest is not the four-arm "
            "joint/parent/fallback design"
        )
    return _four_arm_conditioning_statistics(frame, manifest, contract, source)


def _four_arm_conditioning_statistics(
    frame: pd.DataFrame,
    manifest: dict[str, str],
    contract: dict,
    source: dict,
) -> dict:
    """Conditioning statistics for the matched, GRIT, parent and fallback arms."""

    design = CONDITIONING_DESIGN
    variants = {
        "nn_static", "nn_estimator", "nn_parent_estimator",
        "nn_fixed_fallback",
    }
    required = {
        "status", "variant", "terrain", "path", "speed_mps", "bumpiness",
        "seed", "sim_s", "rms_cte_m", "mean_speed_mps", "mean_solve_ms",
    }
    _require_columns(frame, required, "conditioning")
    if (
        len(frame) != 540
        or set(frame["status"].astype(str)) != {"ok"}
        or set(frame["variant"].astype(str)) != variants
        or frame.groupby("variant").size().to_dict()
        != {variant: 135 for variant in sorted(variants)}
        or contract.get("design") != design
    ):
        raise RuntimeError("joint conditioning source violates its frozen design")
    _numeric(
        frame,
        (
            "speed_mps", "bumpiness", "seed", "sim_s", "rms_cte_m",
            "mean_speed_mps", "mean_solve_ms",
        ),
        "conditioning",
    )
    if (
        float(manifest.get("metric_start", "nan")) != 8.0
        or float(manifest.get("time", "nan")) != 20.0
        or not np.isclose(pd.to_numeric(frame["sim_s"]), 20.0, atol=1e-9).all()
    ):
        raise RuntimeError("joint conditioning endpoint is not fixed t=8..20 s")
    keys = ["terrain", "path", "speed_mps", "bumpiness", "seed"]
    wide = _paired_wide(
        frame,
        keys=keys,
        arm_column="variant",
        metric="rms_cte_m",
        arms=variants,
        study="conditioning",
    )
    roles = {
        "nn_static": "matched-terrain oracle-information baseline",
        "nn_estimator": "promoted independent-n-phi online estimator",
        "nn_parent_estimator": "historical matched scalar-parent estimator",
        "nn_fixed_fallback": "fixed promoted low-grip fallback endpoint",
    }
    variant_means = {
        str(variant): {
            "role": roles[str(variant)],
            "runs": int(len(part)),
            "mean_rms_cte_m": float(pd.to_numeric(part["rms_cte_m"]).mean()),
            "mean_achieved_speed_mps": float(
                pd.to_numeric(part["mean_speed_mps"]).mean()
            ),
            "mean_solver_time_ms": float(
                pd.to_numeric(part["mean_solve_ms"]).mean()
            ),
        }
        for variant, part in frame.groupby("variant", sort=True)
    }
    comparisons = {
        "joint_vs_fixed_fallback": _paired_comparison(
            wide,
            reference="nn_fixed_fallback",
            treatment="nn_estimator",
            resamples=BOOTSTRAP_RESAMPLES["conditioning"],
        ),
        "joint_vs_historical_scalar_parent": _paired_comparison(
            wide,
            reference="nn_parent_estimator",
            treatment="nn_estimator",
            resamples=BOOTSTRAP_RESAMPLES["conditioning"],
        ),
        "joint_vs_matched_terrain_oracle": _paired_comparison(
            wide,
            reference="nn_static",
            treatment="nn_estimator",
            resamples=BOOTSTRAP_RESAMPLES["conditioning"],
        ),
    }
    return {
        "source": source,
        "design": design,
        "rows": 540,
        "paired_cells": 135,
        "pairing_keys": keys,
        "endpoint": {
            "name": "per-run RMS crosstrack error",
            "column": "rms_cte_m",
            "unit": "m",
            "definition": "RMS crosstrack error over acquired controller-diagnostic samples",
            "acquisition_start": "simulation time t >= 8.0 s (inclusive)",
            "acquisition_end": "available samples through nominal 20.0 s endpoint",
            "start_time_s": 8.0,
            "nominal_end_time_s": 20.0,
            "readiness_alignment": (
                "fixed across all arms; not shifted to estimator readiness"
            ),
        },
        "variant_means": variant_means,
        "comparisons": comparisons,
        "readiness_and_updates": _readiness_by(
            frame, "variant", prefix="extra_"
        ),
    }


def build_profile_live_statistics(
    conditioning_dir: Path,
    *,
    root: Path,
) -> dict:
    """Build the canonical artifact from already selected result directories."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "statistics_contract": STATISTICS_CONTRACT,
        "generated_from_selected_results_only": True,
        "definitions": {
            "paired_reduction": "reference endpoint minus treatment endpoint, paired on the study's declared keys",
            "wins": "strictly positive paired reductions; exact zeros are ties",
            "confidence_interval": "nonparametric paired percentile interval of the mean reduction",
            "wilcoxon": "two-sided paired signed-rank test on unrounded paired reductions",
        },
        "bootstrap_contract": {
            "seed": BOOTSTRAP_SEED,
            "resamples_by_study": dict(BOOTSTRAP_RESAMPLES),
            "confidence_level": 0.95,
            "percentile_method": "linear",
            "paired_resampling": True,
        },
        "studies": {
            "conditioning": _conditioning_statistics(Path(conditioning_dir), Path(root)),
        },
    }
    # Fail if a future calculation accidentally leaks NaN/Infinity into JSON.
    json.dumps(payload, allow_nan=False)
    return payload


def validate_profile_live_statistics(
    payload: dict,
    conditioning_dir: Path,
    *,
    root: Path,
) -> None:
    """Fail unless an artifact exactly matches deterministic recomputation."""

    expected = build_profile_live_statistics(
        conditioning_dir,
        root=root,
    )
    if payload != expected:
        raise RuntimeError("profile-live statistics do not match selected source results")


def canonical_json(payload: dict) -> str:
    """Serialize with stable key ordering and no non-standard numeric values."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
