"""Truth-free diagnostics for the active terrain estimator.

The joint estimator publishes an immutable accepted snapshot and exposes the
controller's complete fail-closed readiness decision. These helpers read that
decision directly from a run's diagnostics, so a benchmark can report how
often the estimator was actually available to control without consulting any
plant quantity.

Readiness is the conjunction of the confidence, freshness, boundary,
observability, and control-envelope gates. ``terrain_dynamics_active`` records
that the estimator is computing, which is a necessary but not sufficient
condition, so reading it as readiness would overstate availability by omitting
every one of those gates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from common import (
        GRIT_ESTIMATOR_BACKEND,
        GRIT_ESTIMATOR_CONTRACT,
        parent_estimator_diagnostics,
    )
except ModuleNotFoundError:  # package import in tests/tools
    from benchmarking.common import (
        GRIT_ESTIMATOR_BACKEND,
        GRIT_ESTIMATOR_CONTRACT,
        parent_estimator_diagnostics,
    )


_TRUTH_COLUMNS = {
    "actual_Fx_front",
    "actual_Fx_rear",
    "actual_Fy_front",
    "actual_Fy_rear",
}
_JOINT_COLUMNS = {
    "sim_time",
    "terrain_update_applied",
    "terrain_dynamics_windows",
    "terrain_accepted_dynamics_windows",
    "terrain_rejected_dynamics_windows",
    "terrain_profile_force_gain",
    "terrain_profile_ax_bias",
    "terrain_profile_ay_bias",
    "terrain_profile_bound_hits",
    "terrain_feature_envelope_excursions",
    "terrain_joint_snapshot_seq",
    "terrain_joint_evidence_age_s",
    "terrain_joint_publication_ready",
    "terrain_joint_fallback_reason",
    "terrain_joint_snapshot_confidence",
    "terrain_joint_n_boundary_mass",
    "terrain_joint_phi_boundary_mass",
    "terrain_joint_max_boundary_mass",
    "terrain_joint_boundary_limited",
    "terrain_joint_observability_rank",
    "terrain_joint_observability_min_singular_value",
    "terrain_joint_projection_wall_ms",
    "terrain_joint_profile_wall_ms",
    "terrain_joint_observability_wall_ms",
    "terrain_joint_posterior_wall_ms",
    "terrain_joint_publication_wall_ms",
    "terrain_joint_update_wall_ms",
}


def _base_output(*, applicable: bool) -> dict[str, Any]:
    required_windows = int(
        GRIT_ESTIMATOR_CONTRACT["min_concurrent_windows"]
    )
    return {
        "profile_estimator_diagnostics_applicable": applicable,
        "profile_estimator_diagnostics_complete": not applicable,
        "profile_estimator_diagnostics_error": "",
        "profile_estimator_required_concurrent_windows": required_windows,
        "profile_estimator_publication_ready": False,
        "profile_estimator_publication_applied": False,
        "profile_estimator_abstained": False,
        "profile_estimator_readiness_rows": 0,
        "profile_estimator_update_rows": 0,
        "profile_estimator_time_to_first_ready_s": None,
        "profile_estimator_time_to_first_update_s": None,
        "profile_estimator_max_concurrent_windows": 0,
        "profile_estimator_lifetime_accepted_windows": 0,
        "profile_estimator_lifetime_rejected_windows": 0,
        "profile_estimator_profile_force_gain_final": None,
        "profile_estimator_profile_ax_bias_final_mps2": None,
        "profile_estimator_profile_ay_bias_final_mps2": None,
        "profile_estimator_profile_bound_hits_max": 0,
        "profile_estimator_feature_envelope_excursions_max": 0,
        "profile_estimator_readiness_consistent": True,
        "profile_estimator_snapshot_rows": 0,
        "profile_estimator_unique_snapshot_count": 0,
        "profile_estimator_ready_snapshot_count": 0,
        "profile_estimator_applied_snapshot_count": 0,
        "profile_estimator_final_snapshot_seq": 0,
        "profile_estimator_fallback_rows": 0,
        "profile_estimator_max_evidence_age_s": None,
        "profile_estimator_min_snapshot_confidence": None,
        "profile_estimator_max_boundary_mass": None,
        "profile_estimator_min_observability_singular_value": None,
        "profile_estimator_update_wall_ms_median": None,
        "profile_estimator_update_wall_ms_p95": None,
        "profile_estimator_update_wall_ms_max": None,
    }


def _first_elapsed(times: pd.Series, mask: pd.Series) -> float | None:
    selected = times[mask]
    if selected.empty:
        return None
    return float(max(0.0, float(selected.min()) - float(times.min())))


def _finite_max_int(values: pd.Series) -> int:
    finite = values[np.isfinite(values)]
    return int(finite.max()) if not finite.empty else 0


def _last_finite(values: pd.Series) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.iloc[-1]) if not finite.empty else None


def _joint_diagnostics(
    diag: pd.DataFrame | None,
    *,
    applicable: bool,
) -> dict[str, Any]:
    output = _base_output(applicable=applicable)
    if not applicable:
        return output
    output["profile_estimator_abstained"] = True
    if diag is None or diag.empty:
        output["profile_estimator_diagnostics_error"] = (
            "missing_or_empty_controller_diag"
        )
        return output

    missing = sorted((_JOINT_COLUMNS | _TRUTH_COLUMNS) - set(diag.columns))
    if missing:
        output["profile_estimator_diagnostics_error"] = (
            "missing_columns:" + ",".join(missing)
        )
        return output

    numeric_columns = _JOINT_COLUMNS - {"terrain_joint_fallback_reason"}
    numeric = {
        column: pd.to_numeric(diag[column], errors="coerce")
        for column in numeric_columns
    }
    times = numeric["sim_time"]
    always_finite = {
        "sim_time",
        "terrain_update_applied",
        "terrain_dynamics_windows",
        "terrain_accepted_dynamics_windows",
        "terrain_rejected_dynamics_windows",
        "terrain_profile_force_gain",
        "terrain_profile_ax_bias",
        "terrain_profile_ay_bias",
        "terrain_profile_bound_hits",
        "terrain_feature_envelope_excursions",
        "terrain_joint_snapshot_seq",
        "terrain_joint_publication_ready",
    }
    nonfinite = sorted(
        column
        for column in always_finite
        if not np.isfinite(numeric[column].to_numpy(dtype=float)).all()
    )
    if nonfinite:
        output["profile_estimator_diagnostics_error"] = (
            "nonfinite_columns:" + ",".join(nonfinite)
        )
        return output
    if not times.is_monotonic_increasing:
        output["profile_estimator_diagnostics_error"] = "nonmonotonic_sim_time"
        return output

    binary_columns = (
        "terrain_update_applied",
        "terrain_joint_publication_ready",
    )
    invalid_binary = sorted(
        column
        for column in binary_columns
        if not numeric[column].isin((0.0, 1.0)).all()
    )
    if invalid_binary:
        output["profile_estimator_diagnostics_error"] = (
            "invalid_binary_columns:" + ",".join(invalid_binary)
        )
        return output

    counter_columns = (
        "terrain_dynamics_windows",
        "terrain_accepted_dynamics_windows",
        "terrain_rejected_dynamics_windows",
        "terrain_profile_bound_hits",
        "terrain_feature_envelope_excursions",
        "terrain_joint_snapshot_seq",
    )
    invalid_counters = sorted(
        column
        for column in counter_columns
        if (
            (numeric[column] < 0.0).any()
            or not np.allclose(
                numeric[column].to_numpy(dtype=float),
                np.rint(numeric[column].to_numpy(dtype=float)),
                rtol=0.0,
                atol=1.0e-12,
            )
        )
    )
    if invalid_counters:
        output["profile_estimator_diagnostics_error"] = (
            "invalid_counter_columns:" + ",".join(invalid_counters)
        )
        return output
    for column in (
        "terrain_accepted_dynamics_windows",
        "terrain_rejected_dynamics_windows",
        "terrain_feature_envelope_excursions",
        "terrain_joint_snapshot_seq",
    ):
        if (numeric[column].diff().dropna() < 0.0).any():
            output["profile_estimator_diagnostics_error"] = (
                "nonmonotonic_counter:" + column
            )
            return output

    gain_bounds = GRIT_ESTIMATOR_CONTRACT["force_gain_bounds"]
    bias_bound = float(
        GRIT_ESTIMATOR_CONTRACT["acceleration_bias_bound_mps2"]
    )
    if not numeric["terrain_profile_force_gain"].between(
        float(gain_bounds[0]), float(gain_bounds[1]), inclusive="both"
    ).all():
        output["profile_estimator_diagnostics_error"] = (
            "force_gain_out_of_bounds"
        )
        return output
    for column in ("terrain_profile_ax_bias", "terrain_profile_ay_bias"):
        if not numeric[column].between(
            -bias_bound, bias_bound, inclusive="both"
        ).all():
            output["profile_estimator_diagnostics_error"] = (
                "acceleration_bias_out_of_bounds:" + column
            )
            return output

    ready = numeric["terrain_joint_publication_ready"] > 0.0
    updated = numeric["terrain_update_applied"] > 0.0
    snapshots = numeric["terrain_joint_snapshot_seq"] > 0.0
    fallback = diag["terrain_joint_fallback_reason"].fillna("").astype(str)
    snapshot_fields = (
        "terrain_joint_evidence_age_s",
        "terrain_joint_snapshot_confidence",
        "terrain_joint_n_boundary_mass",
        "terrain_joint_phi_boundary_mass",
        "terrain_joint_max_boundary_mass",
        "terrain_joint_boundary_limited",
        "terrain_joint_observability_rank",
        "terrain_joint_observability_min_singular_value",
        "terrain_joint_projection_wall_ms",
        "terrain_joint_profile_wall_ms",
        "terrain_joint_observability_wall_ms",
        "terrain_joint_posterior_wall_ms",
        "terrain_joint_publication_wall_ms",
        "terrain_joint_update_wall_ms",
    )
    if snapshots.any():
        invalid_snapshot = sorted(
            column
            for column in snapshot_fields
            if not np.isfinite(
                numeric[column].loc[snapshots].to_numpy(dtype=float)
            ).all()
        )
        if invalid_snapshot:
            output["profile_estimator_diagnostics_error"] = (
                "nonfinite_snapshot_columns:" + ",".join(invalid_snapshot)
            )
            return output

    boundary_flag = numeric["terrain_joint_boundary_limited"]
    rank = numeric["terrain_joint_observability_rank"]
    evidence_age = numeric["terrain_joint_evidence_age_s"]
    confidence = numeric["terrain_joint_snapshot_confidence"]
    n_boundary_mass = numeric["terrain_joint_n_boundary_mass"]
    phi_boundary_mass = numeric["terrain_joint_phi_boundary_mass"]
    boundary_mass = numeric["terrain_joint_max_boundary_mass"]
    singular = numeric[
        "terrain_joint_observability_min_singular_value"
    ]
    timing_columns = (
        "terrain_joint_projection_wall_ms",
        "terrain_joint_profile_wall_ms",
        "terrain_joint_observability_wall_ms",
        "terrain_joint_posterior_wall_ms",
        "terrain_joint_publication_wall_ms",
        "terrain_joint_update_wall_ms",
    )
    invalid_timing = any(
        (numeric[column].loc[snapshots] < 0.0).any()
        for column in timing_columns
    )
    timing_sum_valid = np.allclose(
        numeric["terrain_joint_update_wall_ms"].loc[snapshots],
        (
            numeric["terrain_joint_projection_wall_ms"].loc[snapshots]
            + numeric["terrain_joint_posterior_wall_ms"].loc[snapshots]
            + numeric["terrain_joint_publication_wall_ms"].loc[snapshots]
        ),
        rtol=0.0,
        # Controller CSV fields are serialized to six decimal places in ms.
        atol=2.0e-6,
    )
    boundary_consistent = bool(
        boundary_flag.loc[snapshots].isin((0.0, 1.0)).all()
        and np.allclose(
            boundary_mass.loc[snapshots],
            np.maximum(
                n_boundary_mass.loc[snapshots],
                phi_boundary_mass.loc[snapshots],
            ),
            rtol=0.0,
            atol=1.0e-6,
        )
        and (
            (
                boundary_mass.loc[snapshots]
                >= float(
                    GRIT_ESTIMATOR_CONTRACT[
                        "publication_boundary_mass_limit"
                    ]
                )
            )
            == (boundary_flag.loc[snapshots] > 0.0)
        ).all()
    )
    max_age_s = float(
        GRIT_ESTIMATOR_CONTRACT["publication_max_evidence_age_s"]
    )
    min_confidence = float(
        GRIT_ESTIMATOR_CONTRACT["controller_min_confidence"]
    )
    boundary_limit = float(
        GRIT_ESTIMATOR_CONTRACT[
            "publication_boundary_mass_limit"
        ]
    )
    min_rank = int(
        GRIT_ESTIMATOR_CONTRACT[
            "publication_min_observability_rank"
        ]
    )
    min_singular = float(
        GRIT_ESTIMATOR_CONTRACT[
            "publication_min_observability_singular_value"
        ]
    )
    ready_valid = bool(
        snapshots.loc[ready].all()
        and (evidence_age.loc[ready] >= -1.0e-9).all()
        and (evidence_age.loc[ready] <= max_age_s + 1.0e-9).all()
        and (confidence.loc[ready] >= min_confidence).all()
        and (boundary_mass.loc[ready] < boundary_limit).all()
        and (rank.loc[ready] >= min_rank).all()
        and (singular.loc[ready] + 1.0e-12 >= min_singular).all()
        and fallback.loc[ready].eq("none").all()
    )
    fallback_valid = bool(
        fallback.loc[~ready].ne("none").all()
        and fallback.loc[ready].eq("none").all()
    )
    readiness_consistent = bool(
        not (updated & ~ready).any()
        and int(updated.sum()) <= int(ready.sum())
        and ready_valid
        and fallback_valid
        and boundary_consistent
        and not invalid_timing
        and timing_sum_valid
        and _finite_max_int(numeric["terrain_dynamics_windows"])
        <= _finite_max_int(numeric["terrain_accepted_dynamics_windows"])
    )

    snapshot_sequences = numeric["terrain_joint_snapshot_seq"].loc[
        snapshots
    ].astype(int)
    # The controller repeats an immutable accepted snapshot on every control
    # row until the next estimator publication.  Runtime timing belongs to the
    # publication, not to its residence time in the controller CSV, so retain
    # exactly one row per positive snapshot sequence for timing statistics.
    unique_snapshot_rows = snapshot_sequences.drop_duplicates().index
    update_times = numeric["terrain_joint_update_wall_ms"].loc[
        unique_snapshot_rows
    ]
    ready_snapshot_count = int(
        snapshot_sequences.loc[ready & snapshots].nunique()
    )
    applied_snapshot_count = int(
        snapshot_sequences.loc[updated & snapshots].nunique()
    )
    output.update({
        "profile_estimator_diagnostics_complete": readiness_consistent,
        "profile_estimator_diagnostics_error": (
            "" if readiness_consistent else "joint_snapshot_readiness_inconsistent"
        ),
        "profile_estimator_publication_ready": bool(ready.any()),
        "profile_estimator_publication_applied": bool(updated.any()),
        "profile_estimator_abstained": not bool(updated.any()),
        "profile_estimator_readiness_rows": int(ready.sum()),
        "profile_estimator_update_rows": int(updated.sum()),
        "profile_estimator_time_to_first_ready_s": _first_elapsed(times, ready),
        "profile_estimator_time_to_first_update_s": _first_elapsed(
            times, updated
        ),
        "profile_estimator_max_concurrent_windows": _finite_max_int(
            numeric["terrain_dynamics_windows"]
        ),
        "profile_estimator_lifetime_accepted_windows": _finite_max_int(
            numeric["terrain_accepted_dynamics_windows"]
        ),
        "profile_estimator_lifetime_rejected_windows": _finite_max_int(
            numeric["terrain_rejected_dynamics_windows"]
        ),
        "profile_estimator_profile_force_gain_final": _last_finite(
            numeric["terrain_profile_force_gain"]
        ),
        "profile_estimator_profile_ax_bias_final_mps2": _last_finite(
            numeric["terrain_profile_ax_bias"]
        ),
        "profile_estimator_profile_ay_bias_final_mps2": _last_finite(
            numeric["terrain_profile_ay_bias"]
        ),
        "profile_estimator_profile_bound_hits_max": _finite_max_int(
            numeric["terrain_profile_bound_hits"]
        ),
        "profile_estimator_feature_envelope_excursions_max": _finite_max_int(
            numeric["terrain_feature_envelope_excursions"]
        ),
        "profile_estimator_readiness_consistent": readiness_consistent,
        "profile_estimator_snapshot_rows": int(snapshots.sum()),
        "profile_estimator_unique_snapshot_count": int(
            snapshot_sequences.nunique()
        ),
        "profile_estimator_ready_snapshot_count": ready_snapshot_count,
        "profile_estimator_applied_snapshot_count": applied_snapshot_count,
        "profile_estimator_final_snapshot_seq": _finite_max_int(
            numeric["terrain_joint_snapshot_seq"]
        ),
        "profile_estimator_fallback_rows": int((~ready).sum()),
        "profile_estimator_max_evidence_age_s": (
            float(evidence_age.loc[snapshots].max()) if snapshots.any() else None
        ),
        "profile_estimator_min_snapshot_confidence": (
            float(confidence.loc[snapshots].min()) if snapshots.any() else None
        ),
        "profile_estimator_max_boundary_mass": (
            float(boundary_mass.loc[snapshots].max()) if snapshots.any() else None
        ),
        "profile_estimator_min_observability_singular_value": (
            float(singular.loc[snapshots].min()) if snapshots.any() else None
        ),
        "profile_estimator_update_wall_ms_median": (
            float(update_times.median()) if snapshots.any() else None
        ),
        "profile_estimator_update_wall_ms_p95": (
            float(update_times.quantile(0.95)) if snapshots.any() else None
        ),
        "profile_estimator_update_wall_ms_max": (
            float(update_times.max()) if snapshots.any() else None
        ),
    })
    return output


def live_estimator_diagnostics(
    diag: pd.DataFrame | None,
    *,
    backend: str,
    enabled: bool,
) -> dict[str, Any]:
    """Audit the selected estimator with its own frozen readiness contract."""

    if str(backend) == GRIT_ESTIMATOR_BACKEND:
        return _joint_diagnostics(
            diag,
            applicable=bool(enabled),
        )
    return parent_estimator_diagnostics(
        diag,
        backend=backend,
        enabled=enabled,
    )
