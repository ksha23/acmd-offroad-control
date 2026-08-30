#!/usr/bin/env python3
"""Closed-loop tracking against the source of the controller's soil parameters.

This study produces the terrain-conditioning robustness result cited in Sec. 3.2
of the manuscript: with the tire model, path, speed reference, and controller
weights held identical, the source of the controller's soil parameters is varied
over four arms and closed-loop tracking is measured for each.

  * ``nn_static``            -- the true soil preset, an oracle-information
                                reference no deployable controller can have.
  * ``nn_estimator``         -- GRIT, estimating the Bekker exponent ``n`` and
                                the friction angle ``phi`` independently and
                                online from vehicle motion alone.
  * ``nn_parent_estimator``  -- a matched scalar estimator that infers only the
                                manifold coordinate and reads ``phi`` off the
                                soil manifold, isolating what independent
                                two-coordinate estimation contributes.
  * ``nn_fixed_fallback``    -- the fixed low-grip endpoint the controller holds
                                whenever GRIT withholds a snapshot.

The fourth arm is deliberately the controller's own declared fallback rather
than an arbitrary wrong prior, so the comparison measures the cost of abstention
as the deployed stack actually experiences it.

The speed reference is fixed across arms, which confines the measured effect to
the cornering model; the value of the estimate in the speed channel is measured
separately by ``grit_adaptive_speed_matrix.py``. Every arm shares one truth-free
controller packet and one per-scenario reference profile, both enforced by
``_write_reference_contract``. Sensor noise is enabled in every run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BUMPS,
    DEFAULT_NN_MODEL,
    PATHS,
    SPEEDS,
    TERRAINS,
    RunResult,
    RIG_ACTIVE_ESTIMATOR_BACKEND,
    GRIT_ESTIMATOR_BACKEND,
    GRIT_ESTIMATOR_CONTRACT,
    PARENT_ESTIMATOR_BACKEND,
    bounded_ros_workers,
    controller_tire_force_truth_rows,
    launch_and_collect,
    plot_metric_distribution_grid,
    require_active_joint_estimator,
    estimator_contract,
    estimator_runtime_args,
    estimator_artifact_hashes,
    save_summary_markdown,
    summarize_by_variant,
    timestamped_result_dir,
    write_manifest,
    write_results_csv,
)
try:  # noqa: E402
    from active_estimator_diagnostics import live_estimator_diagnostics
except ModuleNotFoundError:  # imported as a package by the tests
    from benchmarking.active_estimator_diagnostics import (
        live_estimator_diagnostics,
    )
try:  # noqa: E402
    from paper_provenance import (
        acquire_paper_ros_lease,
        downstream_repository_provenance,
        launch_identity_contract,
    )
except ModuleNotFoundError:  # imported as a package by the tests
    from benchmarking.paper_provenance import (
        acquire_paper_ros_lease,
        downstream_repository_provenance,
        launch_identity_contract,
    )


# Arms the study can run. ``estimator`` marks the rows that carry
# ``--terrain-estimator``; ``controller_prior`` names the soil the controller
# assumes before, or in place of, an online estimate; ``role`` is the arm's
# recorded interpretation and is carried into every result row and the reference
# contract so each row states what it stands for. The two analytical rows are
# component comparators and are not part of the published four-arm result.
VARIANTS = {
    "pacejka_static":           dict(mpc="pacejka", nn=DEFAULT_NN_MODEL, estimator=False),
    "tmeasy_static":            dict(mpc="tmeasy",  nn=DEFAULT_NN_MODEL, estimator=False),
    "nn_static":                dict(
        mpc="nn", nn=DEFAULT_NN_MODEL, estimator=False,
        role="matched-terrain oracle-information baseline",
    ),
    "nn_estimator":             dict(mpc="nn", nn=DEFAULT_NN_MODEL, estimator=True,
                                     mode="n", controller_prior="dirt",
                                     role="promoted independent-n-phi online estimator"),
    "nn_parent_estimator":      dict(
        mpc="nn", nn=DEFAULT_NN_MODEL, estimator=True,
        mode="n", controller_prior="dirt",
        estimator_backend=PARENT_ESTIMATOR_BACKEND,
        role="historical matched scalar-parent online estimator",
    ),
    # Static counterpart of the controller's startup and fail-closed endpoint:
    # terrain_params_for_n(0.5) with an independent phi of 13 deg reproduces the
    # clay preset exactly, so this arm holds that endpoint for the whole run.
    # The endpoint is control-feasible and low-grip; it is not a claim that clay
    # is a universal physical worst case.
    "nn_fixed_fallback":        dict(
        mpc="nn", nn=DEFAULT_NN_MODEL, estimator=False,
        controller_prior="clay",
        role="fixed promoted low-grip fallback endpoint",
    ),
}
DEFAULT_VARIANTS = [
    "nn_static",
    "nn_estimator",
    "nn_parent_estimator",
    "nn_fixed_fallback",
]

VARIANT_LABELS = {
    "pacejka_static": "Pacejka static",
    "tmeasy_static": "TMeasy static",
    "nn_static": "NN matched-terrain oracle",
    "nn_estimator": "NN joint online estimator",
    "nn_parent_estimator": "NN scalar-parent estimator",
    "nn_fixed_fallback": "NN fixed low-grip fallback",
}


def display_variant(value: str) -> str:
    return VARIANT_LABELS.get(str(value), str(value).replace("_", " "))


def _conditioning_arm_roles(variants: list[str]) -> dict[str, str]:
    """Describe exactly the requested arms, preserving their requested order."""

    return {
        name: str(
            VARIANTS[name].get("role", "component tire-model comparator")
        )
        for name in variants
    }


def add_scenario_labels(ok: pd.DataFrame) -> pd.DataFrame:
    """Scenario labels that show achieved speed beside the requested speed."""
    out = ok.copy()
    stats = (
        out.groupby(["terrain", "speed_mps"], sort=False)
        .agg(mean_speed=("mean_speed_mps", "mean"))
        .reset_index()
    )
    stats["scenario_label"] = stats.apply(
        lambda r: (
            f"{r['terrain']} - cmd {float(r['speed_mps']):.0f}, "
            f"ubar {float(r['mean_speed']):.2f} m/s"
        ),
        axis=1,
    )
    return out.merge(stats, on=["terrain", "speed_mps"], how="left")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                   choices=list(VARIANTS))
    p.add_argument("--terrains", nargs="+", default=list(TERRAINS), choices=list(TERRAINS))
    p.add_argument("--paths", nargs="+", default=list(PATHS))
    p.add_argument("--speeds", nargs="+", type=float, default=list(SPEEDS))
    p.add_argument("--bumpiness", nargs="+", type=int, default=list(BUMPS))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=400)
    p.add_argument("--time", type=float, default=20.0,
                   help="Run duration. Longer than the tire-model benchmark so "
                        "that the online arms have time to acquire an estimate.")
    p.add_argument("--lead-in", type=float, default=5.0)
    p.add_argument("--metric-start", type=float, default=8.0,
                   help="Start of the scoring window, placed after the online "
                        "arms have had time to acquire. The window is fixed and "
                        "identical for every arm, so it includes acquisition "
                        "rather than assuming convergence at its start.")
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--base-port", type=int, default=9000)
    p.add_argument(
        "--estimator-backend",
        choices=[GRIT_ESTIMATOR_BACKEND],
        default=RIG_ACTIVE_ESTIMATOR_BACKEND,
        help="Backend evaluated by the nn_estimator arm. The scalar-parent arm "
             "always uses scalar_parent, so the two estimator arms differ "
             "in method rather than in configuration.",
    )
    p.add_argument("--workers", type=bounded_ros_workers, default=6,
                   help="Parallel worker processes. Worker 0 runs solo first "
                        "to warm the acados/CasADi codegen cache (1--101).")
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--resume-dir",
        default="",
        help="Existing result directory to resume. Successful scenario rows are "
             "kept and only missing/non-ok cells are rerun.",
    )
    return p.parse_args()


@dataclass(frozen=True)
class _Task:
    """Pickle-friendly description of one closed-loop run."""
    idx: int
    variant: str
    mpc_model: str
    nn_model: str
    extra: tuple[str, ...]
    terrain: str
    path: str
    speed: float
    bumpiness: int
    seed: int
    run_dir_str: str
    sim_port: int
    ctrl_port: int
    sim_time: float
    timeout: float
    lead_in: float
    metric_start: float
    estimator_backend: str
    estimator_enabled: bool
    conditioning_role: str
    controller_prior: str


def _run_one(task: _Task) -> RunResult:
    """ProcessPool worker. Each call runs one launch_decoupled closed-loop."""
    os.environ.setdefault("ACADOS_UNIQUE_BUILD_DIR", "1")
    result = launch_and_collect(
        experiment="tire_model_with_estimator_ablation",
        variant=task.variant,
        controller_mode="standard",
        mpc_model=task.mpc_model,
        nn_model=task.nn_model,
        terrain=task.terrain, path=task.path,
        speed=task.speed, bumpiness=task.bumpiness, seed=task.seed,
        run_dir=Path(task.run_dir_str),
        sim_port=task.sim_port, ctrl_port=task.ctrl_port,
        sim_time=task.sim_time, timeout=task.timeout,
        rocks=0, lead_in=task.lead_in,
        extra_args=list(task.extra),
        metric_start=task.metric_start,
    )
    identity = launch_identity_contract(
        Path(task.run_dir_str),
        expected_path=task.path,
        expected_speed_mps=task.speed,
        expected_seed=task.seed,
        expected_sim_port=task.sim_port,
        expected_ctrl_port=task.ctrl_port,
    )
    result.extra.update(identity)
    if not identity["launch_identity_match"]:
        result.status = "launch_identity_violation"
    profiles = sorted(Path(task.run_dir_str).rglob("reference_path_*.csv"))
    diag = None
    if result.status == "ok" and result.diag_csv and Path(result.diag_csv).is_file():
        diag = pd.read_csv(result.diag_csv)
    truth_rows = controller_tire_force_truth_rows(diag) if diag is not None else 0
    profile_diagnostics = live_estimator_diagnostics(
        diag,
        backend=task.estimator_backend,
        enabled=task.estimator_enabled,
    )
    result.extra.update(profile_diagnostics)
    result.extra["estimator_backend"] = task.estimator_backend
    result.extra["estimator_contract_version"] = estimator_contract(
        task.estimator_backend
    )["contract_version"]
    result.extra["conditioning_role"] = task.conditioning_role
    result.extra["controller_prior_terrain"] = task.controller_prior
    if (
        result.status == "ok"
        and profile_diagnostics["profile_estimator_diagnostics_applicable"]
        and (
            not profile_diagnostics["profile_estimator_diagnostics_complete"]
            or not profile_diagnostics["profile_estimator_readiness_consistent"]
        )
    ):
        result.status = "estimator_diagnostics_violation"
    if truth_rows:
        result.status = "truth_packet_violation"
    if result.status == "ok" and len(profiles) == 1:
        result.extra["reference_profile_sha256"] = hashlib.sha256(
            profiles[0].read_bytes()
        ).hexdigest()
    else:
        result.extra["reference_profile_sha256"] = ""
    result.extra["reference_policy"] = "shared_worst_case_phi13_curvature_v1"
    result.extra["truth_free_controller_packet"] = True
    result.extra["controller_tire_force_truth_rows"] = truth_rows
    return result


def _write_reference_contract(out_dir: Path) -> None:
    """Fail closed unless every arm used one identical per-scenario profile."""
    frame = pd.read_csv(out_dir / "results.csv")
    ok = frame[frame["status"].astype(str) == "ok"].copy()
    hash_column = "extra_reference_profile_sha256"
    policy_column = "extra_reference_policy"
    truth_column = "extra_truth_free_controller_packet"
    truth_rows_column = "extra_controller_tire_force_truth_rows"
    identity_column = "extra_launch_identity_match"
    diagnostics_applicable_column = "extra_profile_estimator_diagnostics_applicable"
    diagnostics_complete_column = "extra_profile_estimator_diagnostics_complete"
    readiness_consistent_column = "extra_profile_estimator_readiness_consistent"
    required = {
        hash_column, policy_column, truth_column, truth_rows_column,
        identity_column, diagnostics_applicable_column,
        diagnostics_complete_column, readiness_consistent_column,
    }
    if not required.issubset(ok.columns):
        raise RuntimeError(f"missing reference/truth contract columns: {sorted(required - set(ok.columns))}")
    if ok[hash_column].isna().any() or not ok[hash_column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise RuntimeError("one or more successful runs lack a reference-profile hash")
    if set(ok[policy_column].astype(str)) != {"shared_worst_case_phi13_curvature_v1"}:
        raise RuntimeError("conditioning arms did not use the fixed reference policy")
    if not ok[truth_column].astype(bool).all():
        raise RuntimeError("conditioning controller packets were not truth-free")
    if not (pd.to_numeric(ok[truth_rows_column], errors="coerce") == 0).all():
        raise RuntimeError("simulator-truth tire diagnostics reached the controller")
    if set(ok[identity_column].astype(str).str.lower()) != {"true"}:
        raise RuntimeError("conditioning benchmark contains a launch-identity mismatch")
    if len(ok) != len(frame):
        raise RuntimeError("conditioning benchmark contains failed/incomplete rows")
    keys = ["variant", "terrain", "path", "speed_mps", "bumpiness", "seed"]
    if ok[keys].duplicated().any():
        raise RuntimeError("conditioning benchmark contains duplicate cells")
    metrics = ok[["rms_cte_m", "mean_speed_mps", "mean_solve_ms"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not metrics.notna().all().all():
        raise RuntimeError("conditioning benchmark contains non-finite paper metrics")

    group_hashes = {}
    for (path, speed), group in ok.groupby(["path", "speed_mps"], sort=True):
        hashes = sorted(set(group[hash_column].astype(str)))
        if len(hashes) != 1:
            raise RuntimeError(
                f"reference profile differs across arms for {path}/v={speed}: {hashes}"
            )
        group_hashes[f"{path}|{float(speed):g}"] = hashes[0]
    profile_rows = ok[ok[diagnostics_applicable_column].astype(bool)].copy()
    if (
        not profile_rows.empty
        and (
            not profile_rows[diagnostics_complete_column].astype(bool).all()
            or not profile_rows[readiness_consistent_column].astype(bool).all()
        )
    ):
        raise RuntimeError("profile-estimator readiness diagnostics are incomplete/inconsistent")
    contract = {
        "schema_version": 4,
        "design": "conditioning_joint_parent_fallback_ros_isolated",
        "reference_policy": "shared_worst_case_phi13_curvature_v1",
        "reference_profile_friction_angle_deg": 13.0,
        "lateral_acceleration_bound_policy": "shared_worst_case_phi13_bound_v1",
        "lateral_acceleration_bound_friction_angle_deg": 13.0,
        "controller_packet_truth": False,
        "launch_identity_contract": "path_speed_seed_ports_domain",
        "ros_concurrency_policy": "exclusive_process_lease_and_batched_workers",
        "group_keys": ["path", "speed_mps"],
        "group_profile_sha256": group_hashes,
        "n_successful_rows": int(len(ok)),
        "profile_estimator_applicable_rows": int(len(profile_rows)),
        "profile_estimator_publication_ready_rows": int(
            profile_rows["extra_profile_estimator_publication_ready"].astype(bool).sum()
        ) if not profile_rows.empty else 0,
        "profile_estimator_abstained_rows": int(
            profile_rows["extra_profile_estimator_abstained"].astype(bool).sum()
        ) if not profile_rows.empty else 0,
        "estimator_backend_counts": {
            str(key): int(value)
            for key, value in ok.groupby(
                "extra_estimator_backend", sort=True
            ).size().items()
        },
        "conditioning_roles": {
            str(key): str(value)
            for key, value in ok.groupby("variant", sort=True)[
                "extra_conditioning_role"
            ].first().items()
        },
        "fixed_fallback_contract": (
            {
                "variant": "nn_fixed_fallback",
                "controller_prior_terrain": "clay",
                "n": float(GRIT_ESTIMATOR_CONTRACT["fallback_n"]),
                "phi_deg": float(
                    GRIT_ESTIMATOR_CONTRACT["fallback_phi_deg"]
                ),
                "policy": str(
                    GRIT_ESTIMATOR_CONTRACT["fallback_policy"]
                ),
            }
            if "nn_fixed_fallback" in set(ok["variant"].astype(str))
            else None
        ),
    }
    (out_dir / "reference_profile_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )


def plot_figures(results_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(results_csv)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = ok.groupby("variant", sort=False).agg(
        rms_cte=("rms_cte_m", "mean"),
        rms_cte_std=("rms_cte_m", "std"),
        speed_ratio=("speed_ratio", "mean"),
        speed_ratio_std=("speed_ratio", "std"),
        mean_speed=("mean_speed_mps", "mean"),
        mean_speed_std=("mean_speed_mps", "std"),
        solve_ms=("mean_solve_ms", "mean"),
    ).reset_index()
    summary["display_variant"] = summary["variant"].map(display_variant)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    x = np.arange(len(summary))
    for ax, mean_key, std_key, ylabel in [
        (axes[0], "rms_cte", "rms_cte_std", "RMS CTE (m)"),
        (axes[1], "mean_speed", "mean_speed_std", "Achieved mean speed (m/s)"),
        (axes[2], "solve_ms", None, "Mean solve time (ms)"),
    ]:
        err = summary[std_key].fillna(0.0) if std_key else None
        ax.bar(x, summary[mean_key], yerr=err, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(summary["display_variant"], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Tire model x live terrain estimator")
    fig.tight_layout()
    fig.savefig(fig_dir / "tire_estimator_summary.png", dpi=220)
    plt.close(fig)

    ok = add_scenario_labels(ok)
    pivot = ok.pivot_table(index="scenario_label", columns="variant",
                           values="rms_cte_m", aggfunc="mean")
    pivot = pivot.rename(columns=display_variant)
    fig, ax = plt.subplots(figsize=(1.5 * len(pivot.columns) + 4,
                                    0.5 * len(pivot.index) + 3))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > pivot.values.mean() else "black")
    ax.set_title("RMS CTE (m) by (terrain, speed)")
    fig.colorbar(im, ax=ax, fraction=0.035)
    fig.tight_layout()
    fig.savefig(fig_dir / "tire_estimator_rms_cte_heatmap.png", dpi=220)
    plt.close(fig)

    plot_metric_distribution_grid(
        results_csv,
        out_dir,
        [
            ("rms_cte_m", "RMS CTE (m)", "Tracking error"),
            ("mean_speed_mps", "Achieved mean speed (m/s)", "Actual speed"),
            ("mean_solve_ms", "Mean solve (ms)", "Runtime"),
        ],
        "tire_estimator_metric_distributions.png",
        "Tire model x live terrain estimator",
    )


def finalize_results(out_dir: Path, results: list[RunResult]) -> None:
    """Write summaries and plots, including for an already-complete resume."""
    write_results_csv(out_dir / "results.csv", results)
    _write_reference_contract(out_dir)
    summary = summarize_by_variant(
        results,
        ["rms_cte_m", "speed_ratio", "mean_speed_mps", "mean_solve_ms"],
    )
    summary.to_csv(out_dir / "summary_by_variant.csv", index=False)
    save_summary_markdown(
        out_dir,
        "Tire model x live terrain estimator",
        summary,
        [
            "Noise policy: sensor noise enabled in every run.",
            "Every arm drives the same declared worst-case (phi=13 deg) curvature "
            "reference and receives a truth-free controller packet, so the soil "
            "parameter source is the only difference between arms.",
            "Both online arms start from the same dirt prior. The fixed-fallback "
            "arm holds the controller's n=0.5, phi=13 deg low-grip endpoint for "
            "the whole run.",
            "The scalar-parent arm infers the manifold coordinate alone and reads "
            "phi off the manifold; nn_estimator infers n and phi independently. "
            "The pair measures what independent estimation contributes.",
            "The scoring window is fixed and identical across arms, so it "
            "includes online acquisition rather than assuming convergence.",
        ],
    )
    plot_figures(out_dir / "results.csv", out_dir)


def _resume_estimator_contract_matches(
    recorded_backend: object,
    recorded_contract: object,
    requested_backend: str,
) -> bool:
    """Require an identical frozen estimator contract before retaining rows."""

    return bool(
        recorded_backend == requested_backend
        and isinstance(recorded_contract, dict)
        and recorded_contract == estimator_contract(requested_backend)
    )


_RESUME_MUTABLE_ARGUMENTS = frozenset({"resume_dir", "timeout", "workers"})
_SCENARIO_KEY_COLUMNS = (
    "variant", "terrain", "path", "speed_mps", "bumpiness", "seed",
)


def _resume_manifest_mismatches(
    manifest_values: dict[str, object],
    args: argparse.Namespace,
) -> list[str]:
    """Compare every frozen run argument while allowing only execution tuning."""

    mismatches: list[str] = []
    for key, expected in sorted(vars(args).items()):
        if key in _RESUME_MUTABLE_ARGUMENTS:
            continue
        if key not in manifest_values:
            mismatches.append(f"{key}:missing")
            continue
        raw = manifest_values[key]
        try:
            recorded = ast.literal_eval(str(raw))
        except (SyntaxError, ValueError):
            mismatches.append(f"{key}:unparseable")
            continue
        if recorded != expected:
            mismatches.append(
                f"{key}:recorded={recorded!r}:requested={expected!r}"
            )
    return mismatches


def _task_scenario_key(task: _Task) -> tuple[object, ...]:
    return (
        task.variant,
        task.terrain,
        task.path,
        float(task.speed),
        int(task.bumpiness),
        int(task.seed),
    )


def _row_scenario_key(record: dict[str, object]) -> tuple[object, ...]:
    try:
        return (
            str(record["variant"]),
            str(record["terrain"]),
            str(record["path"]),
            float(record["speed_mps"]),
            int(record["bumpiness"]),
            int(record["seed"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "successful resume row has a malformed scenario key"
        ) from exc


def _recorded_bool(value: object, *, column: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise RuntimeError(f"successful resume row has invalid {column}={value!r}")


def _recorded_int_list(value: object, *, column: str) -> list[int]:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise RuntimeError(
                f"successful resume row has invalid {column}={value!r}"
            ) from exc
    if not isinstance(parsed, (list, tuple)):
        raise RuntimeError(
            f"successful resume row has invalid {column}={value!r}"
        )
    try:
        return [int(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"successful resume row has invalid {column}={value!r}"
        ) from exc


def _validated_successful_resume_rows(
    existing: pd.DataFrame,
    tasks: list[_Task],
) -> pd.DataFrame:
    """Return successful rows only after binding each to its exact task."""

    required = {
        *_SCENARIO_KEY_COLUMNS,
        "experiment",
        "controller_mode",
        "mpc_model",
        "nn_model",
        "run_dir",
        "status",
        "extra_launch_identity_match",
        "extra_observed_path",
        "extra_observed_speed_mps",
        "extra_observed_sim_seed",
        "extra_observed_ros_domain_id",
        "extra_observed_sim_ports",
        "extra_observed_ctrl_ports",
        "extra_estimator_backend",
        "extra_estimator_contract_version",
        "extra_conditioning_role",
        "extra_controller_prior_terrain",
        "extra_reference_policy",
        "extra_truth_free_controller_packet",
        "extra_controller_tire_force_truth_rows",
    }
    missing = sorted(required - set(existing.columns))
    if missing:
        raise RuntimeError(
            "resume results lack required task/provenance columns: "
            + ", ".join(missing)
        )

    successful = existing[
        existing["status"].astype(str) == "ok"
    ].copy()
    requested: dict[tuple[object, ...], _Task] = {}
    for task in tasks:
        key = _task_scenario_key(task)
        if key in requested:
            raise RuntimeError(f"requested task matrix contains duplicate cell {key}")
        requested[key] = task

    records: dict[tuple[object, ...], dict[str, object]] = {}
    for record in successful.to_dict(orient="records"):
        key = _row_scenario_key(record)
        if key in records:
            raise RuntimeError(f"resume results contain duplicate successful cell {key}")
        if key not in requested:
            raise RuntimeError(
                f"resume results contain out-of-matrix successful cell {key}"
            )
        records[key] = record

    for key, record in records.items():
        task = requested[key]
        expected_strings = {
            "experiment": "tire_model_with_estimator_ablation",
            "controller_mode": "standard",
            "mpc_model": task.mpc_model,
            "nn_model": task.nn_model,
            "run_dir": str(Path(task.run_dir_str).resolve()),
            "extra_observed_path": task.path,
            "extra_estimator_backend": task.estimator_backend,
            "extra_estimator_contract_version": estimator_contract(
                task.estimator_backend
            )["contract_version"],
            "extra_conditioning_role": task.conditioning_role,
            "extra_controller_prior_terrain": task.controller_prior,
            "extra_reference_policy": "shared_worst_case_phi13_curvature_v1",
        }
        for column, expected in expected_strings.items():
            observed = str(record[column])
            if column == "run_dir":
                observed = str(Path(observed).resolve())
            if observed != str(expected):
                raise RuntimeError(
                    f"resume cell {key} has incompatible {column}: "
                    f"{observed!r} != {expected!r}"
                )
        numeric_expected = {
            "extra_observed_speed_mps": float(task.speed),
            "extra_observed_sim_seed": int(task.seed),
            "extra_observed_ros_domain_id": int(task.sim_port) % 101,
            "extra_controller_tire_force_truth_rows": 0,
        }
        for column, expected in numeric_expected.items():
            try:
                observed = float(record[column])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"resume cell {key} has malformed {column}"
                ) from exc
            if observed != float(expected):
                raise RuntimeError(
                    f"resume cell {key} has incompatible {column}: "
                    f"{observed!r} != {expected!r}"
                )
        if _recorded_int_list(
            record["extra_observed_sim_ports"],
            column="extra_observed_sim_ports",
        ) != [int(task.sim_port)]:
            raise RuntimeError(f"resume cell {key} has incompatible simulator ports")
        if _recorded_int_list(
            record["extra_observed_ctrl_ports"],
            column="extra_observed_ctrl_ports",
        ) != [int(task.ctrl_port)]:
            raise RuntimeError(f"resume cell {key} has incompatible controller ports")
        for column in (
            "extra_launch_identity_match",
            "extra_truth_free_controller_packet",
        ):
            if not _recorded_bool(record[column], column=column):
                raise RuntimeError(f"resume cell {key} violates {column}")
    return successful


def _domain_safe_batches(
    tasks: list[_Task],
    width: int,
) -> list[list[_Task]]:
    """Partition sparse tasks without overlapping modulo-101 DDS domains."""

    workers = bounded_ros_workers(width)
    batches: list[list[_Task]] = []
    batch: list[_Task] = []
    domains: set[int] = set()
    for task in tasks:
        domain = int(task.sim_port) % 101
        if batch and (len(batch) >= workers or domain in domains):
            batches.append(batch)
            batch = []
            domains = set()
        batch.append(task)
        domains.add(domain)
    if batch:
        batches.append(batch)
    return batches


def main() -> None:
    args = parse_args()
    require_active_joint_estimator(args.estimator_backend)
    ros_lease = acquire_paper_ros_lease("tire_model_with_estimator_ablation")
    args.design_version = "conditioning_joint_parent_fallback_ros_isolated"
    args.launch_identity_contract = "path_speed_seed_ports_domain"
    args.ros_concurrency_policy = "exclusive_process_lease_and_batched_workers"
    args.reference_policy = "shared_worst_case_phi13_curvature_v1"
    args.reference_profile_friction_angle_deg = 13.0
    args.lateral_acceleration_bound_policy = "shared_worst_case_phi13_bound_v1"
    args.lateral_acceleration_bound_friction_angle_deg = 13.0
    args.estimator_initial_prior = "dirt"
    args.tire_force_truth_enabled = False
    args.estimator_contract = estimator_contract(args.estimator_backend)
    args.historical_parent_estimator_backend = PARENT_ESTIMATOR_BACKEND
    args.historical_parent_estimator_contract = estimator_contract(
        PARENT_ESTIMATOR_BACKEND
    )
    args.conditioning_arms = _conditioning_arm_roles(args.variants)
    args.fixed_fallback_contract = (
        {
            "controller_prior_terrain": "clay",
            "n": GRIT_ESTIMATOR_CONTRACT["fallback_n"],
            "phi_deg": GRIT_ESTIMATOR_CONTRACT["fallback_phi_deg"],
            "policy": GRIT_ESTIMATOR_CONTRACT["fallback_policy"],
        }
        if "nn_fixed_fallback" in args.variants
        else None
    )
    for key, value in estimator_artifact_hashes(args.estimator_backend).items():
        setattr(args, key, value)
    for key, value in estimator_artifact_hashes(
        PARENT_ESTIMATOR_BACKEND
    ).items():
        setattr(args, "historical_parent_" + key, value)
    provenance = downstream_repository_provenance()
    args.code_git_head = provenance["code_git_head"]
    args.tracked_worktree_dirty = provenance["tracked_worktree_dirty"]
    args.uncommitted_source_files = provenance["uncommitted_source_files"]
    args.paper_evidence_eligible = provenance["paper_evidence_eligible"]
    args.source_sha256 = provenance["source_sha256"]
    if args.quick:
        args.terrains = ["clay"]
        args.paths = ["sinusoidal"]
        args.speeds = [5.0]
        args.bumpiness = [0]
        args.seeds = 1
        args.time = min(args.time, 12.0)

    result_prefix = "tire_model_with_estimator_ablation"
    if args.resume_dir:
        out_dir = Path(args.resume_dir).expanduser().resolve()
        if not (out_dir / "results.csv").exists():
            raise FileNotFoundError(out_dir / "results.csv")
        manifest_path = out_dir / "manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest_frame = pd.read_csv(manifest_path)
        if (
            not {"key", "value"}.issubset(manifest_frame.columns)
            or manifest_frame["key"].astype(str).duplicated().any()
        ):
            raise RuntimeError("--resume-dir manifest is malformed or has duplicate keys")
        manifest_values = dict(
            manifest_frame[["key", "value"]].itertuples(index=False, name=None)
        )
        recorded_backend = ast.literal_eval(
            manifest_values.get("estimator_backend", "None")
        )
        recorded_contract = ast.literal_eval(
            manifest_values.get("estimator_contract", "None")
        )
        if not _resume_estimator_contract_matches(
            recorded_backend, recorded_contract, args.estimator_backend
        ):
            raise RuntimeError(
                "--resume-dir estimator backend/full contract does not match this run; "
                "start a new timestamped directory instead of mixing evidence"
            )
        manifest_mismatches = _resume_manifest_mismatches(
            manifest_values, args
        )
        if manifest_mismatches:
            raise RuntimeError(
                "--resume-dir frozen design/provenance does not match this run: "
                + "; ".join(manifest_mismatches[:8])
                + ("; ..." if len(manifest_mismatches) > 8 else "")
            )
    else:
        out_dir = timestamped_result_dir(result_prefix)
        write_manifest(out_dir, args,
                       "Closed-loop tracking against the source of the "
                       "controller's soil parameters, at a fixed speed reference.")
    print(f"Output: {out_dir}")

    tasks: list[_Task] = []
    idx = 0
    for variant in args.variants:
        spec = VARIANTS[variant]
        variant_backend = (
            str(spec.get("estimator_backend", args.estimator_backend))
            if spec["estimator"]
            else "disabled"
        )
        # Fix the speed reference across arms. The terrain-aware g-g planner
        # derives its grip budget from the applied soil parameters, so leaving it
        # enabled would let each arm drive at a different speed and would mix the
        # estimate's speed-channel value into a cornering-model comparison. That
        # speed-channel value is measured separately by the adaptive-speed matrix.
        extra = [
            "--legacy-speed-ref",
            "--reference-profile-friction-angle-deg", "13.0",
            "--shared-ay-bound-friction-angle-deg", "13.0",
            "--no-tire-forces",
        ]
        if spec["estimator"]:
            extra += ["--terrain-estimator", "--terrain-estimator-backend",
                      variant_backend,
                      "--terrain-estimator-mode", spec.get("mode", "n")]
            extra += estimator_runtime_args(variant_backend)
        if "controller_prior" in spec:
            extra += ["--controller-prior-terrain", spec["controller_prior"]]
        for terrain in args.terrains:
            for path in args.paths:
                for speed in args.speeds:
                    for bump in args.bumpiness:
                        for seed_i in range(args.seeds):
                            seed = args.base_seed + seed_i
                            sim_port = args.base_port + 2 * idx
                            ctrl_port = sim_port + 1
                            run_dir = out_dir / "raw" / (
                                f"{idx:04d}_{variant}_{terrain}_{path}_v{speed:g}_b{bump}_s{seed}"
                            )
                            tasks.append(_Task(
                                idx=idx, variant=variant,
                                mpc_model=spec["mpc"], nn_model=spec["nn"],
                                extra=tuple(extra),
                                terrain=terrain, path=path, speed=speed,
                                bumpiness=bump, seed=seed,
                                run_dir_str=str(run_dir),
                                sim_port=sim_port, ctrl_port=ctrl_port,
                                sim_time=args.time, timeout=args.timeout,
                                lead_in=args.lead_in,
                                metric_start=args.metric_start,
                                estimator_backend=variant_backend,
                                estimator_enabled=bool(spec["estimator"]),
                                conditioning_role=str(spec.get(
                                    "role", "component tire-model comparator"
                                )),
                                controller_prior=str(
                                    spec.get("controller_prior", "matched_plant")
                                ),
                            ))
                            idx += 1

    results: list[RunResult] = []
    if args.resume_dir:
        existing = pd.read_csv(out_dir / "results.csv")
        existing = _validated_successful_resume_rows(existing, tasks)
        successful = {
            (str(row.variant), str(row.terrain), str(row.path), float(row.speed_mps),
             int(row.bumpiness), int(row.seed))
            for row in existing.itertuples(index=False)
        }
        tasks = [task for task in tasks if (
            task.variant, task.terrain, task.path, float(task.speed),
            int(task.bumpiness), int(task.seed)
        ) not in successful]
        fields = RunResult.__dataclass_fields__
        for record in existing.to_dict(orient="records"):
            values = {name: record[name] for name in fields if name != "extra" and name in record}
            values["extra"] = {
                key.removeprefix("extra_"): value
                for key, value in record.items()
                if key.startswith("extra_")
            }
            results.append(RunResult(**values))
        print(f"Resume: kept {len(results)} successful rows; rerunning {len(tasks)} cells")

    total = len(tasks)
    if not tasks:
        print("Resume is already complete.")
        finalize_results(out_dir, results)
        return

    print(f"[1/{total}] (warmup) {tasks[0].variant} {tasks[0].terrain}/{tasks[0].path} "
          f"v={tasks[0].speed:g} b={tasks[0].bumpiness} seed={tasks[0].seed}")
    first = _run_one(tasks[0])
    results.append(first)
    write_results_csv(out_dir / "results.csv", results)
    print(f"    {first.status}: rms_cte={first.rms_cte_m:.3f} "
          f"speed_ratio={first.speed_ratio:.2f}")

    if len(tasks) > 1:
        completed = 1
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            # Submit one bounded batch at a time. The launcher derives DDS
            # domains from ports modulo 101, so without this barrier a slow task
            # could still be running when a later task whose domain has wrapped
            # starts, placing two simulations on one domain and cross-wiring
            # their topics.
            pending = tasks[1:]
            for batch in _domain_safe_batches(pending, args.workers):
                futs = {ex.submit(_run_one, task): task for task in batch}
                for fut in as_completed(futs):
                    t = futs[fut]
                    res = fut.result()
                    results.append(res)
                    completed += 1
                    write_results_csv(out_dir / "results.csv", results)
                    print(f"[{completed}/{total}] {t.variant} {t.terrain}/{t.path} "
                          f"v={t.speed:g} b={t.bumpiness} seed={t.seed}")
                    print(f"    {res.status}: rms_cte={res.rms_cte_m:.3f} "
                          f"speed_ratio={res.speed_ratio:.2f}")

    finalize_results(out_dir, results)
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
