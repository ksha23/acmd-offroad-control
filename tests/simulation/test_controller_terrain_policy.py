"""Contracts for the controller's terrain-parameter source.

The estimator arms are only meaningful if the controller cannot see the plant
soil, so these tests establish where each arm's terrain parameters come from:
a live estimator bootstraps from a blind prior and the plant configuration
cannot override it, while a matched arm may read the plant label by design.
They also fix the readiness gates that decide when an accepted snapshot is
allowed to reach the controller.
"""

from types import SimpleNamespace

from simulation.control.acados_mpc_controller_node import (
    RIG_JOINT_ACCEPTED_SNAPSHOT_VERSION,
    RIG_JOINT_BOUNDARY_MASS_LIMIT,
    RIG_JOINT_CONTROL_MIN_PHI_DEG,
    RIG_JOINT_MAX_EVIDENCE_AGE_S,
    RIG_JOINT_MIN_OBSERVABILITY_SINGULAR_VALUE,
    RIG_JOINT_MIN_PUBLICATION_CONFIDENCE,
    _JointSnapshotLatch,
    _controller_prior_name,
    _controller_visible_sim_config,
    _joint_fixed_fallback_params,
    _joint_snapshot_parameters,
    _joint_snapshot_control_parameters,
    _joint_snapshot_readiness,
    _joint_snapshot_sequence,
    _mpc_build_friction_angle,
    _reference_profile_friction_angle,
)


def _args(**updates):
    values = {
        "model": "nn",
        "terrain": "clay",
        "terrain_estimator": False,
        "terrain_estimator_prior": "dirt",
        "controller_prior_terrain": None,
        "legacy_speed_ref": False,
        "terrain_independent_ay_bound": False,
        "reference_profile_friction_angle_deg": None,
        "shared_ay_bound_friction_angle_deg": None,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_live_estimator_bootstraps_from_blind_dirt_prior():
    assert _controller_prior_name(_args(terrain_estimator=True)) == "dirt"


def test_estimator_disabled_matched_arm_bootstraps_from_plant_label():
    assert _controller_prior_name(_args()) == "clay"


def test_explicit_controller_prior_wins_for_estimator_arm():
    args = _args(terrain_estimator=True, controller_prior_terrain="sand")
    assert _controller_prior_name(args) == "sand"


def test_plant_config_cannot_overwrite_live_estimator_prior():
    payload = {
        "terrain_preset": "clay",
        "terrain_params": {"n": 0.5},
        "path_type": "sinusoidal",
        "v_target": 5.0,
    }
    visible = _controller_visible_sim_config(
        _args(terrain_estimator=True), payload
    )
    assert "terrain_preset" not in visible
    assert "terrain_params" not in visible
    assert visible["path_type"] == "sinusoidal"
    assert visible["v_target"] == 5.0


def test_matched_static_arm_can_receive_plant_config():
    payload = {"terrain_preset": "sand", "terrain_params": {"n": 1.1}}
    assert _controller_visible_sim_config(_args(), payload) == payload


def test_legacy_reference_and_ablation_bound_are_terrain_independent():
    terrain = {"phi": 13.0}
    args = _args(
        legacy_speed_ref=True,
        terrain_independent_ay_bound=True,
    )
    assert _reference_profile_friction_angle(args, terrain) is None
    assert _mpc_build_friction_angle(args, terrain) is None


def test_default_reference_and_bound_use_controller_belief():
    terrain = {"phi": 29.0}
    args = _args()
    assert _reference_profile_friction_angle(args, terrain) == 29.0
    assert _mpc_build_friction_angle(args, terrain) == 29.0


def test_fixed_design_envelope_overrides_terrain_and_legacy_policy():
    args = _args(
        legacy_speed_ref=True,
        terrain_independent_ay_bound=True,
        reference_profile_friction_angle_deg=13.0,
        shared_ay_bound_friction_angle_deg=13.0,
    )
    assert _reference_profile_friction_angle(args, {"phi": 30.0}) == 13.0
    assert _mpc_build_friction_angle(args, {"phi": 30.0}) == 13.0


def _joint_snapshot(**updates):
    values = {
        "snapshot_version": RIG_JOINT_ACCEPTED_SNAPSHOT_VERSION,
        "update_seq": 1,
        "evidence_time_s": 10.0,
        "n": 0.5,
        "phi_deg": 13.0,
        "confidence": RIG_JOINT_MIN_PUBLICATION_CONFIDENCE,
        "observability_rank": 2,
        "observability_min_singular_value": (
            RIG_JOINT_MIN_OBSERVABILITY_SINGULAR_VALUE
        ),
        "max_boundary_mass": RIG_JOINT_BOUNDARY_MASS_LIMIT - 1.0e-6,
        "boundary_limited": False,
        "n_sigma": 0.1,
        "phi_sigma_deg": 2.0,
        "projection_wall_time_s": 0.004,
        "posterior_wall_time_s": 0.008,
        "publication_wall_time_s": 0.001,
        "update_wall_time_s": 0.013,
        "terrain_params": _joint_fixed_fallback_params(),
    }
    values.update(updates)
    return values


def test_joint_snapshot_exact_freshness_and_observability_limits_are_ready():
    ready, age, reason = _joint_snapshot_readiness(
        _joint_snapshot(),
        10.0 + RIG_JOINT_MAX_EVIDENCE_AGE_S,
    )
    assert ready
    assert abs(age - RIG_JOINT_MAX_EVIDENCE_AGE_S) < 1.0e-12
    assert reason == "ready"


def test_joint_snapshot_latch_defers_new_generation_until_next_control_tick():
    old_snapshot = _joint_snapshot(
        update_seq=1,
        evidence_time_s=10.0,
        confidence=0.3,
    )
    new_snapshot = _joint_snapshot(
        update_seq=2,
        evidence_time_s=10.1,
        confidence=0.7,
    )
    latch = _JointSnapshotLatch()
    latch.accept(old_snapshot)

    current_tick = latch.begin_control_tick()
    latch.accept(new_snapshot)

    assert latch.control_snapshot is current_tick
    assert current_tick["update_seq"] == 1
    assert current_tick["confidence"] == 0.3
    ready, age, reason = _joint_snapshot_readiness(current_tick, 10.1)
    assert ready
    assert abs(age - 0.1) < 1.0e-12
    assert reason == "ready"
    assert _joint_snapshot_control_parameters(current_tick) == (
        old_snapshot["terrain_params"]
    )

    next_tick = latch.begin_control_tick()
    assert next_tick is new_snapshot
    assert next_tick["update_seq"] == 2
    assert next_tick["confidence"] == 0.7


def test_joint_snapshot_sequence_is_zero_before_first_accepted_snapshot():
    assert _joint_snapshot_sequence(None) == 0
    assert _joint_snapshot_sequence({}) == 0
    assert _joint_snapshot_sequence({"seq": 3}) == 3
    assert _joint_snapshot_sequence({"seq": 3, "update_seq": 4}) == 4


def test_joint_snapshot_policy_fails_closed_for_each_publication_gate():
    cases = (
        (None, 10.0, "no_snapshot"),
        (
            _joint_snapshot(evidence_time_s=float("nan")),
            10.0,
            "invalid_snapshot",
        ),
        (
            _joint_snapshot(snapshot_version="unknown"),
            10.0,
            "invalid_snapshot",
        ),
        (_joint_snapshot(update_seq=0), 10.0, "invalid_snapshot"),
        (
            _joint_snapshot(observability_rank=2.5),
            10.0,
            "invalid_snapshot",
        ),
        (_joint_snapshot(confidence=1.1), 10.0, "invalid_snapshot"),
        (
            _joint_snapshot(update_wall_time_s=0.014),
            10.0,
            "invalid_snapshot",
        ),
        (
            _joint_snapshot(max_boundary_mass=-0.1),
            10.0,
            "invalid_snapshot",
        ),
        (_joint_snapshot(evidence_time_s=10.1), 10.0, "future_snapshot"),
        (
            _joint_snapshot(),
            10.0 + RIG_JOINT_MAX_EVIDENCE_AGE_S + 1.0e-4,
            "stale",
        ),
        (_joint_snapshot(observability_rank=1), 10.0, "rank"),
        (
            _joint_snapshot(
                observability_min_singular_value=(
                    RIG_JOINT_MIN_OBSERVABILITY_SINGULAR_VALUE - 1.0e-4
                )
            ),
            10.0,
            "singular_value",
        ),
        (
            _joint_snapshot(
                max_boundary_mass=RIG_JOINT_BOUNDARY_MASS_LIMIT,
                boundary_limited=True,
            ),
            10.0,
            "boundary",
        ),
        (
            _joint_snapshot(
                confidence=RIG_JOINT_MIN_PUBLICATION_CONFIDENCE - 1.0e-4
            ),
            10.0,
            "confidence",
        ),
        (
            _joint_snapshot(
                phi_deg=RIG_JOINT_CONTROL_MIN_PHI_DEG - 1.0e-4,
                terrain_params={
                    **_joint_fixed_fallback_params(),
                    "phi": RIG_JOINT_CONTROL_MIN_PHI_DEG - 1.0e-4,
                },
            ),
            10.0,
            "control_envelope",
        ),
        (
            # Sub-clay n: on the estimator's grid, below the controller's
            # validated envelope -- a labelled envelope rejection, not
            # "invalid_snapshot".
            _joint_snapshot(
                n=0.45,
                terrain_params={
                    **_joint_fixed_fallback_params(),
                    "n": 0.45,
                },
            ),
            10.0,
            "control_envelope",
        ),
    )
    for snapshot, sim_time, expected_reason in cases:
        ready, _age, reason = _joint_snapshot_readiness(snapshot, sim_time)
        assert not ready
        assert reason == expected_reason


def test_joint_fallback_is_fixed_at_control_feasible_low_grip_endpoint():
    parameters = _joint_fixed_fallback_params()
    assert set(parameters) == {"Kphi", "Kc", "n", "c", "phi", "k"}
    assert parameters["n"] == 0.5
    assert parameters["phi"] == 13.0


def test_joint_snapshot_parameters_fail_closed_on_missing_or_nonfinite_value():
    valid = _joint_snapshot()
    assert _joint_snapshot_parameters(valid) == valid["terrain_params"]
    missing = _joint_snapshot()
    del missing["terrain_params"]["phi"]
    assert _joint_snapshot_parameters(missing) is None
    nonfinite = _joint_snapshot()
    nonfinite["terrain_params"]["n"] = float("nan")
    assert _joint_snapshot_parameters(nonfinite) is None
    inconsistent = _joint_snapshot(n=0.7)
    assert _joint_snapshot_parameters(inconsistent) is None
    # n = 0.40 sits on the estimator's contract grid (the sub-clay hold
    # extension), so the snapshot itself is valid; the controller rejects it
    # through the labelled control-envelope gate instead, mirroring phi.
    sub_clay = _joint_snapshot(n=0.4)
    sub_clay["terrain_params"]["n"] = 0.4
    assert _joint_snapshot_parameters(sub_clay) is not None
    assert _joint_snapshot_control_parameters(sub_clay) is None
    outside_grid = _joint_snapshot(n=0.35)
    outside_grid["terrain_params"]["n"] = 0.35
    assert _joint_snapshot_parameters(outside_grid) is None


def test_joint_control_mapping_rejects_only_below_validated_phi_envelope():
    below = _joint_snapshot(phi_deg=6.0)
    below["terrain_params"]["phi"] = 6.0
    raw = _joint_snapshot_parameters(below)
    applied = _joint_snapshot_control_parameters(below)
    assert raw is not None
    assert raw["phi"] == 6.0
    assert applied is None

    edge = _joint_snapshot(phi_deg=RIG_JOINT_CONTROL_MIN_PHI_DEG)
    edge["terrain_params"]["phi"] = RIG_JOINT_CONTROL_MIN_PHI_DEG
    assert _joint_snapshot_control_parameters(edge) == edge["terrain_params"]
