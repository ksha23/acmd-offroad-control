"""Safety and completion contracts for the terrain-identification probe.

The probe deliberately injects steering excitation to make the soil
observable, so it must never trade path safety for identifiability. These
tests establish that it refuses to start without a clear-road certificate,
slews its target under a bound even on an irregular clock, counts dwell from
measured slip rather than elapsed command time, and hands steering back to
the nominal controller on any safety or sensor gate. They also establish that
an aborted probe cannot be recorded as a completed signed excitation.
"""

import math
import unittest
from dataclasses import replace

from simulation.control.terrain_id_probe import (
    TerrainIDProbe,
    TerrainIDProbeConfig,
    TerrainIDProbeInputs,
    TerrainIDProbePhase,
)


def safe_inputs(**overrides) -> TerrainIDProbeInputs:
    values = dict(
        requested=True,
        speed_mps=5.0,
        measured_front_alpha_rad=0.0,
        lateral_accel_mps2=0.0,
        cross_track_error_m=0.0,
        nominal_steering_rad=0.0,
        reference_curvature_inv_m=0.0,
        obstacle_feed_valid=True,
        clear_road=True,
        braking=False,
        solver_ok=True,
        safety_intervened=False,
        path_complete=False,
        latency_s=0.05,
    )
    values.update(overrides)
    return TerrainIDProbeInputs(**values)


def enter_phase(
    probe: TerrainIDProbe,
    wanted: TerrainIDProbePhase,
    *,
    measured_alpha_rad: float = 0.0,
    dt_pattern=(0.017, 0.031, 0.023, 0.049, 0.011),
    max_steps: int = 2000,
):
    """Advance with a deterministic irregular clock until ``wanted``."""

    commands = []
    for index in range(max_steps):
        dt = dt_pattern[index % len(dt_pattern)]
        command = probe.update(
            dt,
            safe_inputs(measured_front_alpha_rad=measured_alpha_rad),
        )
        commands.append((dt, command))
        if command.phase == wanted:
            return commands
        if command.aborted:
            raise AssertionError(f"probe aborted before {wanted}: {command.reason}")
    raise AssertionError(f"probe did not reach {wanted}")


class TerrainIDProbeTest(unittest.TestCase):
    def setUp(self):
        self.config = TerrainIDProbeConfig(
            arming_dwell_s=0.11,
            signed_dwell_s=0.19,
            hold_timeout_s=1.20,
            recovery_dwell_s=0.11,
            recovery_timeout_s=1.20,
            abort_timeout_s=1.20,
        )

    def test_requires_positive_clear_road_certificate(self):
        probe = TerrainIDProbe(self.config)

        for _ in range(20):
            command = probe.update(
                0.03, safe_inputs(obstacle_feed_valid=False, clear_road=True)
            )
        self.assertEqual(command.phase, TerrainIDProbePhase.IDLE)
        self.assertEqual(command.target_alpha_rad, 0.0)
        self.assertEqual(command.reason, "obstacle_feed_invalid")

        for _ in range(20):
            command = probe.update(
                0.03, safe_inputs(obstacle_feed_valid=True, clear_road=False)
            )
        self.assertEqual(command.phase, TerrainIDProbePhase.IDLE)
        self.assertEqual(command.reason, "road_not_clear")

        commands = enter_phase(probe, TerrainIDProbePhase.RAMP_POSITIVE)
        self.assertTrue(commands[-1][1].steering_override)

    def test_target_is_slew_limited_and_bounded_with_irregular_dt(self):
        probe = TerrainIDProbe(self.config)
        enter_phase(probe, TerrainIDProbePhase.RAMP_POSITIVE)
        previous = probe.target_alpha_rad
        maximum = self.config.max_target_abs_alpha_rad

        commands = enter_phase(probe, TerrainIDProbePhase.HOLD_POSITIVE)
        for dt, command in commands:
            self.assertLessEqual(abs(command.target_alpha_rad), maximum + 1e-12)
            self.assertLessEqual(
                abs(command.target_alpha_rad - previous),
                self.config.target_slew_rad_s * dt + 1e-12,
            )
            previous = command.target_alpha_rad
        self.assertAlmostEqual(previous, self.config.target_abs_alpha_rad)

    def test_positive_hold_counts_measured_slip_not_command_time(self):
        probe = TerrainIDProbe(self.config)
        enter_phase(probe, TerrainIDProbePhase.HOLD_POSITIVE)

        # The target is positive, but an unachieved +0.05 measurement must not
        # advance the maneuver even after more than the requested dwell time.
        for dt in (0.07, 0.04, 0.08, 0.06):
            command = probe.update(
                dt, safe_inputs(measured_front_alpha_rad=0.05)
            )
        self.assertEqual(command.phase, TerrainIDProbePhase.HOLD_POSITIVE)
        self.assertEqual(command.measured_dwell_s, 0.0)

        command = probe.update(
            0.08, safe_inputs(measured_front_alpha_rad=0.18)
        )
        command = probe.update(
            0.07, safe_inputs(measured_front_alpha_rad=0.18)
        )
        self.assertEqual(command.phase, TerrainIDProbePhase.HOLD_POSITIVE)
        self.assertAlmostEqual(command.measured_dwell_s, 0.15)

        command = probe.update(
            0.05, safe_inputs(measured_front_alpha_rad=0.18)
        )
        self.assertEqual(command.phase, TerrainIDProbePhase.RAMP_NEGATIVE)
        self.assertEqual(command.measured_dwell_s, 0.0)

    def test_negative_hold_rejects_wrong_sign_measurements(self):
        probe = TerrainIDProbe(self.config)
        enter_phase(probe, TerrainIDProbePhase.HOLD_POSITIVE)
        probe.update(0.20, safe_inputs(measured_front_alpha_rad=0.18))
        enter_phase(probe, TerrainIDProbePhase.HOLD_NEGATIVE)

        for dt in (0.06, 0.07, 0.08):
            command = probe.update(
                dt, safe_inputs(measured_front_alpha_rad=0.18)
            )
        self.assertEqual(command.phase, TerrainIDProbePhase.HOLD_NEGATIVE)
        self.assertEqual(command.measured_dwell_s, 0.0)

        command = probe.update(
            0.20, safe_inputs(measured_front_alpha_rad=-0.18)
        )
        self.assertEqual(command.phase, TerrainIDProbePhase.RECOVERY)

    def test_safety_intervention_aborts_and_recovers_at_bounded_slew(self):
        probe = TerrainIDProbe(self.config)
        enter_phase(probe, TerrainIDProbePhase.HOLD_POSITIVE)
        previous = probe.target_alpha_rad

        dt = 0.037
        command = probe.update(dt, safe_inputs(safety_intervened=True))
        self.assertEqual(command.phase, TerrainIDProbePhase.ABORTING)
        self.assertEqual(command.reason, "safety_intervention")
        # A safety violation must hand steering back to the nominal NMPC on
        # this sample.  The probe may keep slewing its internal target to zero
        # for deterministic logging, but it must not fight path recovery.
        self.assertFalse(command.steering_override)
        self.assertLessEqual(
            abs(command.target_alpha_rad - previous),
            self.config.target_slew_rad_s * dt + 1e-12,
        )
        previous = command.target_alpha_rad

        pattern = (0.019, 0.043, 0.027, 0.061)
        for index in range(200):
            dt = pattern[index % len(pattern)]
            command = probe.update(
                dt, safe_inputs(measured_front_alpha_rad=0.0)
            )
            self.assertFalse(command.steering_override)
            self.assertLessEqual(
                abs(command.target_alpha_rad - previous),
                self.config.target_slew_rad_s * dt + 1e-12,
            )
            previous = command.target_alpha_rad
            if command.aborted:
                break
        self.assertTrue(command.aborted)
        self.assertEqual(command.phase, TerrainIDProbePhase.ABORTED)
        self.assertEqual(command.reason, "safety_intervention")
        self.assertEqual(command.target_alpha_rad, 0.0)
        self.assertFalse(command.steering_override)

    def test_each_active_sensor_or_safety_gate_causes_abort(self):
        unsafe_cases = {
            "obstacle_feed_invalid": dict(obstacle_feed_valid=False),
            "road_not_clear": dict(clear_road=False),
            "solver_failure": dict(solver_ok=False),
            "safety_intervention": dict(safety_intervened=True),
            "path_complete": dict(path_complete=True),
            "braking": dict(braking=True),
            "speed_out_of_bounds": dict(speed_mps=3.0),
            "latency_limit": dict(latency_s=0.31),
            "cross_track_limit": dict(cross_track_error_m=0.76),
            "lateral_accel_limit": dict(lateral_accel_mps2=3.01),
            "measured_slip_limit": dict(measured_front_alpha_rad=0.301),
            "invalid_sensor_input": dict(lateral_accel_mps2=math.nan),
        }
        for reason, overrides in unsafe_cases.items():
            with self.subTest(reason=reason):
                probe = TerrainIDProbe(self.config)
                enter_phase(probe, TerrainIDProbePhase.HOLD_POSITIVE)
                command = probe.update(0.02, safe_inputs(**overrides))
                self.assertEqual(command.phase, TerrainIDProbePhase.ABORTING)
                self.assertEqual(command.reason, reason)
                self.assertFalse(command.steering_override)

    def test_full_sequence_completes_under_deterministic_irregular_clock(self):
        probe = TerrainIDProbe(self.config)
        pattern = (0.017, 0.031, 0.023, 0.049, 0.011)
        seen = set()
        previous_target = 0.0

        for index in range(2000):
            dt = pattern[index % len(pattern)]
            phase = probe.phase
            if phase == TerrainIDProbePhase.HOLD_POSITIVE:
                measured = 0.18
            elif phase == TerrainIDProbePhase.HOLD_NEGATIVE:
                measured = -0.18
            elif phase in {TerrainIDProbePhase.RECOVERY, TerrainIDProbePhase.ABORTING}:
                measured = 0.0
            else:
                # A simple one-cycle-lag actuator surrogate during ramps.
                measured = previous_target
            command = probe.update(
                dt, safe_inputs(measured_front_alpha_rad=measured)
            )
            seen.add(command.phase)
            self.assertLessEqual(
                abs(command.target_alpha_rad),
                self.config.max_target_abs_alpha_rad + 1e-12,
            )
            self.assertLessEqual(
                abs(command.target_alpha_rad - previous_target),
                self.config.target_slew_rad_s * dt + 1e-12,
            )
            previous_target = command.target_alpha_rad
            if command.completed:
                break

        self.assertTrue(command.completed)
        self.assertFalse(command.aborted)
        self.assertEqual(command.target_alpha_rad, 0.0)
        self.assertFalse(command.steering_override)
        self.assertTrue(
            {
                TerrainIDProbePhase.RAMP_POSITIVE,
                TerrainIDProbePhase.HOLD_POSITIVE,
                TerrainIDProbePhase.RAMP_NEGATIVE,
                TerrainIDProbePhase.HOLD_NEGATIVE,
                TerrainIDProbePhase.RECOVERY,
                TerrainIDProbePhase.COMPLETE,
            }.issubset(seen)
        )

    def test_recovery_is_observation_only_and_not_reclassified_by_nominal_control(self):
        probe = TerrainIDProbe(self.config)
        enter_phase(probe, TerrainIDProbePhase.HOLD_POSITIVE)
        probe.update(0.20, safe_inputs(measured_front_alpha_rad=0.18))
        enter_phase(probe, TerrainIDProbePhase.HOLD_NEGATIVE)
        command = probe.update(
            0.20, safe_inputs(measured_front_alpha_rad=-0.18)
        )
        self.assertEqual(command.phase, TerrainIDProbePhase.RECOVERY)
        self.assertFalse(command.steering_override)

        # These gates would abort an active override.  During observation-only
        # recovery they belong to the nominal controller and cannot erase a
        # successfully completed signed excitation.
        for _ in range(100):
            command = probe.update(
                0.02,
                safe_inputs(
                    measured_front_alpha_rad=0.0,
                    braking=True,
                    lateral_accel_mps2=4.0,
                    solver_ok=False,
                ),
            )
            self.assertNotEqual(command.phase, TerrainIDProbePhase.ABORTING)
            if command.completed:
                break
        self.assertTrue(command.completed)

    def test_request_withdrawal_aborts_and_terminal_state_is_latched(self):
        probe = TerrainIDProbe(self.config)
        enter_phase(probe, TerrainIDProbePhase.RAMP_POSITIVE)
        command = probe.update(0.02, safe_inputs(requested=False))
        self.assertEqual(command.phase, TerrainIDProbePhase.ABORTING)
        self.assertEqual(command.reason, "request_withdrawn")
        self.assertFalse(command.steering_override)

        for _ in range(200):
            command = probe.update(
                0.02,
                safe_inputs(requested=False, measured_front_alpha_rad=0.0),
            )
            if command.aborted:
                break
        self.assertTrue(command.aborted)
        latched = probe.update(0.5, safe_inputs())
        self.assertEqual(latched.phase, TerrainIDProbePhase.ABORTED)
        self.assertEqual(latched.reason, "request_withdrawn")

        probe.reset()
        self.assertEqual(probe.phase, TerrainIDProbePhase.IDLE)

    def test_public_inputs_have_no_oracle_terrain_or_force_fields(self):
        field_names = set(TerrainIDProbeInputs.__dataclass_fields__)
        forbidden = {
            "n_true",
            "terrain_true",
            "terrain_params",
            "tire_forces",
            "true_lateral_force",
        }
        self.assertTrue(field_names.isdisjoint(forbidden))

    def test_invalid_config_and_dt_fail_closed(self):
        with self.assertRaises(ValueError):
            replace(self.config, target_abs_alpha_rad=0.27)
        with self.assertRaises(ValueError):
            replace(self.config, target_slew_rad_s=0.0)

        probe = TerrainIDProbe(self.config)
        for dt in (0.0, -0.01, math.nan, math.inf):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                probe.update(dt, safe_inputs())


if __name__ == "__main__":
    unittest.main()
