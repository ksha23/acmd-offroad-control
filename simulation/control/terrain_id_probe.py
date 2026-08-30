"""State machine for bounded terrain-identification steering probes.

The probe commands a signed *measured front slip angle* rather than adding a
waveform to the normalized steering command, so that its excitation is
specified in the quantity the terrain estimator observes.  Conversion to a
steering command belongs to the controller, which forms the road-wheel target
as::

    delta_target = atan2(v + Lf * yaw_rate, u) + command.target_alpha_rad

This module takes no simulation or terrain-truth input.  ``requested`` is
raised by an estimator-side uncertainty or information test, or by an explicit
experiment trigger, and every dwell transition is decided from the measured
slip angle supplied in :class:`TerrainIDProbeInputs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class TerrainIDProbePhase(str, Enum):
    """Phases of one positive/negative terrain-identification maneuver."""

    IDLE = "idle"
    ARMING = "arming"
    RAMP_POSITIVE = "ramp_positive"
    HOLD_POSITIVE = "hold_positive"
    RAMP_NEGATIVE = "ramp_negative"
    HOLD_NEGATIVE = "hold_negative"
    RECOVERY = "recovery"
    ABORTING = "aborting"
    COMPLETE = "complete"
    ABORTED = "aborted"


@dataclass(frozen=True)
class TerrainIDProbeConfig:
    """Tunable limits for :class:`TerrainIDProbe`.

    The defaults describe a conservative clear-road probe at approximately
    5 m/s.  The target is inside the useful measured-slip band and below the
    hard sensor guard.  Times are accumulated from caller-supplied ``dt`` so
    the state machine is deterministic under an irregular controller clock.
    """

    target_abs_alpha_rad: float = 0.10
    # Dwell is accumulated over a band of achieved slip rather than at the
    # exact target, since holding out for the exact value lets the cross-track
    # displacement grow for no additional information.  The default lower edge
    # is 70% of the default target and stays inside the envelope the tire
    # surrogate was supervised on.
    informative_alpha_min_rad: float = 0.07
    informative_alpha_max_rad: float = 0.25
    max_target_abs_alpha_rad: float = 0.25
    max_measured_abs_alpha_rad: float = 0.30
    target_slew_rad_s: float = 0.40

    arming_dwell_s: float = 0.50
    signed_dwell_s: float = 0.15
    hold_timeout_s: float = 2.50
    ramp_timeout_s: float = 2.50
    recovery_alpha_tol_rad: float = 0.05
    recovery_dwell_s: float = 0.20
    recovery_timeout_s: float = 3.00
    abort_timeout_s: float = 3.00

    min_speed_mps: float = 3.50
    max_speed_mps: float = 5.50
    max_arming_abs_lateral_accel_mps2: float = 1.50
    max_abort_abs_lateral_accel_mps2: float = 3.00
    max_arming_abs_cross_track_error_m: float = 0.30
    max_abort_abs_cross_track_error_m: float = 0.75
    max_arming_abs_nominal_steering_rad: float = 0.10
    max_arming_abs_reference_curvature_inv_m: float = 0.015
    max_latency_s: float = 0.30

    def __post_init__(self) -> None:
        finite_values = (
            self.target_abs_alpha_rad,
            self.informative_alpha_min_rad,
            self.informative_alpha_max_rad,
            self.max_target_abs_alpha_rad,
            self.max_measured_abs_alpha_rad,
            self.target_slew_rad_s,
            self.arming_dwell_s,
            self.signed_dwell_s,
            self.hold_timeout_s,
            self.ramp_timeout_s,
            self.recovery_alpha_tol_rad,
            self.recovery_dwell_s,
            self.recovery_timeout_s,
            self.abort_timeout_s,
            self.min_speed_mps,
            self.max_speed_mps,
            self.max_arming_abs_lateral_accel_mps2,
            self.max_abort_abs_lateral_accel_mps2,
            self.max_arming_abs_cross_track_error_m,
            self.max_abort_abs_cross_track_error_m,
            self.max_arming_abs_nominal_steering_rad,
            self.max_arming_abs_reference_curvature_inv_m,
            self.max_latency_s,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("terrain probe configuration must be finite")
        if not (
            0.0 < self.informative_alpha_min_rad
            <= self.target_abs_alpha_rad
            <= self.informative_alpha_max_rad
            <= self.max_target_abs_alpha_rad
            < self.max_measured_abs_alpha_rad
        ):
            raise ValueError(
                "expected 0 < informative_min <= target <= informative_max "
                "<= max_target < max_measured"
            )
        if self.target_slew_rad_s <= 0.0:
            raise ValueError("target_slew_rad_s must be positive")
        if min(
            self.arming_dwell_s,
            self.signed_dwell_s,
            self.hold_timeout_s,
            self.ramp_timeout_s,
            self.recovery_alpha_tol_rad,
            self.recovery_dwell_s,
            self.recovery_timeout_s,
            self.abort_timeout_s,
        ) <= 0.0:
            raise ValueError("probe dwell, timeout, and recovery limits must be positive")
        if self.hold_timeout_s < self.signed_dwell_s:
            raise ValueError("hold_timeout_s must cover signed_dwell_s")
        if self.ramp_timeout_s < 2.0 * self.target_abs_alpha_rad / self.target_slew_rad_s:
            raise ValueError("ramp_timeout_s cannot cover the positive-to-negative ramp")
        minimum_return_s = (
            self.max_target_abs_alpha_rad / self.target_slew_rad_s
            + self.recovery_dwell_s
        )
        if self.recovery_timeout_s < minimum_return_s:
            raise ValueError("recovery_timeout_s must cover a bounded return and dwell")
        if self.abort_timeout_s < minimum_return_s:
            raise ValueError("abort_timeout_s must cover a bounded return and dwell")
        if not 0.0 <= self.min_speed_mps < self.max_speed_mps:
            raise ValueError("expected 0 <= min_speed_mps < max_speed_mps")
        if not (
            0.0 < self.max_arming_abs_lateral_accel_mps2
            <= self.max_abort_abs_lateral_accel_mps2
        ):
            raise ValueError("lateral-acceleration arm limit must not exceed abort limit")
        if not (
            0.0 < self.max_arming_abs_cross_track_error_m
            <= self.max_abort_abs_cross_track_error_m
        ):
            raise ValueError("cross-track arm limit must not exceed abort limit")
        if self.max_arming_abs_nominal_steering_rad <= 0.0:
            raise ValueError("nominal-steering arm limit must be positive")
        if self.max_arming_abs_reference_curvature_inv_m <= 0.0:
            raise ValueError("reference-curvature arm limit must be positive")
        if self.max_latency_s < 0.0:
            raise ValueError("max_latency_s must be nonnegative")


@dataclass(frozen=True)
class TerrainIDProbeInputs:
    """Deployable measurements and safety state consumed by the probe.

    ``obstacle_feed_valid`` is carried separately from ``clear_road`` because
    an empty or failed perception feed is not a positive certificate that the
    road ahead is clear.  No ground-truth soil parameter or tire force appears
    in this interface.
    """

    requested: bool
    speed_mps: float
    measured_front_alpha_rad: float
    lateral_accel_mps2: float
    cross_track_error_m: float
    nominal_steering_rad: float
    reference_curvature_inv_m: float
    obstacle_feed_valid: bool
    clear_road: bool
    braking: bool = False
    solver_ok: bool = True
    safety_intervened: bool = False
    path_complete: bool = False
    latency_s: float = 0.0


@dataclass(frozen=True)
class TerrainIDProbeCommand:
    """State-machine output consumed by the controller each control step."""

    phase: TerrainIDProbePhase
    target_alpha_rad: float
    steering_override: bool
    arm_elapsed_s: float
    measured_dwell_s: float
    completed: bool
    aborted: bool
    reason: str = ""


_TERMINAL_PHASES = {
    TerrainIDProbePhase.COMPLETE,
    TerrainIDProbePhase.ABORTED,
}
_COMMAND_PHASES = {
    TerrainIDProbePhase.RAMP_POSITIVE,
    TerrainIDProbePhase.HOLD_POSITIVE,
    TerrainIDProbePhase.RAMP_NEGATIVE,
    TerrainIDProbePhase.HOLD_NEGATIVE,
}


class TerrainIDProbe:
    """Deterministic achieved-slip probe with explicit abort and recovery."""

    def __init__(self, config: TerrainIDProbeConfig | None = None):
        self.config = config or TerrainIDProbeConfig()
        self.reset()

    @property
    def phase(self) -> TerrainIDProbePhase:
        return self._phase

    @property
    def target_alpha_rad(self) -> float:
        return self._target_alpha_rad

    def reset(self) -> None:
        """Return the state machine to IDLE, ready for a new maneuver."""

        self._phase = TerrainIDProbePhase.IDLE
        self._target_alpha_rad = 0.0
        self._arm_elapsed_s = 0.0
        self._phase_elapsed_s = 0.0
        self._measured_dwell_s = 0.0
        self._recovery_dwell_s = 0.0
        self._reason = ""

    @staticmethod
    def _approach(value: float, target: float, max_change: float) -> float:
        if value < target:
            return min(value + max_change, target)
        return max(value - max_change, target)

    @staticmethod
    def _finite_inputs(inputs: TerrainIDProbeInputs) -> bool:
        return all(
            math.isfinite(value)
            for value in (
                inputs.speed_mps,
                inputs.measured_front_alpha_rad,
                inputs.lateral_accel_mps2,
                inputs.cross_track_error_m,
                inputs.nominal_steering_rad,
                inputs.reference_curvature_inv_m,
                inputs.latency_s,
            )
        )

    def _active_violation(self, inputs: TerrainIDProbeInputs) -> str:
        cfg = self.config
        if not self._finite_inputs(inputs):
            return "invalid_sensor_input"
        if not inputs.obstacle_feed_valid:
            return "obstacle_feed_invalid"
        if not inputs.clear_road:
            return "road_not_clear"
        if not inputs.solver_ok:
            return "solver_failure"
        if inputs.safety_intervened:
            return "safety_intervention"
        if inputs.path_complete:
            return "path_complete"
        if inputs.braking:
            return "braking"
        if not cfg.min_speed_mps <= inputs.speed_mps <= cfg.max_speed_mps:
            return "speed_out_of_bounds"
        if not 0.0 <= inputs.latency_s <= cfg.max_latency_s:
            return "latency_limit"
        if abs(inputs.cross_track_error_m) > cfg.max_abort_abs_cross_track_error_m:
            return "cross_track_limit"
        if abs(inputs.lateral_accel_mps2) > cfg.max_abort_abs_lateral_accel_mps2:
            return "lateral_accel_limit"
        if abs(inputs.measured_front_alpha_rad) > cfg.max_measured_abs_alpha_rad:
            return "measured_slip_limit"
        return ""

    def _arming_block(self, inputs: TerrainIDProbeInputs) -> str:
        violation = self._active_violation(inputs)
        if violation:
            return violation
        cfg = self.config
        if abs(inputs.cross_track_error_m) > cfg.max_arming_abs_cross_track_error_m:
            return "cross_track_not_settled"
        if abs(inputs.lateral_accel_mps2) > cfg.max_arming_abs_lateral_accel_mps2:
            return "lateral_accel_not_settled"
        if abs(inputs.nominal_steering_rad) > cfg.max_arming_abs_nominal_steering_rad:
            return "nominal_steering_not_settled"
        if (
            abs(inputs.reference_curvature_inv_m)
            > cfg.max_arming_abs_reference_curvature_inv_m
        ):
            return "reference_not_straight"
        return ""

    def _transition(self, phase: TerrainIDProbePhase) -> None:
        self._phase = phase
        self._phase_elapsed_s = 0.0
        self._measured_dwell_s = 0.0
        self._recovery_dwell_s = 0.0

    def _begin_abort(self, reason: str) -> None:
        if self._phase not in {TerrainIDProbePhase.ABORTING, TerrainIDProbePhase.ABORTED}:
            self._reason = reason
            self._transition(TerrainIDProbePhase.ABORTING)

    def _bounded_target(self, value: float) -> float:
        bound = self.config.max_target_abs_alpha_rad
        return max(-bound, min(bound, value))

    def _in_signed_informative_band(
        self, measured_alpha_rad: float, sign: float
    ) -> bool:
        signed_alpha = sign * measured_alpha_rad
        return (
            self.config.informative_alpha_min_rad
            <= signed_alpha
            <= self.config.informative_alpha_max_rad
        )

    def _step_recovery(
        self,
        dt: float,
        measured_alpha_rad: float,
        *,
        terminal_phase: TerrainIDProbePhase,
        timeout_s: float,
    ) -> None:
        cfg = self.config
        self._phase_elapsed_s += dt
        self._target_alpha_rad = self._approach(
            self._target_alpha_rad, 0.0, cfg.target_slew_rad_s * dt
        )
        recovered = (
            math.isfinite(measured_alpha_rad)
            and abs(self._target_alpha_rad) <= 1.0e-12
            and abs(measured_alpha_rad) <= cfg.recovery_alpha_tol_rad
        )
        if recovered:
            self._recovery_dwell_s += dt
        else:
            # Recovery must be continuous; a new excursion restarts settling.
            self._recovery_dwell_s = 0.0
        if self._recovery_dwell_s >= cfg.recovery_dwell_s:
            self._target_alpha_rad = 0.0
            self._transition(terminal_phase)
        elif self._phase_elapsed_s >= timeout_s:
            self._target_alpha_rad = 0.0
            if terminal_phase == TerrainIDProbePhase.COMPLETE:
                self._reason = "recovery_timeout"
                self._transition(TerrainIDProbePhase.ABORTED)
            else:
                self._transition(terminal_phase)

    def _command(self, reason: str | None = None) -> TerrainIDProbeCommand:
        target = self._bounded_target(self._target_alpha_rad)
        self._target_alpha_rad = target
        return TerrainIDProbeCommand(
            phase=self._phase,
            target_alpha_rad=target,
            steering_override=self._phase in _COMMAND_PHASES,
            arm_elapsed_s=self._arm_elapsed_s,
            measured_dwell_s=self._measured_dwell_s,
            completed=self._phase == TerrainIDProbePhase.COMPLETE,
            aborted=self._phase == TerrainIDProbePhase.ABORTED,
            reason=self._reason if reason is None else reason,
        )

    def update(self, dt: float, inputs: TerrainIDProbeInputs) -> TerrainIDProbeCommand:
        """Advance the probe by ``dt`` seconds.

        Terminal states are latched, so a caller must call :meth:`reset`
        explicitly after ``COMPLETE`` or ``ABORTED``.  Latching keeps the
        experiment log deterministic and prevents a persistent uncertainty
        flag from immediately triggering a second maneuver.
        """

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self._phase in _TERMINAL_PHASES:
            return self._command()

        if self._phase in {TerrainIDProbePhase.IDLE, TerrainIDProbePhase.ARMING}:
            if not inputs.requested:
                self._phase = TerrainIDProbePhase.IDLE
                self._arm_elapsed_s = 0.0
                self._reason = ""
                return self._command()
            blocked = self._arming_block(inputs)
            if blocked:
                self._phase = TerrainIDProbePhase.IDLE
                self._arm_elapsed_s = 0.0
                return self._command(blocked)
            self._phase = TerrainIDProbePhase.ARMING
            self._arm_elapsed_s += dt
            if self._arm_elapsed_s >= self.config.arming_dwell_s:
                self._reason = ""
                self._transition(TerrainIDProbePhase.RAMP_POSITIVE)
            return self._command()

        # RECOVERY does not own steering: the common rate limiter and the
        # nominal NMPC are already returning the vehicle to the path.  Braking,
        # solver, or lateral-acceleration activity from that nominal controller
        # must therefore not retroactively invalidate two signed holds that
        # completed safely.  Recovery certifies that the measured slip has
        # settled, and its own timeout still fails closed.
        if self._phase not in {
            TerrainIDProbePhase.ABORTING,
            TerrainIDProbePhase.RECOVERY,
        }:
            if not inputs.requested:
                self._begin_abort("request_withdrawn")
            else:
                violation = self._active_violation(inputs)
                if violation:
                    self._begin_abort(violation)

        cfg = self.config
        target = cfg.target_abs_alpha_rad
        # Capture the active phase once per sample.  A transition taken at the
        # end of this sample takes effect on the next one, so a single dt is
        # never charged to two phases, for instance to HOLD_NEGATIVE and
        # RECOVERY both.
        phase_to_step = self._phase

        if phase_to_step == TerrainIDProbePhase.RAMP_POSITIVE:
            self._phase_elapsed_s += dt
            self._target_alpha_rad = self._approach(
                self._target_alpha_rad, target, cfg.target_slew_rad_s * dt
            )
            if self._target_alpha_rad >= target - 1.0e-12:
                self._target_alpha_rad = target
                self._transition(TerrainIDProbePhase.HOLD_POSITIVE)
            elif self._phase_elapsed_s >= cfg.ramp_timeout_s:
                self._begin_abort("positive_ramp_timeout")

        elif phase_to_step == TerrainIDProbePhase.HOLD_POSITIVE:
            self._phase_elapsed_s += dt
            self._target_alpha_rad = target
            if self._in_signed_informative_band(inputs.measured_front_alpha_rad, 1.0):
                self._measured_dwell_s += dt
            if self._measured_dwell_s >= cfg.signed_dwell_s:
                self._transition(TerrainIDProbePhase.RAMP_NEGATIVE)
            elif self._phase_elapsed_s >= cfg.hold_timeout_s:
                self._begin_abort("positive_dwell_timeout")

        elif phase_to_step == TerrainIDProbePhase.RAMP_NEGATIVE:
            self._phase_elapsed_s += dt
            self._target_alpha_rad = self._approach(
                self._target_alpha_rad, -target, cfg.target_slew_rad_s * dt
            )
            if self._target_alpha_rad <= -target + 1.0e-12:
                self._target_alpha_rad = -target
                self._transition(TerrainIDProbePhase.HOLD_NEGATIVE)
            elif self._phase_elapsed_s >= cfg.ramp_timeout_s:
                self._begin_abort("negative_ramp_timeout")

        elif phase_to_step == TerrainIDProbePhase.HOLD_NEGATIVE:
            self._phase_elapsed_s += dt
            self._target_alpha_rad = -target
            if self._in_signed_informative_band(inputs.measured_front_alpha_rad, -1.0):
                self._measured_dwell_s += dt
            if self._measured_dwell_s >= cfg.signed_dwell_s:
                self._transition(TerrainIDProbePhase.RECOVERY)
            elif self._phase_elapsed_s >= cfg.hold_timeout_s:
                self._begin_abort("negative_dwell_timeout")

        elif phase_to_step == TerrainIDProbePhase.RECOVERY:
            self._step_recovery(
                dt,
                inputs.measured_front_alpha_rad,
                terminal_phase=TerrainIDProbePhase.COMPLETE,
                timeout_s=cfg.recovery_timeout_s,
            )
        elif phase_to_step == TerrainIDProbePhase.ABORTING:
            self._step_recovery(
                dt,
                inputs.measured_front_alpha_rad,
                terminal_phase=TerrainIDProbePhase.ABORTED,
                timeout_s=cfg.abort_timeout_s,
            )

        return self._command()
