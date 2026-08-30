"""Longitudinal-model contracts for the DOB-CBF barrier filter.

The disturbance observer and the barrier share one model of longitudinal
motion, ``a_x(alpha) = f_drag + g*alpha + hdv0``. The observer subtracts the
whole nominal at the commanded input; the barrier adds the
control-independent half back as its autonomous term and carries a state-only
authority coefficient on its decision variable. These tests pin the
properties adversarial review found unguarded, and they exercise the braking
branch, which the previous suite never entered:

* the barrier's autonomous term is recoverable from the logged constraint
  rows, is zero at coast, and under a brake command uses the unscaled
  envelope -- so a shortfall in modelled brake authority surfaces as a
  phantom *forward* push (conservative) rather than free deceleration
  (permissive);
* the authority coefficient is a function of state alone: identical for a
  brake command and a throttle command at the same state;
* the terrain speed cap bounds throttle from above and can never strip the
  driver's braking or demand throttle past the limit;
* the surrogate fallbacks are consistent between the two halves of the
  model, and resistance opposes the direction of motion.
"""

import contextlib
import io
import unittest

from simulation.safety.dob_cbf import CBFSafetyFilter


VEHICLE_PARAMS = {
    "M": 2573.0,
    "Izz": 3570.0,
    "Lf": 1.593,
    "Lr": 1.709,
    "h_cg": 0.65,
    "T": 1.8194,
}

# Zero-slip resistance the stub reports, matching the deployed surrogate's
# magnitude on dirt.
DRAG_AX = -0.699

# Slip response, deliberately asymmetric so a sign error in the authority
# query cannot cancel: drive slip returns 12 m/s^2 per unit slip, brake slip
# 20. The state-only authority must come from the drive side.
DRIVE_SLOPE = 12.0
BRAKE_SLOPE = 20.0

W_LONG = 0.15
FORWARD_BIAS = 1.5


class _AsymmetricTire:
    """Surrogate stand-in: resistance at zero slip, asymmetric slip response.

    The lateral channel returns a real force. A zero-Fy stub drives the
    filter's traction speed cap to its 2 m/s floor, silently activating the
    speed row in every at-speed test and coupling it into contracts that are
    about the observer.
    """

    def predict_numeric(self, alpha, Fz, u, kappa, **kwargs):
        slope = DRIVE_SLOPE if kappa >= 0 else BRAKE_SLOPE
        ax = DRAG_AX + slope * float(kappa)
        # Fx_total = 2 * (Fx_f + Fx_r) = M * ax
        return (ax * VEHICLE_PARAMS["M"] / 4.0, 2000.0)


class _BaselineFailsTire(_AsymmetricTire):
    """Raises on the zero-slip query only, so the baseline comes back None
    while the slipped query would succeed."""

    def predict_numeric(self, alpha, Fz, u, kappa, **kwargs):
        if kappa == 0.0:
            raise RuntimeError("baseline query unavailable")
        return super().predict_numeric(alpha, Fz, u, kappa, **kwargs)


def make_filter(tire=None):
    cbf = CBFSafetyFilter(vehicle_params=VEHICLE_PARAMS,
                          nn_casadi=tire if tire is not None else _AsymmetricTire(),
                          obstacle_buffer=0.5, control_dt=0.02)
    cbf._csv_writer = None
    cbf._obs_csv_writer = None
    cbf._beta = 0.0
    return cbf


def settle(speed, obstacle_gap=12.0, throttle=0.0, brake=0.0,
           roughness=0.0, steps=200, cbf=None, obstacles=None):
    """Hold one state until the observer settles; return (result, filter).

    The state is dead-ahead of the obstacle with zero steering, lateral
    velocity, and yaw rate, so the barrier's cross terms vanish and the
    logged ``h_ddot_auto`` row is exactly ``2*w_long*d_along*a_x_auto``.
    """
    if cbf is None:
        cbf = make_filter()
    state = dict(x=0.0, y=0.0, psi=0.0, u=speed, v=0.0, omega=0.0, delta=0.0)
    if obstacles is None:
        obstacles = [(obstacle_gap, 0.0, 0.828)]
    result = None
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(steps):
            result = cbf.filter(desired_steering=0.0, desired_throttle=throttle,
                                desired_brake=brake, vehicle_state=state,
                                obstacles=obstacles,
                                terrain_roughness=roughness)
    return result, cbf


def logged_a_x_auto(cbf):
    """Recover the barrier's autonomous acceleration from the logged row."""
    assert cbf._pending_obs_logs, "no obstacle row was logged"
    row = cbf._pending_obs_logs[0]
    d_along, h_ddot_auto = row[5], row[10]
    return h_ddot_auto / (2.0 * W_LONG * d_along)


def logged_a_alpha(cbf):
    assert cbf._pending_obs_logs, "no obstacle row was logged"
    return cbf._pending_obs_logs[0][12]


class TestAutonomousTerm(unittest.TestCase):
    """The barrier's zero-input acceleration, read back from the constraint."""

    def test_coasting_at_speed_credits_no_free_acceleration(self):
        """At coast the drag is real and the observer cancels it exactly;
        dropping f_drag from the barrier term, or flipping its sign, leaves
        a residual of |DRAG_AX| or more."""
        _, cbf = settle(speed=3.0, throttle=0.0, brake=0.0)
        self.assertAlmostEqual(logged_a_x_auto(cbf), 0.0, delta=0.05)

    def test_braking_uses_the_one_g_envelope(self):
        """The state is held, so measured acceleration is zero and the
        decomposition's brake coefficient is read back directly from the
        executed full brake. A secant here would book the difference between
        itself and true braking as free deceleration; the 1 g bound books the
        whole shortfall as a forward phantom instead."""
        _, cbf = settle(speed=3.0, brake=1.0)
        self.assertAlmostEqual(
            logged_a_x_auto(cbf), CBFSafetyFilter._BRAKE_NOMINAL_ENVELOPE,
            delta=0.05)

    def test_brake_envelope_ignores_the_grip_scaled_actuator_model(self):
        """update_terrain shrinks max_decel with estimated grip; the observer
        bound must not follow it down, or a grip underestimate flips the bias
        into crediting free deceleration."""
        cbf = make_filter()
        cbf.update_terrain({"Kphi": 500000.0, "Kc": 1000.0, "n": 0.45,
                            "c": 500.0, "phi": 13.0, "k": 0.01})
        self.assertLess(abs(cbf.max_decel),
                        CBFSafetyFilter._BRAKE_NOMINAL_ENVELOPE)
        _, cbf = settle(speed=3.0, brake=1.0, cbf=cbf)
        self.assertAlmostEqual(
            logged_a_x_auto(cbf), CBFSafetyFilter._BRAKE_NOMINAL_ENVELOPE,
            delta=0.05)

    def test_braking_never_credits_free_deceleration(self):
        """Whatever the coefficients, the conservative direction is a
        non-negative autonomous term while a brake is commanded and no
        deceleration is measured."""
        for brake in (0.3, 0.7, 1.0):
            with self.subTest(brake=brake):
                _, cbf = settle(speed=3.0, brake=brake)
                self.assertGreaterEqual(logged_a_x_auto(cbf), -0.05)

    def test_throttle_adds_no_autonomous_credit(self):
        """Under executed throttle the nominal coefficient is zero, so the
        autonomous term equals the measured acceleration -- an overstatement
        of what alpha = 0 would sustain, which is the conservative direction.
        A secant here was measured permissive at speed, where the
        power-limited plant delivers less than tire capacity."""
        _, cbf = settle(speed=3.0, throttle=1.0)
        self.assertAlmostEqual(logged_a_x_auto(cbf), 0.0, delta=0.05)

    def test_nominal_uses_the_executed_alpha_not_the_desired(self):
        """During an intervention the desired and executed commands differ by
        up to the full pedal range; a nominal at the desired command books
        the gap as free deceleration and deactivates the constraint being
        enforced. Held at the closing-fast state, the filter brakes against a
        full-throttle driver; the autonomous term must reflect the executed
        brake (positive phantom), not the desired throttle."""
        _, cbf = settle(speed=7.0, obstacle_gap=8.35, throttle=1.0, steps=400)
        self.assertGreater(logged_a_x_auto(cbf), 1.0)


class TestControlAuthority(unittest.TestCase):
    """State-only, drive-slip-derived, envelope-capped."""

    def test_authority_is_identical_for_brake_and_throttle_commands(self):
        _, braked = settle(speed=3.0, brake=1.0)
        _, driven = settle(speed=3.0, throttle=1.0)
        self.assertAlmostEqual(logged_a_alpha(braked), logged_a_alpha(driven),
                               places=6)

    def test_authority_comes_from_the_drive_slip_side(self):
        """The asymmetric stub returns 1.8 at +0.15 slip and 3.0 at -0.15;
        a sign error in the query is therefore visible."""
        cbf = make_filter()
        baseline = cbf._compute_nn_tire_forces(3.0, 0.0, 0.0, 0.0)
        g = cbf._longitudinal_authority(3.0, 0.0, 0.0, 0.0, baseline)
        self.assertAlmostEqual(
            g, DRIVE_SLOPE * CBFSafetyFilter._AUTHORITY_SLIP_RATIO, places=3)

    def test_authority_is_independent_of_resistance_magnitude(self):
        class _TenXDrag(_AsymmetricTire):
            def predict_numeric(self, alpha, Fz, u, kappa, **kwargs):
                slope = DRIVE_SLOPE if kappa >= 0 else BRAKE_SLOPE
                ax = 10.0 * DRAG_AX + slope * float(kappa)
                return (ax * VEHICLE_PARAMS["M"] / 4.0, 0.0)

        for tire in (_AsymmetricTire(), _TenXDrag()):
            cbf = make_filter(tire)
            baseline = cbf._compute_nn_tire_forces(3.0, 0.0, 0.0, 0.0)
            g = cbf._longitudinal_authority(3.0, 0.0, 0.0, 0.0, baseline)
            self.assertAlmostEqual(
                g, DRIVE_SLOPE * CBFSafetyFilter._AUTHORITY_SLIP_RATIO,
                places=3)

    def test_failed_baseline_falls_back_to_the_envelope(self):
        """A lone slipped query must not masquerade as a difference."""
        cbf = make_filter(_BaselineFailsTire())
        self.assertEqual(
            cbf._longitudinal_authority(3.0, 0.0, 0.0, 0.0, None),
            cbf.max_accel)
        # And the full filter step survives the failing baseline.
        result, _ = settle(speed=3.0, throttle=0.5, cbf=cbf)
        self.assertIsNotNone(result)

    def test_no_traction_response_is_floored(self):
        class _NoTraction(_AsymmetricTire):
            def predict_numeric(self, alpha, Fz, u, kappa, **kwargs):
                return (DRAG_AX * VEHICLE_PARAMS["M"] / 4.0, 0.0)

        cbf = make_filter(_NoTraction())
        baseline = cbf._compute_nn_tire_forces(3.0, 0.0, 0.0, 0.0)
        g = cbf._longitudinal_authority(3.0, 0.0, 0.0, 0.0, baseline)
        self.assertAlmostEqual(g, 0.3, places=6)


class TestTerrainSpeedCap(unittest.TestCase):
    """The cap bounds throttle from above; braking is never constrained.

    The probing speed sits just *below* the cap (v_max is floored at 2.0 for
    this roughness). Over the cap both a correct and a sign-reverted row
    collapse into the infeasible-QP emergency brake, which masks the sign;
    just below it the two are distinguishable -- the correct row caps
    throttle and leaves braking free, the reverted one does the opposite.
    """

    def test_full_brake_near_the_limit_is_preserved(self):
        result, _ = settle(speed=1.9, brake=1.0, roughness=50.0,
                           obstacles=[])
        self.assertGreaterEqual(result.braking, 0.9)
        self.assertEqual(result.throttle, 0.0)

    def test_full_throttle_near_the_limit_is_cut(self):
        result, _ = settle(speed=1.9, throttle=1.0, roughness=50.0,
                           obstacles=[])
        self.assertLess(result.throttle, 0.15)

    def test_full_brake_over_the_limit_is_preserved(self):
        result, _ = settle(speed=2.5, brake=1.0, roughness=50.0,
                           obstacles=[])
        self.assertGreaterEqual(result.braking, 0.9)
        self.assertEqual(result.throttle, 0.0)

    def test_throttle_below_the_cap_window_passes(self):
        result, _ = settle(speed=1.0, throttle=0.6, roughness=50.0,
                           obstacles=[])
        self.assertGreater(result.throttle, 0.5)

    def test_overspeed_arrival_at_full_throttle_keeps_the_qp_feasible(self):
        """A step arriving over the cap with executed alpha at +1 cannot
        satisfy any brake-demanding bound within the alpha rate limit; the
        row must shed alpha at the rate limit instead of going infeasible
        and tripping the emergency fallback, which discards the obstacle
        constraints. The fallback would emit full brake; the rate-limited
        path emits reduced throttle."""
        cbf = make_filter()
        cbf._alpha = 1.0
        state = dict(x=0.0, y=0.0, psi=0.0, u=2.5, v=0.0, omega=0.0,
                     delta=0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            result = cbf.filter(0.0, 1.0, 0.0, state,
                                obstacles=[(12.0, 0.0, 0.828)],
                                terrain_roughness=50.0)
        self.assertEqual(result.braking, 0.0)
        self.assertGreater(result.throttle, 0.5)
        self.assertLess(result.throttle, 0.95)

    def test_overspeed_from_coast_brakes_at_the_rate_limit_not_the_fallback(self):
        """Far over the cap, an unfloored bound demands more braking than the
        rate rows allow and the QP collapses into the full-brake fallback.
        The floored row demands exactly what the rate limit can deliver this
        step."""
        cbf = make_filter()
        state = dict(x=0.0, y=0.0, psi=0.0, u=4.0, v=0.0, omega=0.0,
                     delta=0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            result = cbf.filter(0.0, 0.0, 1.0, state,
                                obstacles=[(12.0, 0.0, 0.828)],
                                terrain_roughness=50.0)
        self.assertGreater(result.braking, 0.05)
        self.assertLess(result.braking, 0.5)


class TestResistance(unittest.TestCase):
    def test_zero_at_rest_and_full_above_the_clamp(self):
        cbf = make_filter()
        forces = cbf._compute_nn_tire_forces(0.5, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(cbf._resistive_acceleration(forces, 0.0), 0.0)
        self.assertAlmostEqual(
            cbf._resistive_acceleration(forces, 0.25), DRAG_AX / 2.0, places=6)
        for speed in (cbf._NN_QUERY_SPEED_FLOOR, 1.0, 7.0):
            self.assertAlmostEqual(
                cbf._resistive_acceleration(forces, speed), DRAG_AX, places=6)

    def test_resistance_opposes_backward_motion(self):
        cbf = make_filter()
        forces = cbf._compute_nn_tire_forces(0.5, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(
            cbf._resistive_acceleration(forces, -2.0), -DRAG_AX, places=6)

    def test_no_surrogate_leaves_the_model_defined(self):
        cbf = CBFSafetyFilter(vehicle_params=VEHICLE_PARAMS, nn_casadi=None)
        cbf._csv_writer = None
        cbf._obs_csv_writer = None
        self.assertEqual(cbf._resistive_acceleration(None, 3.0), 0.0)


class TestStandstill(unittest.TestCase):
    """The deadlock contracts from the original fix, unchanged."""

    def test_observer_reports_no_disturbance_at_rest_without_throttle(self):
        _, cbf = settle(speed=0.0, obstacle_gap=60.0, steps=400)
        hdv0 = cbf.dob.p0v + cbf.dob.a_v * 0.1
        self.assertAlmostEqual(hdv0, 0.0, places=3)

    def test_stopped_vehicle_with_margin_may_pull_away(self):
        """At the edge of the standoff the barrier meters the launch (the
        constraint caps alpha at psi0/|A_alpha|) but must admit real
        throttle with no brake; the original deadlock emitted ~0.11 throttle
        against a standing brake. Well clear of the standoff the launch must
        be unmetered."""
        result, _ = settle(speed=0.0, obstacle_gap=8.35, throttle=1.0,
                           steps=400)
        self.assertGreater(result.throttle, 0.3)
        self.assertEqual(result.braking, 0.0)
        clear, _ = settle(speed=0.0, obstacle_gap=15.0, throttle=1.0,
                          steps=400)
        self.assertGreater(clear.throttle, 0.9)
        self.assertEqual(clear.braking, 0.0)

    def test_stopped_vehicle_without_margin_is_still_refused(self):
        for gap in (2.9, 2.0):
            with self.subTest(gap=gap):
                result, _ = settle(speed=0.0, obstacle_gap=gap, throttle=1.0,
                                   steps=400)
                self.assertEqual(result.throttle, 0.0)
                self.assertGreater(result.braking, 0.5)

    def test_closing_fast_on_an_obstacle_still_brakes(self):
        result, _ = settle(speed=7.0, obstacle_gap=8.35, throttle=1.0,
                           steps=400)
        self.assertEqual(result.throttle, 0.0)
        self.assertGreater(result.braking, 0.5)


if __name__ == "__main__":
    unittest.main()
