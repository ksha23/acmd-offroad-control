"""Model Predictive Safety Filter (MPSF), implemented with acados.

A predictive, least-restrictive safety filter and an interchangeable
alternative to the pointwise barrier QP. Each step it solves a short-horizon
optimal control problem that keeps the whole predicted trajectory clear of
obstacles while deviating as little as possible from the incoming command:

    min_U  sum_k || u_k - u_operator ||_W^2   (+ soft terminal-speed penalty)
    s.t.   kinematic-bicycle + grip-limited dynamics,
           (X_k - o_j)^2 + (Y_k - o_j)^2 >= safe_r_j^2   for all k, j   (slacked),
           |steer|, |alpha| <= 1.

Posing safety as trajectory-level feasibility rather than as an instantaneous
inequality admits the command essentially untouched until no safe future
exists. It also removes the sensitivity of a pointwise barrier to its class-K
gain, which sets how sharply the filter reacts and whose appropriate value
depends on the available grip.

The OCP is self-contained and posed in the filter's own normalized command
space, ``[steer, alpha]``, using the same normalized-command to (road-wheel
angle, grip-scaled longitudinal acceleration) bridge as CBFSafetyFilter. It
does not reuse the tracking NMPC's internal state or its (delta, jerk) control
space, so the two remain independent and the filter stays interchangeable. The
obstacle constraints are slacked under a large penalty, which keeps the problem
feasible; safety of the executed command does not rest on the slack penalty's
gradient. Four guards sit between the solver and the vehicle: the lateral
authority rows are imposed at stage 0, where the executed input lives; an
accepted solution whose own predicted trajectory still enters an obstacle's
envelope -- the unavoidable-collision case -- is replaced by maximal braking
with the solver's evasive steering retained; the executed steering is clamped
to the soil's cornering authority on every exit path; and a per-stage speed
funnel keeps the vehicle at speeds it can null within the horizon it actually
constrains. A stale teleop command stream triggers braking, matching
CBFSafetyFilter.

Refs: Wabersich & Zeilinger, Automatica 2021 (arXiv:1812.05506); Tearle et al.,
IEEE RA-L 2021 (arXiv:2102.11907).
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import casadi as ca

from . import SafetyFilterResult
from .authority import peak_braking_decel, peak_lateral_accel
from .grade_preview import G, gravity_lookup, unknown_sine

_ACADOS_BUILD_ROOT = Path(
    os.environ.get("MPSF_BUILD_ROOT",
                   Path.home() / ".cache" / "scm_mpsf_acados"))


def _nn_loader():
    """The rig surrogate loader, whether the repo root or simulation/ is on sys.path.

    chrono_sim_node puts simulation/ on the path and the tests put the repo root
    there. A failed import here used to drop MPSF to soil-blind authority.
    """
    try:
        from simulation.tire_models.nn_tire_model import load_nn_tire_model
    except ModuleNotFoundError:
        from tire_models.nn_tire_model import load_nn_tire_model
    return load_nn_tire_model


class MPSFSafetyFilter:
    """Predictive safety filter over a short kinematic-bicycle horizon.

    Exposes the same interface as CBFSafetyFilter -- ``filter``,
    ``update_terrain``, ``update_command_age``, ``set_teleop_delay``,
    ``get_diagnostics`` -- so the plant node treats the two interchangeably.
    The acados solver is built once per configuration fingerprint and cached on
    disk, so repeated runs and parallel workers share one compiled solver.
    """

    def __init__(self,
                 vehicle_params: dict,
                 horizon: int = 20,
                 dt: float = 0.1,
                 n_obstacles: int = 3,
                 obstacle_buffer: float = 0.25,
                 vehicle_radius: float = 1.0,
                 ego_forward_extent: float = 1.6,
                 max_accel: float = 3.0,
                 max_decel: float = 6.0,
                 max_road_steer_angle: float = 0.49,
                 v_terminal: float = 1.5,
                 max_speed: float = 12.0,
                 w_steer: float = 1.0,
                 w_alpha: float = 1.0,
                 w_terminal: float = 0.0,
                 w_progress: float = 0.0,
                 slack_lin: float = 1.0e3,
                 slack_quad: float = 1.0e3,
                 nn_brake_authority: bool = True,
                 lat_accel_max_default: float = 4.0,
                 nn_model_dir: str = "nn_models/tire_force_static",
                 stale_cmd_timeout: float = 2.0,
                 teleop_delay: float = 0.0,
                 terminal_set: bool = False,
                 sense_range: float = 30.0,
                 unseen_obstacle_radius: float = 1.0,
                 brake_margin: float = 0.85,
                 w_rest: float = 0.01,
                 time_steps=None,
                 pedal_rate: float = 8.0,
                 dynamic_model: bool = False,
                 tube_c1: float = 0.03,
                 tube_eps: float = 0.06,
                 cert_dt: float = 0.05,
                 nominal_soil: str = "sand",
                 psf_max_iter: int = 20,
                 psf_cost: str = "first",
                 psf_lm: float = 0.01,
                 psf_w_rest: float = 1e-4,
                 psf_globalization: str = "FIXED_STEP",
                 traffic_speed_err: float = 0.0,
                 traffic_accel: float = 1.0,
                 unknown_downgrade: float = 0.15,
                 derived: bool = False,
                 powertrain: str = "shafts",
                 ident: dict | None = None,
                 odd_obstacle_radius: float | None = None,
                 build_solver: bool = True,
                 **_ignored):
        self.L = float(vehicle_params.get("Lf", 1.65)) + float(vehicle_params.get("Lr", 1.65))
        # Formulation switch. False keeps the published filter: deviation cost
        # on every stage and a per-stage speed funnel tied to the horizon
        # length. True selects the terminal-set filter. Every executed command
        # is certified by one step of it followed by straight braking to a
        # stop, clear of every known obstacle and inside the sensed region. The
        # OCP, whose plan must end in that same braking set and whose deviation
        # cost sits on the first input, proposes the correction when the
        # operator's command fails. There is no speed funnel. Safety does not
        # depend on the horizon length, and a longer horizon only improves the
        # proposals.
        self.terminal_set = bool(terminal_set)
        # Prediction model switch. True gives the standard predictive safety
        # filter (Wabersich and Zeilinger) on the tracking NMPC's own model, the
        # surrogate dynamic bicycle of surrogate_dynamics.py: the published cost
        # (deviation from the operator's command on every stage), obstacle and
        # sensed-region constraints tightened by the model's error against the
        # plant, and a terminal set from which straight braking stops clear. The
        # first input is executed when the solution, rolled out on the same
        # model, satisfies the constraints. Otherwise the next input of the last
        # such plan is executed. It is independent of terminal_set.
        self.dynamic_model = bool(dynamic_model)
        self.vehicle_params = dict(vehicle_params)
        # Error tube, validated against the plant (model_check.py): the 0.25 m
        # obstacle buffer plus tube_c1 per metre travelled plus tube_eps times
        # the integral over distance of the accumulated heading change.
        self.tube_c1 = float(tube_c1)
        self.tube_eps = float(tube_eps)
        self.cert_dt = float(cert_dt)
        self.nominal_soil = str(nominal_soil)
        # Footprint: three discs along the heading cover the body and wheels
        # (4.3 m x 2.1 m) and reach as far ahead as the published CG circle.
        self.disc_offsets = (-1.3, 0.0, 1.2) if self.dynamic_model else (0.0,)
        self.disc_radius = 1.4
        # Margin between the OCP, which constrains the stage nodes, and the
        # check of its plan, which tests every integration substep (m).
        self.ocp_margin = 0.1
        # Prediction of other vehicles (set-based). Each keeps its measured
        # heading, and its speed can do anything between braking as hard as
        # the soil allows and speeding up at traffic_accel, starting from the
        # measured speed give or take traffic_speed_err. At each future time
        # it is then somewhere on a segment along its heading, and every
        # footprint disc must clear that whole segment.
        self.traffic_speed_err = float(traffic_speed_err)
        self.traffic_accel = float(traffic_accel)
        # Grade preview (dynamic model): ground the preview has not seen counts
        # as this downgrade (rise over run).
        self.unknown_downgrade = float(unknown_downgrade)
        # Terminal set by construction: after the planned stages the OCP carries
        # a tail of stages whose input is fixed to the backup, straight full
        # braking, and the last state must be stopped. The terminal set is then
        # exactly the states from which the backup keeps every constraint, and
        # the check rolls out the same stages.
        self._tail_steps = (0.2,) * 22
        self.stop_speed = 0.3
        # OCP integrator: classic RK4, psf_rk_steps per stage. The check of each
        # plan integrates with exactly the same scheme and steps. Up to
        # psf_max_iter SQP iterations per call, warm-started from the previous
        # call's iterate. A multiple-shooting iterate satisfies its own dynamics
        # only near convergence, and the check rejects one that does not, so
        # fewer iterations mean more steps on the kept plan (about 20 are needed
        # with the surrogate in the model, some 300 ms per call on one core).
        self.psf_rk_steps = 1
        self.psf_max_iter = int(psf_max_iter)
        self.psf_change_tol = 0.01
        self.psf_lm = float(psf_lm)
        self.psf_w_rest = float(psf_w_rest)
        # Cost of the dynamic OCP. "first": the textbook predictive safety
        # filter (Wabersich and Zeilinger 2021), deviation on the executed input
        # only. The later inputs carry psf_w_rest, a numerical regularization:
        # summed over the stages even 0.01 makes planned braking cost as much
        # as cutting the current throttle by three quarters, which slows the
        # filter early. "horizon": the published MPSF's cost, deviation on every
        # planned stage, which charges the braking a plan needs as deviation,
        # so it slows early when the sensed region is short, more so with a
        # longer horizon.
        if psf_cost not in ("first", "horizon"):
            raise ValueError(f"psf_cost must be 'first' or 'horizon', got {psf_cost!r}")
        self.psf_cost = psf_cost
        self.psf_globalization = str(psf_globalization)
        self.sense_range = float(sense_range) if sense_range else 0.0
        self.unseen_obstacle_radius = float(unseen_obstacle_radius)
        self.brake_margin = float(brake_margin)
        self.w_rest = float(w_rest)
        self.brake_samples = (0.5, 1.0)
        # Pedal (alpha) rate limit of the plant's actuator model, per second.
        # The plant node applies it after the filter in operator and replay
        # runs, so throttle 0.9 to full brake takes about 0.24 s. The backup
        # check models that ramp, which it otherwise books as instant braking.
        self.pedal_rate = float(pedal_rate)
        self._alpha_applied = 0.0
        if time_steps:
            self.time_steps = tuple(float(t) for t in time_steps)
            self.N = len(self.time_steps)
            self.dt = self.time_steps[0]
        else:
            self.N = int(horizon)
            self.dt = float(dt)
            self.time_steps = tuple([self.dt] * self.N)
        self.n_obs = int(n_obstacles)
        self.obstacle_buffer = float(obstacle_buffer)
        self.vehicle_radius = float(vehicle_radius)
        self.ego_forward_extent = float(ego_forward_extent)
        self.max_accel = float(max_accel)
        self.max_decel = float(max_decel)
        self.max_steer = float(max_road_steer_angle)
        self.v_terminal = float(v_terminal)
        self.max_speed = float(max_speed)
        self.w_steer = float(w_steer)
        self.w_alpha = float(w_alpha)
        self.w_terminal = float(w_terminal)
        self.w_progress = float(w_progress)
        self.slack_lin = float(slack_lin)
        self.slack_quad = float(slack_quad)
        self.nn_brake_authority = bool(nn_brake_authority)
        self.lat_accel_max_default = float(lat_accel_max_default)
        # Resolve a relative checkpoint directory against the repository root,
        # not the caller's working directory: benchmark drivers launch the
        # plant node from several places, and a CWD-relative path made the
        # surrogate load silently fail (soil-blind authority) in some of them.
        _dir = Path(str(nn_model_dir)).expanduser()
        if not _dir.is_absolute():
            _dir = Path(__file__).resolve().parents[2] / _dir
        self.nn_model_dir = str(_dir)
        self.stale_cmd_timeout = float(stale_cmd_timeout)

        # Kinematic [X, Y, psi, v], or dynamic [X, Y, psi, u, v, r, a] plus the
        # tube states [path length, accumulated turn, their integral].
        self.nx = 10 if self.dynamic_model else 4
        self.nu = 2  # [steer, alpha]
        # params: [s_op, a_op, accel_grip, brake_grip, lat_accel_max,
        #          (ox, oy, orad)*n_obs], or in the dynamic model the segment
        #          a vehicle can be on at that stage, (ax, ay, bx, by, orad)*n_obs.
        #          Braking and cornering authority are
        # separate parameters so the tire surrogate sets the stopping and
        # turning capability the soil actually supports; the filter's avoidance
        # mode -- brake or steer -- then follows from those limits rather than
        # from a fixed preference.
        self._po = 5 if self.dynamic_model else 3
        self._np = 5 + self._po * self.n_obs
        if self.terminal_set or self.dynamic_model:
            # + [sense centre x, y, effective sensed radius, terminal braking decel]
            self._np += 4
        if self.dynamic_model:
            # + the soil in the surrogate's feature convention, the per-wheel
            # longitudinal force curve the braking torque balance reads, and
            # gravity along the heading at the stage (0 on flat ground).
            from .surrogate_dynamics import N_CURVE
            self._np += 6 + N_CURVE + 1
        self._bike = None
        self._dyn_soil = None
        self._ax_app = 0.0          # applied longitudinal acceleration (model state)
        self._backup = None         # the last checked plan's remaining substep commands
        self._sol = None            # last solution, for the warm start
        self._verified_count = 0
        self._backup_count = 0

        self._teleop_delay = max(float(teleop_delay), 0.0)
        self._last_cmd_wall = None  # arrival of the latest command, on self._clock
        self._clock = time.time     # set_clock() swaps in simulation time
        self._last_sense = None
        self._grip = 1.0          # traction/accel grip (kept as external alias)
        self._accel_grip = 1.0
        self._brake_grip = 1.0    # 1.0 => full max_decel (soil-blind default)
        self._lat_accel_max = self.lat_accel_max_default  # soil-blind default
        self._use_terrain_nn = True  # runtime gate over nn_brake_authority
        # Cached surrogate and last soil key, so braking authority is
        # recomputed only when the estimated terrain changes.
        self._nn = None
        self._nn_soil_key = None
        self._nn_load_failed = False
        try:
            import simulation.shared.param_consistency as _pc
            _fzf, _fzr = _pc.get_static_fz_per_wheel()
            self._fz_f, self._fz_r = float(_fzf), float(_fzr)
            self._mass = 2.0 * (self._fz_f + self._fz_r) / 9.81
        except Exception:  # noqa: BLE001
            self._fz_f, self._fz_r, self._mass = 6532.0, 6089.0, 2573.0
        self._solver = None
        self._last_result = None
        self._filter_count = 0
        self._modify_count = 0
        self._fail_stop_count = 0   # accepted solve whose plan was still unsafe
        self._solve_fail_count = 0  # solver status outside {0, 2}: fail-closed brake
        self._pass_count = 0        # terminal set: operator command certified as is
        self._hold_count = 0        # standstill holds (terminal set)
        self._fallback_count = 0    # OCP correction not certified, harder braking used
        self.hold_speed = 0.5       # m/s, below which an uncertified command is held
        self._lat_clamp_count = 0   # executed steering clamped to soil authority
        self._stale_brake_count = 0
        self._solve_ms_ema = 0.0
        # Derived mode (--safety-derived): every number from the plant's spec, the rig surrogate or the bounds
        # identified on separate plant manoeuvres (README, "Derived parameters").
        self.derived = bool(derived)
        self.powertrain = str(powertrain)
        if self.derived:
            from . import vehicle_spec as spec
            if ident is None:
                raise ValueError("the derived MPSF needs the identified bounds (--safety-ident)")
            if not self.dynamic_model:
                raise ValueError("the derived MPSF is the dynamic one (--mpsf-dynamic)")
            b = ident["powertrains"][self.powertrain]
            self.tube_c1 = float(b["tube"]["c1"])
            self.tube_eps = float(b["tube"]["eps"])
            self.obstacle_buffer = float(b["tube"]["c0"])
            self.brake_margin = 1.0
            self.disc_offsets, self.disc_radius = spec.footprint_discs()
            self.max_steer = spec.MAX_ROAD_WHEEL
            self.pedal_rate = spec.PEDAL_RATE
            self.max_speed = 1.0e3
            self.kappa_max = float(np.tan(spec.MAX_ROAD_WHEEL)) / self.L
            if odd_obstacle_radius is not None:
                self.unseen_obstacle_radius = float(odd_obstacle_radius)
            # + the stage's length, for the pedal the first input reaches and the gap between nodes
            self._np += 1
        if self.dynamic_model:
            self._init_dynamic_model()
        if build_solver:
            self._build_solver()

    def _psf_steps(self):
        """Stage lengths of the dynamic OCP: the planned stages, then the tail."""
        return tuple(self.time_steps) + tuple(self._tail_steps)

    def _init_dynamic_model(self):
        """Load the surrogate and set the dynamic model to the nominal soil.

        A soil-blind filter never receives a terrain belief, so its model runs
        on the nominal soil, the firmest in the study (dry sand), which like the
        kinematic filter's nominal grip is optimistic on softer ground.
        """
        from .surrogate_dynamics import SurrogateBicycle
        try:
            import simulation.shared.param_consistency as _pc
        except ModuleNotFoundError:
            import shared.param_consistency as _pc
        terrain = _pc.terrain_preset_to_internal(_pc.get_terrain_preset(self.nominal_soil))
        if self._nn is None:
            self._nn = _nn_loader()(self.nn_model_dir, terrain)
        if self.derived:
            from .derived_model import DerivedBicycle
            self._bike = DerivedBicycle(self._nn, self.vehicle_params, self.powertrain)
        else:
            self._bike = SurrogateBicycle(self._nn, self.vehicle_params, max_steer=self.max_steer)
        self._bike.set_terrain(terrain)
        if self.derived:
            self.traffic_accel = self._bike.drive_capability()
        self._dyn_soil = f"nominal:{self.nominal_soil}"

    # -- interface parity with CBFSafetyFilter ---------------------------------
    def set_clock(self, clock):
        """Use ``clock()`` (seconds) instead of wall time for command ages.

        The plant node passes simulation time, which makes the stale-command
        brake independent of how fast the machine runs the simulation.
        """
        self._clock = clock

    def set_teleop_delay(self, delay_s: float):
        self._teleop_delay = max(float(delay_s), 0.0)

    def update_command_age(self, cmd_wall_time: float):
        """Record the arrival of a command for stale-link detection.

        Mirrors CBFSafetyFilter: under teleop, a link that stops delivering
        commands must not leave the last command latched -- ``filter`` brakes
        once no command has arrived within ``stale_cmd_timeout``.
        """
        self._last_cmd_wall = self._clock()

    def _is_command_stale(self) -> bool:
        """True if no command received within stale_cmd_timeout (teleop only)."""
        if self._teleop_delay <= 0.0 or self._last_cmd_wall is None:
            return False
        return (self._clock() - self._last_cmd_wall) > self.stale_cmd_timeout

    def update_terrain(self, terrain_params, phi_uncertainty_deg=None, n_sigma=None,
                       hedge_k=None, use_terrain_nn=None, grip_scale=None,
                       use_grip_scale=None, **_):
        try:
            n = float(terrain_params["n"] if isinstance(terrain_params, dict) else terrain_params)
        except (TypeError, ValueError, KeyError):
            return
        if use_terrain_nn is not None:
            self._use_terrain_nn = bool(use_terrain_nn)
        # Belief-robust authority, mirroring CBFSafetyFilter: with a posterior
        # std and a risk factor supplied, authority is evaluated at the
        # pessimistic quantile -- the filter trusts its brakes and its tires
        # only as much as the worst plausible soil allows, so an uncertain
        # estimate shortens the trusted stopping distance instead of silently
        # assuming the mean.
        n_eff = n
        hk = float(hedge_k) if hedge_k is not None else 0.0
        if n_sigma is not None and hk > 0.0:
            n_eff = float(np.clip(n - hk * float(n_sigma), 0.4, 1.3))
        eval_terrain = terrain_params
        if isinstance(terrain_params, dict) and hk > 0.0:
            eval_terrain = dict(terrain_params)
            eval_terrain["n"] = n_eff
            if phi_uncertainty_deg is not None and eval_terrain.get("phi") is not None:
                eval_terrain["phi"] = max(
                    5.0, float(eval_terrain["phi"]) - hk * float(phi_uncertainty_deg))
        # Traction and acceleration grip: a linear map of the Bekker exponent
        # onto a grip fraction, saturated at both ends.
        g = float(np.clip(0.6 + 0.4 * (n_eff - 0.4) / 0.9, 0.5, 1.0))
        if use_grip_scale and grip_scale is not None:
            g *= float(np.clip(grip_scale, 0.4, 1.2))
        self._grip = g
        self._accel_grip = g
        # Braking and cornering authority from the force surrogate trained on
        # the controlled single-tire Chrono SCM rig. The filter avoids by
        # braking, by steering, or by both, so its stopping distance and its
        # achievable turn must each reflect the soil rather than a constant.
        # Both shrink on low grip, which is what leaves braking as the only
        # feasible avoidance there.
        if (self.nn_brake_authority and self._use_terrain_nn
                and isinstance(eval_terrain, dict)):
            bg = self._nn_brake_grip(eval_terrain)
            if bg is not None:
                self._brake_grip = bg
            la = self._nn_lat_accel_max(eval_terrain)
            if la is not None:
                self._lat_accel_max = la
            if self.dynamic_model and self._bike is not None:
                try:
                    self._bike.set_terrain(eval_terrain)
                    if self.derived:
                        self.traffic_accel = self._bike.drive_capability()
                    self._dyn_soil = (f"belief:n={float(eval_terrain['n']):.2f},"
                                      f"phi={float(eval_terrain['phi']):.1f}")
                except Exception as e:  # noqa: BLE001
                    self._warn_nn_failure("dynamic model soil", e)

    def _nn_lat_accel_max(self, terrain):
        """Peak achievable lateral (cornering) acceleration on ``terrain``, in
        m/s^2, from the single-tire rig force surrogate. Returns None when the
        surrogate cannot be queried."""
        try:
            required = ("Kphi", "Kc", "c", "phi", "k", "n")
            if any(terrain.get(key) is None for key in required):
                return None
            n = float(terrain["n"])
            if self._nn is None:
                load_nn_tire_model = _nn_loader()
                self._nn = load_nn_tire_model(self.nn_model_dir, terrain)
            best = peak_lateral_accel(self._nn, terrain, self._fz_f, self._fz_r, self._mass)
            if best <= 0.0:
                return None
            # A margin keeps the planned turn inside the achievable envelope.
            return float(np.clip(0.9 * best, 0.5, 8.0))
        except Exception as e:  # noqa: BLE001
            self._warn_nn_failure("lateral authority", e)
            return None

    def _nn_brake_grip(self, terrain):
        """Peak achievable braking deceleration on ``terrain``, from the
        single-tire rig force surrogate, as a fraction of max_decel. Returns
        None when the surrogate cannot be queried."""
        try:
            required = ("Kphi", "Kc", "c", "phi", "k", "n")
            if any(terrain.get(key) is None for key in required):
                return None
            n = float(terrain["n"])
            soil_key = (round(n, 3), round(float(terrain["phi"]), 2),
                        round(float(terrain["c"]), 1))
            if self._nn is None:
                load_nn_tire_model = _nn_loader()
                self._nn = load_nn_tire_model(self.nn_model_dir, terrain)
            best = peak_braking_decel(self._nn, terrain, self._fz_f, self._fz_r, self._mass)
            if best <= 0.0:
                return None
            self._nn_soil_key = soil_key
            # The clamp prevents claiming more than the nominal envelope and
            # holds a floor, so a degenerate query cannot zero out braking.
            return float(np.clip(best / self.max_decel, 0.15, 1.0))
        except Exception as e:  # noqa: BLE001
            self._warn_nn_failure("braking authority", e)
            return None

    def _warn_nn_failure(self, channel: str, err: Exception):
        """Announce a surrogate failure once instead of degrading silently.

        The soil-blind defaults the filter falls back to claim *nominal* grip,
        so a quiet failure here would overstate authority on soft soil for the
        rest of the run. The condition is surfaced on stdout and in the
        diagnostics so a benchmark log shows which arm actually ran.
        """
        if not self._nn_load_failed:
            self._nn_load_failed = True
            print(f"  [MPSF] tire surrogate unavailable for {channel} "
                  f"({type(err).__name__}: {err}); soil-blind authority in effect "
                  f"(model_dir={self.nn_model_dir})")

    def get_diagnostics(self) -> dict:
        rate = (self._modify_count / self._filter_count) if self._filter_count else 0.0
        return {"filter_calls": self._filter_count,
                "interventions": self._modify_count,
                "intervention_rate": rate,
                "mean_solve_ms": self._solve_ms_ema,
                "accel_grip": self._accel_grip,
                "brake_grip": self._brake_grip,
                "lat_accel_max": self._lat_accel_max,
                "nn_brake_authority": self.nn_brake_authority,
                "nn_authority_active": bool(
                    self.nn_brake_authority and self._use_terrain_nn
                    and not self._nn_load_failed),
                "fail_stops": self._fail_stop_count,
                "solve_fails": self._solve_fail_count,
                "certified_passes": self._pass_count,
                "standstill_holds": self._hold_count,
                "certified_fallbacks": self._fallback_count,
                "terminal_set": self.terminal_set,
                "dynamic_model": self.dynamic_model,
                "dynamic_soil": self._dyn_soil,
                "verified_solves": self._verified_count,
                "backup_steps": self._backup_count,
                "horizon_s": float(sum(self.time_steps)),
                "lat_clamps": self._lat_clamp_count,
                "stale_brakes": self._stale_brake_count}

    # -- acados OCP build ------------------------------------------------------
    def _fingerprint(self) -> str:
        cfg = ("mpsf_nn_steer_v2", self.L, self.N, self.dt, self.n_obs, self.obstacle_buffer,
               self.vehicle_radius, self.ego_forward_extent, self.max_accel,
               self.max_decel, self.max_steer, self.v_terminal, self.max_speed,
               self.w_steer, self.w_alpha, self.w_terminal, self.w_progress, self.slack_lin,
               self.slack_quad, self.lat_accel_max_default)
        if self.terminal_set:
            cfg = cfg + ("terminal_set_v2_ellipse", self.time_steps, self.w_rest,
                         self.brake_samples)
        if self.dynamic_model:
            from . import surrogate_dynamics as sd
            b = self._bike
            cfg = cfg + ("psf_dynamic_v8_grade", self.psf_cost, self.psf_lm, self.psf_w_rest, self.psf_rk_steps, self.psf_max_iter,
                         self.psf_globalization, self.time_steps, self._tail_steps, self.stop_speed,
                         self.disc_offsets,
                         self.disc_radius, self.ocp_margin, self.tube_c1, self.tube_eps,
                         self.brake_margin, Path(self.nn_model_dir).name, self._nn.n_params,
                         float(np.sum(self._nn._X_mean)), b.M, b.Izz, b.Lf, b.Lr, b.h_cg, b.T,
                         b.llt, b.blend_lo, b.blend_hi, b.tau_kin, b.tau_long, sd.ALPHA_MAX,
                         tuple(sd.KAPPA_BRAKE), tuple(sd.KAPPA_DRIVE), sd.BRAKE_TORQUE_MAX,
                         sd.WHEEL_RADIUS, sd.BRAKE_FADE_SPEED)
        if self.derived:
            cfg = cfg + ("derived_v1", self.powertrain, self.pedal_rate, self.kappa_max,
                         float(np.sum(self._bike.table)), tuple(self._bike.throttles))
        return hashlib.sha1(repr(cfg).encode()).hexdigest()[:12]

    def _build_model_dynamic(self):
        """The predictive safety filter's OCP on the surrogate dynamic model.

        Cost: the published deviation from the operator's command on every
        planned stage. Constraints, slacked for numerical robustness and
        checked afterwards without slack: every footprint disc keeps the
        planning radius plus the error tube from every selected obstacle, and
        the path stays inside the sensed region less the tube. Terminal set:
        the planned stages are followed by a tail of stages with the input fixed
        to straight full braking, and the vehicle must be stopped at its end.
        That makes safety independent of the planned horizon length.
        """
        from acados_template import AcadosModel
        from .surrogate_dynamics import N_CURVE
        x = ca.SX.sym("x", self.nx)   # X, Y, psi, u, v, r, a, s, Psi, S
        u = ca.SX.sym("u", self.nu)   # steer, alpha
        p = ca.SX.sym("p", self._np)
        X, Y, psi, uu, r = x[0], x[1], x[2], x[3], x[5]
        s_len, turned, S = x[7], x[8], x[9]
        st, al = u[0], u[1]
        q = 5 + self._po * self.n_obs
        cx, cy, r_eff, a_term = p[q], p[q + 1], p[q + 2], p[q + 3]
        terrain = p[q + 4:q + 10]
        curve = p[q + 10:q + 10 + N_CURVE]
        g_along = p[q + 10 + N_CURVE]
        if self.derived:
            # The second input is the pedal rate over its limit. The seventh state is the applied pedal.
            h_stage = p[q + 11 + N_CURVE]
            f_vehicle = self._bike.f_expl_sym(x[0:7], st * self.max_steer, al * self.pedal_rate, terrain,
                                              curve, g_along=g_along)
        else:
            f_vehicle = self._bike.f_expl_sym(x[0:7], st * self.max_steer, al, terrain, curve,
                                              p[2] * self.max_accel, self.brake_margin, g_along)
        eps_r = 0.01  # smooths |r|
        f_expl = ca.vertcat(f_vehicle, uu, ca.sqrt(r ** 2 + eps_r ** 2) - eps_r, uu * turned)
        model = AcadosModel()
        model.name = f"mpsf_{self._fingerprint()}"
        model.x = x
        model.u = u
        model.p = p
        model.f_expl_expr = f_expl
        model.xdot = ca.SX.sym("xdot", self.nx)
        model.f_impl_expr = model.xdot - f_expl

        hx, hy = ca.cos(psi), ca.sin(psi)

        def clear_rows(px, py, tube):
            # Squared distance from each disc centre to the obstacle's segment,
            # less the clearance radius squared. A rock's segment is one point.
            rows = []
            for j in range(self.n_obs):
                ax, ay, bx, by, orad = (p[5 + 5 * j + i] for i in range(5))
                dx, dy = bx - ax, by - ay
                radius = orad + self.disc_radius + self.obstacle_buffer + self.ocp_margin + tube
                if self.derived:
                    # Between nodes: chord and sagitta of the stage (ego travel only, the check is exact).
                    step = uu * h_stage
                    radius = ca.sqrt((radius + self.kappa_max * step ** 2 / 8.0) ** 2 + (step / 2.0) ** 2)
                for off in self.disc_offsets:
                    qx, qy = px + off * hx - ax, py + off * hy - ay
                    w = ca.fmin(ca.fmax((qx * dx + qy * dy) / (dx * dx + dy * dy + 1e-6), 0.0), 1.0)
                    rows.append((qx - w * dx) ** 2 + (qy - w * dy) ** 2 - radius ** 2)
            return rows

        tube = self.tube_c1 * s_len + self.tube_eps * S
        inner = r_eff - tube - self.ocp_margin
        model.con_h_expr = ca.vertcat(*clear_rows(X, Y, tube),
                                      inner ** 2 - ((X - cx) ** 2 + (Y - cy) ** 2))
        # Terminal: stopped at the end of the braking tail, clear and inside.
        model.con_h_expr_e = ca.vertcat(*clear_rows(X, Y, tube),
                                        inner ** 2 - ((X - cx) ** 2 + (Y - cy) ** 2),
                                        self.stop_speed - uu)

        s_op, a_op = p[0], p[1]
        deviation = self.w_steer * (st - s_op) ** 2 + self.w_alpha * (al - a_op) ** 2
        if self.derived:
            # The pedal the input reaches by the end of the stage, against the operator's.
            deviation = (self.w_steer * (st - s_op) ** 2
                         + self.w_alpha * (x[6] + al * self.pedal_rate * h_stage - a_op) ** 2)
        progress = self.w_progress * ca.fmax(0.0, 2.0 - uu) ** 2
        model.cost_expr_ext_cost_0 = deviation + progress
        rest = 1.0 if self.psf_cost == "horizon" else self.psf_w_rest
        model.cost_expr_ext_cost = rest * deviation + progress
        model.cost_expr_ext_cost_e = self.w_terminal * uu ** 2
        return model

    def _build_model(self):
        if self.dynamic_model:
            return self._build_model_dynamic()
        x = ca.SX.sym("x", self.nx)   # X, Y, psi, v
        u = ca.SX.sym("u", self.nu)   # steer, alpha
        p = ca.SX.sym("p", self._np)
        accel_grip = p[2]
        brake_grip = p[3]
        lat_accel_max = p[4]
        X, Y, psi, v = x[0], x[1], x[2], x[3]
        s, a = u[0], u[1]
        delta = s * self.max_steer
        # Traction and braking authority are separate parameters: braking is
        # the safety-critical channel and is set per soil by the surrogate.
        a_gain = ca.if_else(a >= 0.0, accel_grip * self.max_accel,
                            brake_grip * self.max_decel)
        f_expl = ca.vertcat(
            v * ca.cos(psi),
            v * ca.sin(psi),
            v / self.L * ca.tan(delta),
            a_gain * a,
        )
        from acados_template import AcadosModel
        model = AcadosModel()
        model.name = f"mpsf_{self._fingerprint()}"
        model.x = x
        model.u = u
        model.p = p
        model.f_expl_expr = f_expl
        model.xdot = ca.SX.sym("xdot", self.nx)
        model.f_impl_expr = model.xdot - f_expl

        # obstacle avoidance: h_j = dist^2 - safe_r^2 >= 0 (slacked). safe_r adds
        # the ego's forward extent so the CG-based barrier reflects the real body.
        obs_h = []
        for j in range(self.n_obs):
            ox, oy, orad = p[5 + 3 * j], p[6 + 3 * j], p[7 + 3 * j]
            safe_r = orad + self.vehicle_radius + self.ego_forward_extent + self.obstacle_buffer
            obs_h.append((X - ox) ** 2 + (Y - oy) ** 2 - safe_r ** 2)
        # Cornering (lateral) authority: |a_lat| = |v^2 tan(delta)/L| <= lat_max.
        # This row is hard and never slacked, because stopping is always a
        # feasible way to clear an obstacle and the vehicle must therefore
        # never be required to corner harder than the soil supports. On clay
        # lat_max is small, the swerve is infeasible, and the slacked obstacle
        # rows drive braking instead; on firm soil the limit is loose, the
        # swerve is permitted, and being cheaper than braking it is preferred.
        # The avoidance mode is thus set by the terrain, not by a fixed rule.
        a_lat = v ** 2 * ca.tan(delta) / self.L
        lat_h = [lat_accel_max - a_lat, lat_accel_max + a_lat]
        if self.terminal_set:
            # Friction ellipse: braking and cornering share one tire force
            # budget. With the two limits applied separately, the plan could
            # brake at the full soil deceleration while cornering at the full
            # lateral limit, which the tires cannot deliver on low-grip soil,
            # so a braking-and-swerving continuation looked feasible when it
            # was not. Zero steering always satisfies the row.
            a_lon = a_gain * a
            a_lon_lim = brake_grip * self.max_decel
            lat_h.append(1.0 - (a_lon / a_lon_lim) ** 2 - (a_lat / lat_accel_max) ** 2)
        if self.terminal_set:
            # Sensed region: a disc of the filter's sensing range around the
            # vehicle at solve time, shrunk by the footprint of an obstacle the
            # filter cannot yet see. Every planned position must stay inside it.
            q = 5 + 3 * self.n_obs
            cx, cy, r_eff, a_term = p[q], p[q + 1], p[q + 2], p[q + 3]
            sense_h = r_eff ** 2 - ((X - cx) ** 2 + (Y - cy) ** 2)
            model.con_h_expr = ca.vertcat(*obs_h, sense_h, *lat_h)
            # Terminal set: from the last planned state, braking straight ahead
            # at the soil's (margined) deceleration must stop clear of every
            # known obstacle and inside the sensed region. Points along the
            # braking segment are checked against each obstacle. This set is
            # invariant under full braking, so a plan that ends in it can always
            # be continued, which is what makes safety independent of N.
            l_brk = v ** 2 / (2.0 * a_term)
            hx, hy = ca.cos(psi), ca.sin(psi)
            term = list(obs_h)
            for sig in self.brake_samples:
                px, py = X + sig * l_brk * hx, Y + sig * l_brk * hy
                for j in range(self.n_obs):
                    ox, oy, orad = p[5 + 3 * j], p[6 + 3 * j], p[7 + 3 * j]
                    safe_r = (orad + self.vehicle_radius + self.ego_forward_extent
                              + self.obstacle_buffer)
                    term.append((px - ox) ** 2 + (py - oy) ** 2 - safe_r ** 2)
            term.append(r_eff ** 2 - ((X + l_brk * hx - cx) ** 2
                                      + (Y + l_brk * hy - cy) ** 2))
            model.con_h_expr_e = ca.vertcat(*term)
        else:
            model.con_h_expr = ca.vertcat(*obs_h, *lat_h)   # path: obstacles + lateral
            model.con_h_expr_e = ca.vertcat(*obs_h)          # terminal: obstacles only
        # Stage 0 carries the lateral rows alone. Without an explicit stage-0
        # constraint set, acados leaves the first stage unconstrained, and the
        # first stage's input is the one the vehicle executes -- the solver
        # could command a first-step swerve the soil cannot deliver while every
        # later stage obeyed the limit. The obstacle rows are omitted at stage
        # 0 because there they are functions of the fixed initial state only:
        # they constrain nothing, and made hard they would render the problem
        # infeasible from any already-violating state, discarding the solve
        # exactly when it is needed. Steering zero always satisfies the
        # lateral rows, so keeping them hard cannot cause infeasibility.
        model.con_h_expr_0 = ca.vertcat(*lat_h)

        # Costs (EXTERNAL): the running term is deviation from the operator
        # command, plus an optional progress term penalizing speeds below a
        # floor, so that where a safe route through a dense field exists the
        # filter threads it rather than stalling. The penalty is one-sided and
        # zero above the floor, so it never rewards exceeding operator intent,
        # and the obstacle constraints still force a stop where no safe route
        # exists. Setting w_progress to 0 makes braking the primary response.
        s_op, a_op = p[0], p[1]
        progress_floor = 2.0  # m/s -- below this, stalling is penalized
        deviation = self.w_steer * (s - s_op) ** 2 + self.w_alpha * (a - a_op) ** 2
        progress = self.w_progress * ca.fmax(0.0, progress_floor - v) ** 2
        if self.terminal_set:
            # The executed input is the first one, so that is where deviation
            # is charged. Later inputs carry a small weight that keeps the QP
            # well posed. Any braking the plan needs is then placed as late as
            # the constraints allow, and the operator's command passes
            # unchanged until a later start would no longer be safe.
            model.cost_expr_ext_cost_0 = deviation
            model.cost_expr_ext_cost = self.w_rest * deviation + progress
        else:
            model.cost_expr_ext_cost = deviation + progress
        # There is no blanket terminal-speed penalty: it would brake even on a
        # clear road, opposing the operator for no safety gain. Safety over the
        # horizon comes from the slacked obstacle constraints, which are
        # imposed at every stage including the terminal one.
        model.cost_expr_ext_cost_e = self.w_terminal * v ** 2
        return model

    def _build_solver(self):
        from acados_template import AcadosOcp, AcadosOcpSolver
        model = self._build_model()
        ocp = AcadosOcp()
        ocp.model = model
        steps = self._psf_steps() if self.dynamic_model else self.time_steps
        ocp.dims.N = len(steps)
        ocp.dims.np = self._np
        ocp.parameter_values = np.zeros(self._np)

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        # control bounds
        ocp.constraints.lbu = np.array([-1.0, -1.0])
        ocp.constraints.ubu = np.array([1.0, 1.0])
        ocp.constraints.idxbu = np.array([0, 1])
        # state bound: v >= 0 (no reverse in the kinematic model). The dynamic
        # model's braking fades at standstill instead, and a hard zero bound
        # makes the QP infeasible once the brake saturates at wheel lock.
        v_lo = -0.5 if self.dynamic_model else 0.0
        ocp.constraints.lbx = np.array([v_lo])
        ocp.constraints.ubx = np.array([self.max_speed])
        ocp.constraints.idxbx = np.array([3])
        if self.derived:
            ocp.constraints.lbx = np.array([v_lo, -1.0])
            ocp.constraints.ubx = np.array([self.max_speed, 1.0])
            ocp.constraints.idxbx = np.array([3, 6])
        ocp.constraints.lbx_e = np.array([v_lo])
        ocp.constraints.ubx_e = np.array([self.max_speed])
        ocp.constraints.idxbx_e = np.array([3])
        ocp.constraints.x0 = np.zeros(self.nx)

        # path constraints: n_obs obstacle rows + 2 lateral-authority rows;
        # terminal: n_obs obstacle rows only (no control at the terminal node).
        no = self.n_obs * len(self.disc_offsets)            # obstacle rows (x discs)
        with_sense = self.terminal_set or self.dynamic_model
        n_sl = no + 1 if with_sense else no                 # slacked path rows
        n_e = (no * (1 + len(self.brake_samples)) + 1) if with_sense else no
        if self.dynamic_model:
            n_e = no + 2                                   # clear, inside, stopped
        n_lat = 3 if self.terminal_set else 2              # lateral rows (+ ellipse)
        if self.dynamic_model:
            n_lat = 0                                      # the tire forces bound the turn
        nh = n_sl + n_lat
        ocp.constraints.lh = np.zeros(nh)
        ocp.constraints.uh = 1e9 * np.ones(nh)
        ocp.constraints.lh_e = np.zeros(n_e)
        ocp.constraints.uh_e = 1e9 * np.ones(n_e)
        # stage 0: the two hard lateral-authority rows on the executed input.
        if n_lat:
            ocp.constraints.lh_0 = np.zeros(n_lat)
            ocp.constraints.uh_0 = 1e9 * np.ones(n_lat)
        # slack ONLY the obstacle constraints (path + terminal) so the QP stays
        # feasible; the lateral rows are hard (braking is the feasible escape).
        ocp.constraints.idxsh = np.arange(n_sl)
        ocp.constraints.idxsh_e = np.arange(n_e)
        ocp.cost.zl = self.slack_lin * np.ones(n_sl)
        ocp.cost.zu = self.slack_lin * np.ones(n_sl)
        ocp.cost.Zl = self.slack_quad * np.ones(n_sl)
        ocp.cost.Zu = self.slack_quad * np.ones(n_sl)
        ocp.cost.zl_e = self.slack_lin * np.ones(n_e)
        ocp.cost.zu_e = self.slack_lin * np.ones(n_e)
        ocp.cost.Zl_e = self.slack_quad * np.ones(n_e)
        ocp.cost.Zu_e = self.slack_quad * np.ones(n_e)
        if self.terminal_set or self.dynamic_model:
            ocp.cost.cost_type_0 = "EXTERNAL"

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 30
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 2
        ocp.solver_options.sim_method_num_steps = 1
        if self.dynamic_model:
            # Classic RK4, the scheme the plan check repeats. A small
            # Levenberg-Marquardt term keeps the QP Hessian definite, as the cost
            # has no state term.
            ocp.solver_options.sim_method_num_stages = 4
            ocp.solver_options.sim_method_num_steps = self.psf_rk_steps
            ocp.solver_options.nlp_solver_max_iter = self.psf_max_iter
            ocp.solver_options.levenberg_marquardt = self.psf_lm
            ocp.solver_options.globalization = self.psf_globalization
        if self.dynamic_model or len(set(self.time_steps)) > 1:
            ocp.solver_options.tf = float(sum(steps))
            ocp.solver_options.time_steps = np.array(steps)
        else:
            ocp.solver_options.tf = self.N * self.dt   # the published value, bit for bit
        ocp.solver_options.print_level = 0

        import fcntl
        fp = self._fingerprint()
        _ACADOS_BUILD_ROOT.mkdir(parents=True, exist_ok=True)
        exp = _ACADOS_BUILD_ROOT / f"mpsf_{fp}"
        exp.mkdir(parents=True, exist_ok=True)
        ocp.code_export_directory = str(exp / "c_generated_code")
        json_file = str(exp / "acados_ocp.json")
        so_file = exp / "c_generated_code" / f"libacados_ocp_solver_{model.name}.so"

        def _load_cached() -> bool:
            if so_file.exists() and Path(json_file).exists():
                try:
                    self._solver = AcadosOcpSolver(None, json_file=json_file,
                                                   build=False, generate=False)
                    return True
                except Exception:  # noqa: BLE001
                    return False
            return False

        # Serialize code generation across parallel workers sharing a
        # fingerprint directory: the first worker to take the lock generates
        # and compiles, and the rest block and then load the cached shared
        # object. This matches the build lock used by the tracking NMPC.
        if _load_cached():
            return
        lock_path = _ACADOS_BUILD_ROOT / f"mpsf_{fp}.lock"
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                if not _load_cached():
                    self._solver = AcadosOcpSolver(ocp, json_file=json_file,
                                                   verbose=False)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    # -- core ------------------------------------------------------------------
    def _select_obstacles(self, x, y, psi, obstacles):
        """Select the n_obs most threatening obstacles for the OCP.

        Ranking by Euclidean distance alone can drop an obstacle sitting
        directly on the collision path in favour of a nearer one off to the
        side. Obstacles that are ahead and inside the heading-aligned corridor
        (lateral offset below the safe radius plus a margin) are therefore
        ranked by how soon they are reached, and any remaining slots are filled
        with the nearest of the rest. The obstacle the vehicle is driving into
        thus stays in the constraint set even where the field is denser than
        n_obs.
        """
        cpsi, spsi = np.cos(psi), np.sin(psi)
        scored = []
        for o in obstacles:
            ox, oy, orad = float(o[0]), float(o[1]), float(o[2])
            dx, dy = ox - x, oy - y
            d_along = dx * cpsi + dy * spsi
            d_perp = -dx * spsi + dy * cpsi
            safe_r = (orad + self.vehicle_radius + self.ego_forward_extent
                      + self.obstacle_buffer)
            corridor = safe_r + 1.5  # lateral relevance margin (m)
            on_path_ahead = (d_along > -safe_r) and (abs(d_perp) < corridor)
            # tier 0: on-path threats, soonest first; tier 1: others, nearest.
            if on_path_ahead and (self.terminal_set or self.dynamic_model):
                # Rank by where the ego's safety circle first touches the
                # obstacle, then by lateral offset. Ranking by d_along alone
                # ties across a wall of rocks at one distance, and the tie fell
                # to list order, which kept off-centre rocks and dropped the
                # one dead ahead (the published selection, kept below).
                touch = d_along - np.sqrt(max(safe_r * safe_r - d_perp * d_perp, 0.0))
                key = (0, touch, abs(d_perp))
            elif on_path_ahead:
                key = (0, d_along)
            else:
                key = (1, dx * dx + dy * dy)
            scored.append((key, (ox, oy, orad)))
        scored.sort(key=lambda t: t[0])
        return [o for _, o in scored[: self.n_obs]]

    def _lat_steer_limit(self, v: float) -> float:
        """Largest |normalized steer| the soil's cornering authority admits at
        speed ``v``: |v^2 tan(delta)/L| <= lat_accel_max, in [0, 1]."""
        v_eff = max(abs(float(v)), 0.5)  # below 0.5 m/s any steer is admissible
        d_max = float(np.arctan(self._lat_accel_max * self.L / (v_eff * v_eff)))
        return float(np.clip(d_max / self.max_steer, 0.0, 1.0))

    def _plan_is_unsafe(self, obstacles) -> bool:
        """True when the accepted solution's own predicted trajectory enters an
        obstacle's physical envelope.

        The obstacle rows are slacked so the OCP stays feasible, which means an
        "accepted" solve can carry an arbitrarily large violation -- and a
        violated plan is exactly the case the paper resolves by maximal
        braking, not by executing whatever input the penalty gradient landed
        on. The check tolerates penetration of the planning buffer (that is
        the slack's legitimate role) and trips once the predicted path crosses
        into the buffer-stripped safety radius. Checked against the full
        obstacle field, not the selected subset.
        """
        if not obstacles:
            return False
        try:
            xs = [self._solver.get(k, "x") for k in range(self.N + 1)]
        except Exception:  # noqa: BLE001
            return True  # cannot read the plan back: treat as unsafe
        pts = [(float(xk[0]), float(xk[1])) for xk in xs]
        if self.terminal_set and getattr(self, "_last_sense", None) is not None:
            # The terminal braking segment is part of the plan's safety claim.
            cx, cy, r_eff, a_term = self._last_sense
            xn = xs[-1]
            l_brk = float(xn[3]) ** 2 / (2.0 * a_term)
            hx, hy = np.cos(float(xn[2])), np.sin(float(xn[2]))
            for sig in self.brake_samples:
                pts.append((float(xn[0]) + sig * l_brk * hx,
                            float(xn[1]) + sig * l_brk * hy))
            # A stop point outside the sensed region means no plan stays in
            # known space: the case for maximal braking.
            sx, sy = pts[-1]
            if np.hypot(sx - cx, sy - cy) > r_eff + 0.5:
                return True
        for o in obstacles:
            ox, oy, orad = float(o[0]), float(o[1]), float(o[2])
            hard_r = (orad + self.vehicle_radius + self.ego_forward_extent)
            for px, py in pts:
                if np.hypot(px - ox, py - oy) < hard_r:
                    return True
        return False

    def _step(self, x, s_cmd, a_cmd, brake_grip, h):
        """One step of the OCP's model (midpoint rule), for rollouts."""
        X, Y, psi, v = x
        delta = s_cmd * self.max_steer
        a_gain = (self._accel_grip * self.max_accel if a_cmd >= 0.0
                  else brake_grip * self.max_decel)
        v_next = float(np.clip(v + a_gain * a_cmd * h, 0.0, self.max_speed))
        v_mid = 0.5 * (v + v_next)
        psi_next = psi + v_mid / self.L * np.tan(delta) * h
        psi_mid = 0.5 * (psi + psi_next)
        return np.array([X + v_mid * np.cos(psi_mid) * h,
                         Y + v_mid * np.sin(psi_mid) * h, psi_next, v_next])

    def _within_friction_ellipse(self, s_cmd, a_cmd, v, brake_grip) -> bool:
        """The command's longitudinal and lateral demand fit the tire budget."""
        a_gain = (self._accel_grip * self.max_accel if a_cmd >= 0.0
                  else brake_grip * self.max_decel)
        a_lon = a_gain * a_cmd / max(brake_grip * self.max_decel, 1e-6)
        a_lat = v * v * np.tan(s_cmd * self.max_steer) / self.L / max(self._lat_accel_max, 1e-6)
        return a_lon * a_lon + a_lat * a_lat <= 1.0 + 1e-6

    def _backup_rollout(self, z0, s0, a0, brake_grip):
        """The first input, then full braking with the wheels straight.

        The applied pedal follows the commands through the plant's rate limit,
        starting from where the last executed command has brought it. Returns
        the N+1 states and N commanded inputs over the horizon grid.
        """
        xs = [np.array(z0, dtype=float)]
        us = []
        x = xs[0]
        alpha = self._alpha_applied
        sub = 0.02
        t = 0.0
        for k, h in enumerate(self.time_steps):
            u = (s0, a0) if k == 0 else (0.0, -1.0)
            us.append(np.array(u, dtype=float))
            n_sub = max(1, int(round(h / sub)))
            for _ in range(n_sub):
                # The applied pedal moves toward the command at the plant's rate.
                d = self.pedal_rate * (h / n_sub)
                alpha += float(np.clip(u[1] - alpha, -d, d))
                x = self._step(x, u[0], alpha, brake_grip, h / n_sub)
                t += h / n_sub
            xs.append(x)
        return xs, us

    def _path_is_clear(self, xs, obstacles) -> bool:
        """True when the backup path, continued by straight braking to a stop
        from its last state, keeps the planning radius from every obstacle in
        the field and stays inside the sensed region."""
        cx, cy, r_eff, a_term = self._last_sense
        pts = [(float(x[0]), float(x[1])) for x in xs[1:]]
        xn = xs[-1]
        l_brk = float(xn[3]) ** 2 / (2.0 * a_term)
        hx, hy = np.cos(float(xn[2])), np.sin(float(xn[2]))
        for sig in (0.25, 0.5, 0.75, 1.0):
            pts.append((float(xn[0]) + sig * l_brk * hx, float(xn[1]) + sig * l_brk * hy))
        for px, py in pts:
            if np.hypot(px - cx, py - cy) > r_eff:
                return False
        for o in obstacles:
            ox, oy, orad = float(o[0]), float(o[1]), float(o[2])
            r = orad + self.vehicle_radius + self.ego_forward_extent + self.obstacle_buffer
            for px, py in pts:
                if np.hypot(px - ox, py - oy) < r:
                    return False
        return True

    def _certified(self, z0, s_cmd, a_cmd, brake_grip, obstacles) -> bool:
        """One step of the command, then straight braking to a stop, is safe."""
        if not self._within_friction_ellipse(s_cmd, a_cmd, z0[3], brake_grip):
            return False
        bx, _ = self._backup_rollout(z0, s_cmd, a_cmd, brake_grip)
        return self._path_is_clear(bx, obstacles)

    def _certified_correction(self, z0, s_op, a_op, u0, ok, brake_grip, obstacles,
                              result):
        """Execute the OCP's correction if it certifies, else brake harder.

        Candidates in order: the OCP's first input, then its steering with
        progressively harder braking, then straight full braking. The first
        candidate that certifies is executed. If none does, the collision cannot
        be avoided by braking from here and straight full braking is applied.
        """
        candidates = []
        if u0 is not None and ok and np.all(np.isfinite(u0)):
            s_c = float(np.clip(u0[0], -1, 1))
            a_c = float(np.clip(u0[1], -1, 1))
            candidates.append((s_c, a_c))
            for a in np.linspace(a_c, -1.0, 5)[1:]:
                candidates.append((s_c, float(a)))
        else:
            self._solve_fail_count += 1
        candidates.append((0.0, -1.0))
        for s_c, a_c in candidates:
            if self._certified(z0, s_c, a_c, brake_grip, obstacles):
                if (s_c, a_c) != candidates[0]:
                    self._fallback_count += 1
                return result(s_c, a_c, True)
        self._fail_stop_count += 1
        return result(0.0, -1.0, True)

    def _sensed_radius(self) -> float:
        """Radius around the vehicle that every planned position, and the
        terminal stop point, must stay inside. An obstacle the filter cannot
        see yet may sit just past the sensing range, so the range is shrunk by
        that obstacle's footprint plus the ego's. No range means no limit."""
        if self.sense_range <= 0.0:
            return 1.0e4
        if self.derived:
            # The footprint's front disc, at the planned position, may not reach an obstacle just past the range.
            margin = (self.unseen_obstacle_radius + max(self.disc_offsets) + self.disc_radius
                      + self.obstacle_buffer)
            return max(self.sense_range - margin, 1.0)
        margin = (self.unseen_obstacle_radius + self.vehicle_radius
                  + self.ego_forward_extent + self.obstacle_buffer)
        return max(self.sense_range - margin, 1.0)

    # -- dynamic model: the predictive safety filter step -----------------------
    def _traffic_decel(self) -> float:
        """Hardest braking assumed for another vehicle: locked wheels on the
        believed soil, which is how the plant's vehicles brake at full pedal."""
        return max(self._bike.peak_decel(), 0.5)

    def _segment(self, o, t):
        """Ends (ax, ay), (bx, by) of the segment an obstacle can be on at each
        time t (s after its measurement). A rock, or an obstacle given without
        a velocity, stays where it is."""
        t = np.asarray(t, dtype=float)
        ox, oy = float(o[0]), float(o[1])
        if len(o) < 6:
            x0, y0 = np.full_like(t, ox), np.full_like(t, oy)
            return x0, y0, x0, y0
        vx, vy = float(o[4]), float(o[5])
        if len(o) >= 7:
            # Heading given: the speed is the velocity along it, which a noisy
            # measurement can make negative.
            hx, hy = np.cos(float(o[6])), np.sin(float(o[6]))
            speed = vx * hx + vy * hy
        else:
            speed = float(np.hypot(vx, vy))
            if speed < 1e-9:
                x0, y0 = np.full_like(t, ox), np.full_like(t, oy)
                return x0, y0, x0, y0
            hx, hy = vx / speed, vy / speed
        v_lo = max(speed - self.traffic_speed_err, 0.0)
        v_hi = max(speed + self.traffic_speed_err, 0.0)
        a_b = self._traffic_decel()
        # Slowest: brake from v_lo to a stop and stay there.
        d_lo = np.where(t < v_lo / a_b, v_lo * t - 0.5 * a_b * t * t, 0.5 * v_lo * v_lo / a_b)
        # Fastest: speed up from v_hi.
        d_hi = v_hi * t + 0.5 * self.traffic_accel * t * t
        return ox + d_lo * hx, oy + d_lo * hy, ox + d_hi * hx, oy + d_hi * hy

    def _check(self, z0, plan, obstacles, gravity=None):
        """Roll the plan out on the model with the OCP's own integrator, then
        straight full braking to a stop, and test every constraint of the OCP
        without slack against the whole obstacle field. ``gravity(X, Y)`` is
        gravity along the heading where the vehicle is (none: flat ground)."""
        cmds, hs = self._substep_inputs(plan)
        a_min = max(self.brake_margin * self._bike.peak_decel(), 0.5)
        if gravity is not None:
            # A downhill stretches the stop: allow for the steepest admissible grade.
            a_min = max(a_min - G * unknown_sine(self.unknown_downgrade), 0.3)
        xs = self._bike.rollout_rk4(z0, cmds, hs, self.cert_dt,
                                    self._accel_grip * self.max_accel, self.brake_margin,
                                    float(z0[3]) / a_min + 3.0, gravity=gravity)
        n_extra = len(xs) - 1 - len(hs)
        t = np.concatenate([np.cumsum(hs), hs.sum() + self.cert_dt * np.arange(1, n_extra + 1)])
        P, psi = xs[:, :2], xs[:, 2]
        seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
        turned = np.cumsum(np.abs(np.diff(np.unwrap(psi))))
        tube = self.tube_c1 * np.cumsum(seg) + self.tube_eps * np.cumsum(seg * turned)
        P, psi = P[1:], psi[1:]
        cx, cy, r_eff, _ = self._last_sense
        if np.any(np.hypot(P[:, 0] - cx, P[:, 1] - cy) + tube > r_eff):
            return False
        for o in obstacles or ():
            radius = float(o[2]) + self.disc_radius + self.obstacle_buffer
            ax, ay, bx, by = self._segment(o, t)
            dx, dy = bx - ax, by - ay
            for off in self.disc_offsets:
                qx = P[:, 0] + off * np.cos(psi) - ax
                qy = P[:, 1] + off * np.sin(psi) - ay
                w = np.clip((qx * dx + qy * dy) / (dx * dx + dy * dy + 1e-6), 0.0, 1.0)
                if np.any(np.hypot(qx - w * dx, qy - w * dy) < radius + tube):
                    return False
        return True

    def _substep_inputs(self, plan):
        """Per-substep commands and step lengths for the planned stages followed
        by the braking tail, as the OCP integrates them."""
        steps = self._psf_steps()
        inputs = list(plan) + [(0.0, -1.0)] * (len(steps) - len(plan))
        cmds, hs = [], []
        for (s_c, a_c), dt_k in zip(inputs, steps):
            for _ in range(self.psf_rk_steps):
                cmds.append((s_c, a_c))
                hs.append(dt_k / self.psf_rk_steps)
        return np.array(cmds), np.array(hs)

    def _warm_start(self, z0, plan, gravity=None):
        """OCP node states from rolling the plan and the tail out on the model."""
        cmds, hs = self._substep_inputs(plan)
        xs = self._bike.rollout_rk4(z0[:7], cmds, hs, self.cert_dt,
                                    self._accel_grip * self.max_accel, self.brake_margin, 0.0,
                                    gravity=gravity)
        P = xs[:, :2]
        seg = np.concatenate([[0.0], np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))])
        turned = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(np.unwrap(xs[:, 2]))))])
        s_len = np.cumsum(seg)
        S = np.cumsum(seg * turned)
        nodes = range(0, len(xs), self.psf_rk_steps)
        return [np.concatenate([xs[i], [s_len[i], turned[i], S[i]]]) for i in nodes]

    def _filter_dynamic(self, vehicle_state, s_op, a_op, obstacles, x_sense, y_sense, result):
        """One step of the predictive safety filter on the surrogate model.

        Solve the OCP from the measured state and roll its plan, with the
        braking tail, out on the same model. If every constraint holds without
        slack, execute the first input and keep the plan. Otherwise execute the
        next input of the last kept plan, which ends in straight full braking.

        The filter runs on the vehicle, so its command acts at once and there is
        no advance over the command round trip (the published MPSF has one). A
        plan checked from an advanced state and run from the measured one is not
        the plan that was checked.
        """
        u0 = min(max(float(vehicle_state.get("u", 0.0)), 0.0), self.max_speed)
        z0 = np.array([float(vehicle_state["x"]), float(vehicle_state["y"]),
                       float(vehicle_state["psi"]), u0,
                       float(vehicle_state.get("v", 0.0)), float(vehicle_state.get("omega", 0.0)),
                       self._ax_app])
        r_eff = self._sensed_radius()
        a_term = max(self.brake_margin * self._bike.peak_decel(), 0.5)
        self._last_sense = (x_sense, y_sense, r_eff, a_term)

        obs = self._select_obstacles(z0[0], z0[1], z0[2], obstacles)
        full = {(float(o[0]), float(o[1])): o for o in obstacles or ()}
        obs = [full.get((o[0], o[1]), o) for o in obs]
        while len(obs) < self.n_obs:
            obs.append((z0[0] + 1e4, z0[1] + 1e4, 0.1))
        tail = ([x_sense, y_sense, r_eff, a_term]
                + list(self._bike.terrain_vec) + list(self._bike.curve))
        steps = self._psf_steps()
        n_total = len(steps)
        t_nodes = np.concatenate([[0.0], np.cumsum(steps)])
        # Each obstacle's segment at every node, as columns (ax, ay, bx, by).
        segs = [np.stack(self._segment(o, t_nodes), axis=1) for o in obs]
        # Gravity along the heading from the grade preview, if the simulator
        # gives one. The OCP takes it at each node's warm-start position, the
        # check at every position of the rolled-out plan.
        gravity = gravity_lookup(vehicle_state, self._bike.L, self.unknown_downgrade)
        guess = (list(self._sol[1:]) + [self._sol[-1]] if self._sol is not None
                 else [(s_op, a_op)] * self.N)
        xw = self._warm_start(z0, guess, gravity)

        def stage_p(k):
            o_k = [c for o, sg in zip(obs, segs) for c in (*sg[k], float(o[2]))]
            head = [s_op, a_op] if k < self.N else [0.0, -1.0]   # the tail's input is fixed
            g_k = gravity(float(xw[k][0]), float(xw[k][1])) if gravity is not None else 0.0
            return np.array(head + [self._accel_grip, self._brake_grip, self._lat_accel_max]
                            + o_k + tail + [g_k], dtype=float)
        brake = np.array([0.0, -1.0])
        plan = None
        sv = self._solver
        try:
            x_init = np.concatenate([z0, [0.0, 0.0, 0.0]])
            sv.set(0, "lbx", x_init)
            sv.set(0, "ubx", x_init)
            for k in range(n_total + 1):
                sv.set(k, "p", stage_p(k))
                sv.set(k, "x", xw[k])
            for k in range(n_total):
                if k < self.N:
                    sv.set(k, "u", np.array(guess[k], dtype=float))
                else:
                    sv.set(k, "u", brake)
                    sv.constraints_set(k, "lbu", brake)
                    sv.constraints_set(k, "ubu", brake)
            status = sv.solve()
            if status in (0, 2):
                plan = [tuple(float(v) for v in np.clip(sv.get(k, "u"), -1.0, 1.0))
                        for k in range(self.N)]
        except Exception:  # noqa: BLE001
            plan = None

        # The solver's iterate warm-starts the next call either way.
        self._sol = plan
        if plan is not None and self._check(z0, plan, obstacles, gravity):
            self._verified_count += 1
            self._backup = list(plan[1:])
            s_c, a_c = plan[0]
            # A few SQP iterations per call leave the first input within about
            # 0.01 (of full scale) of the optimum, so an intervention is a
            # change larger than that.
            tol = self.psf_change_tol
            modified = abs(s_c - s_op) > tol or abs(a_c - a_op) > tol
            return result(s_c, a_c, modified)
        if plan is None:
            self._solve_fail_count += 1
        # No checked solution this step: continue the last checked plan, whose
        # tail is straight full braking.
        if self._backup is None:
            # No plan has been checked yet: straight full braking.
            self._fail_stop_count += 1
            return result(0.0, -1.0, True)
        self._backup_count += 1
        if self._backup:
            s_c, a_c = self._backup.pop(0)
            return result(s_c, a_c, True)
        # The kept plan is into its braking tail.
        return result(0.0, -1.0, True)

    # -- derived mode (--safety-derived) ---------------------------------------------------------
    def _node_gap(self, radius, step):
        """Radius a node must keep so the path to the next node, ``step`` away along an arc of curvature up
        to the full-lock value, keeps ``radius``: chord half-length and sagitta."""
        return np.sqrt((radius + self.kappa_max * step ** 2 / 8.0) ** 2 + (0.5 * step) ** 2)

    def _check_derived(self, z0, plan, obstacles, gravity=None):
        """The check of ``_check`` on the derived model: the plan (steering, pedal rate) and the braking tail,
        then straight braking with the pedal ramped down until the model stops, which it must. Every node keeps
        its clearance widened by the gap to the next node, the ego's travel plus the obstacle's."""
        cmds, hs = self._substep_inputs(plan)
        a_min = max(self._bike.peak_decel(), 0.5)
        if gravity is not None:
            a_min = max(a_min - G * unknown_sine(self.unknown_downgrade), 0.3)
        t_tail = float(z0[3]) / a_min + 2.0 / self.pedal_rate + 5.0
        xs = self._bike.rollout_rk4(z0, cmds, hs, self.cert_dt, t_tail=t_tail, gravity=gravity)
        if xs[-1, 3] > 0.05:
            return False
        n_extra = len(xs) - 1 - len(hs)
        t = np.concatenate([np.cumsum(hs), hs.sum() + self.cert_dt * np.arange(1, n_extra + 1)])
        P, psi = xs[:, :2], xs[:, 2]
        seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
        turned = np.cumsum(np.abs(np.diff(np.unwrap(psi))))
        tube = self.tube_c1 * np.cumsum(seg) + self.tube_eps * np.cumsum(seg * turned)
        P, psi = P[1:], psi[1:]
        # Travel to the next node (the last node's own step for the last one).
        step = np.append(seg[1:], seg[-1]) if len(seg) > 1 else seg
        cx, cy, r_eff, _ = self._last_sense
        if np.any(np.hypot(P[:, 0] - cx, P[:, 1] - cy) + tube + step > r_eff):
            return False
        for o in obstacles or ():
            radius = float(o[2]) + self.disc_radius + self.obstacle_buffer
            ax, ay, bx, by = self._segment(o, t)
            far = np.hypot(np.diff(bx, append=bx[-1]), np.diff(by, append=by[-1]))
            need = self._node_gap(radius + tube, step + far)
            dx, dy = bx - ax, by - ay
            for off in self.disc_offsets:
                qx = P[:, 0] + off * np.cos(psi) - ax
                qy = P[:, 1] + off * np.sin(psi) - ay
                w = np.clip((qx * dx + qy * dy) / (dx * dx + dy * dy + 1e-6), 0.0, 1.0)
                if np.any(np.hypot(qx - w * dx, qy - w * dy) < need):
                    return False
        return True

    def _filter_derived(self, vehicle_state, s_op, a_op, obstacles, x_sense, y_sense, result):
        """``_filter_dynamic`` on the derived model. The plan's inputs are (steering, pedal rate over its limit),
        the model's pedal starts at the applied one, and the executed command is the pedal the first input
        reaches at the end of the period."""
        from .surrogate_dynamics import N_CURVE  # noqa: F401
        p0 = float(self._alpha_applied)
        h0 = self.time_steps[0]
        z0 = np.array([float(vehicle_state["x"]), float(vehicle_state["y"]), float(vehicle_state["psi"]),
                       max(float(vehicle_state.get("u", 0.0)), 0.0), float(vehicle_state.get("v", 0.0)),
                       float(vehicle_state.get("omega", 0.0)), p0])
        r_eff = self._sensed_radius()
        a_term = max(self._bike.peak_decel(), 0.5)
        self._last_sense = (x_sense, y_sense, r_eff, a_term)

        def pedal_after(rate):
            return float(np.clip(p0 + rate * self.pedal_rate * h0, -1.0, 1.0))

        a_reach = float(np.clip(a_op, p0 - self.pedal_rate * h0, p0 + self.pedal_rate * h0))

        obs = self._select_obstacles(z0[0], z0[1], z0[2], obstacles)
        full = {(float(o[0]), float(o[1])): o for o in obstacles or ()}
        obs = [full.get((o[0], o[1]), o) for o in obs]
        while len(obs) < self.n_obs:
            obs.append((z0[0] + 1e4, z0[1] + 1e4, 0.1))
        tail = ([x_sense, y_sense, r_eff, a_term] + list(self._bike.terrain_vec) + list(self._bike.curve))
        steps = self._psf_steps()
        n_total = len(steps)
        t_nodes = np.concatenate([[0.0], np.cumsum(steps)])
        segs = [np.stack(self._segment(o, t_nodes), axis=1) for o in obs]
        gravity = gravity_lookup(vehicle_state, self._bike.L, self.unknown_downgrade)
        if self._sol is not None:
            guess = list(self._sol[1:]) + [self._sol[-1]]
        else:
            first = float(np.clip((a_op - p0) / (self.pedal_rate * h0), -1.0, 1.0))
            guess = [(s_op, first)] + [(s_op, 0.0)] * (self.N - 1)
        cmds, hs = self._substep_inputs(guess)
        xs = self._bike.rollout_rk4(z0, cmds, hs, self.cert_dt, t_tail=0.0, gravity=gravity)
        P = xs[:, :2]
        seg = np.concatenate([[0.0], np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))])
        turned = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(np.unwrap(xs[:, 2]))))])
        xw = [np.concatenate([xs[i], [np.cumsum(seg)[i], turned[i], np.cumsum(seg * turned)[i]]])
              for i in range(0, len(xs), self.psf_rk_steps)]

        def stage_p(k):
            o_k = [c for o, sg in zip(obs, segs) for c in (*sg[k], float(o[2]))]
            head = [s_op, a_op] if k < self.N else [0.0, -1.0]
            g_k = gravity(float(xw[k][0]), float(xw[k][1])) if gravity is not None else 0.0
            h_k = steps[k] if k < n_total else steps[-1]
            return np.array(head + [1.0, self._brake_grip, self._lat_accel_max] + o_k + tail + [g_k, h_k],
                            dtype=float)
        brake = np.array([0.0, -1.0])
        plan = None
        sv = self._solver
        try:
            x_init = np.concatenate([z0, [0.0, 0.0, 0.0]])
            sv.set(0, "lbx", x_init)
            sv.set(0, "ubx", x_init)
            for k in range(n_total + 1):
                sv.set(k, "p", stage_p(k))
                sv.set(k, "x", xw[k])
            for k in range(n_total):
                if k < self.N:
                    sv.set(k, "u", np.array(guess[k], dtype=float))
                else:
                    # The tail's pedal ramps past -1 (the model clips it), so its bound is lifted there.
                    sv.set(k, "u", brake)
                    sv.constraints_set(k, "lbu", brake)
                    sv.constraints_set(k, "ubu", brake)
                    if k > 0:
                        sv.constraints_set(k, "lbx", np.array([-0.5, -1.0e3]))
            status = sv.solve()
            if status in (0, 2):
                plan = [tuple(float(v) for v in np.clip(sv.get(k, "u"), -1.0, 1.0)) for k in range(self.N)]
        except Exception:  # noqa: BLE001
            plan = None

        self._sol = plan
        if plan is not None and self._check_derived(z0, plan, obstacles, gravity):
            self._verified_count += 1
            self._backup = list(plan[1:])
            s_c, rate = plan[0]
            p_c = pedal_after(rate)
            tol = self.psf_change_tol
            modified = abs(s_c - s_op) > tol or abs(p_c - a_reach) > tol
            return result(s_c, p_c, modified)
        if plan is None:
            self._solve_fail_count += 1
        if self._backup is None:
            self._fail_stop_count += 1
            return result(0.0, -1.0, True)
        self._backup_count += 1
        if self._backup:
            s_c, rate = self._backup.pop(0)
            return result(s_c, pedal_after(rate), True)
        return result(0.0, -1.0, True)

    def filter(self, desired_steering, desired_throttle, desired_brake,
               vehicle_state, obstacles, **_):
        t0 = time.time()
        self._filter_count += 1
        x = float(vehicle_state["x"]); y = float(vehicle_state["y"])
        psi = float(vehicle_state["psi"]); v = float(vehicle_state.get("u", 0.0))
        s_op = float(np.clip(desired_steering, -1, 1))
        a_op = float(np.clip(float(desired_throttle) - float(desired_brake), -1, 1))

        def _result(s_val, a_val, modified):
            if self.dynamic_model:
                # The model's tire forces bound the turn, so there is no
                # kinematic authority clamp on this path.
                s_c = float(np.clip(s_val, -1, 1))
            else:
                s_lim = self._lat_steer_limit(v)
                s_c = float(np.clip(s_val, -s_lim, s_lim))
                if abs(s_c - s_val) > 1e-9:
                    self._lat_clamp_count += 1
            a_c = float(np.clip(a_val, -1, 1))
            # Where the plant's rate-limited pedal will be at the next call.
            step = self.pedal_rate * self.time_steps[0]
            self._alpha_applied += float(np.clip(a_c - self._alpha_applied, -step, step))
            if self.dynamic_model and not self.derived:
                # The model's applied acceleration, one period later.
                target = self._bike.settled_accel(a_c, self._accel_grip * self.max_accel,
                                                  self.brake_margin)
                lag = 1.0 - np.exp(-self.time_steps[0] / self._bike.tau_long)
                self._ax_app += lag * (target - self._ax_app)
            tol = self.psf_change_tol if self.dynamic_model else 1e-3
            modified = bool(modified or abs(s_c - s_op) > tol)
            if modified:
                self._modify_count += 1
            ms = (time.time() - t0) * 1e3
            self._solve_ms_ema = 0.9 * self._solve_ms_ema + 0.1 * ms
            self._last_result = SafetyFilterResult(
                s_c, max(a_c, 0.0), max(-a_c, 0.0),
                modified, self.n_obs, ms, self.max_speed, min_clear)
            return self._last_result

        # Clearance is reported over the whole obstacle field rather than the
        # selected subset, so the diagnostic stays valid in a dense field.
        min_clear = float("inf")
        if obstacles:
            min_clear = float(min(np.hypot(o[0] - x, o[1] - y) for o in obstacles))

        # Teleop: a dead link must not leave the last command latched.
        if self._is_command_stale():
            self._stale_brake_count += 1
            return _result(0.0, -1.0, True)

        # The sensed region is centred where the vehicle is now, before the
        # delay advance below, because that is where the obstacles were sensed.
        x_sense, y_sense = x, y

        if self.derived:
            return self._filter_derived(vehicle_state, s_op, a_op, obstacles, x_sense, y_sense, _result)
        if self.dynamic_model:
            return self._filter_dynamic(vehicle_state, s_op, a_op, obstacles,
                                        x_sense, y_sense, _result)

        # advance the initial state over the command round-trip delay
        if self._teleop_delay > 0.0:
            adv = min(2.0 * self._teleop_delay, 0.5)
            x += v * np.cos(psi) * adv
            y += v * np.sin(psi) * adv
        z0 = np.array([x, y, psi, min(v, self.max_speed)])

        # Most-threatening obstacles, on-path-ahead first, padded to n_obs.
        obs = self._select_obstacles(x, y, psi, obstacles)
        while len(obs) < self.n_obs:
            obs.append((x + 1e4, y + 1e4, 0.1))  # inactive

        brake_grip = self._brake_grip
        extra = []
        if self.terminal_set:
            # One braking model for the stages and the terminal set, margined
            # below the surrogate's peak: the plant reaches 87 to 94% of it.
            brake_grip = self.brake_margin * self._brake_grip
            a_term = max(brake_grip * self.max_decel, 0.5)
            r_eff = self._sensed_radius()
            extra = [x_sense, y_sense, r_eff, a_term]
            self._last_sense = (x_sense, y_sense, r_eff, a_term)
        p = np.array([s_op, a_op, self._accel_grip, brake_grip,
                      self._lat_accel_max]
                     + [c for o in obs for c in o] + extra, dtype=float)

        # Stoppable-speed funnel: the filter sees obstacles only through a
        # horizon of N*dt seconds, so it must never carry a speed it cannot
        # bring to zero within that window on the estimated soil -- beyond it,
        # avoidance is deferred to stages the OCP does not constrain. The cap
        # descends from the current speed at the achievable braking rate, so
        # full braking always satisfies it and an over-cap entry state cannot
        # make the problem infeasible.
        a_brk = max(self._brake_grip * self.max_decel, 0.5)
        v_stop = self.N * self.dt * a_brk
        v0 = float(z0[3])

        sv = self._solver
        try:
            sv.set(0, "lbx", z0)
            sv.set(0, "ubx", z0)
            for k in range(self.N + 1):
                sv.set(k, "p", p)
                if k > 0 and not self.terminal_set:
                    v_cap = min(self.max_speed,
                                max(v_stop, v0 - k * self.dt * a_brk))
                    sv.constraints_set(k, "ubx", np.array([v_cap]))
            # warm start toward holding the operator command
            for k in range(self.N):
                sv.set(k, "u", np.array([s_op, a_op]))
            if self.terminal_set:
                # Every executed command is certified by one explicit test: apply
                # it for one step, then brake straight to a stop at the margined
                # soil deceleration, inside the friction ellipse, and the path
                # must stay clear of the whole field and inside the sensed
                # region. The test runs to the stop, not to the end of the
                # horizon, so safety does not depend on N. The OCP only proposes
                # corrections, and a longer horizon can only propose better ones.
                # A plan that relies on a later swerve is not certified, because
                # the kinematic model is optimistic about turning on low grip.
                if self._certified(z0, s_op, a_op, brake_grip, obstacles):
                    self._pass_count += 1
                    return _result(s_op, a_op, False)
                if z0[3] < self.hold_speed:
                    # Standstill hold. Nearly stopped and the command cannot be
                    # certified: hold the backup itself, full brake with the
                    # wheels straight. A minimum-deviation correction here
                    # alternates with certified throttle steps and prefers hard
                    # steering, which is cheap at low speed, so the vehicle
                    # inches along an arc into the obstacle beside the one it
                    # stopped for.
                    self._hold_count += 1
                    return _result(0.0, -1.0, True)
                # Correction: start from braking now, the continuation most
                # likely to be feasible, and let the OCP find the smallest
                # first-step change, swerve included.
                bx, bu = self._backup_rollout(z0, s_op, -1.0, brake_grip)
                for k in range(self.N + 1):
                    sv.set(k, "x", bx[k])
                for k in range(self.N):
                    sv.set(k, "u", bu[k])
            status = sv.solve()
            u0 = sv.get(0, "u")
            ok = status in (0, 2)  # 0 success, 2 max-iter (still usable)
        except Exception:  # noqa: BLE001
            u0 = None; ok = False

        if self.terminal_set:
            return self._certified_correction(z0, s_op, a_op, u0, ok, brake_grip,
                                              obstacles, _result)

        if u0 is None or not ok or not np.all(np.isfinite(u0)):
            # fail closed: hold (authority-clamped) steer, brake
            self._solve_fail_count += 1
            return _result(s_op, -1.0, True)

        # An accepted solve is executed only if its own predicted trajectory
        # stays clear. Where the slack has bought a predicted intrusion, the
        # situation is the unavoidable-collision case and the response is
        # maximal braking, keeping the solver's (clamped) evasive steering.
        if self._plan_is_unsafe(obstacles):
            self._fail_stop_count += 1
            return _result(float(u0[0]), -1.0, True)

        s_out = float(np.clip(u0[0], -1, 1))
        a_out = float(np.clip(u0[1], -1, 1))
        was_modified = bool(abs(s_out - s_op) > 1e-3 or abs(a_out - a_op) > 1e-3)
        return _result(s_out, a_out, was_modified)
