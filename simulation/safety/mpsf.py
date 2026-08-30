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

_ACADOS_BUILD_ROOT = Path(
    os.environ.get("MPSF_BUILD_ROOT",
                   Path.home() / ".cache" / "scm_mpsf_acados"))


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
                 build_solver: bool = True,
                 **_ignored):
        self.L = float(vehicle_params.get("Lf", 1.65)) + float(vehicle_params.get("Lr", 1.65))
        self.N = int(horizon)
        self.dt = float(dt)
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

        self.nx = 4  # [X, Y, psi, v]
        self.nu = 2  # [steer, alpha]
        # params: [s_op, a_op, accel_grip, brake_grip, lat_accel_max,
        #          (ox, oy, orad)*n_obs]. Braking and cornering authority are
        # separate parameters so the tire surrogate sets the stopping and
        # turning capability the soil actually supports; the filter's avoidance
        # mode -- brake or steer -- then follows from those limits rather than
        # from a fixed preference.
        self._np = 5 + 3 * self.n_obs

        self._teleop_delay = max(float(teleop_delay), 0.0)
        self._last_cmd_wall = None  # wall-clock arrival of the latest command
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
        self._lat_clamp_count = 0   # executed steering clamped to soil authority
        self._stale_brake_count = 0
        self._solve_ms_ema = 0.0
        if build_solver:
            self._build_solver()

    # -- interface parity with CBFSafetyFilter ---------------------------------
    def set_teleop_delay(self, delay_s: float):
        self._teleop_delay = max(float(delay_s), 0.0)

    def update_command_age(self, cmd_wall_time: float):
        """Record the arrival of a command for stale-link detection.

        Mirrors CBFSafetyFilter: under teleop, a link that stops delivering
        commands must not leave the last command latched -- ``filter`` brakes
        once no command has arrived within ``stale_cmd_timeout``.
        """
        self._last_cmd_wall = time.time()

    def _is_command_stale(self) -> bool:
        """True if no command received within stale_cmd_timeout (teleop only)."""
        if self._teleop_delay <= 0.0 or self._last_cmd_wall is None:
            return False
        return (time.time() - self._last_cmd_wall) > self.stale_cmd_timeout

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
                from simulation.tire_models.nn_tire_model import load_nn_tire_model
                self._nn = load_nn_tire_model(self.nn_model_dir, terrain)
            nn = self._nn
            best = 0.0
            for slip in np.linspace(0.02, 0.5, 30):   # slip angle sweep (rad)
                fy_f = nn.predict_numeric(float(slip), self._fz_f, 5.0, kappa=0.0,
                                          n_terrain=n, terrain_params=terrain,
                                          rates=np.zeros(3))[1]
                fy_r = nn.predict_numeric(float(slip), self._fz_r, 5.0, kappa=0.0,
                                          n_terrain=n, terrain_params=terrain,
                                          rates=np.zeros(3))[1]
                if not (np.isfinite(fy_f) and np.isfinite(fy_r)):
                    continue
                lat = (2.0 * abs(fy_f) + 2.0 * abs(fy_r)) / self._mass
                if lat > best:
                    best = lat
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
                from simulation.tire_models.nn_tire_model import load_nn_tire_model
                self._nn = load_nn_tire_model(self.nn_model_dir, terrain)
            nn = self._nn
            best = 0.0
            for kappa in np.linspace(-0.9, -0.02, 30):
                fx_f, _ = nn.predict_numeric(0.0, self._fz_f, 5.0, kappa=float(kappa),
                                             n_terrain=n, terrain_params=terrain,
                                             rates=np.zeros(3))
                fx_r, _ = nn.predict_numeric(0.0, self._fz_r, 5.0, kappa=float(kappa),
                                             n_terrain=n, terrain_params=terrain,
                                             rates=np.zeros(3))
                if not (np.isfinite(fx_f) and np.isfinite(fx_r)):
                    continue
                decel = (2.0 * abs(fx_f) + 2.0 * abs(fx_r)) / self._mass
                if decel > best:
                    best = decel
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
                "lat_clamps": self._lat_clamp_count,
                "stale_brakes": self._stale_brake_count}

    # -- acados OCP build ------------------------------------------------------
    def _fingerprint(self) -> str:
        cfg = ("mpsf_nn_steer_v2", self.L, self.N, self.dt, self.n_obs, self.obstacle_buffer,
               self.vehicle_radius, self.ego_forward_extent, self.max_accel,
               self.max_decel, self.max_steer, self.v_terminal, self.max_speed,
               self.w_steer, self.w_alpha, self.w_terminal, self.w_progress, self.slack_lin,
               self.slack_quad, self.lat_accel_max_default)
        return hashlib.sha1(repr(cfg).encode()).hexdigest()[:12]

    def _build_model(self):
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
        model.cost_expr_ext_cost = (self.w_steer * (s - s_op) ** 2
                                    + self.w_alpha * (a - a_op) ** 2
                                    + self.w_progress
                                    * ca.fmax(0.0, progress_floor - v) ** 2)
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
        ocp.dims.N = self.N
        ocp.dims.np = self._np
        ocp.parameter_values = np.zeros(self._np)

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        # control bounds
        ocp.constraints.lbu = np.array([-1.0, -1.0])
        ocp.constraints.ubu = np.array([1.0, 1.0])
        ocp.constraints.idxbu = np.array([0, 1])
        # state bound: v >= 0 (no reverse in the kinematic model)
        ocp.constraints.lbx = np.array([0.0])
        ocp.constraints.ubx = np.array([self.max_speed])
        ocp.constraints.idxbx = np.array([3])
        ocp.constraints.lbx_e = np.array([0.0])
        ocp.constraints.ubx_e = np.array([self.max_speed])
        ocp.constraints.idxbx_e = np.array([3])
        ocp.constraints.x0 = np.zeros(self.nx)

        # path constraints: n_obs obstacle rows + 2 lateral-authority rows;
        # terminal: n_obs obstacle rows only (no control at the terminal node).
        no = self.n_obs
        nh = no + 2
        ocp.constraints.lh = np.zeros(nh)
        ocp.constraints.uh = 1e9 * np.ones(nh)
        ocp.constraints.lh_e = np.zeros(no)
        ocp.constraints.uh_e = 1e9 * np.ones(no)
        # stage 0: the two hard lateral-authority rows on the executed input.
        ocp.constraints.lh_0 = np.zeros(2)
        ocp.constraints.uh_0 = 1e9 * np.ones(2)
        # slack ONLY the obstacle constraints (path + terminal) so the QP stays
        # feasible; the lateral rows are hard (braking is the feasible escape).
        ocp.constraints.idxsh = np.arange(no)
        ocp.constraints.idxsh_e = np.arange(no)
        ocp.cost.zl = self.slack_lin * np.ones(no)
        ocp.cost.zu = self.slack_lin * np.ones(no)
        ocp.cost.Zl = self.slack_quad * np.ones(no)
        ocp.cost.Zu = self.slack_quad * np.ones(no)
        ocp.cost.zl_e = self.slack_lin * np.ones(no)
        ocp.cost.zu_e = self.slack_lin * np.ones(no)
        ocp.cost.Zl_e = self.slack_quad * np.ones(no)
        ocp.cost.Zu_e = self.slack_quad * np.ones(no)

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 30
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 2
        ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.tf = self.N * self.dt
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
            key = (0, d_along) if on_path_ahead else (1, dx * dx + dy * dy)
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
        for o in obstacles:
            ox, oy, orad = float(o[0]), float(o[1]), float(o[2])
            hard_r = (orad + self.vehicle_radius + self.ego_forward_extent)
            for xk in xs:
                if np.hypot(float(xk[0]) - ox, float(xk[1]) - oy) < hard_r:
                    return True
        return False

    def filter(self, desired_steering, desired_throttle, desired_brake,
               vehicle_state, obstacles, **_):
        t0 = time.time()
        self._filter_count += 1
        x = float(vehicle_state["x"]); y = float(vehicle_state["y"])
        psi = float(vehicle_state["psi"]); v = float(vehicle_state.get("u", 0.0))
        s_op = float(np.clip(desired_steering, -1, 1))
        a_op = float(np.clip(float(desired_throttle) - float(desired_brake), -1, 1))

        def _result(s_val, a_val, modified):
            s_lim = self._lat_steer_limit(v)
            s_c = float(np.clip(s_val, -s_lim, s_lim))
            if abs(s_c - s_val) > 1e-9:
                self._lat_clamp_count += 1
            a_c = float(np.clip(a_val, -1, 1))
            modified = bool(modified or abs(s_c - s_op) > 1e-3)
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

        p = np.array([s_op, a_op, self._accel_grip, self._brake_grip,
                      self._lat_accel_max]
                     + [c for o in obs for c in o], dtype=float)

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
                if k > 0:
                    v_cap = min(self.max_speed,
                                max(v_stop, v0 - k * self.dt * a_brk))
                    sv.constraints_set(k, "ubx", np.array([v_cap]))
            # warm start toward holding the operator command
            for k in range(self.N):
                sv.set(k, "u", np.array([s_op, a_op]))
            status = sv.solve()
            u0 = sv.get(0, "u")
            ok = status in (0, 2)  # 0 success, 2 max-iter (still usable)
        except Exception:  # noqa: BLE001
            u0 = None; ok = False

        if u0 is None or not ok or not np.all(np.isfinite(u0)):
            # fail closed: hold (authority-clamped) steer, brake
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
