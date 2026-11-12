"""Model predictive safety filter (MPSF), implemented with acados.

The standard predictive safety filter (Wabersich and Zeilinger, Automatica 2021) on the vehicle model of
``vehicle_model.py``. Every number comes from the plant's specification, the rig surrogate or the bounds identified
on separate plant manoeuvres (``ident/``). The README lists them, with the free design parameters.

Each call solves an OCP from the measured state:

    min   |u_0 - u_op|^2 + w_rest sum_k>0 |u_k - u_op|^2
    s.t.  the vehicle model (state [X, Y, psi, u, v, r, p] plus the tube states [s, turned, S]),
          the body rectangle clear of every selected obstacle's set by the tube at every node,
          every node inside the sensed region less the tube,
          a braking tail (straight, the pedal ramped down at its limit) after the planned stages,
          stopped at the end of the tail.

The inputs are the steering (over full lock) and the pedal rate (over its limit). The deviation is charged on the
pedal the input reaches by the end of the stage, against the operator's pedal. w_rest is a small numerical
regularization of the later inputs.

The plan is then checked without slack on the same model with the same integrator, against every obstacle: the
plan and the braking tail roll out, straight braking continues until the model stops (it must), and the
prediction is extended straight past the stop by the identified overshoot of its braking part. At every pose every
obstacle's set must clear the rectangle by the sideways tube c0 + c1 s + eps S, widened by the gap to the next pose
(the travel of the rectangle's farthest point and the sagitta of its arc). A rock's set is its disc. A vehicle's is
its own rectangle (``footprint.traffic_box``), stretched along its heading between where it is when braking as
hard as the soil allows and where it is when speeding up at the soil's drive capability, from its measured speed
give or take the measurement error. If the plan passes, its first input is executed and the plan kept. Otherwise
the next input of the last kept plan is executed, which ends in straight full braking.

The filter runs on the vehicle and acts on the measured state, so there is no advance over a teleop delay.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import casadi as ca
import numpy as np

from . import footprint as fp
from . import vehicle_spec as spec
from . import vehicle_model as vm
from .common import SafetyFilterResult, nn_loader, nominal_terrain
from .grade_preview import G, gravity_lookup, unknown_sine

_ACADOS_BUILD_ROOT = Path(os.environ.get("MPSF_BUILD_ROOT", Path.home() / ".cache" / "scm_mpsf_acados"))


class MPSFSafetyFilter:
    """The MPSF of the module docstring, with the plant node's filter interface.

    ``time_steps`` is the planned horizon (default 20 x 0.1 s), ``n_obstacles`` how many obstacles the OCP holds
    (the check tests every one), ``psf_max_iter`` the SQP iterations per call. All three are free design
    parameters: the check keeps safety, and they move how often a plan passes it. The acados solver is built once
    per configuration fingerprint and cached on disk, so repeated runs and parallel workers share one compiled
    solver.
    """

    def __init__(self,
                 vehicle_params: dict,
                 ident: dict,
                 powertrain: str = "shafts",
                 time_steps=None,
                 n_obstacles: int = 3,
                 sense_range: float = 30.0,
                 odd_obstacle_radius: float = 1.0,
                 traffic_speed_err: float = 0.0,
                 unknown_downgrade: float = 0.15,
                 psf_max_iter: int = 20,
                 nominal_soil: str = "sand",
                 nn=None,
                 nn_model_dir: str = "nn_models/tire_force_static",
                 stale_cmd_timeout: float = 2.0,
                 teleop_delay: float = 0.0,
                 build_solver: bool = True):
        if ident.get("footprint") != "box":
            raise ValueError("these bounds were identified for another footprint (need the body rectangle)")
        b = ident["powertrains"][powertrain]
        self.vehicle_params = dict(vehicle_params)
        self.powertrain = str(powertrain)
        # Identified bounds and the body rectangle.
        self.box = fp.body_box()
        self.rho_max = fp.corner_reach()
        self.stop_s0 = float(b["stop"]["s0"])
        self.stop_s1 = float(b["stop"]["s1"])
        self.obstacle_buffer = float(b["tube"]["c0"])
        self.tube_c1 = float(b["tube"]["c1"])
        self.tube_eps = float(b["tube"]["eps"])
        # The plant's actuators.
        self.max_steer = spec.MAX_ROAD_WHEEL
        self.pedal_rate = spec.PEDAL_RATE
        self.max_speed = 1.0e3          # no speed cap (the OCP's state bound)
        # Prediction of other vehicles: the speed measurement's error bound. The acceleration bound is the
        # soil's drive capability, set with the soil.
        self.traffic_speed_err = float(traffic_speed_err)
        self.traffic_accel = 1.0
        # Grade preview: ground the preview has not seen counts as this downgrade (rise over run).
        self.unknown_downgrade = float(unknown_downgrade)
        self.sense_range = float(sense_range) if sense_range else 0.0
        self.unseen_obstacle_radius = float(odd_obstacle_radius)
        self.nominal_soil = str(nominal_soil)
        # Horizon: the planned stages, then a braking tail whose input is fixed to the backup, and the last state
        # must be stopped. The check rolls out the same stages.
        self.time_steps = tuple(float(t) for t in time_steps) if time_steps else (0.1,) * 20
        self.N = len(self.time_steps)
        self._tail_steps = (0.2,) * 22
        self.stop_speed = 0.3
        self.n_obs = int(n_obstacles)
        # Numerical choices of the OCP: RK4 with one step per stage (the check repeats it), up to psf_max_iter SQP
        # iterations warm-started from the last call, a small Levenberg-Marquardt term (the cost has no state
        # term), the weight w_rest on the later inputs, slack on the obstacle and sensing rows (the check has
        # none), and a margin between the OCP's nodes and the check's poses.
        self.cert_dt = 0.05
        self.psf_rk_steps = 1
        self.psf_max_iter = int(psf_max_iter)
        self.psf_change_tol = 0.01
        self.psf_lm = 0.01
        self.psf_w_rest = 1e-4
        self.psf_globalization = "FIXED_STEP"
        self.slack_lin = 1.0e3
        self.slack_quad = 1.0e3
        self.ocp_margin = 0.1
        # Obstacle selection for the OCP (the check tests every obstacle): obstacles in a corridor along the
        # heading come first, soonest reached, then the nearest of the rest. The corridor's reach is the published
        # filter's footprint circle (1.0 m radius plus 1.6 m forward extent), plus the tube's c0 and 1.5 m.
        self._select_radius = 1.0
        self._select_extent = 1.6
        self._select_corridor = 1.5

        _dir = Path(str(nn_model_dir)).expanduser()
        if not _dir.is_absolute():
            _dir = Path(__file__).resolve().parents[2] / _dir
        self.nn_model_dir = str(_dir)
        self.stale_cmd_timeout = float(stale_cmd_timeout)

        # State [X, Y, psi, u, v, r, p] and the tube states [path length, accumulated turn, their integral].
        # Inputs [steer, pedal rate], both normalized. Parameters per stage: the operator's (steer, pedal), the
        # segment each obstacle can be on at that stage (ax, ay, bx, by, radius), the sensed region (centre x, y,
        # radius), the soil in the surrogate's feature convention, the per-wheel longitudinal force curve,
        # gravity along the heading at the stage and the stage's length.
        self.nx = 10
        self.nu = 2
        self._p_obs = 2
        self._p_sense = self._p_obs + 5 * self.n_obs
        self._np = self._p_sense + 3 + 6 + vm.N_CURVE + 2

        self._alpha_applied = 0.0   # the applied pedal, tracked through the rate limit
        self._backup = None         # the last checked plan's remaining inputs
        self._sol = None            # last solution, for the warm start
        self._last_sense = None     # centre of the sensed region at the last call
        self._teleop_delay = max(float(teleop_delay), 0.0)
        self._last_cmd_wall = None  # arrival of the latest command, on self._clock
        self._clock = time.time     # set_clock() swaps in simulation time
        self._solver = None
        self._last_result = None
        self._filter_count = 0
        self._modify_count = 0
        self._verified_count = 0
        self._backup_count = 0
        self._fail_stop_count = 0   # no plan checked yet: straight full braking
        self._solve_fail_count = 0  # solver status outside {0, 2} or an exception
        self._stale_brake_count = 0
        self._solve_ms_ema = 0.0

        self._nn = nn if nn is not None else nn_loader()(self.nn_model_dir, nominal_terrain(self.nominal_soil))
        self._bike = vm.SurrogateBicycle(self._nn, self.vehicle_params, self.powertrain)
        self._set_soil(nominal_terrain(self.nominal_soil))
        self._soil = f"nominal:{self.nominal_soil}"
        if build_solver:
            self._build_solver()

    def _psf_steps(self):
        """Stage lengths of the OCP: the planned stages, then the tail."""
        return tuple(self.time_steps) + tuple(self._tail_steps)

    def _set_soil(self, terrain):
        self._bike.set_terrain(terrain)
        self.traffic_accel = self._bike.drive_capability()

    # -- the plant node's interface ------------------------------------------------------------------
    def set_clock(self, clock):
        """Use ``clock()`` (seconds) for command ages. The plant node passes simulation time."""
        self._clock = clock

    def set_teleop_delay(self, delay_s: float):
        """The link's one-way delay. Above zero, commands come over a link, and a stale one brakes."""
        self._teleop_delay = max(float(delay_s), 0.0)

    def update_command_age(self):
        """A command has arrived."""
        self._last_cmd_wall = self._clock()

    def _is_command_stale(self) -> bool:
        if self._teleop_delay <= 0.0 or self._last_cmd_wall is None:
            return False
        return (self._clock() - self._last_cmd_wall) > self.stale_cmd_timeout

    def update_terrain(self, terrain_params):
        """The soil belief, in the surrogate's internal convention."""
        if not isinstance(terrain_params, dict):
            return
        try:
            self._set_soil(terrain_params)
            self._soil = (f"belief:n={float(terrain_params['n']):.2f},"
                          f"phi={float(terrain_params['phi']):.1f}")
        except Exception as e:  # noqa: BLE001
            print(f"  [MPSF] soil update failed ({type(e).__name__}: {e})")

    def get_diagnostics(self) -> dict:
        rate = (self._modify_count / self._filter_count) if self._filter_count else 0.0
        return {"filter_calls": self._filter_count,
                "interventions": self._modify_count,
                "intervention_rate": rate,
                "mean_solve_ms": self._solve_ms_ema,
                "soil": self._soil,
                "verified_solves": self._verified_count,
                "backup_steps": self._backup_count,
                "fail_stops": self._fail_stop_count,
                "solve_fails": self._solve_fail_count,
                "horizon_s": float(sum(self.time_steps)),
                "stale_brakes": self._stale_brake_count}

    # -- acados OCP build ----------------------------------------------------------------------------
    def _fingerprint(self) -> str:
        b = self._bike
        cfg = ("mpsf_v1", self.time_steps, self._tail_steps, self.stop_speed, self.n_obs, self.obstacle_buffer,
               self.tube_c1, self.tube_eps, self.box, self.rho_max, self.stop_s0, self.stop_s1, self.max_steer,
               self.pedal_rate, self.max_speed, self.ocp_margin, self.slack_lin, self.slack_quad,
               self.psf_rk_steps, self.psf_max_iter, self.psf_lm, self.psf_w_rest, self.psf_globalization,
               Path(self.nn_model_dir).name, self._nn.n_params, float(np.sum(self._nn._X_mean)),
               b.M, b.Izz, b.Lf, b.Lr, b.h_cg, b.T, b.blend_lo, b.blend_hi, b.tau_kin, b.f_brake_max,
               vm.ALPHA_MAX, tuple(vm.KAPPA_BRAKE), tuple(vm.KAPPA_DRIVE), vm.BRAKE_FADE_SPEED,
               self.powertrain, float(np.sum(b.table)), tuple(b.throttles))
        return hashlib.sha1(repr(cfg).encode()).hexdigest()[:12]

    def _build_model(self):
        """The OCP's model, constraints and cost (module docstring)."""
        from acados_template import AcadosModel
        x = ca.SX.sym("x", self.nx)   # X, Y, psi, u, v, r, p, s, turned, S
        u = ca.SX.sym("u", self.nu)   # steer, pedal rate
        p = ca.SX.sym("p", self._np)
        X, Y, psi, uu, r = x[0], x[1], x[2], x[3], x[5]  # noqa: E741
        s_len, turned, S = x[7], x[8], x[9]
        st, al = u[0], u[1]
        q = self._p_sense
        cx, cy, r_eff = p[q], p[q + 1], p[q + 2]
        terrain = p[q + 3:q + 9]
        curve = p[q + 9:q + 9 + vm.N_CURVE]
        g_along = p[q + 9 + vm.N_CURVE]
        h_stage = p[q + 10 + vm.N_CURVE]
        f_vehicle = self._bike.f_expl_sym(x[0:7], st * self.max_steer, al * self.pedal_rate, terrain, curve,
                                          g_along=g_along)
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
        x_r, x_f, w = self.box

        def clear_rows(tube, x_front):
            # Squared distance from the obstacle to the body rectangle, less the clearance squared. A vehicle's
            # segment is represented by its point nearest the centre of mass (the check is exact).
            rows = []
            for j in range(self.n_obs):
                ax, ay, bx, by, orad = (p[self._p_obs + 5 * j + i] for i in range(5))
                dx, dy = bx - ax, by - ay
                t = ca.fmin(ca.fmax(((X - ax) * dx + (Y - ay) * dy) / (dx * dx + dy * dy + 1e-6), 0.0), 1.0)
                ox, oy = ax + t * dx, ay + t * dy
                qx = (ox - X) * hx + (oy - Y) * hy
                qy = -(ox - X) * hy + (oy - Y) * hx
                radius = orad + self.obstacle_buffer + self.ocp_margin + tube
                # Between nodes: travel of the box's farthest point and the sagitta of its arc.
                step = uu * h_stage + self.rho_max * ca.sqrt(r ** 2 + 1e-4) * h_stage
                sag = step * ca.sqrt(r ** 2 + 1e-4) * h_stage / 8.0
                need = ca.sqrt((radius + sag) ** 2 + (step / 2.0) ** 2)
                rows.append(fp.box_sym(qx, qy, x_r, x_front, w) - need ** 2)
            return rows

        tube = self.tube_c1 * s_len + self.tube_eps * S
        inner = r_eff - tube - self.ocp_margin
        # At the last node the rectangle reaches past the predicted stop by the identified overshoot (here on the
        # whole path, a conservative stand-in for its braking part; the check uses the braking part).
        overshoot = self.stop_s0 + (self.stop_s1 - 1.0) * s_len
        model.con_h_expr = ca.vertcat(*clear_rows(tube, x_f),
                                      inner ** 2 - ((X - cx) ** 2 + (Y - cy) ** 2))
        model.con_h_expr_e = ca.vertcat(*clear_rows(tube, x_f + overshoot),
                                        inner ** 2 - ((X - cx) ** 2 + (Y - cy) ** 2),
                                        self.stop_speed - uu)

        # The pedal the input reaches by the end of the stage, against the operator's.
        s_op, a_op = p[0], p[1]
        deviation = (st - s_op) ** 2 + (x[6] + al * self.pedal_rate * h_stage - a_op) ** 2
        model.cost_expr_ext_cost_0 = deviation
        model.cost_expr_ext_cost = self.psf_w_rest * deviation
        model.cost_expr_ext_cost_e = ca.SX(0.0)
        return model

    def _build_solver(self):
        from acados_template import AcadosOcp, AcadosOcpSolver
        model = self._build_model()
        ocp = AcadosOcp()
        ocp.model = model
        steps = self._psf_steps()
        ocp.dims.N = len(steps)
        ocp.dims.np = self._np
        ocp.parameter_values = np.zeros(self._np)

        ocp.cost.cost_type_0 = "EXTERNAL"
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        ocp.constraints.lbu = np.array([-1.0, -1.0])
        ocp.constraints.ubu = np.array([1.0, 1.0])
        ocp.constraints.idxbu = np.array([0, 1])
        # Speed and pedal. Braking fades at a standstill, and a hard zero speed bound makes the QP infeasible
        # once the brake saturates at wheel lock, so the speed may dip slightly below zero.
        v_lo = -0.5
        ocp.constraints.lbx = np.array([v_lo, -1.0])
        ocp.constraints.ubx = np.array([self.max_speed, 1.0])
        ocp.constraints.idxbx = np.array([3, 6])
        ocp.constraints.lbx_e = np.array([v_lo])
        ocp.constraints.ubx_e = np.array([self.max_speed])
        ocp.constraints.idxbx_e = np.array([3])
        ocp.constraints.x0 = np.zeros(self.nx)

        # Path rows: one per obstacle, then the sensed region. Terminal: those, then stopped. All slacked.
        n_sl = self.n_obs + 1
        n_e = self.n_obs + 2
        ocp.constraints.lh = np.zeros(n_sl)
        ocp.constraints.uh = 1e9 * np.ones(n_sl)
        ocp.constraints.lh_e = np.zeros(n_e)
        ocp.constraints.uh_e = 1e9 * np.ones(n_e)
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

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        # Classic RK4, the scheme the check repeats.
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = self.psf_rk_steps
        ocp.solver_options.nlp_solver_max_iter = self.psf_max_iter
        ocp.solver_options.levenberg_marquardt = self.psf_lm
        ocp.solver_options.globalization = self.psf_globalization
        ocp.solver_options.tf = float(sum(steps))
        ocp.solver_options.time_steps = np.array(steps)
        ocp.solver_options.print_level = 0

        import fcntl
        fpr = self._fingerprint()
        _ACADOS_BUILD_ROOT.mkdir(parents=True, exist_ok=True)
        exp = _ACADOS_BUILD_ROOT / f"mpsf_{fpr}"
        exp.mkdir(parents=True, exist_ok=True)
        ocp.code_export_directory = str(exp / "c_generated_code")
        json_file = str(exp / "acados_ocp.json")
        so_file = exp / "c_generated_code" / f"libacados_ocp_solver_{model.name}.so"

        def _load_cached() -> bool:
            if so_file.exists() and Path(json_file).exists():
                try:
                    self._solver = AcadosOcpSolver(None, json_file=json_file, build=False, generate=False)
                    return True
                except Exception:  # noqa: BLE001
                    return False
            return False

        # Serialize code generation across parallel workers sharing a fingerprint directory: the first worker to
        # take the lock generates and compiles, the rest wait and load the cached shared object.
        if _load_cached():
            return
        lock_path = _ACADOS_BUILD_ROOT / f"mpsf_{fpr}.lock"
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                if not _load_cached():
                    self._solver = AcadosOcpSolver(ocp, json_file=json_file, verbose=False)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    # -- obstacles -------------------------------------------------------------------------------------
    def _select_obstacles(self, x, y, psi, obstacles):
        """The n_obs obstacles the OCP holds: those in the corridor ahead, soonest touched first (then by
        lateral offset), then the nearest of the rest. The check tests every obstacle."""
        cpsi, spsi = np.cos(psi), np.sin(psi)
        scored = []
        for o in obstacles:
            ox, oy, orad = float(o[0]), float(o[1]), float(o[2])
            dx, dy = ox - x, oy - y
            d_along = dx * cpsi + dy * spsi
            d_perp = -dx * spsi + dy * cpsi
            safe_r = orad + self._select_radius + self._select_extent + self.obstacle_buffer
            corridor = safe_r + self._select_corridor
            if (d_along > -safe_r) and (abs(d_perp) < corridor):
                touch = d_along - np.sqrt(max(safe_r * safe_r - d_perp * d_perp, 0.0))
                key = (0, touch, abs(d_perp))
            else:
                key = (1, dx * dx + dy * dy)
            scored.append((key, (ox, oy, orad)))
        scored.sort(key=lambda t: t[0])
        return [o for _, o in scored[: self.n_obs]]

    def _sensed_radius(self) -> float:
        """Radius around the vehicle that every planned node must stay inside: the sensing range less the
        largest obstacle the scenario holds, the rectangle's farthest corner and the tube's c0 (an obstacle just
        past the range may not be reached). No range means no limit."""
        if self.sense_range <= 0.0:
            return 1.0e4
        margin = self.unseen_obstacle_radius + self.rho_max + self.obstacle_buffer
        return max(self.sense_range - margin, 1.0)

    def _traffic_decel(self) -> float:
        """Hardest braking assumed for another vehicle: locked wheels on the believed soil, which is how the
        plant's vehicles brake at full pedal."""
        return max(self._bike.peak_decel(), 0.5)

    def _segment(self, o, t):
        """Ends (ax, ay), (bx, by) of the segment an obstacle can be on at each time t (s after its measurement).
        A rock, or an obstacle given without a velocity, stays where it is."""
        t = np.asarray(t, dtype=float)
        ox, oy = float(o[0]), float(o[1])
        if len(o) < 6:
            x0, y0 = np.full_like(t, ox), np.full_like(t, oy)
            return x0, y0, x0, y0
        vx, vy = float(o[4]), float(o[5])
        if len(o) >= 7:
            # Heading given: the speed is the velocity along it, which a noisy measurement can make negative.
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

    @staticmethod
    def _is_vehicle(o):
        return len(o) >= 6 and bool(o[3])

    @staticmethod
    def _heading_of(o):
        """Unit heading of a vehicle obstacle: its reported heading, else its velocity's direction."""
        if len(o) >= 7:
            return float(np.cos(float(o[6]))), float(np.sin(float(o[6])))
        speed = float(np.hypot(float(o[4]), float(o[5])))
        if speed > 1e-9:
            return float(o[4]) / speed, float(o[5]) / speed
        return 1.0, 0.0

    # -- the check ---------------------------------------------------------------------------------------
    def _substep_inputs(self, plan):
        """Per-substep commands and step lengths for the planned stages followed by the braking tail, as the OCP
        integrates them."""
        steps = self._psf_steps()
        inputs = list(plan) + [(0.0, -1.0)] * (len(steps) - len(plan))
        cmds, hs = [], []
        for (s_c, a_c), dt_k in zip(inputs, steps):
            for _ in range(self.psf_rk_steps):
                cmds.append((s_c, a_c))
                hs.append(dt_k / self.psf_rk_steps)
        return np.array(cmds), np.array(hs)

    def _check(self, z0, plan, obstacles, gravity=None):
        """The check of the module docstring. ``gravity(X, Y)`` is gravity along the heading where the vehicle
        is (none: flat ground). A vehicle's set only moves forward along its heading, so between two poses it
        lies in its rectangle stretched from its slow position at the first to its fast position at the second,
        which the check uses."""
        cmds, hs = self._substep_inputs(plan)
        a_min = max(self._bike.peak_decel(), 0.5)
        if gravity is not None:
            a_min = max(a_min - G * unknown_sine(self.unknown_downgrade), 0.3)
        t_tail = float(z0[3]) / a_min + 2.0 / self.pedal_rate + 5.0
        xs = self._bike.rollout_rk4(z0, cmds, hs, self.cert_dt, t_tail=t_tail, gravity=gravity)
        if xs[-1, 3] > 0.05:
            return False
        n_extra = len(xs) - 1 - len(hs)
        t = np.concatenate([[0.0], np.cumsum(hs), hs.sum() + self.cert_dt * np.arange(1, n_extra + 1)])
        X, Y, psi = xs[:, 0], xs[:, 1], np.unwrap(xs[:, 2])
        seg = np.hypot(np.diff(X), np.diff(Y))
        s_len = np.concatenate([[0.0], np.cumsum(seg)])
        turned = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(psi)))])
        S = np.concatenate([[0.0], np.cumsum(seg * turned[1:])])
        # The final full-brake command: from the last planned input that is not a full ramp down (the tail is one).
        rates = np.concatenate([np.asarray(plan, dtype=float)[:, 1], np.full(len(hs) - len(plan) + n_extra, -1.0)])
        k = len(rates)
        while k > 0 and rates[k - 1] <= -0.999:
            k -= 1
        ext = self.stop_s0 + (self.stop_s1 - 1.0) * float(s_len[-1] - s_len[min(k, len(s_len) - 1)])
        n_ext = max(int(np.ceil(ext / 0.1)), 1)
        a = np.arange(1, n_ext + 1) * (ext / n_ext)
        X = np.concatenate([X, X[-1] + a * np.cos(psi[-1])])
        Y = np.concatenate([Y, Y[-1] + a * np.sin(psi[-1])])
        psi = np.concatenate([psi, np.full(n_ext, psi[-1])])
        s_len = np.concatenate([s_len, s_len[-1] + a])
        S = np.concatenate([S, np.full(n_ext, S[-1])])
        t_end = t[-1]
        t = np.concatenate([t, np.full(n_ext, t_end)])
        tube = self.obstacle_buffer + self.tube_c1 * s_len + self.tube_eps * S
        # Travel to the next pose of the rectangle's farthest point, and the heading change.
        dpsi = np.abs(np.diff(psi, append=psi[-1]))
        step = np.hypot(np.diff(X, append=X[-1]), np.diff(Y, append=Y[-1])) + self.rho_max * dpsi
        x_r, x_f, w = self.box
        if self.sense_range > 0.0:
            # Every point of the rectangle, with its tube, inside the sensed range less the largest unseen obstacle.
            cx, cy = self._last_sense
            limit = self.sense_range - self.unseen_obstacle_radius
            if np.any(np.hypot(X - cx, Y - cy) + self.rho_max + tube + step > limit):
                return False
        for o in obstacles or ():
            ax, ay, bx, by = self._segment(o, t)
            # Past the predicted stop the plant may arrive later, by up to the overshoot ratio.
            if n_ext:
                _, _, bx_late, by_late = self._segment(o, np.array([t_end * max(self.stop_s1, 1.0)]))
                bx[-n_ext:], by[-n_ext:] = bx_late[0], by_late[0]
            # The obstacle's set from this pose to the next: slow end now, fast end at the next pose.
            bx = np.append(bx[1:], bx[-1])
            by = np.append(by[1:], by[-1])
            sag = step * dpsi / 8.0
            if self._is_vehicle(o):
                hx, hy = self._heading_of(o)
                t_xr, t_xf, t_w = fp.traffic_box()
                length = np.maximum((bx - ax) * hx + (by - ay) * hy, 0.0)
                cs = fp.rect_corners(ax, ay, hx, hy, t_xr, t_xf + length, t_w)
                qx, qy = fp.to_body(cs[..., 0], cs[..., 1], X[:, None], Y[:, None], psi[:, None])
                dist = fp.box_quad(qx, qy, x_r, x_f, w)
                radius = tube
            else:
                qax, qay = fp.to_body(ax, ay, X, Y, psi)
                qbx, qby = fp.to_body(bx, by, X, Y, psi)
                dist = fp.segment_box(qax, qay, qbx, qby, x_r, x_f, w)
                radius = float(o[2]) + tube
            need = np.sqrt((radius + sag) ** 2 + (step / 2.0) ** 2)
            if np.any(dist < need):
                return False
        return True

    # -- the filter step -------------------------------------------------------------------------------
    def _step(self, vehicle_state, s_op, a_op, obstacles, x_sense, y_sense, result):
        """Solve the OCP from the measured state, check its plan, and execute (module docstring). The plan's
        inputs are (steering, pedal rate over its limit), the model's pedal starts at the applied one, and the
        executed command is the pedal the first input reaches at the end of the period."""
        p0 = float(self._alpha_applied)
        h0 = self.time_steps[0]
        z0 = np.array([float(vehicle_state["x"]), float(vehicle_state["y"]), float(vehicle_state["psi"]),
                       max(float(vehicle_state.get("u", 0.0)), 0.0), float(vehicle_state.get("v", 0.0)),
                       float(vehicle_state.get("omega", 0.0)), p0])
        r_eff = self._sensed_radius()
        self._last_sense = (x_sense, y_sense)

        def pedal_after(rate):
            return float(np.clip(p0 + rate * self.pedal_rate * h0, -1.0, 1.0))

        a_reach = float(np.clip(a_op, p0 - self.pedal_rate * h0, p0 + self.pedal_rate * h0))

        obs = self._select_obstacles(z0[0], z0[1], z0[2], obstacles)
        full = {(float(o[0]), float(o[1])): o for o in obstacles or ()}
        obs = [full.get((o[0], o[1]), o) for o in obs]
        while len(obs) < self.n_obs:
            obs.append((z0[0] + 1e4, z0[1] + 1e4, 0.1))
        tail = [x_sense, y_sense, r_eff] + list(self._bike.terrain_vec) + list(self._bike.curve)
        steps = self._psf_steps()
        n_total = len(steps)
        t_nodes = np.concatenate([[0.0], np.cumsum(steps)])
        segs, radii = [], []
        t_xr, t_xf, t_w = fp.traffic_box()
        for o in obs:
            sg = np.stack(self._segment(o, t_nodes), axis=1)
            if self._is_vehicle(o):
                # The planner sees a vehicle as the capsule round its centreline from rear to front edge, of radius
                # its half-width: its rectangle lies inside, no wider. The check uses the rectangle.
                hx, hy = self._heading_of(o)
                sg = sg + np.array([t_xr * hx, t_xr * hy, t_xf * hx, t_xf * hy])
                radii.append(t_w)
            else:
                radii.append(float(o[2]))
            segs.append(sg)
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
            o_k = [c for rad, sg in zip(radii, segs) for c in (*sg[k], rad)]
            head = [s_op, a_op] if k < self.N else [0.0, -1.0]
            g_k = gravity(float(xw[k][0]), float(xw[k][1])) if gravity is not None else 0.0
            h_k = steps[k] if k < n_total else steps[-1]
            return np.array(head + o_k + tail + [g_k, h_k], dtype=float)
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

        # The solver's iterate warm-starts the next call either way.
        self._sol = plan
        if plan is not None and self._check(z0, plan, obstacles, gravity):
            self._verified_count += 1
            self._backup = list(plan[1:])
            s_c, rate = plan[0]
            p_c = pedal_after(rate)
            # A few SQP iterations leave the first input within about 0.01 (of full scale) of the optimum, so an
            # intervention is a change larger than that.
            tol = self.psf_change_tol
            modified = abs(s_c - s_op) > tol or abs(p_c - a_reach) > tol
            return result(s_c, p_c, modified)
        if plan is None:
            self._solve_fail_count += 1
        if self._backup is None:
            # No plan has been checked yet: straight full braking.
            self._fail_stop_count += 1
            return result(0.0, -1.0, True)
        self._backup_count += 1
        if self._backup:
            s_c, rate = self._backup.pop(0)
            return result(s_c, pedal_after(rate), True)
        # The kept plan is into its braking tail.
        return result(0.0, -1.0, True)

    def filter(self, desired_steering, desired_throttle, desired_brake, vehicle_state, obstacles, **_):
        t0 = time.time()
        self._filter_count += 1
        x = float(vehicle_state["x"])
        y = float(vehicle_state["y"])
        s_op = float(np.clip(desired_steering, -1, 1))
        a_op = float(np.clip(float(desired_throttle) - float(desired_brake), -1, 1))

        def _result(s_val, a_val, modified):
            s_c = float(np.clip(s_val, -1, 1))
            a_c = float(np.clip(a_val, -1, 1))
            # Where the plant's rate-limited pedal will be at the next call.
            step = self.pedal_rate * self.time_steps[0]
            self._alpha_applied += float(np.clip(a_c - self._alpha_applied, -step, step))
            modified = bool(modified or abs(s_c - s_op) > self.psf_change_tol)
            if modified:
                self._modify_count += 1
            ms = (time.time() - t0) * 1e3
            self._solve_ms_ema = 0.9 * self._solve_ms_ema + 0.1 * ms
            self._last_result = SafetyFilterResult(
                s_c, max(a_c, 0.0), max(-a_c, 0.0),
                modified, self.n_obs, ms, self.max_speed, min_clear)
            return self._last_result

        # Clearance is reported over the whole obstacle field rather than the selected subset.
        min_clear = float("inf")
        if obstacles:
            min_clear = float(min(np.hypot(o[0] - x, o[1] - y) for o in obstacles))

        # Teleop: a dead link must not leave the last command latched.
        if self._is_command_stale():
            self._stale_brake_count += 1
            return _result(0.0, -1.0, True)

        # The sensed region is centred where the vehicle is now, where the obstacles were sensed.
        return self._step(vehicle_state, s_op, a_op, obstacles, x, y, _result)
