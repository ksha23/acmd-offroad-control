"""Dynamic single-track model driven by the rig tire surrogate, for the MPSF.

The tracking NMPC predicts with a dynamic bicycle model whose lateral and yaw
accelerations come from the surrogate's lateral tire forces (paper, model
f_c). The MPSF used a kinematic bicycle instead. Its vehicle turned exactly
where the wheels pointed, and the surrogate entered only as scalar limits on
braking and cornering. On low-grip soil that is optimistic: the tires slip,
the vehicle runs wide, and braking locks the wheels and takes away the
cornering force the kinematic model still granted.

This module gives the MPSF the NMPC's physics:

    X' = u cos(psi) - v sin(psi)
    Y' = u sin(psi) + v cos(psi)
    psi' = r
    u' = a                           (braking fades to zero at standstill)
    v' = (Fyf + Fyr) / M - u r
    r' = (Lf Fyf - Lr Fyr) / Izz
    a' = (a_target(pedal) - a) / tau_long

with Fyf and Fyr the surrogate's lateral forces at each axle's slip angle,
normal load (longitudinal and lateral load transfer, as in the NMPC) and
longitudinal slip ratio.

Braking is a torque balance on the surrogate's longitudinal curve. The pedal
asks each wheel for a braking force of pedal x 4000 N m / 0.467 m (the HMMWV's
brake torque and tire radius). The tire delivers that force at the slip ratio
where the surrogate's Fx(kappa) reaches it. A demand past the curve's
locked-wheel end locks the wheel, which then gives the locked-wheel force and
almost no cornering force. On clay that happens above a pedal of about 0.22,
which is what the plant does. The deceleration target is the four wheels' force over
the mass, times the filter's braking margin. Traction keeps the filter's
scalar map, grip times the acceleration limit, with the slip ratio again read
off the surrogate's curve. The applied acceleration a follows the target with a
0.1 s lag, fitted to the plant (model_check.py), which stands for the plant's
pedal rate limit and the brake force build-up while the wheels spin down.

The steering-rate input of the surrogate is held at zero (quasi-steady). Its
training spread is 0.31 rad/s, and the rates of a swerve are far outside it,
where the network extrapolates.

Below a few m/s the slip angles are ill conditioned (the surrogate's training
speeds start near 2 m/s), so the lateral states relax toward the kinematic
values there instead. The same equations exist in numpy, for the certificate
rollouts, and in CasADi, for the OCP.
"""
from __future__ import annotations

import numpy as np
import casadi as ca

G = 9.81
ALPHA_MAX = 0.55          # slip-angle clamp, the surrogate's training envelope (as in the NMPC)
KAPPA_LOCK = -0.95
BRAKE_TORQUE_MAX = 4000.0  # N m per wheel, HMMWV_BrakeShafts / HMMWV_BrakeSimple
WHEEL_RADIUS = 0.467       # m, HMMWV rigid tire
BRAKE_FADE_SPEED = 0.25    # m/s, scale of the braking fade at standstill
# Slip-ratio knots for the surrogate's longitudinal curve, braking and traction.
KAPPA_BRAKE = np.array([-0.95, -0.7, -0.5, -0.35, -0.25, -0.18, -0.12, -0.07, -0.03, 0.0])
KAPPA_DRIVE = np.array([0.0, 0.03, 0.07, 0.12, 0.2, 0.35, 0.6, 0.95])
N_CURVE = 2 * len(KAPPA_BRAKE) + len(KAPPA_DRIVE)


def _strictly_increasing(f):
    """Force values made strictly increasing, so the inverse is a function."""
    f = np.maximum.accumulate(np.asarray(f, dtype=float))
    for i in range(1, len(f)):
        if f[i] <= f[i - 1]:
            f[i] = f[i - 1] + 1e-3
    return f


class SurrogateBicycle:
    """Dynamic bicycle with surrogate tire forces. State [X, Y, psi, u, v, r]."""

    def __init__(self, nn, vehicle_params: dict, max_steer: float = 0.49,
                 lateral_load_transfer: bool = True,
                 blend_lo: float = 1.5, blend_hi: float = 3.0, tau_kin: float = 0.1,
                 tau_long: float = 0.1):
        self.nn = nn
        vp = vehicle_params
        self.M = float(vp.get("M", 2573.0))
        self.Izz = float(vp.get("Izz", 3570.0))
        self.Lf = float(vp.get("Lf", 1.593))
        self.Lr = float(vp.get("Lr", 1.709))
        self.L = self.Lf + self.Lr
        self.h_cg = float(vp.get("h_cg", 0.65))
        self.T = float(vp.get("T", 1.8194))
        self.max_steer = float(max_steer)
        self.llt = bool(lateral_load_transfer)
        self.blend_lo = float(blend_lo)
        self.blend_hi = float(blend_hi)
        self.tau_kin = float(tau_kin)
        self.tau_long = float(tau_long)
        self.f_brake_max = BRAKE_TORQUE_MAX / WHEEL_RADIUS
        self.fz_f = self.M * G * self.Lr / self.L / 2.0
        self.fz_r = self.M * G * self.Lf / self.L / 2.0
        self.terrain_vec = None    # [Kphi, Kc, n, c, phi_feature, k]
        self.curve = None          # per-wheel Fx at the knots: front brake, rear brake, drive

    # -- soil -----------------------------------------------------------------
    def set_terrain(self, terrain: dict):
        """Store the soil in the surrogate's feature convention and tabulate the
        per-wheel longitudinal curve at the static loads."""
        phi = self.nn.phi_feature_value(float(terrain["phi"]))
        self.terrain_vec = np.array([float(terrain["Kphi"]), float(terrain["Kc"]),
                                     float(terrain["n"]), float(terrain["c"]), phi,
                                     float(terrain["k"])])
        nb, nd = len(KAPPA_BRAKE), len(KAPPA_DRIVE)
        alpha = np.zeros(2 * nb + nd)
        fz = np.concatenate([np.full(nb, self.fz_f), np.full(nb, self.fz_r),
                             np.full(nd, 0.5 * (self.fz_f + self.fz_r))])
        kap = np.concatenate([KAPPA_BRAKE, KAPPA_BRAKE, KAPPA_DRIVE])
        fx, _ = self._tire(alpha, fz, np.full_like(alpha, 5.0), kap)
        self.curve = np.concatenate([_strictly_increasing(fx[:nb]),
                                     _strictly_increasing(fx[nb:2 * nb]),
                                     _strictly_increasing(fx[2 * nb:])])

    def lock_force(self):
        """Per-wheel locked-wheel braking force (N), front and rear, positive."""
        nb = len(KAPPA_BRAKE)
        return float(-self.curve[0]), float(-self.curve[nb])

    def peak_decel(self) -> float:
        """Four-wheel locked-wheel deceleration at the static loads (m/s^2)."""
        cf, cr = self.lock_force()
        return 2.0 * (cf + cr) / self.M

    def longitudinal(self, pedal, accel_gain, brake_margin):
        """(a_x, kappa_front, kappa_rear) for an array of pedal values in [-1, 1].

        ``accel_gain`` is the traction acceleration at full throttle (m/s^2).
        """
        pedal = np.asarray(pedal, dtype=float)
        nb = len(KAPPA_BRAKE)
        cf, cr = self.lock_force()
        demand = np.maximum(-pedal, 0.0) * self.f_brake_max
        brake_force = 2.0 * (np.minimum(demand, cf) + np.minimum(demand, cr))
        ax_brake = -brake_margin * brake_force / self.M
        k_f = np.interp(-demand, self.curve[:nb], KAPPA_BRAKE)
        k_r = np.interp(-demand, self.curve[nb:2 * nb], KAPPA_BRAKE)
        ax_drive = accel_gain * np.maximum(pedal, 0.0)
        k_d = np.interp(self.M * ax_drive / 4.0, self.curve[2 * nb:], KAPPA_DRIVE)
        braking = pedal < 0.0
        ax = np.where(braking, ax_brake, ax_drive)
        return ax, np.where(braking, k_f, k_d), np.where(braking, k_r, k_d)

    # -- numpy ------------------------------------------------------------------
    def _tire(self, alpha, fz, u, kappa):
        """Per-wheel (Fx, Fy) from the surrogate for row vectors of inputs."""
        n_rows = len(alpha)
        feats = np.empty((n_rows, 11))
        feats[:, 0] = kappa
        feats[:, 1] = alpha
        feats[:, 2] = u
        feats[:, 3] = fz
        feats[:, 4] = 0.0
        feats[:, 5:] = self.terrain_vec[[0, 1, 2, 3, 4, 5]]
        out = self.nn.predict_feature_rows(feats)
        return out[:, 0], out[:, 1]

    def _loads(self, ax, ay):
        """Per-wheel loads, front outer/inner and rear outer/inner (NMPC form)."""
        M, h = self.M, self.h_cg
        fz_f = (M * G * self.Lr - M * ax * h) / self.L / 2.0
        fz_r = (M * G * self.Lf + M * ax * h) / self.L / 2.0
        if not self.llt:
            return fz_f, fz_f, fz_r, fz_r
        d = M * ay * h / self.T / 2.0
        return (np.minimum(fz_f + d, 1.9 * fz_f), np.maximum(fz_f - d, 0.1 * fz_f),
                np.minimum(fz_r + d, 1.9 * fz_r), np.maximum(fz_r - d, 0.1 * fz_r))

    def lateral_forces(self, u, v, r, delta, ax, k_f, k_r):
        """Axle lateral forces (Fyf, Fyr) in N, for arrays of states."""
        u_s = np.maximum(np.abs(u), 0.5)
        a_f = np.clip(delta - np.arctan2(v + self.Lf * r, u_s), -ALPHA_MAX, ALPHA_MAX)
        a_r = np.clip(-np.arctan2(v - self.Lr * r, u_s), -ALPHA_MAX, ALPHA_MAX)
        fo_f, fi_f, fo_r, fi_r = self._loads(ax, u * r)
        b = len(u)
        alpha = np.concatenate([a_f, a_f, a_r, a_r])
        fz = np.concatenate([np.broadcast_to(fo_f, b), np.broadcast_to(fi_f, b),
                             np.broadcast_to(fo_r, b), np.broadcast_to(fi_r, b)])
        kap = np.concatenate([np.broadcast_to(k_f, b), np.broadcast_to(k_f, b),
                              np.broadcast_to(k_r, b), np.broadcast_to(k_r, b)])
        _, fy = self._tire(alpha, fz, np.tile(u_s, 4), kap)
        fy = fy.reshape(4, b)
        return -(fy[0] + fy[1]), -(fy[2] + fy[3])

    def deriv(self, x, delta, pedal, accel_gain, brake_margin, g_along=0.0):
        """State derivative for a batch x of shape (B, 7), [X, Y, psi, u, v, r, a].

        ``a`` is the applied longitudinal acceleration. It follows the pedal's
        target with the first-order lag ``tau_long``, which stands for the
        plant's pedal rate limit and its brake force build-up while the wheels
        spin down (model_check.py). The slip ratio, and with it the cornering
        force braking leaves, follows the pedal directly.

        ``g_along`` is gravity along the heading (m/s^2, positive downhill). It
        adds to ``a``, and a net deceleration fades out at a standstill, so the
        vehicle stays stopped on a slope its brakes can hold.
        """
        X, Y, psi, u, v, r, a = x.T
        a_tgt, k_f, k_r = self.longitudinal(pedal, accel_gain, brake_margin)
        fyf, fyr = self.lateral_forces(u, v, r, delta, a, k_f, k_r)
        w = np.clip((u - self.blend_lo) / (self.blend_hi - self.blend_lo), 0.0, 1.0)
        r_kin = u * np.tan(delta) / self.L
        v_kin = r_kin * self.Lr
        v_dot = w * ((fyf + fyr) / self.M - u * r) + (1.0 - w) * (v_kin - v) / self.tau_kin
        r_dot = (w * (self.Lf * fyf - self.Lr * fyr) / self.Izz
                 + (1.0 - w) * (r_kin - r) / self.tau_kin)
        c, s = np.cos(psi), np.sin(psi)
        net = a + g_along
        u_dot = np.where(net < 0.0, net * np.tanh(np.maximum(u, 0.0) / BRAKE_FADE_SPEED), net)
        return np.stack([u * c - v * s, u * s + v * c, r, u_dot, v_dot, r_dot,
                         (a_tgt - a) / self.tau_long], axis=1)

    def step(self, x, delta, pedal, h, accel_gain, brake_margin, g_along=0.0):
        """One midpoint (RK2) step for a batch. Speed is held at zero or above."""
        k1 = self.deriv(x, delta, pedal, accel_gain, brake_margin, g_along)
        k2 = self.deriv(x + 0.5 * h * k1, delta, pedal, accel_gain, brake_margin, g_along)
        xn = x + h * k2
        stopped = xn[:, 3] <= 0.0
        if np.any(stopped):
            xn[stopped, 3] = 0.0
            xn[stopped, 4] = 0.0
            xn[stopped, 5] = 0.0
        return xn

    def step_rk4(self, x, delta, pedal, h, accel_gain, brake_margin, g_along=0.0):
        """One classic RK4 step for a batch, the scheme of the OCP's integrator."""
        k1 = self.deriv(x, delta, pedal, accel_gain, brake_margin, g_along)
        k2 = self.deriv(x + 0.5 * h * k1, delta, pedal, accel_gain, brake_margin, g_along)
        k3 = self.deriv(x + 0.5 * h * k2, delta, pedal, accel_gain, brake_margin, g_along)
        k4 = self.deriv(x + h * k3, delta, pedal, accel_gain, brake_margin, g_along)
        xn = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        stopped = xn[:, 3] <= 0.0
        if np.any(stopped):
            xn[stopped, 3] = 0.0
            xn[stopped, 4] = 0.0
            xn[stopped, 5] = 0.0
        return xn

    def rollout_rk4(self, x0, cmds, hs, h_tail, accel_gain, brake_margin, t_tail, stop_speed=0.05,
                    gravity=None):
        """Roll one command sequence out with RK4: ``cmds`` (K, 2) held for the
        step lengths ``hs`` (K,), then the last command with steps ``h_tail``
        for up to ``t_tail`` seconds or until the vehicle stops. ``gravity(X, Y)``
        gives gravity along the heading at the start of each step (none: flat).
        Returns the states (n+1, 7)."""
        x = np.asarray(x0, dtype=float)[None, :].copy()
        xs = [x[0].copy()]

        def g(x):
            return gravity(float(x[0, 0]), float(x[0, 1])) if gravity is not None else 0.0
        for c, h in zip(cmds, hs):
            x = self.step_rk4(x, np.array([c[0] * self.max_steer]), np.array([c[1]]), float(h),
                              accel_gain, brake_margin, g(x))
            xs.append(x[0].copy())
        c = cmds[-1]
        for _ in range(int(np.ceil(t_tail / h_tail))):
            if x[0, 3] <= stop_speed:
                break
            x = self.step_rk4(x, np.array([c[0] * self.max_steer]), np.array([c[1]]), h_tail,
                              accel_gain, brake_margin, g(x))
            xs.append(x[0].copy())
        return np.array(xs)

    def settled_accel(self, pedal, accel_gain, brake_margin):
        """Applied acceleration once the lag has settled on a held pedal."""
        return float(self.longitudinal(np.array([float(pedal)]), accel_gain, brake_margin)[0][0])

    def rollout(self, x0, cmds, h, accel_gain, brake_margin, t_max, stop_speed=0.05):
        """Roll a batch of command sequences out, to a stop if they end braking.

        ``x0`` is one state (7,). ``cmds`` has shape (B, K, 2), normalized
        (steer, pedal) held over substeps of length ``h``. Past K the last
        command of each sequence holds. Returns the states (B, n+1, 7) and the
        time of each sample.
        """
        B, K, _ = cmds.shape
        n = int(np.ceil(t_max / h))
        x = np.repeat(np.asarray(x0, dtype=float)[None, :], B, axis=0)
        xs = [x.copy()]
        for i in range(n):
            c = cmds[:, min(i, K - 1), :]
            x = self.step(x, c[:, 0] * self.max_steer, c[:, 1], h, accel_gain, brake_margin)
            xs.append(x.copy())
            if i >= K and np.all(x[:, 3] <= stop_speed):
                break
        xs = np.stack(xs, axis=1)
        return xs, h * np.arange(xs.shape[1])

    # -- CasADi -----------------------------------------------------------------
    @staticmethod
    def _interp_sym(xq, xk, yk):
        """Piecewise-linear y(xq) through knots (xk ascending, symbolic), clamped."""
        y = yk[0]
        for i in range(len(yk) - 1):
            dx = xk[i + 1] - xk[i]
            y = y + (yk[i + 1] - yk[i]) / dx * ca.fmin(ca.fmax(xq - xk[i], 0.0), dx)
        return y

    def longitudinal_sym(self, pedal, curve_sym, accel_gain, brake_margin):
        """Symbolic (a_x, kappa_front, kappa_rear). ``curve_sym`` holds the
        per-wheel Fx knots, as ``self.curve`` (front brake, rear brake, drive)."""
        nb, nd = len(KAPPA_BRAKE), len(KAPPA_DRIVE)
        cb_f = [curve_sym[i] for i in range(nb)]
        cb_r = [curve_sym[nb + i] for i in range(nb)]
        cd = [curve_sym[2 * nb + i] for i in range(nd)]
        demand = ca.fmax(-pedal, 0.0) * self.f_brake_max
        cf, cr = -cb_f[0], -cb_r[0]
        ax_brake = -brake_margin * 2.0 * (ca.fmin(demand, cf) + ca.fmin(demand, cr)) / self.M
        k_bf = self._interp_sym(-demand, cb_f, list(KAPPA_BRAKE))
        k_br = self._interp_sym(-demand, cb_r, list(KAPPA_BRAKE))
        ax_drive = accel_gain * ca.fmax(pedal, 0.0)
        k_d = self._interp_sym(self.M * ax_drive / 4.0, cd, list(KAPPA_DRIVE))
        braking = pedal < 0.0
        return (ca.if_else(braking, ax_brake, ax_drive), ca.if_else(braking, k_bf, k_d),
                ca.if_else(braking, k_br, k_d))

    def f_expl_sym(self, x, delta, pedal, terrain_sym, curve_sym, accel_gain, brake_margin, g_along=0.0):
        """Symbolic state derivative of the 7-state model. ``terrain_sym`` is
        [Kphi, Kc, n, c, phi, k] in the surrogate's feature convention,
        ``curve_sym`` the per-wheel longitudinal force knots (``self.curve``) and
        ``g_along`` gravity along the heading, as in ``deriv``."""
        X, Y, psi, u, v, r, a = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        a_tgt, k_f, k_r = self.longitudinal_sym(pedal, curve_sym, accel_gain, brake_margin)
        u_s = ca.fmax(ca.fabs(u), 0.5)
        a_f = ca.fmax(ca.fmin(delta - ca.atan2(v + self.Lf * r, u_s), ALPHA_MAX), -ALPHA_MAX)
        a_r = ca.fmax(ca.fmin(-ca.atan2(v - self.Lr * r, u_s), ALPHA_MAX), -ALPHA_MAX)
        M, h = self.M, self.h_cg
        fz_f = (M * G * self.Lr - M * a * h) / self.L / 2.0
        fz_r = (M * G * self.Lf + M * a * h) / self.L / 2.0
        if self.llt:
            d = M * (u * r) * h / self.T / 2.0
            loads = [ca.fmin(fz_f + d, 1.9 * fz_f), ca.fmax(fz_f - d, 0.1 * fz_f),
                     ca.fmin(fz_r + d, 1.9 * fz_r), ca.fmax(fz_r - d, 0.1 * fz_r)]
        else:
            loads = [fz_f, fz_f, fz_r, fz_r]
        B = self.nn._BATCH
        pad = B - 4
        a_vec = ca.vertcat(a_f, a_f, a_r, a_r, *([0.0] * pad))
        fz_vec = ca.vertcat(*loads, *([self.fz_f] * pad))
        u_vec = ca.repmat(u_s, B, 1)
        k_vec = ca.vertcat(k_f, k_f, k_r, k_r, *([0.0] * pad))
        n_vec = ca.repmat(terrain_sym[2], B, 1)
        sr_vec = ca.DM.zeros(B, 1)
        _, fys = self.nn.predict_batch(a_vec, fz_vec, u_vec, k_vec, n_vec, sr_vec,
                                       terrain_sym[0], terrain_sym[1], terrain_sym[3],
                                       terrain_sym[4], terrain_sym[5])
        fyf = -(fys[0] + fys[1])
        fyr = -(fys[2] + fys[3])
        w = ca.fmin(1.0, ca.fmax(0.0, (u - self.blend_lo) / (self.blend_hi - self.blend_lo)))
        r_kin = u * ca.tan(delta) / self.L
        v_kin = r_kin * self.Lr
        v_dot = w * ((fyf + fyr) / M - u * r) + (1.0 - w) * (v_kin - v) / self.tau_kin
        r_dot = w * (self.Lf * fyf - self.Lr * fyr) / self.Izz + (1.0 - w) * (r_kin - r) / self.tau_kin
        net = a + g_along
        u_dot = ca.if_else(net < 0.0, net * ca.tanh(ca.fmax(u, 0.0) / BRAKE_FADE_SPEED), net)
        return ca.vertcat(u * ca.cos(psi) - v * ca.sin(psi),
                          u * ca.sin(psi) + v * ca.cos(psi),
                          r, u_dot, v_dot, r_dot, (a_tgt - a) / self.tau_long)
