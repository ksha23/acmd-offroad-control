"""The surrogate dynamic bicycle with no hand-set numbers (``--safety-derived``).

Differences from ``SurrogateBicycle``:

* The seventh state is the applied pedal p in [-1, 1] (throttle positive, brake negative), and the
  longitudinal input is its rate, which the plant limits to 8 per second. The acceleration follows from
  p at once: there is no fitted lag.
* Longitudinal force as drawbar pull = tractive effort - motion resistance (Wong, Theory of Ground
  Vehicles). The surrogate's net Fx at zero slip is the motion resistance. A wheel torque T adds T / r to
  it, until the curve's end: the locked-wheel force when braking, its top when driving (the wheel spins).
  So a coasting vehicle slows by the soil's resistance, and there is no braking margin.
* Drive torque is the plant powertrain's quasi-static wheel torque at the current speed (``vehicle_spec``),
  shared over the four wheels. Brake torque is pedal x 4000 N m per wheel.

The steering input is the road-wheel angle, as before.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from . import vehicle_spec as spec
from .surrogate_dynamics import (ALPHA_MAX, BRAKE_FADE_SPEED, G, KAPPA_BRAKE, KAPPA_DRIVE,
                                 SurrogateBicycle)


class DerivedBicycle(SurrogateBicycle):
    """Dynamic bicycle, state [X, Y, psi, u, v, r, p], inputs (road-wheel angle, pedal rate)."""

    def __init__(self, nn, vehicle_params: dict, powertrain: str, **kw):
        kw.setdefault("max_steer", spec.MAX_ROAD_WHEEL)
        super().__init__(nn, vehicle_params, **kw)
        self.powertrain = spec.make_powertrain(powertrain)
        throttles = (np.linspace(0.0, 1.0, 11) if powertrain == "shafts" else np.array([0.0, 1.0]))
        self.speeds, self.throttles, self.table = spec.drive_table(self.powertrain, throttles=throttles)
        self.pedal_rate = spec.PEDAL_RATE
        self.f_brake_max = spec.BRAKE_TORQUE / spec.WHEEL_RADIUS

    # -- longitudinal -------------------------------------------------------------------------
    def drive_force(self, u, throttle):
        """Total powertrain force at the wheels (N), bilinear in the table."""
        u, th = np.broadcast_arrays(np.atleast_1d(np.asarray(u, dtype=float)),
                                    np.atleast_1d(np.asarray(throttle, dtype=float)))
        sp, tt, T = self.speeds, self.throttles, self.table
        i = np.clip(np.searchsorted(sp, u, side="right") - 1, 0, len(sp) - 2)
        j = np.clip(np.searchsorted(tt, th, side="right") - 1, 0, len(tt) - 2)
        fu = np.clip((u - sp[i]) / (sp[i + 1] - sp[i]), 0.0, 1.0)
        ft = np.clip((th - tt[j]) / (tt[j + 1] - tt[j]), 0.0, 1.0)
        return ((1.0 - fu) * (1.0 - ft) * T[i, j] + fu * (1.0 - ft) * T[i + 1, j]
                + (1.0 - fu) * ft * T[i, j + 1] + fu * ft * T[i + 1, j + 1])

    def drive_capability(self):
        """Largest drive acceleration on the current soil (m/s^2): full throttle, at any speed in the table,
        capped by the four wheels at the top of the surrogate's drive curve."""
        nd = len(KAPPA_DRIVE)
        f0, top = float(self.curve[-nd]), float(self.curve[-1])
        return 4.0 * float(np.max(np.minimum(f0 + self.table[:, -1] / 4.0, top))) / self.M

    def longitudinal(self, pedal, u, *_ignored):
        """(a_x, kappa_front, kappa_rear) for arrays of applied pedal and speed."""
        pedal = np.clip(np.atleast_1d(np.asarray(pedal, dtype=float)), -1.0, 1.0)
        u = np.broadcast_to(np.atleast_1d(np.asarray(u, dtype=float)), pedal.shape)
        nb = len(KAPPA_BRAKE)
        cbf, cbr, cd = self.curve[:nb], self.curve[nb:2 * nb], self.curve[2 * nb:]
        brake = np.maximum(-pedal, 0.0) * self.f_brake_max
        net_f = np.maximum(cbf[-1] - brake, cbf[0])
        net_r = np.maximum(cbr[-1] - brake, cbr[0])
        ax_brake = 2.0 * (net_f + net_r) / self.M
        k_f = np.interp(net_f, cbf, KAPPA_BRAKE)
        k_r = np.interp(net_r, cbr, KAPPA_BRAKE)
        drive = self.drive_force(np.maximum(u, 0.0), np.maximum(pedal, 0.0)) / 4.0
        per_wheel = np.minimum(cd[0] + drive, cd[-1])
        ax_drive = 4.0 * per_wheel / self.M
        k_d = np.interp(per_wheel, cd, KAPPA_DRIVE)
        braking = pedal < 0.0
        return np.where(braking, ax_brake, ax_drive), np.where(braking, k_f, k_d), np.where(braking, k_r, k_d)

    def settled_accel(self, pedal, u=0.0, *_ignored):
        """Longitudinal acceleration at a held pedal and speed (m/s^2), before the standstill fade."""
        return float(self.longitudinal(np.array([float(pedal)]), np.array([float(u)]))[0][0])

    def speed_rate(self, pedal, u, g_along=0.0):
        """du/dt of the model on a straight path: the acceleration plus gravity, a net deceleration fading
        out at a standstill (as in ``deriv``)."""
        net = self.settled_accel(pedal, u) + g_along
        return net * np.tanh(max(u, 0.0) / BRAKE_FADE_SPEED) if net < 0.0 else net

    # -- numpy dynamics -----------------------------------------------------------------------
    def deriv(self, x, delta, pdot, *_ignored, g_along=0.0):
        """State derivative for a batch x (B, 7). ``pdot`` is the pedal rate (per second)."""
        X, Y, psi, u, v, r, p = x.T
        a, k_f, k_r = self.longitudinal(p, u)
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
                         np.broadcast_to(pdot, u.shape)], axis=1)

    def step_rk4(self, x, delta, pdot, h, *_ignored, g_along=0.0):
        """One classic RK4 step, the scheme of the OCP's integrator."""
        k1 = self.deriv(x, delta, pdot, g_along=g_along)
        k2 = self.deriv(x + 0.5 * h * k1, delta, pdot, g_along=g_along)
        k3 = self.deriv(x + 0.5 * h * k2, delta, pdot, g_along=g_along)
        k4 = self.deriv(x + h * k3, delta, pdot, g_along=g_along)
        return x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def rollout_rk4(self, x0, cmds, hs, h_tail, *_ignored, t_tail=0.0, stop_speed=0.05, gravity=None):
        """Roll normalized commands (steer, pedal rate over its limit) out with RK4: ``cmds`` (K, 2) held
        for the step lengths ``hs``, then straight full braking (pedal ramped down at the limit) with steps
        ``h_tail`` for up to ``t_tail`` seconds or until the vehicle stops. Returns the states (n+1, 7)."""
        x = np.asarray(x0, dtype=float)[None, :].copy()
        xs = [x[0].copy()]

        def g(x):
            return gravity(float(x[0, 0]), float(x[0, 1])) if gravity is not None else 0.0
        for c, h in zip(cmds, hs):
            x = self.step_rk4(x, np.array([c[0] * self.max_steer]), np.array([c[1] * self.pedal_rate]),
                              float(h), g_along=g(x))
            xs.append(x[0].copy())
        for _ in range(int(np.ceil(t_tail / h_tail))):
            if x[0, 3] <= stop_speed:
                break
            x = self.step_rk4(x, np.array([0.0]), np.array([-self.pedal_rate]), h_tail, g_along=g(x))
            xs.append(x[0].copy())
        return np.array(xs)

    # -- CasADi -------------------------------------------------------------------------------
    def drive_force_sym(self, u, throttle):
        per_knot = [self._interp_sym(u, list(self.speeds), list(self.table[:, j]))
                    for j in range(len(self.throttles))]
        return self._interp_sym(throttle, list(self.throttles), per_knot)

    def longitudinal_sym(self, pedal, u, curve_sym):
        nb, nd = len(KAPPA_BRAKE), len(KAPPA_DRIVE)
        cb_f = [curve_sym[i] for i in range(nb)]
        cb_r = [curve_sym[nb + i] for i in range(nb)]
        cd = [curve_sym[2 * nb + i] for i in range(nd)]
        pedal = ca.fmin(ca.fmax(pedal, -1.0), 1.0)
        brake = ca.fmax(-pedal, 0.0) * self.f_brake_max
        net_f = ca.fmax(cb_f[-1] - brake, cb_f[0])
        net_r = ca.fmax(cb_r[-1] - brake, cb_r[0])
        ax_brake = 2.0 * (net_f + net_r) / self.M
        k_bf = self._interp_sym(net_f, cb_f, list(KAPPA_BRAKE))
        k_br = self._interp_sym(net_r, cb_r, list(KAPPA_BRAKE))
        drive = self.drive_force_sym(ca.fmax(u, 0.0), ca.fmax(pedal, 0.0)) / 4.0
        per_wheel = ca.fmin(cd[0] + drive, cd[-1])
        ax_drive = 4.0 * per_wheel / self.M
        k_d = self._interp_sym(per_wheel, cd, list(KAPPA_DRIVE))
        braking = pedal < 0.0
        return (ca.if_else(braking, ax_brake, ax_drive), ca.if_else(braking, k_bf, k_d),
                ca.if_else(braking, k_br, k_d))

    def f_expl_sym(self, x, delta, pdot, terrain_sym, curve_sym, *_ignored, g_along=0.0):
        """Symbolic derivative of [X, Y, psi, u, v, r, p] for road-wheel angle ``delta`` and pedal rate
        ``pdot`` (per second)."""
        X, Y, psi, u, v, r, p = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        a, k_f, k_r = self.longitudinal_sym(p, u, curve_sym)
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
                          r, u_dot, v_dot, r_dot, pdot)
