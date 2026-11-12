"""The plant vehicle's specification, which the safety filters take their numbers from.

Every number here is read from the Chrono 9.0.0 HMMWV model the simulator runs, or from the plant node's
actuator limits. The README section "Derived parameters" lists the source of each.
"""
from __future__ import annotations

import numpy as np

G = 9.81
RPM = 2.0 * np.pi / 60.0

# -- geometry -------------------------------------------------------------------------------
# Body outline from hmmwv_chassis_col.obj (chassis reference frame). The wheels lie inside it.
BODY_X_REF = (-2.459, 2.212)
BODY_HALF_WIDTH = 1.281
# The filters get the chassis body's position, its centre of mass (HMMWV_Chassis m_body_COM_loc).
COM_X_REF = 0.056

# -- actuators (plant node chrono_sim_node.py) ------------------------------------------------
STEER_RATE = 16.0          # full lock per second
PEDAL_RATE = 8.0           # pedal units per second, throttle (+) and brake (-)
MAX_ROAD_WHEEL = 0.49      # rad at full lock

# -- wheels and brakes -----------------------------------------------------------------------
WHEEL_RADIUS = 0.467       # HMMWV_RigidTire
BRAKE_TORQUE = 4000.0      # N m per wheel at full pedal, HMMWV_BrakeSimple / BrakeShafts
CONICAL_RATIO = 0.2        # HMMWV_Driveline4WD: wheel speed over driveshaft speed


def _interp(x, xs, ys):
    """Piecewise linear, held constant beyond the end points (ChFunctionInterp without extrapolation)."""
    return float(np.interp(x, xs, ys))


class ShaftsPowertrain:
    """HMMWV_EngineShafts through the torque converter of HMMWV_AutomaticTransmissionShafts, quasi-static.

    The engine runs where its torque equals the torque converter's pump torque, (w_e / K(SR))^2 with
    SR = w_turbine / w_e, and the turbine passes TR(SR) times that torque on (ChShaftsTorqueConverter). The
    gear is the one the up-shift rule (turbine above 2500 rpm) leaves while the vehicle speeds up. Shaft
    inertias and the 1 s shift latency are left out.
    """

    name = "shafts"
    torque_map = ([-100, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
                   2100, 2200, 2300, 2400, 2500, 2700],
                  [300, 382, 490, 579, 650, 706, 746, 774, 789, 793, 788, 774, 754, 728,
                   697, 664, 628, 593, 558, -400])
    losses_map = ([-50, 0, 50, 1000, 2000, 3000], [30, 0, -30, -50, -70, -90])
    capacity_map = ([0.0, 0.25, 0.50, 0.75, 0.90, 1.00], [15, 15, 15, 16, 18, 35])
    ratio_map = ([0.0, 0.25, 0.50, 0.75, 1.00], [2.00, 1.80, 1.50, 1.15, 1.00])
    gears = (0.2, 0.4, 0.8)
    upshift = 2500 * RPM

    def __init__(self):
        self._tq_w = np.array(self.torque_map[0]) * RPM
        self._loss_w = np.array(self.losses_map[0]) * RPM

    def _engine(self, w, throttle):
        return throttle * _interp(w, self._tq_w, self.torque_map[1]) + _interp(w, self._loss_w,
                                                                                 self.losses_map[1])

    def _converter(self, w_e, w_t):
        """(pump torque the converter puts on the engine, turbine torque), ChShaftsTorqueConverter."""
        sr = w_t / w_e
        reverse = sr > 1.0
        if reverse:
            sr = max(2.0 - sr, 0.0)
        k = _interp(sr, *self.capacity_map)
        pump = (w_e / k) ** 2
        if reverse:
            return -pump, -pump
        return pump, _interp(sr, *self.ratio_map) * pump

    def _turbine_torque(self, w_t, throttle):
        lo, hi = 1e-3, 4000 * RPM
        f = lambda w: self._engine(w, throttle) - self._converter(w, w_t)[0]   # noqa: E731
        if f(lo) <= 0.0:
            return 0.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return self._converter(0.5 * (lo + hi), w_t)[1]

    def gear(self, v):
        w_ds = max(v, 0.0) / WHEEL_RADIUS / CONICAL_RATIO
        for g in self.gears:
            if w_ds / g <= self.upshift:
                return g
        return self.gears[-1]

    def wheel_force(self, v, throttle):
        """Total drive force at the four wheels (N) at speed v (m/s) and throttle in [0, 1]."""
        g = self.gear(v)
        w_ds = max(v, 0.0) / WHEEL_RADIUS / CONICAL_RATIO
        t_turbine = self._turbine_torque(w_ds / g, float(throttle))
        return t_turbine / g / CONICAL_RATIO / WHEEL_RADIUS


class SimpleEngine:
    """ChEngineSimple with the HMMWV's numbers: throttle x min(330 N m, 110 kW / speed), none above 10000 rad/s."""

    max_torque, max_power, max_speed = 330.0, 110000.0, 10000.0

    def torque(self, w, throttle):
        if w >= self.max_speed:
            return 0.0
        full = self.max_torque if w <= self.max_power / self.max_torque else self.max_power / w
        return throttle * full


class SimpleMapPowertrain(SimpleEngine):
    """ChEngineSimple through HMMWV_AutomaticTransmissionSimpleMap (no torque converter, no efficiency).

    The gear is the one the up-shift points leave while the vehicle speeds up."""

    name = "simple_map"
    gears = (0.1708, 0.2791, 0.4218, 0.6223, 1.0173, 1.5361)
    upshift = tuple(r * RPM for r in (2226, 2225, 2210, 2226, 2225))

    def gear(self, v):
        w_ds = max(v, 0.0) / WHEEL_RADIUS / CONICAL_RATIO
        for g, up in zip(self.gears, self.upshift):
            if w_ds / g <= up:
                return g
        return self.gears[-1]

    def wheel_force(self, v, throttle):
        g = self.gear(v)
        w_ds = max(v, 0.0) / WHEEL_RADIUS / CONICAL_RATIO
        return self.torque(w_ds / g, float(throttle)) / g / CONICAL_RATIO / WHEEL_RADIUS


class SimpleCVTPowertrain(SimpleEngine):
    """ChEngineSimple through HMMWV_AutomaticTransmissionSimpleCVT (newer Chrono than 9.0.0)."""

    name = "simple_cvt"
    # HMMWV_AutomaticTransmissionSimpleCVT: ratio 0.1708 to 1.5361 between 2 and 20 m/s, computed there with a
    # tyre radius of 0.42 and the conical ratio 0.2, efficiency 0.8.
    w_start, w_end = 2.0 / (0.2 * 0.42), 20.0 / (0.2 * 0.42)
    r_start, r_end, eff = 0.1708, 1.5361, 0.8

    def wheel_force(self, v, throttle):
        w_ds = max(v, 0.0) / WHEEL_RADIUS / CONICAL_RATIO
        g = _interp(w_ds, [self.w_start, self.w_end], [self.r_start, self.r_end])
        return self.eff * self.torque(w_ds / g, float(throttle)) / g / CONICAL_RATIO / WHEEL_RADIUS


POWERTRAINS = {p.name: p for p in (ShaftsPowertrain, SimpleMapPowertrain, SimpleCVTPowertrain)}


def make_powertrain(name: str):
    return POWERTRAINS[name]()


def drive_table(powertrain, speeds=None, throttles=None):
    """Tabulate the total wheel drive force (N) on a speed x throttle grid, for the symbolic models."""
    speeds = np.linspace(0.0, 24.0, 49) if speeds is None else np.asarray(speeds, dtype=float)
    throttles = np.linspace(0.0, 1.0, 11) if throttles is None else np.asarray(throttles, dtype=float)
    table = np.array([[powertrain.wheel_force(v, t) for t in throttles] for v in speeds])
    return speeds, throttles, table
