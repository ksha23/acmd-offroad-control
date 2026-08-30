#!/usr/bin/env python3
"""
Chrono plant node
=================

Runs the PyChrono HMMWV on SCM deformable terrain as an independent process,
and is the sole owner of ground truth: obstacles, traffic, collision contacts,
and the soil the vehicle is actually driving on. What it publishes is
restricted to quantities a physical vehicle could measure, so no controller or
estimator downstream can consume simulator truth. ROS 2 through Chrono::ROS is
the default controller transport; ZeroMQ is an explicit development fallback.

Published: VehicleState at a configurable rate, by default 100 Hz decimated
from the 333 Hz physics rate.
Subscribed: ControlCommand from the MPC controller.

The most recently received ControlCommand is applied at each physics step.
Before any command arrives the plant holds zero throttle and zero steering, and
a command older than the stale-command timeout triggers a brake, so a lost link
fails closed rather than latching the last command.

Usage:
    python chrono_sim_node.py --terrain sand --time 30 --path sinusoidal
"""

import os as _os, sys as _sys  # flat-import bootstrap (simulation/flatpath.py)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import flatpath  # noqa: E402,F401
import argparse
import csv
import math
import os
import sys
import time as wall_time
from pathlib import Path

import numpy as np

# Chrono imports (must be available in environment)
import pychrono as chrono
import pychrono.vehicle as veh

# Driver-view camera pose shared by Irrlicht and Chrono Sensor visualization.
# Chrono sensor cameras look forward along the local +X axis.
DRIVER_CAM_POS_LOCAL = chrono.ChVector3d(0.53, 0.7, 1.0)
DRIVER_CAM_ROT_LOCAL = chrono.ChQuaterniond(1, 0, 0, 0)
DRIVER_CAM_LOOKAHEAD_DISTANCE = 12.0

# Sensor imports (optional — only needed for sensor visualization mode)
try:
    import pychrono.sensor as sens
    HAS_SENSOR = True
except ImportError:
    HAS_SENSOR = False

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from hil_messages import (
    VehicleState, ControlCommand, SimStatus,
    STEER_MAX_RAD, DRIVE_TORQUE_MAX_NM, BRAKE_TORQUE_MAX_NM,
    make_publisher, make_subscriber,
    sim_pub_endpoint, ctrl_sub_endpoint,
    TOPIC_VEHICLE_STATE, TOPIC_CONTROL_CMD,
)


def _torque_to_pedals(drive_torque: float) -> tuple[float, float]:
    """Map a physical drive torque (N m, negative = brake) onto the Chrono
    driver's throttle/brake actuator fractions at the plant boundary."""
    tau = float(drive_torque)
    if tau >= 0.0:
        return min(tau / DRIVE_TORQUE_MAX_NM, 1.0), 0.0
    return 0.0, min(-tau / BRAKE_TORQUE_MAX_NM, 1.0)

from param_consistency import (
    TERRAIN_PRESETS, get_vehicle_params_for_demo,
    get_terrain_preset, terrain_preset_to_internal,
)

# Re-use terrain/vehicle setup helpers (extracted modules)
from chrono_setup import (
    setup_chrono_vehicle,
    setup_scm_terrain,
    add_trajectory_markers,
    load_terrain_config,
)
from g29_controller import ManualDriver
from delayed_pov import DelayedPOV

# Safety filter + obstacles (optional)
from sensors.obstacles import add_rock_obstacles, get_rock_positions, get_rock_radii
from safety import make_safety_filter
from collision_detector import (
    COLLISION_SOURCE,
    VEHICLE_CLEARANCE_RADIUS,
    CollisionLogger,
)
from traffic import TrafficManager
from latency_profile import LatencyProfile

# NN tire model for terrain-aware CBF traction limits
try:
    from nn_tire_model import load_nn_tire_model
except ImportError:
    load_nn_tire_model = None


# =============================================================================
# Simple driver that applies external commands
# =============================================================================

class ExternalDriver(veh.ChDriver):
    """Minimal Chrono driver that applies commands received over the network."""

    def __init__(self, vehicle):
        super().__init__(vehicle.GetVehicle())
        self.m_steering = 0.0
        self.m_throttle = 0.0
        self.m_braking = 0.0

    def apply(self, cmd: ControlCommand):
        # Physical command bus (delta rad, drive_torque N m) -> Chrono actuators.
        self.m_steering = float(np.clip(cmd.delta / STEER_MAX_RAD, -1.0, 1.0))
        thr, brk = _torque_to_pedals(cmd.drive_torque)
        self.m_throttle = float(thr)
        self.m_braking = float(brk)

    def Synchronize(self, time):
        pass  # Nothing to do — commands are applied externally

    def Advance(self, step):
        pass

    def GetSteering(self):
        return self.m_steering

    def GetThrottle(self):
        return self.m_throttle

    def GetBraking(self):
        return self.m_braking


class ReplayDriver(veh.ChDriver):
    """Replays a recorded operator command trace for counterfactual replay.

    Reads a command trace CSV, such as a recorded run's sim_diag.csv, and at
    each simulated time supplies the operator's raw command
    (steering_op/throttle_op/braking_op, falling back to the applied columns).
    The safety filter then screens that identical intent, so replaying one
    trace with the filter disabled and with each filter enabled isolates the
    filter as the only cause of any difference in outcome. This is what makes
    harm prevented a causal measurement rather than a comparison across runs
    with different operator inputs.
    """

    def __init__(self, vehicle, csv_path):
        super().__init__(vehicle.GetVehicle())
        import pandas as pd
        d = pd.read_csv(csv_path)
        sc = "steering_op" if "steering_op" in d.columns else "steering"
        tc = "throttle_op" if "throttle_op" in d.columns else "throttle"
        bc = "braking_op" if "braking_op" in d.columns else "braking"
        self._t = pd.to_numeric(d["time"], errors="coerce").to_numpy()
        self._s = pd.to_numeric(d[sc], errors="coerce").to_numpy()
        self._th = pd.to_numeric(d[tc], errors="coerce").to_numpy()
        self._b = pd.to_numeric(d[bc], errors="coerce").to_numpy()
        self.m_steering = 0.0
        self.m_throttle = 0.0
        self.m_braking = 0.0
        print(f"  Replay driver: {len(self._t)} command samples "
              f"({self._t[0]:.2f}-{self._t[-1]:.2f}s) from {os.path.basename(csv_path)} [{sc}]")

    def Synchronize(self, time):
        self.m_steering = float(np.clip(np.interp(time, self._t, self._s), -1.0, 1.0))
        self.m_throttle = float(np.clip(np.interp(time, self._t, self._th), 0.0, 1.0))
        self.m_braking = float(np.clip(np.interp(time, self._t, self._b), 0.0, 1.0))

    def Advance(self, step):
        pass

    def GetSteering(self):
        return self.m_steering

    def GetThrottle(self):
        return self.m_throttle

    def GetBraking(self):
        return self.m_braking


class SyntheticTeleopDriver(veh.ChDriver):
    """Closed-loop synthetic teleoperator for automated latency studies.

    This operator closes the loop on delayed perception: at each step it tracks
    a reference path using the vehicle state as it stood ``perception_delay``
    seconds earlier, which is the video-channel latency. Its command then
    passes through the command-channel delay buffer and the onboard safety
    filter. Because the feedback itself is delayed, tracking quality degrades
    and eventually oscillates as latency grows -- the effect under study, and
    one a replayed open-loop trace cannot exhibit, since a fixed trace responds
    identically at every delay. Steering follows a Stanley-style law and speed
    is a proportional hold on the target speed.
    """

    def __init__(self, vehicle, ref_path, target_speed,
                 k_lat=0.55, k_psi=1.15, max_steer_rad=0.5):
        super().__init__(vehicle.GetVehicle())
        self._vehicle = vehicle
        self._path = ref_path
        self._v_target = float(target_speed)
        self._k_lat = float(k_lat)
        self._k_psi = float(k_psi)
        self._max_steer = float(max_steer_rad)
        self._buf = []  # (t, x, y, psi, u), oldest first
        self.perception_delay = 0.0
        self.m_steering = 0.0
        self.m_throttle = 0.0
        self.m_braking = 0.0

    def _true_state(self):
        c = self._vehicle.GetChassisBody()
        p = c.GetPos()
        r = c.GetRot()
        vl = r.RotateBack(c.GetPosDt())
        psi = math.atan2(2 * (r.e0 * r.e3 + r.e1 * r.e2),
                         1 - 2 * (r.e2 * r.e2 + r.e3 * r.e3))
        return float(p.x), float(p.y), float(psi), float(vl.x)

    def _delayed_state(self, t):
        buf = self._buf
        if not buf:
            return 0.0, 0.0, 0.0, 0.0
        td = t - self.perception_delay
        if td <= buf[0][0]:
            return buf[0][1], buf[0][2], buf[0][3], buf[0][4]
        for i in range(len(buf) - 1, -1, -1):
            if buf[i][0] <= td:
                if i == len(buf) - 1:
                    return buf[i][1], buf[i][2], buf[i][3], buf[i][4]
                t0 = buf[i][0]
                t1 = buf[i + 1][0]
                a = (td - t0) / (t1 - t0) if t1 > t0 else 0.0
                return tuple(buf[i][1 + j] + a * (buf[i + 1][1 + j] - buf[i][1 + j])
                             for j in range(4))
        return buf[-1][1], buf[-1][2], buf[-1][3], buf[-1][4]

    def Synchronize(self, time):
        x, y, psi, u = self._true_state()
        self._buf.append((time, x, y, psi, u))
        while len(self._buf) > 2 and time - self._buf[0][0] > 5.0:
            self._buf.pop(0)
        xd, yd, psid, ud = self._delayed_state(time)
        cp = self._path.closest_point_on_path(xd, yd)
        head_err = math.atan2(math.sin(cp["psi_ref"] - psid),
                              math.cos(cp["psi_ref"] - psid))
        v = max(abs(ud), 1.0)
        steer_rad = self._k_psi * head_err + math.atan2(-self._k_lat * cp["e_lat"], v)
        steer_cmd = float(np.clip(steer_rad / self._max_steer, -1.0, 1.0))
        # First-order smoothing. A human operator does not slew the wheel
        # instantaneously, and the same smoothing keeps the loop well damped at
        # zero delay, so the delay sweep starts from a stable baseline. The
        # blend factor is derived from the elapsed simulation time and a fixed
        # 0.15 s time constant: Synchronize runs once per physics step (~3 ms),
        # and a per-call constant of 0.6 at that rate has a ~3 ms time
        # constant -- no smoothing at all, which defeated the stated intent.
        dt_op = max(time - getattr(self, "_last_sync_time", time), 0.0)
        self._last_sync_time = time
        alpha = 1.0 - math.exp(-dt_op / 0.15) if dt_op > 0.0 else 1.0
        self.m_steering = float((1.0 - alpha) * self.m_steering
                                + alpha * steer_cmd)
        err = self._v_target - ud
        if err >= 0.0:
            self.m_throttle = float(np.clip(0.35 + 0.4 * err, 0.0, 1.0))
            self.m_braking = 0.0
        else:
            self.m_throttle = 0.0
            self.m_braking = float(np.clip(-0.4 * err, 0.0, 1.0))

    def Advance(self, step):
        pass

    def GetSteering(self):
        return self.m_steering

    def GetThrottle(self):
        return self.m_throttle

    def GetBraking(self):
        return self.m_braking


def update_safety_terrain_from_command(
    safety_filter,
    command,
    last_terrain_seq: int,
    *,
    no_sigma_gate: bool = False,
    hedge_k: float = 0.0,
    use_terrain_nn: bool = False,
    use_grip_scale: bool = False,
) -> tuple[int, bool, float]:
    """Apply a piggybacked terrain belief only when its command is accepted."""
    terrain_n = getattr(command, "terrain_n", None)
    update_seq = int(getattr(command, "terrain_update_seq", 0))
    if (
        safety_filter is None
        or terrain_n is None
        or update_seq <= int(last_terrain_seq)
    ):
        return int(last_terrain_seq), False, 0.0
    sigma_deg = (
        0.0 if no_sigma_gate
        else float(getattr(command, "terrain_phi_sigma_deg", None) or 0.0)
    )
    terrain = {
        "Kphi": float(getattr(command, "terrain_Kphi", None) or 0.0),
        "Kc": float(getattr(command, "terrain_Kc", None) or 0.0),
        "n": float(terrain_n),
        "c": float(getattr(command, "terrain_c", None) or 0.0),
        "phi": float(getattr(command, "terrain_phi_deg")),
        "k": float(getattr(command, "terrain_k", None) or 0.0),
    }
    try:
        safety_filter.update_terrain(
            terrain,
            phi_uncertainty_deg=sigma_deg,
            n_sigma=getattr(command, "terrain_n_sigma", None),
            hedge_k=float(hedge_k),
            use_terrain_nn=bool(use_terrain_nn),
            grip_scale=getattr(command, "terrain_grip_scale", None),
            use_grip_scale=bool(use_grip_scale),
        )
    except TypeError:
        safety_filter.update_terrain(terrain)
    return update_seq, True, sigma_deg


def get_driver_camera_view(vehicle):
    """Return Irrlicht eye and look-at points matching the sensor driver POV."""
    chassis = vehicle.GetChassisBody()
    chassis_pos = chassis.GetPos()
    chassis_rot = chassis.GetRot()
    eye = chassis_pos + chassis_rot.Rotate(DRIVER_CAM_POS_LOCAL)

    # Sensor camera orientation is defined by DRIVER_CAM_ROT_LOCAL.  Irrlicht
    # uses a look-at target, so convert the same local +X camera direction into
    # a point in front of the camera.
    camera_forward_local = DRIVER_CAM_ROT_LOCAL.Rotate(chrono.ChVector3d(1, 0, 0))
    lookahead_local = chrono.ChVector3d(
        camera_forward_local.x * DRIVER_CAM_LOOKAHEAD_DISTANCE,
        camera_forward_local.y * DRIVER_CAM_LOOKAHEAD_DISTANCE,
        camera_forward_local.z * DRIVER_CAM_LOOKAHEAD_DISTANCE,
    )
    target = eye + chassis_rot.Rotate(lookahead_local)
    return eye, target


def update_irrlicht_driver_camera(vis, vehicle):
    """Keep the Irrlicht camera at the same chassis-fixed pose as DriverPOV."""
    eye, target = get_driver_camera_view(vehicle)
    # Match the standalone Irrlicht demo: drive the active camera explicitly.
    # The vehicle visual system's chase-camera wrapper can otherwise keep
    # restoring chase behavior on Synchronize/Advance in some PyChrono builds.
    if hasattr(vis, "SetChaseCameraPosition"):
        vis.SetChaseCameraPosition(eye, target)
    vis.SetCameraPosition(eye)
    vis.SetCameraTarget(target)


def set_z_up_if_available(vis):
    """Use Chrono's Z-up camera convention when exposed by the local bindings."""
    if hasattr(chrono, "CameraVerticalDir_Z"):
        vis.SetCameraVertical(chrono.CameraVerticalDir_Z)


def set_visual_color(item, color):
    """Set color on all visual shapes owned by a Chrono item, if exposed."""
    def color_shape(shape):
        shape.SetColor(color)
        try:
            for i in range(shape.GetNumMaterials()):
                material = shape.GetMaterial(i)
                material.SetAmbientColor(color)
                material.SetDiffuseColor(color)
        except Exception:
            pass

    try:
        model = item.GetVisualModel()
    except Exception:
        model = None

    if model:
        try:
            for i in range(model.GetNumShapes()):
                color_shape(model.GetShape(i))
            return
        except Exception:
            pass

    try:
        shape_count = item.GetNumVisualShapes()
    except Exception:
        shape_count = 0

    for i in range(shape_count):
        try:
            color_shape(item.GetVisualShape(i))
        except Exception:
            pass


def color_hmmwv(vehicle):
    """Apply explicit colors for Irrlicht builds that do not load HMMWV materials."""
    body_color = chrono.ChColor(0.88, 0.82, 0.66)
    set_visual_color(vehicle.GetChassisBody(), body_color)


# =============================================================================
# Vehicle state extraction
# =============================================================================

# Default measurement noise standard deviations (sensor-fusion realistic)
DEFAULT_MEAS_NOISE = {
    'x':     0.05,    # Differential GPS position (m)
    'y':     0.05,    # Differential GPS position (m)
    'psi':   0.005,   # ~0.3° heading (rad)
    'u':     0.05,    # Speed (m/s)
    'v':     0.05,    # Lateral speed (m/s)
    'omega': 0.005,   # Yaw rate (rad/s)
}


def extract_tire_forces(vehicle, terrain) -> dict:
    """Extract per-wheel tire forces and slips from Chrono.

    Forces are rotated from the global frame into the vehicle body frame
    so that Fx/Fy/Fz align with the bicycle-model convention used by MPC.
    """
    tf = {}
    veh_obj = vehicle.GetVehicle()
    chassis = vehicle.GetChassisBody()
    rot = chassis.GetRot()
    for axle_idx, axle_name in enumerate(['front', 'rear']):
        for side_idx, side_name in [(veh.LEFT, 'left'), (veh.RIGHT, 'right')]:
            tire = veh_obj.GetTire(axle_idx, side_idx)
            force_global = tire.ReportTireForce(terrain)
            # Rotate global-frame force into body frame
            f_body = rot.RotateBack(force_global.force)
            key = f'{axle_name}_{side_name}'
            tf[f'{key}_Fx'] = f_body.x
            tf[f'{key}_Fy'] = f_body.y
            tf[f'{key}_Fz'] = f_body.z
            tf[f'{key}_slip_angle'] = tire.GetSlipAngle()
            tf[f'{key}_long_slip'] = tire.GetLongitudinalSlip()
    return tf


class ReproducibleIMUNoise:
    """Seeded equivalent of Chrono Sensor's normal-drift IMU noise.

    ``ChNoiseNormalDrift`` seeds its private C++ generator from the wall clock
    and exposes no way for a benchmark seed to replay the same IMU stream. A
    seeded run therefore attaches ``ChNoiseNone`` to the sensors and applies
    the same white-noise-plus-bias-random-walk model here, where the seed
    controls it. An unseeded run uses Chrono's own implementation.
    """

    def __init__(self, rng: np.random.Generator, update_rate: float,
                 acc_stdev: float, acc_bias_drift: float, acc_tau_drift: float,
                 gyro_stdev: float, gyro_bias_drift: float, gyro_tau_drift: float):
        self.rng = rng
        self.update_rate = float(update_rate)
        self.acc_stdev = float(acc_stdev)
        self.acc_bias_drift = float(acc_bias_drift)
        self.acc_tau_drift = float(acc_tau_drift)
        self.gyro_stdev = float(gyro_stdev)
        self.gyro_bias_drift = float(gyro_bias_drift)
        self.gyro_tau_drift = float(gyro_tau_drift)
        self.acc_bias = np.zeros(3, dtype=float)
        self.gyro_bias = np.zeros(3, dtype=float)

    def _sample(self, stdev: float, drift: float, tau: float,
                bias: np.ndarray) -> np.ndarray:
        white = self.rng.normal(0.0, stdev, size=3)
        if self.update_rate > 0.0 and drift > np.finfo(float).eps and tau > np.finfo(float).eps:
            drift_stdev = drift * math.sqrt(1.0 / (self.update_rate * tau))
            bias += self.rng.normal(0.0, drift_stdev, size=3)
        return white + bias

    def add_accel(self, values) -> np.ndarray:
        return np.asarray(values, dtype=float) + self._sample(
            self.acc_stdev, self.acc_bias_drift, self.acc_tau_drift, self.acc_bias)

    def add_gyro(self, values) -> np.ndarray:
        return np.asarray(values, dtype=float) + self._sample(
            self.gyro_stdev, self.gyro_bias_drift, self.gyro_tau_drift, self.gyro_bias)


def _attach_truth_pose_to_force_diagnostics(tf, *, noise, pos, rot, vel_loc):
    """Attach evaluator-only pose only when tire diagnostics are enabled.

    In particular, never manufacture a ``tire_forces`` payload when ``tf`` is
    ``None``: that value is the deployed ``--no-tire-forces`` contract.
    """
    if not noise or tf is None:
        return tf
    tf['true_x_cg'] = pos.x
    tf['true_y_cg'] = pos.y
    tf['true_psi'] = math.atan2(
        2 * (rot.e0 * rot.e3 + rot.e1 * rot.e2),
        1 - 2 * (rot.e2 * rot.e2 + rot.e3 * rot.e3))
    tf['true_u'] = vel_loc.x
    return tf


# Diagnostic counters for the IMU analytical (ground-truth) fallback. The IMU
# accelerometer/gyro normally read the Chrono sensor buffer; when a *configured*
# sensor has no sample on a step, the code falls back to analytical rigid-body
# state, which is ground truth. That truth must never silently enter the
# estimator's ax/ay/az/omega channels, so every fallback step is counted and the
# "with_sensor" cases (a configured sensor that still yielded no sample -- the
# only way truth can leak) are surfaced loudly at shutdown.
_IMU_TRUTH_FALLBACK = {
    "accel_steps": 0, "gyro_steps": 0,
    "accel_with_sensor": 0, "gyro_with_sensor": 0,
}


def reset_imu_truth_fallback_counters() -> None:
    for _k in _IMU_TRUTH_FALLBACK:
        _IMU_TRUTH_FALLBACK[_k] = 0


def imu_truth_fallback_report() -> dict:
    return dict(_IMU_TRUTH_FALLBACK)


def extract_vehicle_state(vehicle, sim_time: float, terrain=None,
                          noise: dict = None,
                          imu_acc_sensor=None,
                          imu_gyro_sensor=None,
                          noise_rng: np.random.Generator = None,
                          reproducible_imu_noise: ReproducibleIMUNoise = None,
                          torque_noise_rng: np.random.Generator = None,
                          torque_noise_std: float = 0.0,
                          wheel_center_noise_rng: np.random.Generator = None,
                          wheel_center_noise_std: float = 0.0,
                          wheel_center_calibration_bias: float = 0.0,
                          obstacles_flat: list = None,
                          driver_io: tuple = None) -> VehicleState:
    """Read Chrono vehicle and pack into a VehicleState message.

    Args:
        terrain: If provided, tire forces are included.
        noise: If provided, dict of std-devs to add Gaussian noise to sensors.
        imu_acc_sensor: ChAccelerometerSensor (if available, replaces GetPosDt2).
        imu_gyro_sensor: ChGyroscopeSensor (if available, replaces GetAngVelLocal).
        noise_rng: Optional seeded generator for repeatable non-IMU noise.
        reproducible_imu_noise: Optional seeded replacement for Chrono's
            wall-clock-seeded IMU noise model.
        torque_noise_rng: Independent seeded generator for torque transducers.
        torque_noise_std: Per-channel additive torque-sensor noise (N m).
        wheel_center_noise_rng: Independent seeded generator for the fused
            wheel-center elevation measurement.
        wheel_center_noise_std: Per-wheel elevation noise (m).
        wheel_center_calibration_bias: Run-constant common-mode residual after
            hard-ground/known-plane height calibration (m).
    """
    chassis = vehicle.GetChassisBody()
    pos = chassis.GetPos()
    rot = chassis.GetRot()
    vel = chassis.GetPosDt()

    # Velocity in body frame
    vel_loc = rot.RotateBack(vel)

    x_cg = pos.x
    y_cg = pos.y
    u = vel_loc.x
    v = vel_loc.y

    # --- IMU accelerometer (body-frame acceleration from sensor module) ---
    # Chrono's ChAccelerometerSensor outputs: a_global - gravity_global (in global frame).
    # Rotating to body frame gives the same result as rot.RotateBack(GetPosDt2()).
    _imu_acc_ok = False
    if imu_acc_sensor is not None:
        buf = imu_acc_sensor.GetMostRecentAccelBuffer()
        if buf.HasData():
            data = buf.GetAccelData()  # numpy (3,): global-frame, gravity subtracted
            acc_global = chrono.ChVector3d(float(data[0]), float(data[1]), float(data[2]))
            acc_body = rot.RotateBack(acc_global)
            ax = acc_body.x
            ay = acc_body.y
            az = acc_body.z
            _imu_acc_ok = True
    if not _imu_acc_ok:
        # Fallback: analytical rigid-body acceleration (ground truth)
        acc = chassis.GetPosDt2()
        acc_loc = rot.RotateBack(acc)
        ax = acc_loc.x
        ay = acc_loc.y
        az = acc_loc.z
        _IMU_TRUTH_FALLBACK["accel_steps"] += 1
        if imu_acc_sensor is not None:
            _IMU_TRUTH_FALLBACK["accel_with_sensor"] += 1

    # --- IMU gyroscope (body-frame, includes noise from sensor module) ---
    _imu_gyro_ok = False
    if imu_gyro_sensor is not None:
        buf = imu_gyro_sensor.GetMostRecentGyroBuffer()
        if buf.HasData():
            data = buf.GetGyroData()  # numpy (3,): [Roll, Pitch, Yaw]
            omega_x = float(data[0])  # Roll rate
            omega_y = float(data[1])  # Pitch rate
            omega   = float(data[2])  # Yaw rate
            _imu_gyro_ok = True
    if not _imu_gyro_ok:
        omega_vec = chassis.GetAngVelLocal()
        omega_x = omega_vec.x
        omega_y = omega_vec.y
        omega   = omega_vec.z
        _IMU_TRUTH_FALLBACK["gyro_steps"] += 1
        if imu_gyro_sensor is not None:
            _IMU_TRUTH_FALLBACK["gyro_with_sensor"] += 1

    # Seeded benchmark runs use noise-free Chrono IMU sensors and apply a
    # reproducible copy of ChNoiseNormalDrift after transforming acceleration
    # into the body frame.  Only apply it when a real sensor sample is present;
    # the analytical fallback retains the existing manual-noise behavior.
    if reproducible_imu_noise is not None:
        if _imu_acc_ok:
            ax, ay, az = reproducible_imu_noise.add_accel((ax, ay, az))
        if _imu_gyro_ok:
            omega_x, omega_y, omega = reproducible_imu_noise.add_gyro(
                (omega_x, omega_y, omega))

    # Wheel angular velocities (wheel-encoder equivalent)
    veh_obj = vehicle.GetVehicle()
    wheel_omega_fl = veh_obj.GetSpindleOmega(0, veh.LEFT)
    wheel_omega_fr = veh_obj.GetSpindleOmega(0, veh.RIGHT)
    wheel_omega_rl = veh_obj.GetSpindleOmega(1, veh.LEFT)
    wheel_omega_rr = veh_obj.GetSpindleOmega(1, veh.RIGHT)

    # Wheel-center elevation is a vehicle-side kinematic observation.  The
    # simulator reads spindle position, while the physical implementation is
    # the equivalent RTK/INS + suspension-encoder reconstruction.  No terrain
    # node height, sinkage, or contact quantity is queried here.
    wheel_center_z = [
        float(veh_obj.GetSpindlePos(axle, side).z)
        + float(wheel_center_calibration_bias)
        for axle in (0, 1) for side in (veh.LEFT, veh.RIGHT)
    ]
    if wheel_center_noise_std > 0.0:
        center_normal = (
            wheel_center_noise_rng.normal
            if wheel_center_noise_rng is not None else np.random.normal
        )
        wheel_center_z = [
            value + float(center_normal(0.0, wheel_center_noise_std))
            for value in wheel_center_z
        ]

    # Actuator-side torque transducers.  These channels are available on a
    # physical driveline/brake system and deliberately avoid ReportTireForce.
    driveline = veh_obj.GetDriveline()
    drive_torques = [
        float(driveline.GetSpindleTorque(axle, side))
        for axle in (0, 1) for side in (veh.LEFT, veh.RIGHT)
    ]
    brake_torques = [
        float(veh_obj.GetBrake(axle, side).GetBrakeTorque())
        for axle in (0, 1) for side in (veh.LEFT, veh.RIGHT)
    ]
    if torque_noise_std > 0.0:
        torque_normal = (
            torque_noise_rng.normal
            if torque_noise_rng is not None else np.random.normal
        )
        drive_torques = [
            value + float(torque_normal(0.0, torque_noise_std))
            for value in drive_torques
        ]
        brake_torques = [
            value + float(torque_normal(0.0, torque_noise_std))
            for value in brake_torques
        ]

    # Road-wheel steering angle (steering-angle sensor equivalent, avg L/R)
    steer_angle = 0.5 * (veh_obj.GetSteeringAngle(0, veh.LEFT)
                         + veh_obj.GetSteeringAngle(0, veh.RIGHT))

    # Compute yaw from quaternion for noise injection
    psi = math.atan2(2 * (rot.e0 * rot.e3 + rot.e1 * rot.e2),
                     1 - 2 * (rot.e2 * rot.e2 + rot.e3 * rot.e3))

    # Sensor noise injection
    # NOTE: ax, ay, omega are already noisy when using Chrono sensor-module IMU.
    # Manual noise is only added to non-IMU channels (GPS, speed, etc.).
    if noise:
        normal = noise_rng.normal if noise_rng is not None else np.random.normal
        x_cg  += normal(0, noise['x'])
        y_cg  += normal(0, noise['y'])
        psi   += normal(0, noise['psi'])
        u     += normal(0, noise['u'])
        v     += normal(0, noise['v'])
        # Only add manual noise to IMU channels if sensor module not active
        if not _imu_gyro_ok:
            omega += normal(0, noise['omega'])
        if not _imu_acc_ok:
            ax    += normal(0, noise.get('ax', 0.05))
            ay    += normal(0, noise.get('ay', 0.05))
        # Replace only yaw.  Preserve Chrono roll/pitch so vertical and pitch
        # IMU channels remain usable by terrain estimators on uneven ground.
        roll = math.atan2(
            2.0 * (rot.e0 * rot.e1 + rot.e2 * rot.e3),
            1.0 - 2.0 * (rot.e1 * rot.e1 + rot.e2 * rot.e2),
        )
        pitch = math.asin(max(-1.0, min(
            1.0, 2.0 * (rot.e0 * rot.e2 - rot.e3 * rot.e1)
        )))
        cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
        cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
        cy, sy = math.cos(0.5 * psi), math.sin(0.5 * psi)
        qe0 = cr * cp * cy + sr * sp * sy
        qe1 = sr * cp * cy - cr * sp * sy
        qe2 = cr * sp * cy + sr * cp * sy
        qe3 = cr * cp * sy - sr * sp * cy
    else:
        qe0, qe1, qe2, qe3 = rot.e0, rot.e1, rot.e2, rot.e3

    # Tire forces (optional)
    tf = extract_tire_forces(vehicle, terrain) if terrain is not None else None

    # Evaluator-only truth accompanies explicitly enabled tire diagnostics.
    # ``terrain is None`` is the --no-tire-forces deployment contract, so
    # sensor noise must not recreate a truth-bearing payload in that mode.
    tf = _attach_truth_pose_to_force_diagnostics(
        tf, noise=noise, pos=pos, rot=rot, vel_loc=vel_loc
    )

    return VehicleState(
        time=sim_time,
        wall_time=wall_time.time(),
        x_cg=x_cg,
        y_cg=y_cg,
        z_cg=pos.z,
        quat_e0=qe0,
        quat_e1=qe1,
        quat_e2=qe2,
        quat_e3=qe3,
        u=u,
        v=v,
        omega=omega,
        ax=ax,
        ay=ay,
        az=az,
        omega_x=omega_x,
        omega_y=omega_y,
        wheel_omega_fl=wheel_omega_fl,
        wheel_omega_fr=wheel_omega_fr,
        wheel_omega_rl=wheel_omega_rl,
        wheel_omega_rr=wheel_omega_rr,
        wheel_center_z_fl=wheel_center_z[0],
        wheel_center_z_fr=wheel_center_z[1],
        wheel_center_z_rl=wheel_center_z[2],
        wheel_center_z_rr=wheel_center_z[3],
        drive_torque_fl=drive_torques[0],
        drive_torque_fr=drive_torques[1],
        drive_torque_rl=drive_torques[2],
        drive_torque_rr=drive_torques[3],
        brake_torque_fl=brake_torques[0],
        brake_torque_fr=brake_torques[1],
        brake_torque_rl=brake_torques[2],
        brake_torque_rr=brake_torques[3],
        steering_angle=steer_angle,
        steering_op=float(driver_io[0]) if driver_io else 0.0,
        throttle_op=float(driver_io[1]) if driver_io else 0.0,
        braking_op=float(driver_io[2]) if driver_io else 0.0,
        steering_app=float(driver_io[3]) if driver_io else 0.0,
        throttle_app=float(driver_io[4]) if driver_io else 0.0,
        braking_app=float(driver_io[5]) if driver_io else 0.0,
        tire_forces=tf,
        obstacles=obstacles_flat,
    )


# =============================================================================
# Main simulation loop
# =============================================================================

def run_sim_node(args):
    print("=" * 60)
    print("Chrono Simulation Node (Decoupled)")
    print("=" * 60)

    # A benchmark seed controls every stochastic measurement channel.  Spawn
    # independent streams so adding a state channel later cannot silently
    # perturb the IMU sequence (or vice versa).
    noise_rng = None
    imu_noise_rng = None
    torque_noise_rng = None
    wheel_center_noise_rng = None
    wheel_center_bias_rng = None
    if args.sim_seed is not None:
        seed_sequence = np.random.SeedSequence(args.sim_seed)
        (
            state_seed,
            imu_seed,
            torque_seed,
            wheel_center_seed,
            wheel_center_bias_seed,
        ) = seed_sequence.spawn(5)
        noise_rng = np.random.default_rng(state_seed)
        imu_noise_rng = np.random.default_rng(imu_seed)
        torque_noise_rng = np.random.default_rng(torque_seed)
        wheel_center_noise_rng = np.random.default_rng(wheel_center_seed)
        wheel_center_bias_rng = np.random.default_rng(wheel_center_bias_seed)
        chrono.ChRandom.SetSeed(float(args.sim_seed))
        print(
            f"  Simulation seed: {args.sim_seed} "
            "(state + IMU + torque + wheel-center measurement streams)"
        )

    wheel_center_calibration_bias = 0.0
    if not args.no_noise and args.wheel_center_calibration_bias_std > 0.0:
        bias_normal = (
            wheel_center_bias_rng.normal
            if wheel_center_bias_rng is not None else np.random.normal
        )
        wheel_center_calibration_bias = float(
            bias_normal(0.0, args.wheel_center_calibration_bias_std)
        )
        print(
            "  Wheel-center residual calibration bias sampled "
            f"(sigma={args.wheel_center_calibration_bias_std:.4f} m)"
        )

    # Determine visualization flags
    use_irrlicht = args.vis_mode in ('irrlicht', 'both')
    use_sensor = args.vis_mode in ('sensor', 'both')
    any_vis = use_irrlicht or use_sensor

    if use_sensor and not HAS_SENSOR:
        print("WARNING: pychrono.sensor not available, falling back to irrlicht")
        use_sensor = False
        use_irrlicht = True
        any_vis = True

    latency_profile = None
    if args.latency_profile_json:
        latency_profile = LatencyProfile.from_json(args.latency_profile_json)
        latency_profile.phase_s = float(getattr(args, "latency_phase_s", 0.0))
        print(f"  Latency profile: {latency_profile.describe()}"
              + (f" phase=+{latency_profile.phase_s:.0f}s"
                 if latency_profile.phase_s else ""))
    initial_control_delay = (
        latency_profile.delay(0.0, "control") if latency_profile is not None
        else float(args.teleop_delay)
    )

    # ------------------------------------------------------------------
    # Setup vehicle
    # ------------------------------------------------------------------
    system, vehicle = setup_chrono_vehicle(
        any_vis, payload_mass=getattr(args, "payload_mass", 0.0),
        simple_powertrain=getattr(args, "simple_powertrain", False))
    # Capture the ego assembly before terrain, rocks, traffic, or visual goal
    # bodies are added. Chrono body identifiers are stable for the life of the
    # system and let the contact reporter reject traffic-vs-rock and
    # traffic-vs-traffic contacts from the ego collision KPI.
    ego_body_ids = {int(body.GetIdentifier()) for body in system.GetBodies()}

    # ------------------------------------------------------------------
    # Setup terrain
    # ------------------------------------------------------------------
    terrain_config = None
    if args.terrain_config:
        terrain_config = load_terrain_config(args.terrain_config)

    # Optional spatial soil transition (one preset blends into another along +x).
    spatial_spec = None
    base_preset = args.terrain
    if args.terrain_transition:
        from spatial_terrain import SpatialTransitionSpec
        start_preset = args.terrain_start or args.terrain
        if args.terrain_end is None:
            raise ValueError("--terrain-transition requires --terrain-end")
        spatial_spec = SpatialTransitionSpec(
            start_preset=start_preset,
            end_preset=args.terrain_end,
            transition_x=args.transition_x,
            transition_width=args.transition_width,
        )
        # Base soil must match the start of the patch so the fallback agrees
        # with the callback before the transition. A custom .yaml endpoint is
        # routed through terrain_config (which overrides the preset).
        if str(start_preset).endswith((".yaml", ".yml")):
            if terrain_config is None:
                terrain_config = load_terrain_config(str(start_preset))
            base_preset = args.terrain
        else:
            base_preset = start_preset

    terrain, terrain_params = setup_scm_terrain(
        system, vehicle=vehicle, visualize=any_vis,
        terrain_preset=base_preset, terrain_config=terrain_config,
        bumpiness=args.bumpiness, spatial_spec=spatial_spec,
        mesh_resolution=args.mesh_resolution,
    )

    if any_vis:
        color_hmmwv(vehicle)

    # ------------------------------------------------------------------
    # Rock obstacles
    # ------------------------------------------------------------------
    rocks = []
    collision_logger = None
    if args.rocks > 0:
        # Only clear the immediate spawn so you're in the field quickly; also
        # keep the goal gate clear if one is set.
        exclusion_zones = [(0.0, 0.0, args.rock_spawn_clear)]
        if args.goal_distance > 0:
            exclusion_zones.append((float(args.goal_distance), 0.0, 6.0))
        rocks = add_rock_obstacles(
            system, num_rocks=args.rocks,
            zone_x=tuple(args.rock_zone_x), zone_y=tuple(args.rock_zone_y),
            size_range=tuple(args.rock_size), seed=args.rock_seed,
            min_spacing=args.rock_min_spacing,
            centerline_clear=args.rock_centerline_clear,
            exclusion_zones=exclusion_zones,
        )
        print(f"  Placed {len(rocks)} rock obstacles")
        # Dump static obstacle geometry next to the diagnostic trace so the
        # top-down renderer can draw the field without re-running Chrono.
        if args.sim_diag_csv:
            try:
                import json as _json
                _op = get_rock_positions(rocks); _or = get_rock_radii(rocks)
                _obs_json = Path(args.sim_diag_csv).with_name("obstacles.json")
                _obs_json.write_text(_json.dumps({
                    "kind": "rocks",
                    "obstacles": [{"x": float(px), "y": float(py), "r": float(rr)}
                                  for (px, py, _pz), rr in zip(_op, _or)],
                }, indent=2))
            except Exception as _e:  # noqa: BLE001
                print(f"  [obstacles.json] dump skipped: {_e}")

    # --- Convoy traffic vehicles (PID-driven, shared system) ---
    traffic_mgr = None
    if args.convoy:
        traffic_mgr = TrafficManager.from_preset(args.convoy, ego_lane_y=0.0)
        _detail = args.traffic_detail if any_vis else "none"
        traffic_mgr.build(system, terrain, detail=_detail)
        print(f"  Convoy '{args.convoy}': {len(traffic_mgr.vehicles)} traffic vehicles")

    # --- Goal gate (visible finish line; round ends early on reaching it) ---
    if args.goal_distance > 0 and any_vis:
        _gx = float(args.goal_distance)
        for _gy in (-3.5, 3.5):                       # two bright posts
            post = chrono.ChBodyEasyBox(0.4, 0.4, 3.5, 100.0, True, False)
            post.SetPos(chrono.ChVector3d(_gx, _gy, 1.75)); post.SetFixed(True)
            try:
                post.GetVisualShape(0).SetColor(chrono.ChColor(0.1, 0.9, 0.2))
            except Exception:
                pass
            system.Add(post)
        banner = chrono.ChBodyEasyBox(0.4, 7.4, 0.5, 100.0, True, False)
        banner.SetPos(chrono.ChVector3d(_gx, 0.0, 3.3)); banner.SetFixed(True)
        try:
            banner.GetVisualShape(0).SetColor(chrono.ChColor(0.1, 0.9, 0.2))
        except Exception:
            pass
        system.Add(banner)
        print(f"  Goal gate at x={_gx:.0f} m (round ends on reaching it)")

    # --- Collision detector (active when rocks OR traffic present) ---
    # Parallel sweeps set HIL_RUN_LOG_DIR to a unique per-run directory so
    # collision and safety-filter logs are never shared between concurrent
    # workers: a single shared logs/ directory races on truncation and
    # cross-contaminates collision counts across runs. Live and operator-driven
    # runs leave the variable unset and write to the repository-level logs/.
    _log_dir = os.environ.get('HIL_RUN_LOG_DIR') or os.path.join(
        os.path.dirname(__file__), '..', '..', 'logs')
    collision_logger = (
        CollisionLogger(
            system=system,
            ego_body_ids=ego_body_ids,
            rocks=rocks,
            traffic_body_map=(traffic_mgr.collision_body_map()
                              if traffic_mgr is not None else None),
            run_dir=_log_dir,
        )
        if (rocks or traffic_mgr) else None
    )

    # ------------------------------------------------------------------
    # CBF safety filter
    # ------------------------------------------------------------------
    safety_filter = None
    if args.safety_filter:
        vehicle_params = get_vehicle_params_for_demo()

        # Load the neural tire model that supplies the filter's traction
        # limits. It is the same checkpoint the MPC controller uses, so the two
        # never disagree about achievable force, and it is supervised only by
        # the controlled single-tire Chrono SCM rig. If the import fails or the
        # checkpoint is missing, the filter falls back to kinematic steering
        # authority and fixed longitudinal limits. The belief soil below is the
        # soil-blind versus terrain-aware ablation control, applied uniformly
        # across filters.
        _belief_mode = getattr(args, "shield_init_terrain_belief", "") or ""
        _belief_soil = None
        if _belief_mode in ("sand", "clay", "dirt"):
            _belief_soil = _belief_mode
        elif _belief_mode == "match":
            _belief_soil = args.terrain

        _nn_cbf = None
        if args.no_safety_nn or _belief_mode == "none":
            _why = "--no-safety-nn" if args.no_safety_nn else "--shield-init-terrain-belief none"
            print(f"  [CBF] NN tire model disabled by {_why}; using kinematic (soil-blind) fallback")
        elif load_nn_tire_model is not None:
            try:
                if _belief_soil is not None:
                    _preset = get_terrain_preset(_belief_soil)
                else:
                    _prior_name = getattr(args, "terrain_belief_prior", None)
                    _preset = (
                        get_terrain_preset(_prior_name)
                        if _prior_name
                        else (terrain_config if terrain_config else get_terrain_preset(args.terrain))
                    )
                _tp = terrain_preset_to_internal(_preset)
                _root = Path(__file__).resolve().parents[2]
                _requested = Path(args.nn_model).expanduser()
                if _requested.is_absolute() or len(_requested.parts) > 1:
                    _model_dir = _requested if _requested.is_absolute() else _root / _requested
                else:
                    _model_dir = _root / "nn_models" / args.nn_model
                _nn_cbf = load_nn_tire_model(str(_model_dir), _tp)
                _prior_label = (_belief_soil or getattr(args, "terrain_belief_prior", None)
                                or args.terrain)
                print(f"  [CBF] NN tire model loaded: {args.nn_model} "
                      f"with belief prior {_prior_label}")
            except Exception as _e:
                print(f"  [CBF] NN load failed ({_e}), using kinematic fallback")

        # The predictive filter sizes its OCP for a fixed obstacle count, so
        # that count is exposed: it selects the most threatening n_obstacles
        # each step, and raising the count handles a field dense enough that
        # several obstacles bind at once.
        _mpsf_kw = ({"n_obstacles": int(args.mpsf_n_obstacles),
                     "w_steer": float(args.mpsf_w_steer),
                     "w_progress": float(args.mpsf_w_progress),
                     # The predictive filter loads its own rig surrogate for
                     # braking/cornering authority; the soil-blind switches
                     # must reach it, or --no-safety-nn would disable only the
                     # barrier filter's tire queries while this filter kept
                     # its own.
                     "nn_brake_authority": not (args.no_safety_nn
                                                or _belief_mode == "none")}
                    if args.safety_flavor in ("mpsf", "predictive", "mpc_safety")
                    else {})
        safety_filter = make_safety_filter(
            args.safety_flavor, vehicle_params=vehicle_params,
            nn_model=_nn_cbf,
            cbf_alpha=args.cbf_alpha,
            obstacle_buffer=args.safety_buffer,
            delay_steps=args.delay_steps,
            control_dt=0.1,
            w_long=args.cbf_w_long,
            w_lat=args.cbf_w_lat,
            forward_bias=args.cbf_forward_bias,
            dob_bandwidth=args.dob_bandwidth,
            cbf_flavor=args.cbf_flavor,
            teleop_delay=initial_control_delay,
            stale_cmd_timeout=args.stale_cmd_timeout,
            **_mpsf_kw,
        )
        delay_msg = (f", teleop_delay={initial_control_delay*1000:.0f}ms"
                     if initial_control_delay > 0 else "")
        print(f"  [SAFETY] {args.safety_flavor} filter enabled: alpha={args.cbf_alpha}, "
              f"buffer={args.safety_buffer}m, flavor={args.cbf_flavor}{delay_msg}")

        # Terrain-aware ablation arm: the belief soil is pushed into every
        # filter exposing update_terrain, so the predictive filter picks up
        # surrogate-derived braking and cornering authority. The call is a
        # no-op for the barrier QP, whose tire model was conditioned on the
        # same soil above. The soil-blind arms -- the empty default and 'none'
        # -- skip this and leave both filters at nominal grip.
        if _belief_soil is not None and hasattr(safety_filter, "update_terrain"):
            try:
                _bp = get_terrain_preset(_belief_soil)
                _belief_rt = {
                    "Kphi": float(_bp.get("Kphi", 0.0)), "Kc": float(_bp.get("Kc", 0.0)),
                    "n": float(_bp.get("n", 0.7)),
                    "c": float(_bp.get("cohesion", _bp.get("c", 0.0))),
                    "phi": float(_bp.get("friction_angle", _bp.get("phi", 0.0))),
                    "k": float(_bp.get("janosi_shear", _bp.get("k", 0.0))),
                }
                safety_filter.update_terrain(_belief_rt)
                print(f"  [SAFETY] terrain-aware belief injected: soil={_belief_soil} "
                      f"n={_belief_rt['n']:.2f} phi={_belief_rt['phi']:.1f}deg")
            except Exception as _e:  # noqa: BLE001
                print(f"  [SAFETY] belief injection failed ({_e})")

    # ------------------------------------------------------------------
    # Trajectory markers (visual only)
    # ------------------------------------------------------------------
    if any_vis:
        marker_z = 0.5 if args.bumpiness > 0 else 0.15
        add_trajectory_markers(
            system, args.path,
            marker_z=marker_z,
            lead_in=args.lead_in,
        )

    # ------------------------------------------------------------------
    # Driver (external commands or manual G29)
    # ------------------------------------------------------------------
    if args.synthetic_operator:
        from reference_path import generate_path_waypoints, ReferencePath
        _op_speed = args.synthetic_operator_speed or args.speed
        _xw, _yw = generate_path_waypoints(args.path, lead_in=5.0)
        _op_path = ReferencePath(_xw, _yw, _op_speed)
        driver = SyntheticTeleopDriver(
            vehicle, _op_path, _op_speed,
            k_lat=args.synthetic_operator_klat, k_psi=args.synthetic_operator_kpsi)
        print(f"  Synthetic teleoperator: tracking '{args.path}' at {_op_speed} m/s "
              f"from delayed perception")
    elif args.replay_cmds:
        print(f"  Replay mode: re-driving from command trace {args.replay_cmds}")
        driver = ReplayDriver(vehicle, args.replay_cmds)
    elif args.wasd:
        print("  Manual mode: using WASD keyboard (via Irrlicht window)")
        driver = veh.ChInteractiveDriver(vehicle.GetVehicle())
        driver.SetSteeringDelta(1.0 / 50)
        driver.SetThrottleDelta(1.0 / 50)
        driver.SetBrakingDelta(1.0 / 50)
        driver.SetGains(4.0, 4.0, 4.0, 4.0)
        driver.Initialize()
    elif args.manual:
        print("  Manual mode: using G29 steering wheel")
        driver = ManualDriver(vehicle)
    else:
        driver = ExternalDriver(vehicle)

    # ------------------------------------------------------------------
    # Visualization — Irrlicht
    # ------------------------------------------------------------------
    vis = None
    if use_irrlicht:
        try:
            vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
            vis.SetWindowTitle("Chrono Sim Node (decoupled)")
            vis.SetWindowSize(args.cam_width, args.cam_height)
            set_z_up_if_available(vis)
            vis.Initialize()
            vis.AddLogo(chrono.GetChronoDataFile("logo_chrono_alpha.png"))
            vis.AddLightDirectional(
                45.0,
                120.0,
                chrono.ChColor(0.28, 0.28, 0.28),
                chrono.ChColor(0.08, 0.08, 0.08),
                chrono.ChColor(0.68, 0.68, 0.68),
            )
            vis.AddSkyBox()
            vis.AttachVehicle(vehicle.GetVehicle())
            update_irrlicht_driver_camera(vis, vehicle)
            if args.wasd:
                vis.AttachDriver(driver)
            print("  Irrlicht: driver POV camera active")
        except Exception as e:
            print(f"Warning: Irrlicht visualization failed: {e}")
            vis = None

    # ------------------------------------------------------------------
    # Visualization — Chrono Sensor (driver POV camera)
    # ------------------------------------------------------------------
    sensor_manager = None
    driver_cam = None
    delayed_pov = None
    if use_sensor:
        try:
            sensor_manager = sens.ChSensorManager(system)
            # Scene lighting and environment
            sensor_manager.scene.AddPointLight(
                chrono.ChVector3f(0, 0, 100),
                chrono.ChColor(1.5, 1.5, 1.5),
                500.0,
            )
            sensor_manager.scene.SetAmbientLight(chrono.ChVector3f(0.1, 0.1, 0.1))
            sensor_manager.scene.SetSceneEpsilon(1e-3)
            sensor_manager.scene.EnableDynamicOrigin(True)
            sensor_manager.scene.SetOriginOffsetThreshold(500.0)

            # Driver POV camera attached to chassis
            # Eye-point matches HMMWV left-hand-drive seat position
            cam_offset = chrono.ChFramed(
                DRIVER_CAM_POS_LOCAL,
                DRIVER_CAM_ROT_LOCAL,
            )
            driver_cam = sens.ChCameraSensor(
                vehicle.GetChassisBody(),  # attached body
                args.cam_rate,             # render rate (Hz) — real-time lever
                cam_offset,                # offset pose
                args.cam_width,            # image width
                args.cam_height,           # image height
                args.cam_fov,              # horizontal FOV (rad)
            )
            driver_cam.SetName("DriverPOV")
            driver_cam.SetLag(latency_profile.delay(0.0, "camera") if latency_profile is not None else 0.0)
            # Operator-driven rounds can display the point of view through a
            # software frame-delay buffer, so the camera-channel latency is
            # visible to the driver. SetLag alone does not achieve this,
            # because it does not delay ChFilterVisualize (see delayed_pov).
            want_delayed_pov = bool(getattr(args, "delayed_pov", False)) and \
                (args.manual or args.wasd)
            if want_delayed_pov:
                driver_cam.PushFilter(sens.ChFilterRGBA8Access())
                delayed_pov = DelayedPOV(
                    args.cam_width, args.cam_height,
                    fullscreen=args.cam_fullscreen,
                    flip_vertical=not bool(getattr(args, "pov_no_flip", False)),
                    frame_period_s=1.0 / max(args.cam_rate, 1.0),
                    debug=os.environ.get("DELAYED_POV_DEBUG") == "1")
                if delayed_pov.ok:
                    # The delay buffer applies the whole camera delay in
                    # wall-clock time, so the sensor's own lag is held at zero.
                    # SetLag delays when GetMostRecentRGBA8Buffer becomes
                    # available, so a non-zero value here would compose with
                    # the buffer and double the delay the operator experiences.
                    driver_cam.SetLag(0.0)
                    print(f"  Chrono Sensor: driver POV shown through a "
                          f"software delay buffer (camera-channel latency visible)")
                else:
                    # No display is available for the delay buffer; fall back
                    # to the live view so the round remains driveable.
                    driver_cam.PushFilter(sens.ChFilterVisualize(
                        args.cam_width, args.cam_height, "Driver POV", args.cam_fullscreen))
                    delayed_pov = None
            else:
                delayed_pov = None
                if not getattr(args, "cam_no_window", False):
                    driver_cam.PushFilter(sens.ChFilterVisualize(
                        args.cam_width, args.cam_height, "Driver POV", args.cam_fullscreen
                    ))
            if getattr(args, "cam_save_dir", ""):
                os.makedirs(args.cam_save_dir, exist_ok=True)
                driver_cam.PushFilter(sens.ChFilterSave(args.cam_save_dir + "/"))
                print(f"  Chrono Sensor: saving driver-POV frames to {args.cam_save_dir}")
            sensor_manager.AddSensor(driver_cam)
            print("  Chrono Sensor: driver POV camera active")
        except Exception as e:
            print(f"Warning: Sensor visualization failed: {e}")
            sensor_manager = None
            driver_cam = None
            delayed_pov = None

    # ------------------------------------------------------------------
    # IMU Sensors (Chrono Sensor module — accelerometer + gyroscope)
    # ------------------------------------------------------------------
    imu_acc_sensor = None
    imu_gyro_sensor = None
    reproducible_imu_noise = None
    if HAS_SENSOR and not args.no_imu:
        try:
            # Create a sensor manager if camera mode didn't already
            if sensor_manager is None:
                sensor_manager = sens.ChSensorManager(system)

            imu_rate = args.imu_rate  # Hz
            imu_offset = chrono.ChFramed(
                chrono.ChVector3d(0, 0, 0),
                chrono.ChQuaterniond(1, 0, 0, 0),
            )

            # --- Noise models ---
            if args.no_noise or args.sim_seed is not None:
                acc_noise = sens.ChNoiseNone()
                gyro_noise = sens.ChNoiseNone()
                if not args.no_noise:
                    reproducible_imu_noise = ReproducibleIMUNoise(
                        imu_noise_rng,
                        update_rate=imu_rate,
                        acc_stdev=args.imu_acc_stdev,
                        acc_bias_drift=args.imu_acc_bias_drift,
                        acc_tau_drift=args.imu_acc_tau_drift,
                        gyro_stdev=args.imu_gyro_stdev,
                        gyro_bias_drift=args.imu_gyro_bias_drift,
                        gyro_tau_drift=args.imu_gyro_tau_drift,
                    )
            else:
                # ChNoiseNormalDrift: Gaussian + slow-varying bias drift
                #   (updateRate, mean, stdev, bias_drift, tau_drift)
                # Typical automotive-grade MEMS accelerometer:
                #   noise density ~150 µg/√Hz → stdev ≈ 0.015 m/s² at 100 Hz
                #   bias stability ~10 µg → drift ~ 1e-4 m/s²
                acc_noise = sens.ChNoiseNormalDrift(
                    float(imu_rate),
                    chrono.ChVector3d(0, 0, 0),                                      # mean
                    chrono.ChVector3d(args.imu_acc_stdev, args.imu_acc_stdev, args.imu_acc_stdev),  # stdev
                    args.imu_acc_bias_drift,                                          # bias drift rate
                    args.imu_acc_tau_drift,                                           # tau drift (s)
                )
                # Typical automotive-grade MEMS gyroscope:
                #   noise density ~0.005 °/s/√Hz → stdev ≈ 0.001 rad/s at 100 Hz
                #   bias stability ~1 °/hr → drift ~ 5e-6 rad/s
                gyro_noise = sens.ChNoiseNormalDrift(
                    float(imu_rate),
                    chrono.ChVector3d(0, 0, 0),                                          # mean
                    chrono.ChVector3d(args.imu_gyro_stdev, args.imu_gyro_stdev, args.imu_gyro_stdev),  # stdev
                    args.imu_gyro_bias_drift,                                            # bias drift rate
                    args.imu_gyro_tau_drift,                                             # tau drift (s)
                )

            # --- Accelerometer ---
            imu_acc_sensor = sens.ChAccelerometerSensor(
                vehicle.GetChassisBody(),
                float(imu_rate),
                imu_offset,
                acc_noise,
            )
            imu_acc_sensor.SetName("IMU_Accelerometer")
            imu_acc_sensor.SetLag(args.imu_lag)
            imu_acc_sensor.SetCollectionWindow(0.0)
            imu_acc_sensor.PushFilter(sens.ChFilterAccelAccess())
            sensor_manager.AddSensor(imu_acc_sensor)

            # --- Gyroscope ---
            imu_gyro_sensor = sens.ChGyroscopeSensor(
                vehicle.GetChassisBody(),
                float(imu_rate),
                imu_offset,
                gyro_noise,
            )
            imu_gyro_sensor.SetName("IMU_Gyroscope")
            imu_gyro_sensor.SetLag(args.imu_lag)
            imu_gyro_sensor.SetCollectionWindow(0.0)
            imu_gyro_sensor.PushFilter(sens.ChFilterGyroAccess())
            sensor_manager.AddSensor(imu_gyro_sensor)

            noise_label = "OFF" if args.no_noise else (
                f"acc_σ={args.imu_acc_stdev}, gyro_σ={args.imu_gyro_stdev}"
                + (f", seeded={args.sim_seed}" if args.sim_seed is not None else "")
            )
            print(f"  IMU sensors: {imu_rate} Hz, lag={args.imu_lag}s, noise={noise_label}")
        except Exception as e:
            print(f"Warning: IMU sensor setup failed: {e}")
            imu_acc_sensor = None
            imu_gyro_sensor = None
    elif not HAS_SENSOR and not args.no_imu:
        print("  WARNING: pychrono.sensor not available — using analytical accel/gyro (ground truth)")

    # ------------------------------------------------------------------
    # ROS 2/Chrono::ROS transport (ZMQ remains an explicit fallback)
    # ------------------------------------------------------------------
    _manual_mode = (args.manual or args.wasd or bool(args.replay_cmds)
                    or args.synthetic_operator)
    state_pub = None
    ctrl_sub = None
    # Vehicle state is published in every mode, because the live overlay and
    # telemetry subscribe to it including during operator-driven rounds. Only
    # the autonomous controller link requires the inbound command subscriber.
    if True:
        state_pub = make_publisher(sim_pub_endpoint(args.sim_port), args.transport,
                                   topic=TOPIC_VEHICLE_STATE)
        print(f"  Publishing state on port {args.sim_port} ({args.transport})")
        if not _manual_mode:
            ctrl_sub = make_subscriber(ctrl_sub_endpoint(args.ctrl_host, args.ctrl_port),
                                       args.transport, topic=TOPIC_CONTROL_CMD)
            print(f"  Subscribing to controls from {args.ctrl_host}:{args.ctrl_port} ({args.transport})")

        # Allow the transport to establish itself before the loop starts.
        # ZeroMQ connects in about 0.3 s; DDS needs longer to discover matched
        # endpoints, and best-effort delivery drops anything published before
        # the match completes. Warming up for about two seconds keeps a short
        # run, of 12 to 15 s, from beginning inside that discovery transient
        # and thereby inheriting a corrupted pre-measurement window.
        wall_time.sleep(2.0 if args.transport == "ros" else 0.3)

        # Chrono::ROS-native publishing: the chassis body appears on the ROS
        # graph as pose, twist, and acceleration under
        # ~/chrono/vehicle/state/*, alongside /clock, through Chrono's own
        # ChROSPythonManager. This runs in addition to the richer VehicleState
        # carried by the application transport above. Chrono::ROS is required
        # by the paper configuration; --no-chrono-ros is a development-only
        # opt-out.
        ros_manager = None
        if args.transport == "ros" and not args.no_chrono_ros:
            try:
                import pychrono.ros as chros
                ros_manager = chros.ChROSPythonManager()
                ros_manager.RegisterHandler(chros.ChROSClockHandler())
                ros_manager.RegisterHandler(chros.ChROSBodyHandler(
                    50, vehicle.GetChassisBody(), "~/chrono/vehicle/state"))
                ros_manager.Initialize()
                print("  Chrono::ROS: publishing chassis state on "
                      "~/chrono/vehicle/state/{pose,twist,accel} + /clock")
            except Exception as _e:
                raise RuntimeError(
                    "Chrono::ROS is required for the default ROS paper "
                    "configuration. Rebuild Chrono with "
                    "CH_ENABLE_MODULE_ROS=ON, or use --no-chrono-ros only "
                    "for a non-paper development run."
                ) from _e

        # Publish initial config so controller knows terrain / vehicle params
        vehicle_params = get_vehicle_params_for_demo()
        internal_terrain = terrain_preset_to_internal(
            terrain_config if terrain_config else get_terrain_preset(base_preset)
        )
        # Named preset string is for logging/telemetry; YAML soil overrides physics.
        terrain_label = base_preset
        if terrain_config is not None:
            terrain_label = "custom"

        config_msg = SimStatus(
            event="config",
            time=0.0,
            wall_time=wall_time.time(),
            config={
                "vehicle_params": vehicle_params,
                "terrain_params": internal_terrain,
                "terrain_preset": terrain_label,
                "path_type": args.path,
                "v_target": args.speed,
                "sim_time": args.time,
                "step_size": args.step_size,
                "sine_amplitude": args.sine_amplitude,
                "sine_wavelength": args.sine_wavelength,
                "lead_in": args.lead_in,
            },
        )
        state_pub.send(config_msg)

        # --------------------------------------------------------------
        # Wait for the controller to signal readiness with a neutral
        # ControlCommand, which it sends once acados code generation and
        # solver warm-up are complete. The controller emits these pings
        # without waiting for VehicleState, so the handshake cannot
        # deadlock, and gating here keeps compilation time from consuming
        # the run's simulated duration.
        # --------------------------------------------------------------
        wait_s = 0.0 if args.no_wait_for_controller else float(args.wait_for_controller)
        if ctrl_sub is not None and wait_s > 0:
            print(f"  Waiting for controller ready signal (timeout {wait_s:.0f}s)...")
            t0_wait = wall_time.time()
            last_cfg_send = t0_wait
            got_ready = False
            while wall_time.time() - t0_wait < wait_s:
                # Re-publish the config while waiting, so a controller that is
                # still compiling its solver when the first copy went out
                # still receives one.
                now_wait = wall_time.time()
                if now_wait - last_cfg_send >= 0.5:
                    config_msg.wall_time = now_wait
                    state_pub.send(config_msg)
                    last_cfg_send = now_wait
                result = ctrl_sub.recv(timeout_ms=100)
                if result is None:
                    continue
                _, msg = result
                if isinstance(msg, ControlCommand):
                    driver.apply(msg)
                    print("  Controller ready — starting simulation.")
                    got_ready = True
                    break
            if not got_ready:
                print("  WARNING: No controller handshake before timeout — "
                      "starting simulation anyway. Chrono time may run ahead of MPC.")

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    reset_imu_truth_fallback_counters()
    step_size = args.step_size
    state_pub_interval = 1.0 / args.state_rate  # Decimated publishing rate
    last_state_pub_time = -state_pub_interval
    last_config_resend = 0.0  # Re-publish config during first 2s so controller catches it

    render_interval = 1.0 / 35.0
    last_render_time = -render_interval
    # Sensor-manager update cadence:
    # - With IMU sensors active, Update() is called every physics step, since
    #   each sensor schedules its own update rate internally.
    # - With only the camera active, updates are gated to the camera frame
    #   rate, which avoids per-step overhead that buys nothing.
    _imu_active = (imu_acc_sensor is not None or imu_gyro_sensor is not None)
    sensor_interval = 0.0 if _imu_active else (1.0 / 30.0)
    last_sensor_time = -1.0
    last_report_time = 0.0
    start_wall = wall_time.time()
    cmd_count = 0
    # Both delayed command channels are simulation-time indexed with
    # latest-issue-wins release (see shared/command_delay.py): the delay under
    # test is exact regardless of real-time pacing, and downward latency
    # jitter can never rewind the actuated command to an older one. Under
    # real-time pacing simulation and wall clocks coincide, so interactive use
    # is unchanged.
    from command_delay import CommandDelayBuffer
    cmd_buffer = CommandDelayBuffer()
    terrain_update_count = 0
    last_terrain_seq = -1
    manual_cmd_buffer = CommandDelayBuffer()

    def accept_command_metadata(command) -> None:
        """Advance safety metadata at the command-actuation boundary."""
        nonlocal last_terrain_seq, terrain_update_count
        if safety_filter is not None and command.wall_time > 0:
            safety_filter.update_command_age(command.wall_time)
        last_terrain_seq, applied, sigma_deg = update_safety_terrain_from_command(
            safety_filter,
            command,
            last_terrain_seq,
            no_sigma_gate=bool(getattr(args, "shield_no_sigma_gate", False)),
            hedge_k=float(getattr(args, "shield_hedge_k", 0.0)),
            use_terrain_nn=bool(getattr(args, "shield_terrain_nn", False)),
            use_grip_scale=bool(getattr(args, "shield_grip_scale", False)),
        )
        if applied:
            terrain_update_count += 1
            if terrain_update_count == 1 or terrain_update_count % 50 == 0:
                print(
                    f"  [SHIELD-TERRAIN] update #{terrain_update_count}: "
                    f"n={command.terrain_n:.3f} "
                    f"phi={command.terrain_phi_deg:.2f}\N{DEGREE SIGN} "
                    f"sigma_phi={sigma_deg:.2f}\N{DEGREE SIGN} "
                    f"class={command.terrain_class}",
                    flush=True,
                )
    delayed_manual_inputs = [0.0, 0.0, 0.0]
    cam_lag_ema = None   # EMA-smoothed camera lag (anti-stutter, see SetLag below)
    steer_diverge_t = 0.0   # accumulated time the actual steer angle defies the command
    steer_broken = False    # latched once the front steering/suspension breaks
    applied_steer = 0.0     # physics-rate steering-actuator state (rate-limited cmd)
    STEER_RATE_MAX = 16.0   # ~8 rad/s road wheel; matches the CBF QP's max_steer_rate
    applied_alpha = 0.0     # physics-rate throttle/brake-actuator state (alpha in [-1,1])
    ALPHA_RATE_MAX = 8.0    # throttle/brake rate (1/s); matches the CBF QP's max_alpha_rate
    step_count = 0
    sim_diag_file = None
    sim_diag_writer = None
    sim_diag_interval = 0.1
    last_sim_diag_time = -sim_diag_interval
    if args.sim_diag_csv:
        diag_path = Path(args.sim_diag_csv)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        sim_diag_file = diag_path.open("w", newline="")
        sim_diag_writer = csv.writer(sim_diag_file)
        sim_diag_writer.writerow([
            "time", "x", "y", "z", "speed", "vx_local", "vy_local", "omega_z",
            "steering", "throttle", "braking", "collisions", "near_misses",
            "collision_source", "nearest_clearance_m", "latency_control_s",
            "latency_manual_s", "latency_camera_s",
            "steering_op", "throttle_op", "braking_op",
            "hazard_x", "hazard_y", "hazard_r",
        ])
        print(f"  Sim diagnostic CSV: {diag_path}")

    latency_log_file = None
    latency_log_writer = None
    latency_log_interval = 0.05
    last_latency_log_time = -latency_log_interval
    if args.latency_profile_log:
        latency_log_path = Path(args.latency_profile_log)
        latency_log_path.parent.mkdir(parents=True, exist_ok=True)
        latency_log_file = latency_log_path.open("w", newline="")
        latency_log_writer = csv.writer(latency_log_file)
        latency_log_writer.writerow(["time", "control_delay_s", "manual_delay_s", "camera_delay_s"])
        print(f"  Latency profile log: {latency_log_path}")

    noise_cfg = None if args.no_noise else DEFAULT_MEAS_NOISE
    print(f"  Sensor noise: {'OFF' if noise_cfg is None else 'ON'}")
    print(f"  Physics step: {step_size * 1000:.0f}ms, state rate: {args.state_rate} Hz")
    if _manual_mode:
        if args.manual_honor_time:
            print(f"  Manual mode: automatic stop after {args.time}s")
        else:
            print(f"  Manual mode: close window to exit")
        if args.manual_input_delay > 0:
            print(f"  Manual input actuation delay: {args.manual_input_delay:.3f}s")
        elif latency_profile is not None:
            print(f"  Manual input actuation delay: profile-driven")
    else:
        print(f"  Running {args.time}s simulation...")

    # --- Timing accumulators (debug) ---
    _t_irr = 0.0; _t_sensor = 0.0; _t_terrain_sync = 0.0; _t_terrain_adv = 0.0
    _t_veh_sync = 0.0; _t_veh_adv = 0.0; _t_driver = 0.0; _t_safety = 0.0
    _t_vis_sync = 0.0; _t_vis_adv = 0.0
    _t_loop_total = 0.0; _t_state_extract = 0.0; _t_rt_sleep = 0.0
    _t_report_steps = 0; _sensor_calls = 0

    while True:
        _t_loop_start = wall_time.time()
        time_chrono = vehicle.GetSystem().GetChTime()
        if latency_profile is not None:
            control_delay_s = latency_profile.delay(time_chrono, "control")
            manual_delay_s = latency_profile.delay(time_chrono, "manual")
            camera_delay_s = latency_profile.delay(time_chrono, "camera")
        else:
            control_delay_s = float(args.teleop_delay)
            manual_delay_s = float(args.manual_input_delay)
            camera_delay_s = float(args.camera_input_delay)
        if safety_filter is not None:
            # Delay-blind ablation: the filter is not told the delay even though
            # the channels are still delayed by the profile.
            safety_filter.set_teleop_delay(
                0.0 if getattr(args, "safety_delay_blind", False) else control_delay_s)
        if args.synthetic_operator:
            # The operator perceives its own state through the video channel.
            driver.perception_delay = camera_delay_s

        if (not _manual_mode or args.manual_honor_time or args.replay_cmds
                or args.synthetic_operator) and time_chrono >= args.time:
            break
        if args.goal_distance > 0 and vehicle.GetVehicle().GetPos().x >= args.goal_distance:
            print(f"  GOAL REACHED: ego x={vehicle.GetVehicle().GetPos().x:.1f} m "
                  f">= goal {args.goal_distance:.0f} m at t={time_chrono:.1f} s -- ending round early")
            break
        if vis is not None and not vis.Run():
            break

        # --- Render Irrlicht (frame-skipped) ---
        if vis is not None and (time_chrono - last_render_time >= render_interval):
            _tw = wall_time.time()
            update_irrlicht_driver_camera(vis, vehicle)
            vis.BeginScene()
            vis.Render()
            vis.EndScene()
            _t_irr += wall_time.time() - _tw
            last_render_time = time_chrono

        # --- Receive latest control command (non-blocking) ---
        if ctrl_sub is not None:
            result = ctrl_sub.recv(timeout_ms=0)
            if result is not None:
                topic, msg = result
                if isinstance(msg, ControlCommand):
                    if control_delay_s > 0:
                        cmd_buffer.push(time_chrono + control_delay_s, msg)
                    else:
                        driver.apply(msg)
                        cmd_count += 1
                        accept_command_metadata(msg)

            delayed_msg = cmd_buffer.pop_latest(time_chrono)
            if delayed_msg is not None:
                driver.apply(delayed_msg)
                cmd_count += 1
                accept_command_metadata(delayed_msg)

        # --- Synchronize ---
        _tw = wall_time.time()
        driver.Synchronize(time_chrono)

        driver_inputs = veh.DriverInputs()
        driver_inputs.m_steering = driver.GetSteering()
        driver_inputs.m_throttle = driver.GetThrottle()
        driver_inputs.m_braking = driver.GetBraking()
        # Operator's raw command (pre-delay, pre-safety-filter) for the HMI ghost.
        op_io = (driver_inputs.m_steering, driver_inputs.m_throttle,
                 driver_inputs.m_braking)
        if _manual_mode and manual_delay_s > 0:
            manual_cmd_buffer.push(time_chrono + manual_delay_s, (
                driver_inputs.m_steering,
                driver_inputs.m_throttle,
                driver_inputs.m_braking,
            ))
            released_manual = manual_cmd_buffer.pop_latest(time_chrono)
            if released_manual is not None:
                delayed_manual_inputs[0], delayed_manual_inputs[1], \
                    delayed_manual_inputs[2] = released_manual
                # A delayed manual command actuating IS a command arriving over
                # the link: record it so the filter's stale-command emergency
                # brake is live in replay and synthetic-operator runs. During
                # a profile outage nothing is released, no arrival is
                # recorded, and a delay-aware filter brakes once the stream
                # exceeds its staleness timeout -- previously this mechanism
                # was dead code in every latency study.
                if safety_filter is not None:
                    safety_filter.update_command_age(wall_time.time())
            driver_inputs.m_steering = delayed_manual_inputs[0]
            driver_inputs.m_throttle = delayed_manual_inputs[1]
            driver_inputs.m_braking = delayed_manual_inputs[2]
        _t_driver += wall_time.time() - _tw

        # --- Safety Filter ---
        # Safety filter at ~10Hz
        sf_interval = max(1, int(1.0 / (10.0 * step_size)))  # 10 Hz, matching MPC rate
        if safety_filter is not None and step_count % sf_interval == 0:
            chassis_body = vehicle.GetChassisBody()
            veh_pos = chassis_body.GetPos()
            veh_rot = chassis_body.GetRot()
            veh_psi = np.arctan2(
                2 * (veh_rot.e0 * veh_rot.e3 + veh_rot.e1 * veh_rot.e2),
                1 - 2 * (veh_rot.e2**2 + veh_rot.e3**2))
            vel_world = chassis_body.GetPosDt()
            vel_loc = veh_rot.RotateBack(vel_world)

            all_obstacles = []
            if args.rocks > 0:
                rock_pos = get_rock_positions(rocks)
                rock_rad = get_rock_radii(rocks)
                for i in range(len(rock_pos)):
                    dist = np.sqrt((rock_pos[i, 0] - veh_pos.x)**2 +
                                   (rock_pos[i, 1] - veh_pos.y)**2)
                    if dist < 30.0:
                        # 4th element False => static rock (CBF prefers steering)
                        all_obstacles.append((rock_pos[i, 0], rock_pos[i, 1], rock_rad[i], False))
            if traffic_mgr is not None:
                for ox, oy, orad in traffic_mgr.obstacles():
                    if (ox - veh_pos.x) ** 2 + (oy - veh_pos.y) ** 2 < 30.0 ** 2:
                        # 4th element True => vehicle (CBF weights braking equally)
                        all_obstacles.append((ox, oy, orad, True))

            veh_state = {
                'x': veh_pos.x, 'y': veh_pos.y, 'psi': veh_psi,
                'u': vehicle.GetVehicle().GetSpeed(),
                'v': vel_loc.y, 'omega': chassis_body.GetAngVelLocal().z,
                'delta': driver_inputs.m_steering * 0.49,
            }
            sf_result = safety_filter.filter(
                desired_steering=driver_inputs.m_steering,
                desired_throttle=driver_inputs.m_throttle,
                desired_brake=driver_inputs.m_braking,
                vehicle_state=veh_state,
                obstacles=all_obstacles,
            )
            driver_inputs.m_steering = sf_result.steering
            driver_inputs.m_throttle = sf_result.throttle
            driver_inputs.m_braking = sf_result.braking
        elif safety_filter is not None and safety_filter._last_result is not None:
            cached = safety_filter._last_result
            if cached.was_modified:
                driver_inputs.m_steering = cached.steering
                driver_inputs.m_throttle = cached.throttle
                driver_inputs.m_braking = cached.braking

        # --- Steering and throttle/brake actuator, at the physics rate ---
        # These impose the actuator rate limits every step, so neither a rapid
        # flick of the physical wheel nor the safety filter's 10 Hz output
        # staircase can slam the road wheels and impulse the front suspension,
        # and the pedals cannot chatter. They apply to operator-driven and
        # replayed rounds only: the autonomous NMPC already rate-limits its own
        # commands, integrating the steering rate and clipping to
        # delta +/- dbeta_max*dt, so a second limiter would be redundant there.
        # The rates are high, roughly 8 rad/s at the road wheel and 8 per
        # second on the pedals, so only a physically impossible instantaneous
        # step is excluded.
        if _manual_mode:
            _dmax = STEER_RATE_MAX * step_size
            applied_steer += max(-_dmax, min(_dmax, driver_inputs.m_steering - applied_steer))
            driver_inputs.m_steering = applied_steer

            _alpha_des = driver_inputs.m_throttle - driver_inputs.m_braking
            _damax = ALPHA_RATE_MAX * step_size
            applied_alpha += max(-_damax, min(_damax, _alpha_des - applied_alpha))
            if applied_alpha >= 0.0:
                driver_inputs.m_throttle = applied_alpha
                driver_inputs.m_braking = 0.0
            else:
                driver_inputs.m_throttle = 0.0
                driver_inputs.m_braking = -applied_alpha

        # Applied command (post delay + safety filter) for the HMI solid trace.
        app_io = (driver_inputs.m_steering, driver_inputs.m_throttle,
                  driver_inputs.m_braking)
        driver_io = op_io + app_io

        # --- Front steering and suspension failure detection ---
        # Compares the measured front road-wheel angle against the commanded
        # one. A mechanical failure makes the wheels stop responding, splay
        # apart, or reach a physically impossible angle. This runs in
        # operator-driven and replayed rounds only: it produces a discard
        # signal for the operator, and restricting it there guarantees that no
        # autonomous benchmark run can be terminated early by it.
        if _manual_mode and not steer_broken:
            try:
                _vo = vehicle.GetVehicle()
                _sa_l = _vo.GetSteeringAngle(0, veh.LEFT)
                _sa_r = _vo.GetSteeringAngle(0, veh.RIGHT)
                _steer_act = 0.5 * (_sa_l + _sa_r)
                _cmd_ang = driver_inputs.m_steering * 0.49
                _insane = ((not math.isfinite(_steer_act)) or abs(_steer_act) > 0.9
                           or abs(_sa_l - _sa_r) > 0.6)   # max physical ~0.49; Ackermann split is small
                if abs(_cmd_ang) > 0.12 and abs(_steer_act - _cmd_ang) > 0.28:
                    steer_diverge_t += step_size
                else:
                    steer_diverge_t = 0.0
                if _insane or steer_diverge_t > 1.0:
                    steer_broken = True
                    _msg = (f"FRONT STEERING/SUSPENSION LIKELY BROKEN at t={time_chrono:.1f}s: "
                            f"commanded {_cmd_ang:+.2f} rad, actual L/R "
                            f"{_sa_l:+.2f}/{_sa_r:+.2f} rad -- vehicle is unresponsive to "
                            f"steering. DISCARD THIS ROUND.")
                    print(f"\n  ** {_msg} **\n", flush=True)
                    try:
                        _ld = os.environ.get('HIL_RUN_LOG_DIR') or os.path.join(
                            os.path.dirname(__file__), '..', 'logs')
                        os.makedirs(_ld, exist_ok=True)
                        with open(os.path.join(_ld, 'steering_break.txt'), 'w') as _fh:
                            _fh.write(_msg + "\n")
                    except Exception:
                        pass
            except Exception:
                pass
        if steer_broken:
            # A mechanically failed vehicle yields no usable measurement, so
            # the round ends here rather than recording meaningless tracking.
            print("  Ending round early (front end broken).", flush=True)
            break

        _tw = wall_time.time()
        terrain.Synchronize(time_chrono)
        _t_terrain_sync += wall_time.time() - _tw

        _tw = wall_time.time()
        vehicle.Synchronize(time_chrono, driver_inputs, terrain)
        if traffic_mgr is not None:
            _rock_obs = None
            if args.rocks > 0 and rocks:
                _rp = get_rock_positions(rocks); _rr = get_rock_radii(rocks)
                _rock_obs = [(float(_rp[i, 0]), float(_rp[i, 1]), float(_rr[i]))
                             for i in range(len(_rp))]
            traffic_mgr.synchronize(time_chrono, terrain,
                                    ego_speed=vehicle.GetVehicle().GetSpeed(),
                                    avoid_obstacles=_rock_obs)
        _t_veh_sync += wall_time.time() - _tw

        if vis is not None:
            _tw = wall_time.time()
            vis.Synchronize(time_chrono, driver_inputs)
            _t_vis_sync += wall_time.time() - _tw

        # --- Advance ---
        driver.Advance(step_size)

        _tw = wall_time.time()
        terrain.Advance(step_size)
        _t_terrain_adv += wall_time.time() - _tw

        _tw = wall_time.time()
        vehicle.Advance(step_size)        # ego owns the system -> steps it once
        if traffic_mgr is not None:
            traffic_mgr.advance(step_size)
        if ros_manager is not None:
            ros_manager.Update(time_chrono, step_size)  # Chrono::ROS body/clock publish
        _t_veh_adv += wall_time.time() - _tw

        if vis is not None:
            _tw = wall_time.time()
            vis.Advance(step_size)
            _t_vis_adv += wall_time.time() - _tw

        # --- Update Chrono Sensor manager (gated to camera FPS) ---
        if sensor_manager is not None and (time_chrono - last_sensor_time >= sensor_interval):
            # Apply camera lag whenever a non-zero delay is active, whether it
            # originates in the time-varying latency profile or in the fixed
            # --camera-input-delay used by the delay sweep.
            #
            # The lag is smoothed by an exponential moving average before it is
            # applied. The profile's per-frame camera delay is jittery, and
            # pushing a different lag into the sensor every frame releases
            # buffered frames at uneven intervals, which reads as stutter once
            # the scene is in motion. The average preserves the slow variation
            # between good and poor link regimes while removing the
            # frame-to-frame jump, so the delay the operator sees is realistic
            # and the motion remains smooth.
            if driver_cam is not None and camera_delay_s > 0.0:
                if cam_lag_ema is None:
                    cam_lag_ema = camera_delay_s
                else:
                    cam_lag_ema += 0.1 * (camera_delay_s - cam_lag_ema)
                # SetLag is used only on the live ChFilterVisualize path. When
                # the delay buffer is active it owns the delay and SetLag stays
                # at zero: a positive value there would leave
                # GetMostRecentRGBA8Buffer without data and the display black.
                # Real-time behaviour comes from the buffer deduplicating
                # frames by render timestamp, so the expensive GetRGBA8Data
                # readback runs once per rendered frame, about 30 per second,
                # rather than once per physics step at roughly 330 per second.
                if delayed_pov is None:
                    try:
                        driver_cam.SetLag(cam_lag_ema)
                    except Exception:
                        pass
            _tw = wall_time.time()
            sensor_manager.Update()
            _dt_s = wall_time.time() - _tw
            _t_sensor += _dt_s
            _sensor_calls += 1
            last_sensor_time = time_chrono

            # Buffer the freshly rendered frame, tagged to appear one camera
            # delay from now on the wall clock, using the smoothed lag so
            # frames cannot be reordered by per-frame jitter. Display itself
            # happens every loop iteration below, on a steady wall-clock
            # cadence.
            if delayed_pov is not None:
                _disp_lag = cam_lag_ema if cam_lag_ema is not None else camera_delay_s
                delayed_pov.capture(driver_cam, _disp_lag)

        step_count += 1
        _t_report_steps += 1

        # --- Re-publish config during first 2s (late subscribers can miss it) ---
        if state_pub is not None and time_chrono < 2.0 and time_chrono - last_config_resend >= 0.2:
            config_msg.wall_time = wall_time.time()
            state_pub.send(config_msg)
            last_config_resend = time_chrono

        # --- Publish vehicle state at decimated rate ---
        if state_pub is not None and time_chrono - last_state_pub_time >= state_pub_interval:
            _tw = wall_time.time()
            # Nearest 3 obstacles (rocks + traffic) within 40m → flat list for
            # MPC horizon planning. Traffic poses are dynamic, refreshed here.
            _obs_flat_msg = []
            _vpos_now = vehicle.GetChassisBody().GetPos()
            _vx, _vy = _vpos_now.x, _vpos_now.y
            _cands = []  # (dist, x, y, r)
            if args.rocks > 0:
                _rpos = get_rock_positions(rocks)
                _rrad = get_rock_radii(rocks)
                for _i in range(len(_rpos)):
                    _cands.append((math.hypot(_rpos[_i, 0] - _vx, _rpos[_i, 1] - _vy),
                                   float(_rpos[_i, 0]), float(_rpos[_i, 1]), float(_rrad[_i])))
            if traffic_mgr is not None:
                for ox, oy, orad in traffic_mgr.obstacles():
                    _cands.append((math.hypot(ox - _vx, oy - _vy), ox, oy, orad))
            _cands = sorted((c for c in _cands if c[0] < 40.0), key=lambda c: c[0])
            for _d, ox, oy, orad in _cands[:3]:
                _obs_flat_msg += [ox, oy, orad]

            state_msg = extract_vehicle_state(
                vehicle, time_chrono,
                terrain=None if args.no_tire_forces else terrain,
                noise=noise_cfg,
                imu_acc_sensor=imu_acc_sensor,
                imu_gyro_sensor=imu_gyro_sensor,
                noise_rng=noise_rng,
                reproducible_imu_noise=reproducible_imu_noise,
                torque_noise_rng=torque_noise_rng,
                torque_noise_std=(0.0 if args.no_noise else args.torque_noise_std),
                wheel_center_noise_rng=wheel_center_noise_rng,
                wheel_center_noise_std=(
                    0.0 if args.no_noise else args.wheel_center_noise_std
                ),
                wheel_center_calibration_bias=wheel_center_calibration_bias,
                # An empty list is a valid, fresh obstacle observation.  Reserve
                # None for a missing/invalid feed so the identification probe can
                # distinguish "clear road" from "no obstacle data".
                obstacles_flat=_obs_flat_msg,
                driver_io=driver_io,
            )
            state_pub.send(state_msg)
            _t_state_extract += wall_time.time() - _tw
            last_state_pub_time = time_chrono

        # --- Delayed POV: release frames on the wall clock every iteration ---
        # This sits outside the sensor block so the display cadence tracks real
        # time rather than the simulation step timing, which varies from step
        # to step and would otherwise appear as jitter to the operator.
        if delayed_pov is not None:
            delayed_pov.show()

        # --- Real-time pacing (active unless --no-rt) ---
        # A headless run advances four to five times faster than real time, and
        # the decoupled controller could then process only a small fraction of
        # the published state. Pacing keeps the controller's solve budget in
        # the same relation to simulated time as it would be on a vehicle.
        if not args.no_rt:
            target_wall = start_wall + time_chrono
            remaining = target_wall - wall_time.time()
            if remaining > 0:
                _t_rt_sleep += remaining
                wall_time.sleep(remaining)

        _t_loop_total += wall_time.time() - _t_loop_start

        # --- Collision detection (every physics step when rocks present) ---
        if collision_logger is not None:
            _veh_cg = vehicle.GetChassisBody().GetPos()
            _veh_spd = vehicle.GetVehicle().GetSpeed()
            _traffic_obs = traffic_mgr.obstacles() if traffic_mgr is not None else None
            collision_logger.check(time_chrono, _veh_cg.x, _veh_cg.y, _veh_spd,
                                   extra_obstacles=_traffic_obs)

        if sim_diag_writer is not None and time_chrono - last_sim_diag_time >= sim_diag_interval:
            chassis = vehicle.GetChassisBody()
            pos = chassis.GetPos()
            rot = chassis.GetRot()
            vel_loc = rot.RotateBack(chassis.GetPosDt())
            # (clearance, x, y, r) per hazard, so the nearest one's geometry can
            # be logged beside the scalar clearance; a moving hazard is otherwise
            # unrecoverable from the ego trace alone.
            _clear = []
            if args.rocks > 0 and rocks:
                _rpos = get_rock_positions(rocks)
                _rrad = get_rock_radii(rocks)
                if len(_rpos):
                    _d = np.sqrt((_rpos[:, 0] - pos.x) ** 2 + (_rpos[:, 1] - pos.y) ** 2)
                    _gap = _d - _rrad - VEHICLE_CLEARANCE_RADIUS
                    _i = int(np.argmin(_gap))
                    _clear.append((float(_gap[_i]), float(_rpos[_i, 0]),
                                   float(_rpos[_i, 1]), float(_rrad[_i])))
            if traffic_mgr is not None:
                for ox, oy, orad in traffic_mgr.obstacles():
                    _clear.append((math.hypot(ox - pos.x, oy - pos.y)
                                   - orad - VEHICLE_CLEARANCE_RADIUS,
                                   float(ox), float(oy), float(orad)))
            if _clear:
                nearest_clearance, hz_x, hz_y, hz_r = min(_clear)
            else:
                nearest_clearance = hz_x = hz_y = hz_r = math.nan
            sim_diag_writer.writerow([
                f"{time_chrono:.6f}",
                f"{pos.x:.6f}", f"{pos.y:.6f}", f"{pos.z:.6f}",
                f"{vehicle.GetVehicle().GetSpeed():.6f}",
                f"{vel_loc.x:.6f}", f"{vel_loc.y:.6f}",
                f"{chassis.GetAngVelLocal().z:.6f}",
                f"{driver_inputs.m_steering:.6f}",
                f"{driver_inputs.m_throttle:.6f}",
                f"{driver_inputs.m_braking:.6f}",
                collision_logger.total_collisions if collision_logger is not None else 0,
                collision_logger.total_near_misses if collision_logger is not None else 0,
                COLLISION_SOURCE,
                f"{nearest_clearance:.6f}" if math.isfinite(nearest_clearance) else "",
                f"{control_delay_s:.6f}",
                f"{manual_delay_s:.6f}",
                f"{camera_delay_s:.6f}",
                f"{op_io[0]:.6f}", f"{op_io[1]:.6f}", f"{op_io[2]:.6f}",
                *(f"{v:.6f}" if math.isfinite(v) else ""
                  for v in (hz_x, hz_y, hz_r)),
            ])
            last_sim_diag_time = time_chrono

        if latency_log_writer is not None and time_chrono - last_latency_log_time >= latency_log_interval:
            latency_log_writer.writerow([
                f"{time_chrono:.6f}",
                f"{control_delay_s:.6f}",
                f"{manual_delay_s:.6f}",
                f"{camera_delay_s:.6f}",
            ])
            last_latency_log_time = time_chrono

        # --- Progress report ---
        if time_chrono - last_report_time >= 2.0:
            last_report_time = time_chrono
            elapsed = wall_time.time() - start_wall
            rt = time_chrono / elapsed if elapsed > 0 else 0
            pos = vehicle.GetChassisBody().GetPos()
            _col_str = ""
            if collision_logger is not None:
                _col_str = (f"  collisions={collision_logger.total_collisions}"
                            f"  near_misses={collision_logger.total_near_misses}")
            print(f"  t={time_chrono:.1f}s  pos=({pos.x:.1f},{pos.y:.1f})  "
                  f"RT={rt:.2f}x  cmds_recv={cmd_count}{_col_str}")
            if traffic_mgr is not None:
                _tz = [round(s['x'], 0) for s in traffic_mgr.states()]
                _zz = [round(s.get('z', 0.0), 2) for s in traffic_mgr.states()]
                print(f"    [TRAFFIC] x={_tz}  z={_zz}")
            # --- Timing breakdown (per 2s window) ---
            n = max(_t_report_steps, 1)
            accounted = (_t_terrain_sync + _t_terrain_adv + _t_veh_sync + _t_veh_adv +
                         _t_irr + _t_sensor + _t_driver + _t_safety +
                         _t_vis_sync + _t_vis_adv + _t_state_extract + _t_rt_sleep)
            unaccounted = _t_loop_total - accounted
            sensor_avg_ms = (_t_sensor / max(_sensor_calls, 1)) * 1000
            print(f"    [TIMING] steps={n}  loop_total={_t_loop_total:.3f}s  "
                  f"rt_sleep={_t_rt_sleep:.3f}s  unaccounted={unaccounted:.3f}s")
            print(f"    [TIMING] terrain_sync={_t_terrain_sync:.3f}s  "
                  f"terrain_adv={_t_terrain_adv:.3f}s  veh_sync={_t_veh_sync:.3f}s  "
                  f"veh_adv={_t_veh_adv:.3f}s")
            print(f"    [TIMING] irrlicht={_t_irr:.3f}s  sensor={_t_sensor:.3f}s "
                  f"({_sensor_calls} calls, avg={sensor_avg_ms:.1f}ms)  "
                  f"state_extract={_t_state_extract:.3f}s")
            print(f"    [TIMING] driver={_t_driver:.3f}s  safety={_t_safety:.3f}s  "
                  f"vis_sync={_t_vis_sync:.3f}s  vis_adv={_t_vis_adv:.3f}s")
            _t_irr = 0.0; _t_sensor = 0.0; _t_terrain_sync = 0.0; _t_terrain_adv = 0.0
            _t_veh_sync = 0.0; _t_veh_adv = 0.0; _t_driver = 0.0; _t_safety = 0.0
            _t_vis_sync = 0.0; _t_vis_adv = 0.0
            _t_loop_total = 0.0; _t_state_extract = 0.0; _t_rt_sleep = 0.0
            _t_report_steps = 0; _sensor_calls = 0

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    if delayed_pov is not None:
        delayed_pov.close()
    if collision_logger is not None:
        collision_logger.close()
    if sim_diag_file is not None:
        sim_diag_file.close()
    if latency_log_file is not None:
        latency_log_file.close()

    if state_pub is not None:
        stop_msg = SimStatus(event="stop", time=time_chrono, wall_time=wall_time.time())
        # The live state stream is intentionally BEST_EFFORT/latest-only. Under
        # a heavily parallel ROS paper sweep, a single terminal sample can be
        # lost during DDS teardown even though every simulation result was
        # already written. Repeat the stop edge briefly so the controller can
        # leave its receive loop cleanly without changing steady-state QoS.
        for _ in range(5):
            state_pub.send(stop_msg)
            wall_time.sleep(0.02)

    elapsed = wall_time.time() - start_wall
    print(f"\n  Simulation complete: {time_chrono:.1f}s in {elapsed:.1f}s "
          f"(RT factor {time_chrono / elapsed:.2f}x)")
    if not _manual_mode:
        print(f"  Total control commands received: {cmd_count}")

    # IMU analytical (ground-truth) fallback audit. In a healthy run the
    # configured Chrono IMU sensors supply every step; any "with_sensor"
    # fallback means noise-free ground-truth acceleration/rate entered the
    # estimator's ax/ay/az/omega stream on that step, which the no-oracle
    # contract forbids. Surface it loudly rather than let it pass silently.
    _imu_fb = imu_truth_fallback_report()
    if _imu_active and (_imu_fb["accel_with_sensor"] or _imu_fb["gyro_with_sensor"]):
        print(f"  [IMU][WARN] ground-truth fallback entered the sensor stream on a "
              f"configured sensor: accel={_imu_fb['accel_with_sensor']} "
              f"gyro={_imu_fb['gyro_with_sensor']} step(s) had no sensor sample.")
    elif _imu_fb["accel_steps"] or _imu_fb["gyro_steps"]:
        print(f"  [IMU] analytical model used (no IMU sensor configured): "
              f"accel={_imu_fb['accel_steps']} gyro={_imu_fb['gyro_steps']} step(s).")

    # Safety filter summary
    if safety_filter is not None:
        diag = safety_filter.get_diagnostics()
        # mean_solve_ms is the filter's own solve cost, distinct from the NMPC's.
        # Predictive filters solve an OCP per call and reactive ones a small QP,
        # so this is the number that makes "predictive safety costs a heavier
        # solve" a measurement rather than an assertion. Filters that do not
        # report it are printed without the field rather than as zero.
        _solve_ms = diag.get('mean_solve_ms')
        _solve_txt = (f", MeanSolveMs: {_solve_ms:.3f}"
                      if isinstance(_solve_ms, (int, float)) and _solve_ms > 0 else "")
        print(f"  [SAFETY] Calls: {diag['filter_calls']}, "
              f"Interventions: {diag['interventions']} "
              f"({diag['intervention_rate']*100:.1f}%){_solve_txt}")

    # A ROS 2 DDS participant can occasionally linger in native teardown when
    # many isolated domains exit concurrently. All simulation outputs are
    # already closed and summaries printed at this point, so publish a durable
    # completion edge for launch_decoupled.py before transport destruction.
    # The launcher may then reap a teardown-only straggler without converting
    # a valid benchmark cell into a timeout failure.
    _completion_dir = os.environ.get("HIL_RUN_LOG_DIR")
    if _completion_dir:
        try:
            _completion_path = Path(_completion_dir) / ".sim_complete"
            _completion_path.write_text(
                f"time={time_chrono:.6f}\ncollision_source={COLLISION_SOURCE}\n"
            )
        except OSError as _e:
            print(f"  [shutdown] Could not write completion marker: {_e}")

    if state_pub is not None:
        state_pub.close()
    if ctrl_sub is not None:
        ctrl_sub.close()
    if vis is not None:
        vis.GetDevice().closeDevice()
    if sensor_manager is not None:
        del sensor_manager


# =============================================================================
# Entry point
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Chrono Simulation Node (decoupled)")

    # Simulation
    p.add_argument("--time", type=float, default=15.0, help="Simulation duration (s)")
    p.add_argument("--step-size", type=float, default=3e-3, help="Physics step (s)")
    # Visualization sizing for the driver point-of-view camera and the Irrlicht
    # window. The defaults describe a single 1080p screen. Every pixel is
    # ray-traced against the deformable terrain each frame, so the resolution
    # sets the render budget directly. Use a height of 1200 for a 16:10 screen.
    p.add_argument("--cam-width", type=int, default=1920,
                   help="Driver POV camera / window width (px).")
    p.add_argument("--cam-height", type=int, default=1080,
                   help="Driver POV camera / window height (px). Use 1200 for 16:10.")
    p.add_argument("--cam-fov", type=float, default=1.05,
                   help="Driver POV camera horizontal field of view (rad). The "
                        "default of ~1.05 is 60 degrees, appropriate for a "
                        "single screen; widen it for a wider display.")
    p.add_argument("--cam-rate", type=float, default=30.0,
                   help="Driver POV camera render rate (Hz). Each render ray-traces "
                        "the deformable terrain; combined with --mesh-resolution "
                        "this sets the real-time budget (1080p@30Hz is real-time "
                        "at mesh 0.12, but only ~0.55x at the fine 0.08 mesh).")
    p.add_argument("--cam-fullscreen", action="store_true",
                   help="Display the driver POV fullscreen (renders at "
                        "--cam-width x --cam-height, scaled to the screen).")
    p.add_argument("--cam-save-dir", type=str, default="",
                   help="If set, also save driver-POV camera frames (PNG) to this "
                        "directory (for figures/screenshots).")
    p.add_argument("--cam-no-window", action="store_true",
                   help="Do not open the live driver-POV window; only save frames "
                        "(requires --cam-save-dir). Use for headless video capture.")
    p.add_argument("--convoy", type=str, default="",
                   help="Spawn PID-driven traffic vehicles for a convoy safety "
                        "scenario (lead_brake/cut_in/stalled/swerver/convoy/platoon/"
                        "oncoming/double_cut/stop_and_go/jam/overtake/gauntlet). The "
                        "ego must avoid them; they appear as dynamic obstacles.")
    p.add_argument("--goal-distance", type=float, default=0.0,
                   help="If >0, place a visible goal gate this far ahead (m) and "
                        "end the round early once the ego reaches it.")
    p.add_argument("--synthetic-operator", action="store_true",
                   help="Closed-loop synthetic teleoperator that tracks the "
                        "reference path from delayed perception, modelling "
                        "video-channel latency. Its tracking degrades as "
                        "latency grows, which a fixed replayed trace cannot "
                        "reproduce, so it measures the effect of latency "
                        "without a human in the loop.")
    p.add_argument("--synthetic-operator-speed", type=float, default=None,
                   help="Target speed for the synthetic operator (default: --speed).")
    p.add_argument("--synthetic-operator-klat", type=float, default=0.55)
    p.add_argument("--synthetic-operator-kpsi", type=float, default=1.15)
    p.add_argument("--safety-delay-blind", action="store_true",
                   help="Force the safety filter's teleop_delay to 0 (delay-blind) "
                        "even under a latency profile, for the blind-vs-aware "
                        "latency ablation.")
    p.add_argument("--replay-cmds", type=str, default="",
                   help="Counterfactual replay: drive the ego from a recorded "
                        "operator command trace CSV (steering_op/throttle_op/"
                        "braking_op, such as a recorded run's sim_diag.csv) in "
                        "place of a live driver. The safety filter still "
                        "screens the replayed intent, so one trace can be run "
                        "with the filter disabled and with each filter enabled "
                        "for a causal harm-prevented comparison.")
    p.add_argument("--traffic-detail", choices=["auto", "mesh", "primitives"],
                   default="mesh",
                   help="Traffic vehicle render detail. 'mesh' (default, full HMMWV "
                        "mesh) is real-time for ~3 vehicles on a large terrain; "
                        "'auto' downgrades big scenes to primitive boxes; 'primitives' "
                        "forces boxes. Ignored when headless (no visual assets).")
    p.add_argument("--mesh-resolution", type=float, default=None,
                   help="SCM mesh spacing (m). The default of 0.08 is the "
                        "paper fidelity; 0.12 holds real time for interactive "
                        "and hardware-in-the-loop runs.")
    p.add_argument("--vis-mode", default="irrlicht",
                   choices=["irrlicht", "sensor", "both", "none"],
                   help="Visualization mode: irrlicht, sensor (driver POV), both, or none")
    p.add_argument("--irrlicht-window-size", type=int, nargs=2,
                   metavar=("WIDTH", "HEIGHT"), default=[4320, 720],
                   help="Irrlicht window size in pixels")
    p.add_argument("--no-rt",  action="store_true",
                   help="Disable real-time pacing (fast-forward; breaks decoupled MPC)")
    p.add_argument("--no-tire-forces", action="store_true",
                   help="Omit per-wheel tire forces from state messages. These "
                        "are evaluation diagnostics only; the controller and "
                        "estimator must operate identically without them.")
    p.add_argument("--speed", type=float, default=5.0, help="Target speed for markers (m/s)")

    # Terrain
    p.add_argument("--terrain", default="sand", choices=["sand", "clay", "dirt"])
    p.add_argument(
        "--terrain-belief-prior",
        default=None,
        choices=["sand", "clay", "dirt"],
        help="Blind prior held by the plant-side safety modules until a live "
             "terrain estimate arrives from the controller. It does not alter "
             "the SCM physics, which stay set by --terrain.",
    )
    p.add_argument("--terrain-config", type=str, default=None, help="YAML terrain config")
    p.add_argument("--bumpiness", type=int, default=0, choices=range(0, 11),
                    help="Terrain bumpiness level 0 (flat) to 10 (extreme)")

    # Spatial soil transition: soil changes type partway across the patch via a
    # per-location SCM callback (vehicle drives +x through the boundary).
    p.add_argument("--terrain-transition", action="store_true",
                   help="Enable a spatial soil transition along +x "
                        "(--terrain-start blends into --terrain-end).")
    p.add_argument("--terrain-start", default=None,
                   help="Soil before the transition: preset name (sand/clay/"
                        "dirt) or a custom soil .yaml path (defaults to "
                        "--terrain).")
    p.add_argument("--terrain-end", default=None,
                   help="Soil after the transition: preset name or a custom "
                        "soil .yaml path.")
    p.add_argument("--transition-x", type=float, default=60.0,
                   help="Center of the soil transition, in terrain x (m).")
    p.add_argument("--transition-width", type=float, default=2.0,
                   help="Full width of the linear soil blend (m); 0 = hard step.")

    # Path (for visual markers only; the controller handles actual path generation)
    p.add_argument("--path", default="lane_change",
                   choices=["lane_change", "double_lane_change", "right_left", "sinusoidal", "straight"])
    p.add_argument("--sine-amplitude", type=float, default=2.0)
    p.add_argument("--sine-wavelength", type=float, default=30.0)
    p.add_argument("--lead-in", type=float, default=0.0,
                   help="Straight lead-in distance (m) before path starts")

    # Network
    p.add_argument("--sim-port", type=int, default=5555, help="Port to publish state")
    p.add_argument("--ctrl-host", default="localhost", help="Controller host")
    p.add_argument("--ctrl-port", type=int, default=5556, help="Controller command port")
    p.add_argument("--transport", choices=["zmq", "ros"], default=os.environ.get("HIL_TRANSPORT", "ros"),
                   help="IPC transport for the sim<->controller link: ros "
                        "(default, direct rclpy/DDS; needs ROS 2 sourced) or zmq.")
    p.add_argument("--no-chrono-ros", action="store_true",
                   help="Development-only: keep the rclpy/DDS application "
                        "transport but disable the default Chrono::ROS body "
                        "state and /clock publishers. Not used by paper runs.")
    p.add_argument("--state-rate", type=int, default=100,
                   help="Vehicle state publish rate (Hz)")
    p.add_argument("--no-noise", action="store_true",
                   help="Disable sensor noise (noise ON by default)")
    p.add_argument("--sim-seed", type=int, default=None,
                   help="Seed all stochastic measurement channels. This replays the "
                        "noise generators, but asynchronous transport/process scheduling "
                        "can still change the closed-loop trajectory. If omitted, "
                        "measurement noise remains nondeterministic.")
    p.add_argument(
        "--torque-noise-std", type=float, default=5.0,
        help="Per-wheel driveline/brake torque-sensor noise stdev in N m.",
    )
    p.add_argument(
        "--wheel-center-noise-std", type=float, default=0.01,
        help="Per-wheel fused wheel-center elevation noise stdev in metres.",
    )
    p.add_argument(
        "--wheel-center-calibration-bias-std", type=float, default=0.0,
        help=(
            "Run-constant common wheel-height bias stdev after hard-ground/"
            "known-plane calibration, in metres."
        ),
    )
    p.add_argument("--sim-diag-csv", default="",
                   help="Write plant-side state and control diagnostics to "
                        "this CSV.")

    # IMU sensor (Chrono sensor module)
    p.add_argument("--no-imu", action="store_true",
                   help="Disable Chrono sensor-module IMU (use analytical ground-truth accel/gyro)")
    p.add_argument("--imu-rate", type=int, default=100,
                   help="IMU update rate in Hz (default 100)")
    p.add_argument("--imu-lag", type=float, default=0.0,
                   help="IMU sensor lag in seconds (default 0)")
    p.add_argument("--imu-acc-stdev", type=float, default=0.015,
                   help="Accelerometer noise stdev in m/s² (default 0.015, ~150µg/√Hz MEMS)")
    p.add_argument("--imu-acc-bias-drift", type=float, default=1e-4,
                   help="Accelerometer bias drift rate (default 1e-4)")
    p.add_argument("--imu-acc-tau-drift", type=float, default=100.0,
                   help="Accelerometer drift time constant in s (default 100)")
    p.add_argument("--imu-gyro-stdev", type=float, default=0.001,
                   help="Gyroscope noise stdev in rad/s (default 0.001, ~0.005°/s/√Hz MEMS)")
    p.add_argument("--imu-gyro-bias-drift", type=float, default=5e-6,
                   help="Gyroscope bias drift rate (default 5e-6)")
    p.add_argument("--imu-gyro-tau-drift", type=float, default=500.0,
                   help="Gyroscope drift time constant in s (default 500)")

    p.add_argument("--wait-for-controller", type=float, default=300.0,
                   help="Wait up to this many seconds for the controller's first control message (ready ping after "
                        "ACADOS init) before advancing Chrono. Default 300. Start the sim first, then the "
                        "controller, or use launch_decoupled.py.")
    p.add_argument("--no-wait-for-controller", action="store_true",
                   help="Enter the sim loop immediately (no MPC handshake). Use for sim-only / debugging without a "
                        "controller node.")

    # Manual control
    p.add_argument("--manual", action="store_true",
                   help="Manual control with G29 steering wheel (no MPC controller)")
    p.add_argument("--wasd", action="store_true",
                   help="Manual control with WASD keyboard (no MPC controller)")
    p.add_argument("--manual-honor-time", action="store_true",
                   help="In manual mode, stop automatically after --time seconds.")
    p.add_argument("--manual-input-delay", type=float, default=0.0,
                   help="Fixed actuation delay applied to manual steering/throttle/brake inputs.")
    p.add_argument("--camera-input-delay", type=float, default=0.0,
                   help="Fixed lag applied to the driver POV camera feed. Models "
                        "downlink video latency to the operator. Overridden by the "
                        "camera channel of --latency-profile-json when supplied.")
    p.add_argument("--delayed-pov", action="store_true",
                   help="Display the driver POV through a software frame-delay "
                        "buffer, so the camera-channel latency is visible to "
                        "the operator. Chrono's SetLag delays only data "
                        "availability and not the ChFilterVisualize display, "
                        "so without this the live view is real-time whatever "
                        "the configured delay. Requires a display and falls "
                        "back to the live view if one cannot be opened. "
                        "Ignored outside operator-driven rounds.")
    p.add_argument("--pov-no-flip", action="store_true",
                   help="Disable the delayed POV's default vertical flip (the Chrono "
                        "RGBA8 buffer is bottom-up, so it is flipped upright by default).")
    p.add_argument("--latency-phase-s", type=float, default=0.0,
                   help="Shift the latency-profile replay window by this many "
                        "seconds (mod trace length); paired arms must share it.")
    p.add_argument("--latency-profile-json", default="",
                   help="JSON profile for time-varying 5G-like one-way latency. "
                        "Overrides fixed --teleop-delay/--manual-input-delay per channel.")
    p.add_argument("--latency-profile-log", default="",
                   help="Optional CSV path for logging active control/manual/camera latency samples.")

    # Rock obstacles
    p.add_argument("--payload-mass", type=float, default=0.0,
                   help="Unmodelled cargo mass (kg) added to the chassis. The "
                        "controller keeps the nominal empty-vehicle mass, so a "
                        "non-zero value creates a persistent plant/model "
                        "mismatch for robustness experiments.")
    p.add_argument("--simple-powertrain", action="store_true",
                   help="Near-direct drive: linear EngineSimple + CVT (no engine "
                        "RPM map, no gear shifts) so throttle->wheel-torque is "
                        "~linear/soil-independent -- the clean actuation map the "
                        "force-balance NMPC needs.")
    p.add_argument("--rocks", type=int, default=0,
                   help="Number of rock obstacles (0 = none)")
    p.add_argument("--rock-zone-x", type=float, nargs=2, default=[-15.0, 50.0])
    p.add_argument("--rock-zone-y", type=float, nargs=2, default=[-10.0, 10.0])
    p.add_argument("--rock-size", type=float, nargs=2, default=[0.5, 3.0])
    p.add_argument("--rock-seed", type=int, default=42)
    p.add_argument("--rock-min-spacing", type=float, default=0.0,
                   help="Min center-to-center spacing (m) between rocks. >0 makes "
                        "a threadable blue-noise boulder field (no free bypass).")
    p.add_argument("--rock-centerline-clear", type=float, default=0.0,
                   help="Lateral half-width (m) around y=0 where rock density is "
                        "thinned so the convoy lead can pick a line (not a clear lane).")
    p.add_argument("--rock-spawn-clear", type=float, default=12.0,
                   help="Radius (m) of the rock-free circle around the spawn.")

    # Safety filter
    p.add_argument("--safety-filter", action="store_true",
                   help="Enable the safety filter (flavor controlled by --safety-flavor)")
    p.add_argument("--safety-flavor", type=str, default="dob_cbf",
                   choices=["dob_cbf", "mpsf"],
                   help="Safety filter: dob_cbf is the pointwise DOB-CBF-QP; "
                        "mpsf is the predictive Model Predictive Safety "
                        "Filter, least-restrictive over a horizon.")
    p.add_argument("--shield-no-sigma-gate", action="store_true",
                   help="Ablation arm: zero the controller's friction-angle "
                        "uncertainty before the safety filter reads it, which "
                        "is equivalent to --shield-sigma-mode off.")
    p.add_argument("--shield-hedge-k", type=float, default=0.0,
                   help="Belief-robust safety authority: evaluate the filter's "
                        "acceleration and braking limits at the pessimistic "
                        "soil quantile n_hat - k*sigma_n, taking sigma_n from "
                        "ControlCommand.terrain_n_sigma, so an uncertain "
                        "estimate widens the stopping buffer. 0 disables it "
                        "and the authority follows the point estimate.")
    p.add_argument("--shield-terrain-nn", action="store_true",
                   help="Condition the safety filter's own tire queries -- "
                        "barrier dynamics, steering authority, and the "
                        "traction speed limit -- on the live terrain estimate "
                        "rather than the model's nominal soil.")
    p.add_argument("--shield-grip-scale", action="store_true",
                   help="Scale the filter's control authority by the measured "
                        "grip ratio from the controller's force-map adapter "
                        "(ControlCommand.terrain_grip_scale). This covers soil "
                        "whose grip departs from what its Bekker exponent "
                        "implies, which the exponent estimate alone cannot see.")
    p.add_argument("--mpsf-w-progress", type=float, default=0.0,
                   help="MPSF only: progress weight. >0 penalizes crawling below "
                        "a 2 m/s floor so the filter threads a dense field instead "
                        "of stalling (when a safe thread exists); the hard "
                        "obstacle constraint still forces a stop when none does. "
                        "0 = brake-primary default (strict intent preservation). "
                        "Note: >0 also nudges the vehicle up to the floor speed in "
                        "obstacle-free driving, so keep it 0 unless dense-field "
                        "threading is required.")
    p.add_argument("--mpsf-w-steer", type=float, default=1.0,
                   help="MPSF only: steering-deviation cost weight (throttle "
                        "weight is 1.0). Lower (<1) makes weaving cheaper than "
                        "braking, so it threads dense fields instead of stopping; "
                        "1.0 keeps braking primary (intent-preserving default).")
    p.add_argument("--mpsf-n-obstacles", type=int, default=3,
                   help="MPSF only: number of obstacles the predictive OCP sizes "
                        "for. The filter feeds it the most-threatening "
                        "(on-path-ahead) n at each step, so a larger world is "
                        "handled by selection; raise this for dense fields where "
                        "several obstacles bind simultaneously. Solve cost grows "
                        "sub-linearly (~4 ms at 16).")
    p.add_argument("--shield-init-terrain-belief", type=str, default="",
                   choices=["", "none", "match", "sand", "clay", "dirt"],
                   help="Set every safety filter's terrain belief uniformly at "
                        "construction, which is the control for the soil-blind "
                        "versus terrain-aware ablation. The empty default "
                        "leaves each filter at its own setting: dob_cbf "
                        "conditions its tire model on the terrain preset while "
                        "mpsf stays soil-blind. 'none' makes every filter "
                        "soil-blind (dob_cbf kinematic, mpsf nominal grip). "
                        "'match' makes every filter terrain-aware to "
                        "--terrain, and naming a soil forces that belief. Each "
                        "terrain-aware setting also calls update_terrain, so "
                        "mpsf takes up surrogate-derived braking and cornering "
                        "authority.")
    p.add_argument("--shield-sigma-mode", type=str, default="off",
                   choices=["tighten", "inflate", "both", "off"],
                   help="How the estimator's friction-angle uncertainty acts "
                        "on the safety filter. The default, off, runs the "
                        "filter on its initial terrain belief. tighten, "
                        "inflate, and both are the arms of the "
                        "uncertainty-gating ablation.")
    p.add_argument("--shield-sigma-buffer-gain", type=float, default=0.05,
                   help="Metres of extra obstacle buffer per degree of phi_sigma.")
    # Parameters shared by both safety filters.
    p.add_argument("--shield-horizon", type=int, default=12,
                   help="Prediction horizon steps (latency adds more dynamically).")
    p.add_argument("--cbf-alpha", type=float, default=5.0)
    p.add_argument("--safety-buffer", type=float, default=0.25)
    p.add_argument("--delay-steps", type=int, default=5)
    p.add_argument("--cbf-w-long", type=float, default=0.06)
    p.add_argument("--cbf-w-lat", type=float, default=0.50)
    p.add_argument("--cbf-forward-bias", type=float, default=3.0)
    p.add_argument("--dob-bandwidth", type=float, default=10.0)
    p.add_argument("--cbf-flavor", type=str, default="balance",
                   choices=["balance", "steer_priority", "throttle_priority"])
    p.add_argument("--nn-model", type=str, default="tire_force_static",
                   help="Directory under nn_models/ holding the tire model the "
                        "safety filter queries for traction limits.")
    p.add_argument("--no-safety-nn", action="store_true",
                   help="Run the safety filter without its neural tire model; "
                        "dob_cbf then uses kinematic steering authority and "
                        "fixed longitudinal limits.")
    p.add_argument("--teleop-delay", type=float, default=0.0,
                   help="Initial one-way teleop delay estimate in seconds "
                        "(0 = local, auto-measured from cmd timestamps)")
    p.add_argument("--stale-cmd-timeout", type=float, default=2.0,
                   help="Auto-brake if no command received for this many seconds")

    args = p.parse_args()
    run_sim_node(args)


if __name__ == "__main__":
    main()
