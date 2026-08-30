#!/usr/bin/env python3
"""
HIL Message Definitions and Transport Factories
================================================

Shared message types and serialization for communication between the decoupled
Chrono simulation and the MPC controller. ROS 2/DDS is the default transport; a
self-contained ZeroMQ backend is available for development runs outside the
paper workflow, and carries identical message bytes.

Topics:
  - vehicle_state  : Sim → Controller  (PUB/SUB)
  - control_cmd    : Controller → Sim   (PUB/SUB)
  - sim_status     : Sim → Controller   (PUB/SUB, lifecycle events)

All messages are serialized as msgpack for minimal overhead (~2-5μs round-trip
on localhost, vs ~50μs for JSON).
"""

import os as _os, sys as _sys  # flat-import bootstrap (simulation/flatpath.py)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import flatpath  # noqa: E402,F401
import struct
import time as _time
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

import numpy as np

try:
    import msgpack

    def _serialize(obj: dict) -> bytes:
        return msgpack.packb(obj, use_bin_type=True)

    def _deserialize(data: bytes) -> dict:
        return msgpack.unpackb(data, raw=False)

except ImportError:
    # Fallback to JSON if msgpack not available
    import json

    def _serialize(obj: dict) -> bytes:
        return json.dumps(obj).encode()

    def _deserialize(data: bytes) -> dict:
        return json.loads(data.decode())


# =============================================================================
# Physical actuator ranges (drive-by-wire command units). The control bus and
# safety filters command a road-wheel steering ANGLE (rad) and a drive TORQUE
# (N m, negative = brake). These are the actuator saturation limits; the Chrono
# driver maps them onto the vehicle's steering/engine actuators at the plant
# boundary. Grounded to the HMMWV: tau_max = M*ax_max*r, brake_tau = M*|ax_min|*r.
# =============================================================================
STEER_MAX_RAD = 0.528          # road-wheel steering lock (rad)
DRIVE_TORQUE_MAX_NM = 2298.0   # M(2573)*ax_max(1.9)*r(0.47), total drive torque
BRAKE_TORQUE_MAX_NM = 3144.0   # M(2573)*|ax_min|(2.6)*r(0.47), total brake torque


# =============================================================================
# Message types
# =============================================================================

@dataclass
class VehicleState:
    """Published by sim node at physics rate (or decimated)."""
    time: float           # Chrono simulation time (s)
    wall_time: float      # Wall-clock timestamp for latency measurement (s)
    # CG position in global frame
    x_cg: float
    y_cg: float
    z_cg: float
    # Orientation (quaternion e0, e1, e2, e3)
    quat_e0: float
    quat_e1: float
    quat_e2: float
    quat_e3: float
    # Velocity in body (local) frame
    u: float              # Longitudinal speed (m/s)
    v: float              # Lateral speed (m/s)
    # Angular velocity
    omega: float          # Yaw rate (rad/s)
    # Body-frame acceleration (IMU)
    ax: float = 0.0       # Longitudinal acceleration (m/s²)
    ay: float = 0.0       # Lateral acceleration (m/s²)
    # Vertical-dynamics IMU channels. Buzhardt & Tallapragada (2024) show that
    # vertical acceleration and pitch rate carry the strongest Bekker-n signal
    # for a half-car vehicle traversing deformable terrain, so both are
    # published alongside the planar channels for the terrain estimator.
    az: float = 0.0       # Vertical acceleration in body frame (m/s²)
    omega_x: float = 0.0  # Roll rate (rad/s)
    omega_y: float = 0.0  # Pitch rate (rad/s)
    # Wheel angular velocities (wheel encoders, rad/s)
    wheel_omega_fl: float = 0.0
    wheel_omega_fr: float = 0.0
    wheel_omega_rl: float = 0.0
    wheel_omega_rr: float = 0.0
    # Wheel-center elevation from the fused pose/suspension measurement used
    # by the level-ground terrain-identification maneuver.  In hardware this
    # is reconstructed from RTK/INS pose and suspension-position encoders; it
    # is not a tire-force or soil-truth channel.
    wheel_center_z_fl: float = 0.0
    wheel_center_z_fr: float = 0.0
    wheel_center_z_rl: float = 0.0
    wheel_center_z_rr: float = 0.0
    # Deployable driveline/brake torque-sensor channels (N m).  These are
    # actuator-side measurements, not Chrono tire/contact-force truth.
    drive_torque_fl: float = 0.0
    drive_torque_fr: float = 0.0
    drive_torque_rl: float = 0.0
    drive_torque_rr: float = 0.0
    brake_torque_fl: float = 0.0
    brake_torque_fr: float = 0.0
    brake_torque_rl: float = 0.0
    brake_torque_rr: float = 0.0
    # Road-wheel steering angle (steering sensor, rad)
    steering_angle: float = 0.0
    # Driver-input telemetry for the live HMI overlay (normalized units).
    # "op" = operator's raw command (pre-safety-filter); "app" = what the
    # vehicle actually applied (post-filter). Their divergence is the takeover.
    steering_op: float = 0.0     # operator steering [-1, 1]
    throttle_op: float = 0.0     # operator throttle [0, 1]
    braking_op: float = 0.0      # operator brake [0, 1]
    steering_app: float = 0.0    # applied steering [-1, 1]
    throttle_app: float = 0.0    # applied throttle [0, 1]
    braking_app: float = 0.0     # applied brake [0, 1]
    # Tire info (optional, for diagnostics)
    tire_forces: Optional[Dict[str, float]] = None
    # Nearby obstacles for MPC planning: flat list [x0,y0,r0, x1,y1,r1, ...].
    # Sorted by distance; at most N_OBS entries. [] means a valid clear-road
    # observation, while None is reserved for a missing/invalid obstacle feed.
    obstacles: Optional[List[float]] = None

    def to_bytes(self) -> bytes:
        d = asdict(self)
        return _topic_frame(b"vehicle_state", _serialize(d))

    @classmethod
    def from_dict(cls, d: dict) -> "VehicleState":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class ControlCommand:
    """Published by controller node at MPC rate.

    The physical command bus is ``(delta, drive_torque)`` -- road-wheel
    steering angle (rad) and drive torque (N m, negative = brake). Every
    command source (NMPC, operator, safety filters) speaks these units; the
    Chrono driver maps them onto the actuators at the plant boundary.
    """
    time: float           # Simulation time this command targets
    wall_time: float      # Wall-clock timestamp
    seq: int              # Sequence number
    # Physical command bus:
    delta: float          # Road-wheel steering angle (rad)
    drive_torque: float   # Drive torque (N m); negative = brake torque
    # MPC internal state (for warm-start continuity)
    acceleration: float   # Longitudinal acceleration state (m/s²)
    delta_dot: float      # Steering rate command (rad/s)
    jerk: float           # Longitudinal jerk command (m/s³)
    # Controller diagnostics
    solve_time_ms: float = 0.0
    mpc_cost: float = 0.0
    # Optional live terrain estimate forwarded to the plant-side safety
    # filter. It travels on ControlCommand because the controller-to-plant
    # channel is latest-only: a separate terrain topic multiplexed onto the
    # same socket would be discarded whenever a ControlCommand arrived between
    # estimator ticks. ``terrain_n=None`` means no live estimate has been
    # produced yet.
    terrain_n: Optional[float] = None
    terrain_n_sigma: Optional[float] = None   # posterior std of the n belief
    # Measured grip scale from the controller's force-map adapter
    # (min of the per-axle realised/modelled Fy ratios); None = no adapter.
    terrain_grip_scale: Optional[float] = None
    terrain_phi_deg: Optional[float] = None
    terrain_phi_sigma_deg: Optional[float] = None
    terrain_Kphi: Optional[float] = None
    terrain_Kc: Optional[float] = None
    terrain_c: Optional[float] = None
    terrain_k: Optional[float] = None
    terrain_class: Optional[str] = None
    terrain_confidence: Optional[float] = None
    terrain_update_seq: int = 0  # monotone counter; shield acts on change

    def to_bytes(self) -> bytes:
        d = asdict(self)
        return _topic_frame(b"control_cmd", _serialize(d))

    @classmethod
    def from_dict(cls, d: dict) -> "ControlCommand":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class SimStatus:
    """Lifecycle / configuration messages from sim node."""
    event: str            # "start", "stop", "config"
    time: float
    wall_time: float
    config: Optional[Dict[str, Any]] = None  # terrain params, vehicle params, etc.

    def to_bytes(self) -> bytes:
        d = asdict(self)
        return _topic_frame(b"sim_status", _serialize(d))

    @classmethod
    def from_dict(cls, d: dict) -> "SimStatus":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class TerrainUpdate:
    """Live terrain-estimator output, controller to plant-side safety filter.

    The filter reads this each tick to re-condition its tire surrogate on the
    estimated soil and to tighten its friction-cone margin by phi_sigma_deg.
    It is multiplexed onto the controller's publishing socket under the
    ``terrain_update`` topic.
    """
    time: float
    wall_time: float
    n: float
    phi_deg: float
    phi_sigma_deg: float      # ensemble/EMA-residual uncertainty
    terrain_class: str = "estimated"
    confidence: float = 0.0
    Kphi: float = 0.0
    Kc: float = 0.0
    c: float = 0.0
    k: float = 0.0

    def to_bytes(self) -> bytes:
        d = asdict(self)
        return _topic_frame(b"terrain_update", _serialize(d))

    @classmethod
    def from_dict(cls, d: dict) -> "TerrainUpdate":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# =============================================================================
# ZMQ topic framing helpers
# =============================================================================

def _topic_frame(topic: bytes, payload: bytes) -> bytes:
    """Prefix payload with a 2-byte topic length + topic (ZMQ multipart alternative)."""
    return struct.pack("!H", len(topic)) + topic + payload


def parse_message(raw: bytes):
    """Parse a topic-framed message. Returns (topic_str, dataclass_instance)."""
    topic_len = struct.unpack("!H", raw[:2])[0]
    topic = raw[2:2 + topic_len].decode()
    payload = _deserialize(raw[2 + topic_len:])

    _registry = {
        "vehicle_state": VehicleState,
        "control_cmd": ControlCommand,
        "sim_status": SimStatus,
        "terrain_update": TerrainUpdate,
    }

    # Check any application-specific extensions registered by a caller.
    cls = _registry.get(topic)
    if cls is None:
        cls = _EXTENDED_REGISTRY.get(topic)
    if cls is None:
        return topic, payload
    return topic, cls.from_dict(payload)


# Extended message registry — populated by downstream modules
_EXTENDED_REGISTRY: Dict[str, type] = {}


# =============================================================================
# ZMQ Transport wrapper (thin layer over raw ZMQ)
# =============================================================================

class ZMQPublisher:
    """Publish messages on a ZMQ PUB socket."""

    def __init__(self, endpoint: str = "tcp://*:5555"):
        import time as _time
        import zmq as _zmq
        self._zmq = _zmq
        self._ctx = _zmq.Context.instance()
        self._sock = self._ctx.socket(_zmq.PUB)
        self._sock.setsockopt(_zmq.SNDHWM, 2)  # Drop old messages if consumer is slow
        self._sock.setsockopt(_zmq.LINGER, 0)   # Release port immediately on close
        # Bind retries with linear backoff. Orchestrated sweeps reuse port
        # blocks rapidly, and a freshly closed socket can hold its port in
        # TIME_WAIT for several seconds, so binding is retried for about eight
        # seconds. Without the retry a long sweep loses a small fraction of its
        # runs to bind failures on ports that are logically free.
        last_err = None
        for attempt in range(16):
            try:
                self._sock.bind(endpoint)
                last_err = None
                break
            except _zmq.error.ZMQError as e:
                last_err = e
                _time.sleep(0.5)
        if last_err is not None:
            raise last_err
        self.endpoint = endpoint

    def send(self, msg) -> None:
        """Send a dataclass message (must have .to_bytes())."""
        self._sock.send(msg.to_bytes(), self._zmq.NOBLOCK)

    def close(self):
        self._sock.close()


class ZMQSubscriber:
    """Subscribe to messages on a ZMQ SUB socket."""

    def __init__(self, endpoint: str = "tcp://localhost:5555",
                 topics: Optional[List[str]] = None):
        import zmq as _zmq
        self._zmq = _zmq
        self._ctx = _zmq.Context.instance()
        self._sock = self._ctx.socket(_zmq.SUB)
        self._sock.setsockopt(_zmq.RCVHWM, 2)
        self._sock.setsockopt(_zmq.CONFLATE, 1)  # Keep only latest message
        self._sock.connect(endpoint)
        # Subscribe to everything: the framing is application-defined rather
        # than ZMQ's native topic prefix, so topic filtering happens in
        # parse_message.
        self._sock.setsockopt(_zmq.SUBSCRIBE, b"")
        self.endpoint = endpoint

    def recv(self, timeout_ms: int = 0):
        """Receive a message. Returns (topic, msg) or None if no message available.

        Args:
            timeout_ms: 0 = non-blocking, -1 = block forever, >0 = wait up to N ms.
        """
        if timeout_ms == 0:
            flags = self._zmq.NOBLOCK
        else:
            self._sock.setsockopt(self._zmq.RCVTIMEO, timeout_ms)
            flags = 0
        try:
            raw = self._sock.recv(flags)
            return parse_message(raw)
        except Exception:
            return None

    def close(self):
        self._sock.close()


# =============================================================================
# Transport factory: selects the ROS 2 rclpy/DDS backend (the default) or the
# self-contained ZeroMQ backend. The ROS backend (ros_transport.py) is imported
# lazily, so the ZeroMQ path never requires ROS 2 to be sourced. Both carry
# identical framed message bytes. Setting HIL_TRANSPORT=zmq selects the ZeroMQ
# backend process-wide without a ROS environment.
# =============================================================================

def _ros_module(kind: str):
    """Import the ROS backend, or raise a clear, actionable error."""
    try:
        import ros_transport  # noqa
        return ros_transport
    except Exception as e:  # pragma: no cover - env-dependent
        raise RuntimeError(
            f"--transport ros selected but the ROS backend is unavailable "
            f"({type(e).__name__}: {e}). Source ROS 2 + the Chrono ROS workspace "
            f"(see SETUP.md §4), or run with the self-contained fallback: "
            f"--transport zmq  (or export HIL_TRANSPORT=zmq)."
        ) from e


# Namespaced ROS 2 topic names, one per message role. The ROS transport uses
# them; the ZeroMQ transport ignores them and keys off the endpoint port.
TOPIC_VEHICLE_STATE = "scm_hil/vehicle_state"    # sim -> controller/classifier/HUD
TOPIC_CONTROL_CMD = "scm_hil/control_cmd"        # controller -> sim (+ terrain piggyback)
TOPIC_TERRAIN_ESTIMATE = "scm_hil/terrain_estimate"  # classifier -> controller


def make_publisher(endpoint: str, transport: str = "ros", topic: Optional[str] = None):
    """Return a publisher (.send/.close) for the chosen transport. ``topic`` is
    the semantic ROS topic (ignored by zmq, which keys off the endpoint port)."""
    if (transport or "ros").lower() == "ros":
        return _ros_module("pub").ROSPublisher(endpoint, topic=topic)
    return ZMQPublisher(endpoint)


def make_subscriber(endpoint: str, transport: str = "ros",
                    topics: Optional[List[str]] = None, topic: Optional[str] = None):
    """Return a subscriber (.recv/.close) for the chosen transport. ``topic`` is
    the semantic ROS topic (ignored by zmq)."""
    if (transport or "ros").lower() == "ros":
        return _ros_module("sub").ROSSubscriber(endpoint, topics, topic=topic)
    return ZMQSubscriber(endpoint, topics)


# =============================================================================
# Default port assignments
# =============================================================================

# Sim publishes vehicle state on this port
SIM_PUB_PORT = 5555
# Controller publishes control commands on this port
CTRL_PUB_PORT = 5556

def sim_pub_endpoint(port: int = SIM_PUB_PORT) -> str:
    return f"tcp://*:{port}"

def sim_sub_endpoint(host: str = "localhost", port: int = SIM_PUB_PORT) -> str:
    return f"tcp://{host}:{port}"

def ctrl_pub_endpoint(port: int = CTRL_PUB_PORT) -> str:
    return f"tcp://*:{port}"

def ctrl_sub_endpoint(host: str = "localhost", port: int = CTRL_PUB_PORT) -> str:
    return f"tcp://{host}:{port}"
