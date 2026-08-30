"""ROS 2 (rclpy) transport backend, mirroring the ZMQPublisher/ZMQSubscriber
interface in ``hil_messages.py``.

This is the paper transport for the decoupled hardware-in-the-loop stack. It
carries the same framed message bytes (``msg.to_bytes()`` / ``parse_message``)
as the ZeroMQ backend, so every layer above it is transport-agnostic and the
only difference is DDS in place of ZeroMQ publish/subscribe.

Three choices align the two backends' semantics:

  * One ROS topic per port (``/hil/port_<n>``), matching the scheme in which a
    single socket multiplexes several message types -- port 5555 carries both
    ``vehicle_state`` and ``sim_status``, for instance. The per-message framing
    in ``to_bytes()`` distinguishes them, and ``parse_message`` recovers the
    type.
  * QoS ``KEEP_LAST`` at depth 1 with ``BEST_EFFORT`` reproduces ZeroMQ's
    conflating, drop-stale subscriber. The control loop depends on this: a
    consumer must act on the newest command, never on a queued one.
  * Parallel sweep workers are isolated by ``ROS_DOMAIN_ID``, set per worker,
    which plays the role of the per-worker port block on the ZeroMQ side.

The payload is wrapped in ``std_msgs/UInt8MultiArray`` as raw bytes, so no
custom .msg definition or rosidl build step is needed and the framing plus
msgpack encoding in ``hil_messages`` remains the single source of truth for the
message schema.
"""
from __future__ import annotations

import threading
from typing import List, Optional

# rclpy and std_msgs are imported only when this backend is selected, so the
# ZeroMQ path never requires ROS 2 to be sourced.
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from std_msgs.msg import UInt8MultiArray

from hil_messages import parse_message

# Latest-only QoS: the DDS analogue of ZeroMQ CONFLATE=1 (depth-1 keep-last).
_LATEST_ONLY = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
)

# One rclpy context, node, and executor per process, shared by every publisher
# and subscriber that process creates, in the manner of zmq.Context.instance().
# No background spin thread is started: rclpy's executor teardown races a daemon
# spin thread and aborts at process exit, which would fail an otherwise valid
# sweep run. Subscribers instead spin the executor inside recv(), which also
# matches the polling model on the ZeroMQ side, where consumers poll once per
# loop iteration.
_LOCK = threading.Lock()
_NODE: Optional[Node] = None
_EXECUTOR: Optional[SingleThreadedExecutor] = None
_REFCOUNT = 0


def _port_topic(endpoint: str) -> str:
    """Map a ZMQ-style endpoint (``tcp://host:PORT``) to a ROS topic name."""
    port = endpoint.rsplit(":", 1)[-1].strip("/")
    return f"/hil/port_{port}"


def _ensure_node() -> Node:
    """Lazily start the shared process node + executor (no spin thread)."""
    global _NODE, _EXECUTOR, _REFCOUNT
    with _LOCK:
        if _NODE is None:
            if not rclpy.ok():
                rclpy.init(args=None)
            _NODE = Node("hil_transport")
            _EXECUTOR = SingleThreadedExecutor()
            _EXECUTOR.add_node(_NODE)
        _REFCOUNT += 1
        return _NODE


def _spin(timeout_sec: float) -> None:
    """Process pending callbacks on the shared executor (called from recv)."""
    if _EXECUTOR is not None:
        _EXECUTOR.spin_once(timeout_sec=timeout_sec)


def _release_node() -> None:
    global _NODE, _EXECUTOR, _REFCOUNT
    with _LOCK:
        _REFCOUNT -= 1
        if _REFCOUNT <= 0 and _NODE is not None:
            try:
                _NODE.destroy_node()
            except Exception:
                pass
            _NODE = None
            _EXECUTOR = None
            _REFCOUNT = 0
            if rclpy.ok():
                rclpy.shutdown()


class ROSPublisher:
    """Publish framed HIL messages on a ROS 2 topic (ZMQPublisher analogue)."""

    def __init__(self, endpoint: str = "tcp://*:5555", topic: Optional[str] = None):
        self._node = _ensure_node()
        # An explicit semantic topic (/scm_hil/vehicle_state, for instance) is
        # preferred, with a port-derived name as the fallback. Parallel runs are
        # separated by ROS_DOMAIN_ID, so fixed semantic topics cannot collide
        # across concurrent runs.
        self.topic = topic or _port_topic(endpoint)
        self.endpoint = endpoint
        self._pub = self._node.create_publisher(UInt8MultiArray, self.topic, _LATEST_ONLY)
        self._closed = False

    def send(self, msg) -> None:
        """Send a dataclass message (must have ``.to_bytes()``)."""
        m = UInt8MultiArray()
        m.data = list(msg.to_bytes())
        self._pub.publish(m)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._node.destroy_publisher(self._pub)
        except Exception:
            pass
        _release_node()


class ROSSubscriber:
    """Subscribe to framed HIL messages on a ROS 2 topic (ZMQSubscriber analogue).

    Retains only the most recent frame, matching the conflating ZeroMQ
    subscriber. ``recv`` returns the newest unconsumed frame.
    """

    def __init__(self, endpoint: str = "tcp://localhost:5555",
                 topics: Optional[List[str]] = None, topic: Optional[str] = None):
        self._node = _ensure_node()
        self.topic = topic or _port_topic(endpoint)
        self.endpoint = endpoint
        self._latest: Optional[bytes] = None
        self._lock = threading.Lock()
        self._closed = False
        self._sub = self._node.create_subscription(
            UInt8MultiArray, self.topic, self._on_msg, _LATEST_ONLY)

    def _on_msg(self, m: UInt8MultiArray) -> None:
        with self._lock:
            self._latest = bytes(bytearray(m.data))  # conflate to the newest

    def recv(self, timeout_ms: int = 0):
        """Return (topic, msg) for the latest frame, or None.

        Spins the shared executor so that pending DDS callbacks run; each
        overwrites ``_latest``, conflating to the newest frame, which is then
        handed off. ``timeout_ms == 0`` is a non-blocking poll and a positive
        value waits up to that long, matching ZMQSubscriber.recv.
        """
        # KEEP_LAST at depth 1 means the middleware holds only the newest
        # sample, so a single spin_once processes it. A non-blocking poll uses
        # 0.0; a positive timeout blocks that long inside the executor.
        _spin(0.0 if timeout_ms <= 0 else timeout_ms / 1000.0)
        with self._lock:
            raw, self._latest = self._latest, None
        if raw is None:
            return None
        try:
            return parse_message(raw)
        except Exception:
            return None

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._node.destroy_subscription(self._sub)
        except Exception:
            pass
        _release_node()
