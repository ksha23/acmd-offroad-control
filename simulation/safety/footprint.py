"""The vehicle's footprint as its body rectangle, and distances to it (``--safety-derived``).

The rectangle is the collision mesh's outline (``vehicle_spec``), in the body frame of the chassis centre of
mass: x in [x_rear, x_front] along the heading, y in [-half_width, half_width]. Distances are Euclidean and zero
where the shapes meet. The distance from a point to a convex set is continuously differentiable outside it, so
the optimizer can use it too.
"""
from __future__ import annotations

import numpy as np

from . import vehicle_spec as spec


def body_box():
    """(x_rear, x_front, half_width) of the body rectangle about the centre of mass (m)."""
    return (spec.BODY_X_REF[0] - spec.COM_X_REF, spec.BODY_X_REF[1] - spec.COM_X_REF, spec.BODY_HALF_WIDTH)


def corner_reach():
    """Largest distance from the centre of mass to a corner of the rectangle (m)."""
    x_r, x_f, w = body_box()
    return float(max(np.hypot(x_r, w), np.hypot(x_f, w)))


def to_body(px, py, X, Y, psi):
    """World points into the body frame of pose (X, Y, psi). Arrays broadcast."""
    c, s = np.cos(psi), np.sin(psi)
    dx, dy = px - X, py - Y
    return dx * c + dy * s, -dx * s + dy * c


def corners(X, Y, psi, box=None):
    """World corners (..., 4, 2) of the rectangle at poses (X, Y, psi): rear-left, front-left, front-right,
    rear-right."""
    x_r, x_f, w = box if box is not None else body_box()
    X, Y, psi = (np.asarray(v, dtype=float) for v in (X, Y, psi))
    c, s = np.cos(psi)[..., None], np.sin(psi)[..., None]
    bx = np.array([x_r, x_f, x_f, x_r])
    by = np.array([w, w, -w, -w])
    return np.stack([X[..., None] + bx * c - by * s, Y[..., None] + bx * s + by * c], axis=-1)


def point_box(qx, qy, x_r, x_f, w):
    """Distance from body-frame points to the rectangle [x_r, x_f] x [-w, w] (0 inside)."""
    dx = np.maximum(np.maximum(x_r - qx, qx - x_f), 0.0)
    dy = np.maximum(np.abs(qy) - w, 0.0)
    return np.hypot(dx, dy)


def _point_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy + 1e-12), 0.0, 1.0)
    return np.hypot(px - ax - t * dx, py - ay - t * dy)


def _cross(ax, ay, bx, by, cx, cy, dx, dy):
    """True where segments ab and cd cross (proper crossing)."""
    def orient(ox, oy, px, py, qx, qy):
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)
    d1 = orient(cx, cy, dx, dy, ax, ay)
    d2 = orient(cx, cy, dx, dy, bx, by)
    d3 = orient(ax, ay, bx, by, cx, cy)
    d4 = orient(ax, ay, bx, by, dx, dy)
    return (d1 * d2 < 0.0) & (d3 * d4 < 0.0)


def segment_box(ax, ay, bx, by, x_r, x_f, w):
    """Distance from body-frame segments AB to the rectangle (0 where they meet). Arrays broadcast."""
    ax, ay, bx, by = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (ax, ay, bx, by)))
    d = np.minimum(point_box(ax, ay, x_r, x_f, w), point_box(bx, by, x_r, x_f, w))
    cx = np.array([x_r, x_f, x_f, x_r])
    cy = np.array([w, w, -w, -w])
    meet = np.zeros(d.shape, dtype=bool)
    for i in range(4):
        j = (i + 1) % 4
        d = np.minimum(d, _point_segment(cx[i], cy[i], ax, ay, bx, by))
        meet |= _cross(ax, ay, bx, by, cx[i], cy[i], cx[j], cy[j])
    return np.where(meet, 0.0, d)


def box_sym(qx, qy, x_r, x_f, w):
    """Squared distance from a body-frame point to the rectangle, for CasADi (continuously differentiable)."""
    import casadi as ca
    dx = ca.fmax(ca.fmax(x_r - qx, qx - x_f), 0.0)
    dy = ca.fmax(ca.fabs(qy) - w, 0.0)
    return dx ** 2 + dy ** 2
