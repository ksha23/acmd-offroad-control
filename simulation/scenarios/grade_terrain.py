"""A straight grade along +x, for the terrain-preview study.

The ground is flat (height 0) up to ``start_x`` and then rises or falls at a
constant grade (rise over run, negative downhill along +x). SCM builds the
patch from a 16-bit height image, so the slope is resolved to about 0.1 mm.
Rocks use ``height`` to sit on the slope.
"""
from __future__ import annotations

import numpy as np


class GradeProfile:
    def __init__(self, grade: float, start_x: float):
        self.grade = float(grade)
        self.start_x = float(start_x)

    def height(self, x, y=None):
        """Ground height at x (y is ignored, the grade runs along x)."""
        return self.grade * np.maximum(np.asarray(x, dtype=float) - self.start_x, 0.0)

    def write_png(self, path: str, length: float, width: float, px_per_m: float = 4.0):
        """Write the patch as a 16-bit image and return (h_min, h_max) for SCMTerrain.Initialize.

        SCM maps the image's columns to x across [-length/2, length/2], left to
        right, and black to h_min, white to h_max.
        """
        from PIL import Image

        nx = int(round(length * px_per_m)) + 1
        ny = max(int(round(width * 0.5)), 2)
        x = np.linspace(-0.5 * length, 0.5 * length, nx)
        h = self.height(x)
        h_min, h_max = float(h.min()), float(h.max())
        span = max(h_max - h_min, 1e-9)
        row = np.round((h - h_min) / span * 65535.0).astype(np.uint16)
        img = np.tile(row, (ny, 1))
        Image.fromarray(img, mode="I;16").save(path)
        return h_min, h_max
