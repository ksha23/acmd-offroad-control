#!/usr/bin/env python3
"""Calibrate one global Pacejka and TMeasy parameter set to the SCM plant.

This calibration establishes the analytical baselines of Table 1. It fits a
single parameter set to force data pooled over the whole soil range measured on
the controlled single-tire Chrono SCM rig, in place of the rigid-road ``.tir``
defaults (mu = 0.74, B = 8.77) that a deformable-soil plant would render
meaningless. The resulting peak friction is adopted in
``simulation/tire_models/analytical_tire_models.py`` and is the value the
manuscript quotes when it states that the baselines are calibrated to the
plant's mean lateral friction.

One global set is fit rather than one per soil, because the analytical arms have
no terrain knowledge in closed loop; the truth-calibrated and rig-fit Pacejka
arms of Table 1 supply the per-soil comparison separately. Measured SCM peak
lateral friction has a median near 0.32 and spans roughly 0.18 on clay-like
soils to 0.47 on sand-like soils, so the calibrated value lies far below a rigid
road and the analytical model no longer over-corners.

Calibrating in this way is what makes the comparison in Table 1 informative: it
attributes the neural surrogate's remaining advantage to the terrain-to-terrain
force variation a single analytical set cannot express, rather than to a
mismatched friction level that any recalibration would remove.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# The static-parent rig corpus: the same file whose SHA-256 is anchored as
# PRIMARY by verify_provenance_chain.py. An earlier revision pointed at
# data/tire_rig/train.csv, a rename casualty that made this script fail
# outright while the adopted mu = 0.42 lived on as an unverifiable
# transcription (2026-08-26 audit, finding F3).
DATA = (Path(__file__).resolve().parents[1]
        / "data" / "tire_rig_static" / "train.csv")


def main():
    df = pd.read_csv(DATA)
    import hashlib
    print(f"corpus: {DATA}")
    print(f"corpus sha256: {hashlib.sha256(DATA.read_bytes()).hexdigest()}")
    m = df[df.slip_ratio.abs() < 0.05]               # pure-cornering subset
    a = m.slip_angle.abs().to_numpy()
    rcoef = (m.Fy.abs() / m.Fz).to_numpy()           # |Fy/Fz| magnitude
    Fy = m.Fy.abs().to_numpy()

    # Pacejka magnitude: |Fy/Fz| = mu*sin(C*atan(B*a - E*(B*a - atan(B*a))))
    def pac(p, x):
        B, C, E, mu = p; Bx = B * x
        return mu * np.sin(C * np.arctan(Bx - E * (Bx - np.arctan(Bx))))
    rp = least_squares(lambda p: pac(p, a) - rcoef, [6, 1.5, 0.4, 0.35],
                       bounds=([1, 1.0, 0.0, 0.1], [20, 2.0, 1.0, 1.2]))
    B, C, E, mu = rp.x

    # TMeasy magnitude (balanced load, per-tire = 0.5*Fym*sin(min(pi/2*a/am, pi/2)))
    def tm(p, x):
        Fym, am = p; hp = np.pi / 2
        return 0.5 * Fym * np.sin(np.minimum(hp * x / max(am, 1e-4), hp))
    rt = least_squares(lambda p: tm(p, a) - Fy, [6000, 0.15], bounds=([1000, 0.03], [20000, 0.40]))
    Fym, am = rt.x

    # SCM peak lateral friction by soil firmness (saturated, |alpha|>0.25)
    big = m[m.slip_angle.abs() > 0.25]
    print(f"rows fit: {len(m)} (pure-cornering subset of {len(df)})")
    print(f"\nSCM peak |Fy|/Fz (saturated):  overall median={np.median((big.Fy.abs()/big.Fz)):.2f}")
    for lo, hi, lbl in [(0, 15, 'clay-like'), (15, 25, 'dirt-like'), (25, 40, 'sand-like')]:
        s = big[(big.mohr_friction >= np.radians(lo)) & (big.mohr_friction < np.radians(hi))]
        if len(s):
            print(f"  phi[{lo:2d},{hi:2d})deg ({lbl:9s}) peak|Fy/Fz|={np.median((s.Fy.abs()/s.Fz)):.2f}")
    print(f"\nleast-squares Pacejka fit:  B={B:.2f} C={C:.3f} E={E:.3f}  mu={mu:.3f}")
    print(f"least-squares TMeasy fit:   Fym={Fym:.0f} N alpha_m={am:.3f} rad ({np.degrees(am):.1f} deg)")
    print(f"\nADOPTED in simulation/tire_models/analytical_tire_models.py: PACEJKA_MU = {mu:.2f}")
    print("  Single global peak friction. The magic-formula and TMeasy shape factors")
    print("  are held at their published values: refitting the full shape collapses")
    print("  cornering stiffness and degrades closed-loop tracking, which would")
    print("  understate the baselines rather than calibrate them.")


if __name__ == "__main__":
    main()
