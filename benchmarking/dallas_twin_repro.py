#!/usr/bin/env python3
"""Reproduce the estimator of Dallas et al. in its own twin-experiment regime.

The accuracies published by Dallas et al. (arXiv:1908.00130, a Bekker-model
filter at 3.8% on clay; arXiv:2003.02635, a neural filter at 1.9%) are
twin-experiment results: the plant generating the measurements and the model
inside the filter belong to the same terramechanics family, and their network is
a surrogate of that same family. Reproducing those numbers therefore requires
reconstructing that regime exactly, which is what this script does:

* plant: the same bicycle and force model the UKF propagates
  (``_bicycle_step_2n`` at the true exponent), driven through their maneuver
  envelope (3 s steering period, speed within their Fig. 7b band, 40 s);
* Bekker arm: the filter's force model is the plant's force model, their
  arXiv:1908.00130 configuration;
* neural arm: the filter's force model is a 3x35 tanh network trained on the
  Bekker model's own outputs, their arXiv:2003.02635 configuration, in which the
  network is a surrogate of the plant by construction;
* their Table V measurement noise, their Table VII initialisation offset of
  +0.2, their filter constants, and three noise seeds per soil.

The reproduction is what makes ``dallas_protocol_cross_eval.py`` interpretable.
That script runs the same filter against the Chrono SCM plant, where the filter
model and the plant are different terramechanics implementations. If the twin
runs here land near the published table while the Chrono runs do not, the
difference between the two sets of results is the evaluation regime rather than
the estimator, and the comparison can be reported as such.

Usage:
  python benchmarking/dallas_twin_repro.py [--out DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT, ROOT / "simulation", ROOT / "simulation" / "estimators"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

spec = importlib.util.spec_from_file_location(
    "dallas_cross", ROOT / "benchmarking" / "dallas_protocol_cross_eval.py")
dallas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dallas)

SEEDS = [11, 12, 13]
SOILS = {"clay": 0.5, "dirt": 0.7}   # preset name -> true exponent
PUBLISHED = {  # their tables, % error on clay
    "bekker": 3.8,
    "nn": 1.9,
}


def simulate_twin(soil_preset: str, n_true: float) -> pd.DataFrame:
    """Forward-simulate the filter's own plant model at the true exponent."""
    p = dallas.TERRAIN_PRESETS[soil_preset]
    soil_true = dallas.SoilParams(
        kc=float(p["Kc"]), kphi=float(p["Kphi"]), n=n_true,
        c=float(p["cohesion"]), phi=math.radians(float(p["friction_angle"])),
        kx=float(p["janosi_shear"]), ky=float(p["janosi_shear"]),
    )
    veh = dallas.Vehicle()
    dt = dallas.EST_DT
    t_end = dallas.LEAD_IN_S + dallas.RUN_S
    z = np.array([0.0, 0.0, 0.0, 5.5, 0.0, 0.0, n_true, n_true])
    rows = []
    t = 0.0
    while t <= t_end:
        # their maneuver envelope: sinusoidal steer after the lead-in,
        # mild longitudinal modulation keeping u inside their Fig. 7b band
        delta = (0.25 * math.sin(2.0 * math.pi * (t - dallas.LEAD_IN_S)
                                 / dallas.STEER_PERIOD_S)
                 if t >= dallas.LEAD_IN_S else 0.0)
        ax = 0.25 * math.sin(2.0 * math.pi * t / 7.0)
        rows.append(dict(sim_time=t, x_cg=z[0], y_cg=z[1], psi=z[2],
                         u_raw=z[3], v_lateral_raw=z[4], omega_raw=z[5],
                         steering_angle=delta, ax_imu=ax))
        z = dallas._bicycle_step_2n(z, delta, ax, dt, soil_true, veh)
        t += dt
    return pd.DataFrame(rows)


def train_bekker_surrogate(soil_preset: str, seed: int = 0):
    """A 3x35 tanh surrogate of the Bekker axle force -- their neural setup."""
    import torch
    import torch.nn as nn

    p = dallas.TERRAIN_PRESETS[soil_preset]
    rng = np.random.default_rng(seed)
    N = 24000
    Fz = rng.uniform(6.0e3, 3.0e4, N)
    vl = rng.uniform(0.3, 9.0, N)
    vc = rng.uniform(-3.5, 3.5, N)
    nn_ = rng.uniform(0.2, 1.4, N)
    y = np.empty(N)
    for i in range(N):
        soil = dallas.SoilParams(
            kc=float(p["Kc"]), kphi=float(p["Kphi"]), n=float(nn_[i]),
            c=float(p["cohesion"]),
            phi=math.radians(float(p["friction_angle"])),
            kx=float(p["janosi_shear"]), ky=float(p["janosi_shear"]),
        )
        _, Fy, _, _ = dallas._bekker_wheel_forces(
            Fz_required=Fz[i], vl=vl[i], vc=vc[i],
            omega=max(vl[i], 0.5) / dallas.HMMWV_WHEEL.r,
            soil=soil, wheel=dallas.HMMWV_WHEEL)
        y[i] = Fy
    X = np.column_stack([Fz, vl, vc, nn_]).astype(np.float32)
    y = y.astype(np.float32)[:, None]
    xm, xs = X.mean(0), X.std(0)
    ym, ys = y.mean(), y.std()
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(4, 35), nn.Tanh(), nn.Linear(35, 35),
                        nn.Tanh(), nn.Linear(35, 35), nn.Tanh(),
                        nn.Linear(35, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    Xt = torch.tensor((X - xm) / xs)
    yt = torch.tensor((y - ym) / ys)
    for epoch in range(400):
        idx = torch.randperm(len(Xt))
        for b in idx.split(512):
            opt.zero_grad()
            loss = ((net(Xt[b]) - yt[b]) ** 2).mean()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = (net(Xt).numpy() * ys + ym)
    ss = 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  surrogate[{soil_preset}] R^2 vs Bekker: {ss:.5f}")

    def axle_force(Fz_axle, vl_, vc_, soil):
        with torch.no_grad():
            xin = torch.tensor(
                ((np.array([[Fz_axle, max(vl_, 0.3), vc_, soil.n]],
                           dtype=np.float32)) - xm) / xs)
            return float(net(xin).item() * ys + ym)
    return axle_force


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "benchmarking" / "results"
                                         / "dallas_twin_repro"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for soil, n_true in SOILS.items():
        obs = simulate_twin(soil, n_true)
        obs_path = out / f"twin_obs_{soil}.csv"
        obs.to_csv(obs_path, index=False)

        for arm in ("bekker", "nn"):
            if arm == "nn":
                surrogate = train_bekker_surrogate(soil)
                original = dallas._axle_forces
                dallas._axle_forces = lambda Fz, vl, vc, s: surrogate(Fz, vl, vc, s)
            try:
                for seed in SEEDS:
                    tr = dallas.run_dallas_ukf(obs_path, soil, n_true, seed)
                    final = float(tr.n_mean.iloc[-1])
                    err = 100.0 * abs(final - n_true) / n_true
                    results.append(dict(soil=soil, arm=arm, seed=seed,
                                        n_true=n_true, n_final=final,
                                        pct_err=err))
                    print(f"  {soil:<6} {arm:<7} seed {seed}: "
                          f"final {final:.4f}  ({err:.1f}% err)")
            finally:
                if arm == "nn":
                    dallas._axle_forces = original

    frame = pd.DataFrame(results)
    frame.to_csv(out / "twin_results.csv", index=False)
    print("\n=== twin regime (plant == filter model), mean over seeds ===")
    summary = frame.groupby(["soil", "arm"]).agg(
        n_final=("n_final", "mean"), pct_err=("pct_err", "mean")).round(4)
    print(summary.to_string())
    print(f"\npublished clay accuracies: bekker {PUBLISHED['bekker']}%, "
          f"neural {PUBLISHED['nn']}%")
    print("the same filter on the Chrono SCM plant: bekker 13-26%, "
          "neural (rig-supervised network) not tracking")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
