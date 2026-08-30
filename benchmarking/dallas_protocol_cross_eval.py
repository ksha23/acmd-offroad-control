#!/usr/bin/env python3
"""Recreate the evaluation protocol of Dallas et al. and run both estimators on it.

Comparing two estimators on one protocol favours whichever was designed for it.
This script therefore supplies the symmetric comparison: it reconstructs the
published evaluation of Dallas et al. and runs GRIT on it alongside a faithful
implementation of their filter, over identical traces, so that their method is
assessed on its own terms rather than on ours.

Faithful to Dallas et al., "Online terrain estimation for autonomous vehicles on
deformable terrains" (arXiv:1908.00130v2; J. Terramechanics 91:11-22, 2020):

* Plant: Chrono with SCM, level terrain, simple powertrain (their Sec. III-A
  describes exactly this configuration for their notional vehicle).
* Soils (their Table VI): sandy loam and clay. These are numerically identical
  to this repository's ``dirt`` and ``clay`` presets, both drawn from the same
  published table, so the plant soils ARE their soils.
* Maneuver (their Sec. V, Fig. 7): sinusoidal steering "fully to the left and
  right over a three second period", throttle varied sinusoidally at the same
  frequency so speed oscillates in roughly the 4.8--6.3 m/s band, no braking.
* Estimation target (their Sec. IV): sinkage exponent as a 2x1 vector
  ``[n_f, n_r]`` appended to a 3-DoF bicycle with trivial dynamics
  (``n_dot = 0``); every other terrain parameter fixed at its TRUE value; n
  initialised 0.2 above truth (their Table VII: 0.9->0.7 sandy loam, 0.7->0.5
  clay).
* Measurements (their Table V): the full state ``[x, y, psi, u, v, omega]``
  from the plant, corrupted with zero-mean Gaussian noise of sigma
  ``[1.2 m, 1.2 m, 0.0175 rad, 0.25 m/s, 0.25 m/s, 0.0175 rad/s]``, delivered
  every 24 ms; the filter integrates at 12 ms. Longitudinal channel of the
  bicycle is driven by measured a_x (their Eq. 24), so terrain sensitivity
  enters only through the Bekker lateral forces -- their stated design.
* Metrics (their Table VII / Fig. 8): the converged estimate is the final value
  at the end of the run; percent error against truth; and their convergence
  remark is checked as time to enter and stay within 10% of the final value.

Declared deviations, each recorded in the manifest's ledger: the vehicle is this
repository's HMMWV rather than their unspecified notional military vehicle; full
steering lock here is 0.528 rad where their Fig. 7a shows about +-0.65 rad;
their UKF process noise is not printed in the paper, so the values are taken
from a reproduction of their Fig. 8; the plant step is the deployed 3 ms rather
than their 2 ms; and each soil is run at three simulator seeds rather than their
single run, with every seed reported.

Both estimators consume the SAME run: the controller node runs passively under
``--replay-cmds`` (its commands do not actuate) and logs unrounded
``terrain_observations.csv``; the UKF reads the raw state columns plus Table V
noise, and GRIT replays the sanitized ``sensor_trace.csv`` capsule through its
frozen contract via ``develop_joint_estimator.py``. GRIT additionally
estimates phi and is denied the other-parameters-at-truth privilege the UKF
enjoys (it prescribes them from its manifold), which on these two anchor soils
coincides with truth; this asymmetry favours the UKF and is part of the ledger.

Run one at a time (sequential by construction). Phases:

  tune     find per-soil throttle profiles landing in the 4.8-6.3 m/s band
  collect  drive the profiles in Chrono, write traces + manifest
  ukf      faithful Dallas UKF over the shared traces
  grit     GRIT frozen replay over the shared traces
  score    their metrics + ours, one summary table
  all      everything above in order
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarking"))
sys.path.insert(0, str(ROOT / "benchmarking" / "lib"))
sys.path.insert(0, str(ROOT / "simulation" / "shared"))

from common import timestamped_result_dir, write_manifest  # noqa: E402
from terrain_estimator_trace import sanitize_exact_observations  # noqa: E402
from param_consistency import TERRAIN_PRESETS  # noqa: E402
from ukf_reference_models import (  # noqa: E402
    SoilParams,
    Vehicle,
    WheelGeom,
    _bekker_wheel_forces,
    _load_nn_tire,
    _nn_wheel_forces,
)

# Dallas et al. published this filter twice: over a reduced-order analytical
# Bekker surrogate (arXiv:1908.00130) and over a neural terramechanics model
# trained on Latin-hypercube single-tire SCM samples (arXiv:2003.02635), the
# latter reaching 1.9% on clay against the former's 3.8%. Both are reproduced
# here so the comparison is against the better of their two estimators.
UKF_FORCE_MODEL = "bekker"
NN_TIRE_MODEL_DIR = str(ROOT / "nn_models" / "tire_force_static_parent")


def _axle_forces(Fz_axle, vl, vc, soil):
    """Lateral force for one axle under the selected force model."""
    if UKF_FORCE_MODEL == "nn":
        model = _load_nn_tire(NN_TIRE_MODEL_DIR, {
            "Kphi": soil.kphi, "Kc": soil.kc, "n": soil.n,
            "c": soil.c, "phi": math.degrees(soil.phi), "k": soil.kx})
        _, Fy = _nn_wheel_forces(model, Fz_axle=Fz_axle, vl=vl, vc=vc, soil=soil)
        return Fy
    _, Fy, _, _ = _bekker_wheel_forces(
        Fz_required=Fz_axle, vl=vl, vc=vc,
        omega=max(vl, 0.5) / HMMWV_WHEEL.r, soil=soil, wheel=HMMWV_WHEEL)
    return Fy

# The UKF's internal Bekker wheel must match the plant vehicle, as Dallas's
# internal model matched his: Chrono HMMWV rolling radius, RIGID-tire width.
HMMWV_WHEEL = WheelGeom(r=0.47, b=0.254)

LAUNCHER = ROOT / "simulation" / "runtime" / "launch_decoupled.py"
STEER_PERIOD_S = 3.0          # their Fig. 7a
LEAD_IN_S = 3.0               # straight lead-in; their t=0 is first turn-in
RUN_S = 40.0                  # their runs: 32.89-47 s
SPEED_BAND = (4.8, 6.3)       # their Fig. 7b envelope
CMD_DT = 0.01
MAX_STEER_RAD = 0.528         # HMMWV full lock; ledger notes their ~0.65

# their Table VI == our presets; assert rather than trust
_TABLE_VI = {
    "dirt": dict(Kc=5300.0, Kphi=1515000.0, n=0.7, k=0.025, c=1700.0, phi_deg=29.0),
    "clay": dict(Kc=13200.0, Kphi=692200.0, n=0.5, k=0.01, c=4140.0, phi_deg=13.0),
}
SOILS = {"sandy_loam": "dirt", "clay": "clay"}
INIT_OFFSET = 0.2             # their Table VII initial guesses
TABLE_V_SIGMA = np.array([1.2, 1.2, 0.0175, 0.25, 0.25, 0.0175])
MEAS_DT = 0.024
EST_DT = 0.012


def _assert_presets_match() -> None:
    for preset, want in _TABLE_VI.items():
        have = TERRAIN_PRESETS[preset]
        checks = [
            (have["Kc"], want["Kc"]), (have["Kphi"], want["Kphi"]),
            (have["n"], want["n"]), (have["janosi_shear"], want["k"]),
            (have["cohesion"], want["c"]),
            (have["friction_angle"], want["phi_deg"]),
        ]
        for got, expect in checks:
            if abs(float(got) - float(expect)) > 1e-9:
                raise SystemExit(
                    f"preset {preset} deviates from Dallas Table VI: "
                    f"{got} != {expect}"
                )


def write_profile(path: Path, throttle_mean: float, throttle_amp: float,
                  duration: float) -> None:
    """Command CSV: straight lead-in, then their sinusoids at 1/3 Hz."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "steering", "throttle", "braking"])
        t = 0.0
        while t <= duration + 1e-9:
            if t < LEAD_IN_S:
                steer = 0.0
            else:
                steer = math.sin(2.0 * math.pi * (t - LEAD_IN_S) / STEER_PERIOD_S)
            thr = throttle_mean + throttle_amp * math.sin(
                2.0 * math.pi * max(0.0, t - LEAD_IN_S) / STEER_PERIOD_S
            )
            w.writerow([f"{t:.3f}", f"{steer:.6f}",
                        f"{min(1.0, max(0.0, thr)):.6f}", "0.0"])
            t += CMD_DT


def launch_run(run_dir: Path, terrain: str, profile: Path, seed: int,
               sim_port: int, duration: float, timeout: float = 400.0) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(LAUNCHER),
        "--transport", "ros",
        "--terrain", terrain, "--path", "straight", "--speed", "5",
        "--replay-cmds", str(profile),
        "--time", str(duration), "--sim-seed", str(seed),
        "--sim-port", str(sim_port), "--ctrl-port", str(sim_port + 1),
        "--plot-dir", str(run_dir), "--no-vis", "--no-plot",
        # their plant: simple powertrain, level SCM; controller passive but
        # must run to log unrounded observations with a fixed prior
        "--simple-powertrain", "--bumpiness", "0",
        "--controller-prior-terrain", "dirt",
    ]
    import os
    env = dict(os.environ)
    env["HIL_RUN_LOG_DIR"] = str(run_dir)
    log = run_dir / "launch.log"
    with log.open("w") as fh:
        completed = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=fh,
                                   timeout=timeout, env=env)
    if completed.returncode:
        raise SystemExit(f"launch failed rc={completed.returncode}: see {log}")


def _observations(run_dir: Path) -> Path:
    hits = sorted(run_dir.rglob("terrain_observations.csv"))
    if not hits:
        raise SystemExit(f"no terrain_observations.csv under {run_dir}")
    return hits[-1]


def mean_speed_of(run_dir: Path) -> float:
    obs = pd.read_csv(_observations(run_dir))
    tail = obs[obs.sim_time >= LEAD_IN_S]
    return float(pd.to_numeric(tail.u_raw, errors="coerce").mean())


def tune(out: Path, base_port: int) -> dict:
    """Bisect the mean throttle per soil into their Fig. 7b speed band."""
    profiles = {}
    for label, terrain in SOILS.items():
        lo, hi = 0.15, 0.85
        mean = 0.45
        amp = 0.12
        chosen = None
        for attempt in range(4):
            profile = out / f"tune_{label}_{attempt}.csv"
            write_profile(profile, mean, amp, LEAD_IN_S + 12.0)
            rd = out / f"tune_{label}_{attempt}"
            launch_run(rd, terrain, profile, seed=900, sim_port=base_port,
                       duration=LEAD_IN_S + 12.0)
            u = mean_speed_of(rd)
            print(f"  tune {label} attempt {attempt}: throttle_mean={mean:.3f} "
                  f"-> mean u={u:.2f} m/s", flush=True)
            if SPEED_BAND[0] + 0.2 <= u <= SPEED_BAND[1] - 0.2:
                chosen = dict(throttle_mean=mean, throttle_amp=amp, mean_u=u)
                break
            if u < 0.5 * (SPEED_BAND[0] + SPEED_BAND[1]):
                lo = mean
            else:
                hi = mean
            mean = 0.5 * (lo + hi)
        if chosen is None:
            chosen = dict(throttle_mean=mean, throttle_amp=amp,
                          mean_u=u, note="band not reached; nearest attempt")
        profiles[label] = chosen
    (out / "throttle_profiles.json").write_text(
        json.dumps(profiles, indent=2) + "\n")
    return profiles


def collect(out: Path, base_port: int, seeds: list[int]) -> None:
    profiles = json.loads((out / "throttle_profiles.json").read_text())
    rows = []
    port = base_port + 40
    for label, terrain in SOILS.items():
        prof = profiles[label]
        profile = out / f"profile_{label}.csv"
        write_profile(profile, prof["throttle_mean"], prof["throttle_amp"],
                      LEAD_IN_S + RUN_S)
        for seed in seeds:
            trace_id = f"{label}_s{seed}"
            rd = out / "raw" / trace_id
            launch_run(rd, terrain, profile, seed=seed, sim_port=port,
                       duration=LEAD_IN_S + RUN_S)
            port += 4
            obs = _observations(rd)
            trace = rd / "sensor_trace.csv"
            sanitize_exact_observations(obs, trace)
            digest = hashlib.sha256(trace.read_bytes()).hexdigest()
            n_rows = sum(1 for _ in trace.open()) - 1
            rows.append(dict(
                trace_id=trace_id, status="ok",
                run_dir=str(rd.relative_to(out)),
                trace_path=str(trace.relative_to(out)),
                trace_sha256=digest, trace_rows=n_rows,
                trace_schema_version=3,
                trace_quality="exact_runtime_observations",
                source_path=str(obs.relative_to(out)),
                failure="", controller_prior="dirt", controller_prior_n=0.7,
                terrain_estimator_enabled=False, sim_seed=seed,
                path="dallas_sine_full_lock", speed_mps=prof["mean_u"],
                sim_time_s=LEAD_IN_S + RUN_S, lead_in_m=0.0,
                maneuver_label="dallas_protocol",
            ))
            print(f"  collected {trace_id}: mean u={mean_speed_of(rd):.2f} "
                  f"m/s rows={n_rows}", flush=True)
    pd.DataFrame(rows).to_csv(out / "trace_manifest.csv", index=False)


# ---------------------------------------------------------------------------
# Faithful Dallas UKF: 8-state [x, y, psi, u, v, omega, n_f, n_r]
# ---------------------------------------------------------------------------

def _bicycle_step_2n(z: np.ndarray, delta: float, ax_meas: float, dt: float,
                     soil_true: SoilParams, veh: Vehicle) -> np.ndarray:
    x, y, psi, u, v, om, n_f, n_r = z
    n_f = float(np.clip(n_f, 0.2, 1.4))
    n_r = float(np.clip(n_r, 0.2, 1.4))
    soil_f = SoilParams(kc=soil_true.kc, kphi=soil_true.kphi, n=n_f,
                        c=soil_true.c, phi=soil_true.phi,
                        kx=soil_true.kx, ky=soil_true.ky)
    soil_r = SoilParams(kc=soil_true.kc, kphi=soil_true.kphi, n=n_r,
                        c=soil_true.c, phi=soil_true.phi,
                        kx=soil_true.kx, ky=soil_true.ky)
    # quasi-static longitudinal transfer from the measured a_x input
    dN = veh.m * ax_meas * 1.20 / (veh.Lf + veh.Lr) / 2.0
    Nf = max(0.3 * veh.m * veh.g, 0.5 * veh.m * veh.g - dN)
    Nr = max(0.3 * veh.m * veh.g, 0.5 * veh.m * veh.g + dN)
    u_safe = max(u, 0.5)
    vc_f = v + veh.Lf * om
    vc_r = v - veh.Lr * om
    vl_f_t = u_safe * math.cos(delta) + vc_f * math.sin(delta)
    vc_f_t = -u_safe * math.sin(delta) + vc_f * math.cos(delta)
    Fyf_t = _axle_forces(Nf, max(vl_f_t, 0.5), vc_f_t, soil_f)
    Fyr = _axle_forces(Nr, u_safe, vc_r, soil_r)
    Fyf = Fyf_t * math.cos(delta)
    dz = np.array([
        u * math.cos(psi) - (v + veh.Lf * om) * math.sin(psi),
        u * math.sin(psi) + (v + veh.Lf * om) * math.cos(psi),
        om,
        ax_meas,
        (Fyf + Fyr) / veh.m - u * om,
        (Fyf * veh.Lf - Fyr * veh.Lr) / veh.Iz,
        0.0, 0.0,
    ])
    return z + dt * dz


def run_dallas_ukf(obs_path: Path, soil_preset: str, n_true: float,
                   seed: int) -> pd.DataFrame:
    """Run their filter, measurements, and initialisation on a shared trace."""
    p = TERRAIN_PRESETS[soil_preset]
    soil_true = SoilParams(
        kc=float(p["Kc"]), kphi=float(p["Kphi"]), n=n_true,
        c=float(p["cohesion"]),
        phi=math.radians(float(p["friction_angle"])),
        kx=float(p["janosi_shear"]), ky=float(p["janosi_shear"]),
    )
    veh = Vehicle()
    obs = pd.read_csv(obs_path)
    obs = obs[obs.sim_time >= LEAD_IN_S].reset_index(drop=True)
    t0 = float(obs.sim_time.iloc[0])
    rng = np.random.default_rng(seed)

    L = 8
    alpha, kappa, zeta = 0.35, 0.0, 2.0
    lam = alpha * alpha * (L + kappa) - L
    wm = np.full(2 * L + 1, 1.0 / (2.0 * (L + lam)))
    wc = wm.copy()
    wm[0] = lam / (L + lam)
    wc[0] = lam / (L + lam) + (1.0 - alpha * alpha + zeta)
    n_init = n_true + INIT_OFFSET

    z = np.array([float(obs.x_cg[0]), float(obs.y_cg[0]), float(obs.psi[0]),
                  float(obs.u_raw[0]), float(obs.v_lateral_raw[0]),
                  float(obs.omega_raw[0]), n_init, n_init])
    P = np.diag([0.5**2, 0.5**2, 0.01**2, 0.3**2, 0.3**2, 0.01**2,
                 0.12**2, 0.12**2])
    Q = np.diag([0.04**2, 0.04**2, 0.002**2, 0.04**2, 0.04**2, 0.004**2,
                 1e-5, 1e-5])
    R = np.diag(TABLE_V_SIGMA**2)
    H = np.zeros((6, L)); H[:6, :6] = np.eye(6)

    rows = []
    next_meas = t0
    last_t = t0
    for _, r in obs.iterrows():
        t = float(r.sim_time)
        if t < next_meas - 1e-9:
            continue
        next_meas += MEAS_DT
        delta = float(r.steering_angle)
        ax_meas = float(r.ax_imu)
        n_sub = max(1, int(round((t - last_t) / EST_DT)))
        dt_sub = (t - last_t) / n_sub if t > last_t else 0.0
        last_t = t

        # sigma points through the bicycle
        try:
            S = np.linalg.cholesky((L + lam) * P)
        except np.linalg.LinAlgError:
            P = P + 1e-9 * np.eye(L)
            S = np.linalg.cholesky((L + lam) * P)
        pts = np.vstack([z, z + S.T, z - S.T])
        for i in range(pts.shape[0]):
            zi = pts[i]
            for _ in range(n_sub):
                zi = _bicycle_step_2n(zi, delta, ax_meas, dt_sub,
                                      soil_true, veh)
            pts[i] = zi
        z_pred = wm @ pts
        dP = pts - z_pred
        P_pred = (wc[:, None] * dP).T @ dP + Q

        y_meas = np.array([r.x_cg, r.y_cg, r.psi, r.u_raw,
                           r.v_lateral_raw, r.omega_raw], dtype=float)
        y_meas = y_meas + rng.normal(0.0, TABLE_V_SIGMA)
        Y = pts @ H.T
        y_pred = wm @ Y
        dY = Y - y_pred
        Pyy = (wc[:, None] * dY).T @ dY + R
        Pzy = (wc[:, None] * dP).T @ dY
        K = Pzy @ np.linalg.inv(Pyy)
        z = z_pred + K @ (y_meas - y_pred)
        P = P_pred - K @ Pyy @ K.T
        z[6] = float(np.clip(z[6], 0.2, 1.4))
        z[7] = float(np.clip(z[7], 0.2, 1.4))
        rows.append(dict(sim_time=t - t0, n_front=z[6], n_rear=z[7],
                         n_mean=0.5 * (z[6] + z[7])))
    return pd.DataFrame(rows)


def phase_ukf(out: Path) -> None:
    manifest = pd.read_csv(out / "trace_manifest.csv")
    all_rows = []
    for _, tr in manifest.iterrows():
        label = str(tr.trace_id).rsplit("_s", 1)[0]
        preset = SOILS[label]
        n_true = float(TERRAIN_PRESETS[preset]["n"])
        hits = sorted((out / str(tr.run_dir)).rglob("terrain_observations.csv"))
        series = run_dallas_ukf(hits[-1], preset, n_true,
                                seed=int(tr.sim_seed))
        series.insert(0, "trace_id", tr.trace_id)
        series.to_csv(out / f"ukf_timeline_{tr.trace_id}.csv", index=False)
        final = series.iloc[-1]
        all_rows.append(dict(
            trace_id=tr.trace_id, soil=label, n_true=n_true,
            n_init=n_true + INIT_OFFSET,
            n_final_mean=float(final.n_mean),
            n_final_front=float(final.n_front),
            n_final_rear=float(final.n_rear),
            pct_err=100.0 * abs(final.n_mean - n_true) / n_true,
        ))
        print(f"  ukf {tr.trace_id}: n_true={n_true} "
              f"final={final.n_mean:.3f} "
              f"({all_rows[-1]['pct_err']:.1f}% err)", flush=True)
    pd.DataFrame(all_rows).to_csv(out / "ukf_results.csv", index=False)


def phase_grit(out: Path) -> None:
    cmd = [sys.executable, "-u",
           str(ROOT / "benchmarking" / "develop_joint_estimator.py"),
           "--trace-manifest", str(out / "trace_manifest.csv"),
           "--output-dir", str(out / "grit_replay"),
           "--workers", "1"]
    log = out / "grit_replay.log"
    with log.open("w") as fh:
        completed = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=fh)
    if completed.returncode:
        raise SystemExit(f"grit replay failed rc={completed.returncode}: {log}")
    print("  grit replay complete", flush=True)


def within_band_time(t: np.ndarray, v: np.ndarray, final: float,
                     frac: float = 0.10) -> float:
    """First time the series enters and stays within frac of its final value."""
    band = frac * abs(final)
    inside = np.abs(v - final) <= band
    for i in range(len(inside)):
        if inside[i:].all():
            return float(t[i])
    return float("nan")


def phase_score(out: Path) -> None:
    manifest = pd.read_csv(out / "trace_manifest.csv")
    ukf = pd.read_csv(out / "ukf_results.csv")
    est_files = sorted((out / "grit_replay").rglob("estimates.csv"))
    grit = pd.read_csv(est_files[-1]) if est_files else pd.DataFrame()
    rows = []
    for _, tr in manifest.iterrows():
        label = str(tr.trace_id).rsplit("_s", 1)[0]
        preset = SOILS[label]
        n_true = float(TERRAIN_PRESETS[preset]["n"])
        phi_true = float(TERRAIN_PRESETS[preset]["friction_angle"])
        u_row = ukf[ukf.trace_id == tr.trace_id].iloc[0]
        ut = pd.read_csv(out / f"ukf_timeline_{tr.trace_id}.csv")
        ukf_conv = within_band_time(ut.sim_time.to_numpy(),
                                    ut.n_mean.to_numpy(),
                                    float(u_row.n_final_mean))
        rec = dict(
            trace_id=tr.trace_id, soil=label, n_true=n_true,
            phi_true_deg=phi_true,
            ukf_n_final=float(u_row.n_final_mean),
            ukf_pct_err=float(u_row.pct_err),
            ukf_time_to_10pct_of_final_s=ukf_conv,
        )
        if len(grit):
            g = grit[grit.trace_id.astype(str) == str(tr.trace_id)]
            if len(g):
                grow = g.iloc[0]
                for src, dst in (("final_snapshot_n", "grit_n_final"),
                                 ("final_snapshot_phi_deg",
                                  "grit_phi_final_deg"),
                                 ("joint_snapshot_update_count",
                                  "grit_snapshots")):
                    if src in g.columns:
                        rec[dst] = float(grow[src])
                if "grit_n_final" in rec:
                    rec["grit_n_pct_err"] = (
                        100.0 * abs(rec["grit_n_final"] - n_true) / n_true)
                if "grit_phi_final_deg" in rec:
                    rec["grit_phi_abs_err_deg"] = abs(
                        rec["grit_phi_final_deg"] - phi_true)
        rows.append(rec)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "cross_eval_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\nDallas Table VII reference: sandy loam 0.722 (3.1%), "
          f"clay 0.519 (3.8%)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["tune", "collect", "ukf", "grit",
                                      "score", "all"])
    ap.add_argument("--out", default="", help="existing result dir to continue")
    ap.add_argument("--ukf-force-model", choices=["bekker", "nn"], default="bekker",
                    help="Force model inside their UKF. Dallas et al. published "
                         "both: analytical Bekker (arXiv:1908.00130, 3.8%% on clay) "
                         "and a neural terramechanics model (arXiv:2003.02635, "
                         "1.9%%). Use nn to reproduce the better one.")
    ap.add_argument("--base-port", type=int, default=28000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13])
    args = ap.parse_args()
    global UKF_FORCE_MODEL
    UKF_FORCE_MODEL = args.ukf_force_model

    _assert_presets_match()
    if args.out:
        out = Path(args.out)
    else:
        out = timestamped_result_dir("dallas_protocol_cross_eval")
        write_manifest(out, args, "Dallas et al. protocol recreation; "
                       "both estimators on shared traces")
        ledger = {
            "faithful": [
                "Chrono SCM plant, level, simple powertrain",
                "soils = their Table VI (verbatim our clay/dirt presets, asserted)",
                "full-lock sinusoidal steering, 3 s period; sinusoidal throttle, "
                "same frequency, tuned into their 4.8-6.3 m/s band",
                "n as 2x1 [n_f, n_r], n_dot = 0, other soil parameters at truth",
                "n initialised truth + 0.2 (their Table VII)",
                "measurements [x, y, psi, u, v, omega] + their Table V noise, "
                "24 ms cadence, 12 ms filter substeps",
                "metric: final value, percent error, time within 10% of final",
            ],
            "deviations": [
                "vehicle: HMMWV, not their unspecified notional vehicle",
                "full lock 0.528 rad vs about 0.65 rad in their Fig. 7a",
                "UKF process noise not printed in the paper; values from our "
                "reproduction that matches their Fig. 8",
                "plant step 3 ms (deployed) vs their 2 ms",
                "three seeds per soil instead of their single run",
                "GRIT estimates phi as well and prescribes non-target soil "
                "parameters from its manifold (harder task; manifold equals "
                "truth on these two anchor soils); the UKF receives them at "
                "truth per their protocol",
            ],
        }
        (out / "deviations_ledger.json").write_text(
            json.dumps(ledger, indent=2) + "\n")
    print(f"output: {out}")

    if args.phase in ("tune", "all"):
        tune(out, args.base_port)
    if args.phase in ("collect", "all"):
        collect(out, args.base_port, args.seeds)
    if args.phase in ("ukf", "all"):
        phase_ukf(out)
    if args.phase in ("grit", "all"):
        phase_grit(out)
    if args.phase in ("score", "all"):
        phase_score(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
