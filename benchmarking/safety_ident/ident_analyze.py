"""Identify the derived filters' error bounds from the plant manoeuvres (PLAN.md, "Identification").

    python ident_analyze.py <out dir> <run dir> [<run dir> ...]

Runs are ident_runs.py outputs. The certificate-shaped set (--certificate: a manoeuvre for 1 or 2 s, then
straight full braking to a stop) is the one the bounds are identified on. Bounds are per plant powertrain and hold
over all three soils.

Split. Within each (powertrain, soil, manoeuvre family) stratum the runs are shuffled with a fixed seed and the
first half (rounded up) identifies, the rest validates. Every bound is the tightest of its form that covers every
identification point (a linear program minimising its mean), and its coverage is reported on the held-out half.

Model: derived_model.py (rig surrogate, the plant's powertrain from its spec, motion resistance = the surrogate's
force at zero slip), the true soil.

MPSF error tube. From the logged state at the manoeuvre start the model is rolled out at the check's 0.05 s steps
with the commanded steering (the model turns the wheels at once, the plant at its rate limit) and the applied
pedal (the filter plans the pedal rate inside the limit). For each plant position (every 0.1 s) and footprint disc,
cover = distance to the nearest predicted position of the same disc. Form: c0 + c1 s + eps S, s the predicted path
length, S the integral over it of the accumulated heading change.

DOB-CBF backup. At the instant the full-brake command starts, from the plant's speed and applied pedal, the model's
straight backup (pedal ramped down at the limit, held at full brake) gives a stopping distance S_model. The plant's
distance along its heading at that instant until it stops is S_plant, and its largest distance off that line is
L_plant. Forms: S_plant <= s0 + s1 S_model and L_plant <= l0 + l1 S_model.

DOB-CBF observer, at the filter's 0.1 s period. d_k = (u_k+1 - u_k) / dt - mean model du/dt over the interval.
The observer of dob_cbf_theory.py (k dt = 0.5) runs on the logged speed and the model's du/dt, started converged.
Identified: the largest observer error (sigma) and the largest |d| (e0). Also reported: the largest lateral body
speed and the kinematic model's largest yaw-rate error.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

CODE = Path(os.environ.get('ACMD_CODE', str(Path(__file__).resolve().parents[2])))
for p in (CODE, CODE / 'simulation', CODE / 'simulation' / 'shared'):
    sys.path.insert(0, str(p))
from param_consistency import get_terrain_preset, get_vehicle_params_for_demo, terrain_preset_to_internal  # noqa: E402
from simulation.safety import vehicle_spec as spec  # noqa: E402
from simulation.safety.derived_model import DerivedBicycle  # noqa: E402
from simulation.tire_models.nn_tire_model import load_nn_tire_model  # noqa: E402

SEED = 20261112
H = 0.05            # the check's step
DT = 0.1            # the filters' period
K_OBS = 0.5 / DT    # the DOB-CBF's observer gain
VEH = get_vehicle_params_for_demo()
DISCS, DISC_R = spec.footprint_discs()
_models = {}


def model(plant, soil):
    key = (plant, soil)
    if key not in _models:
        t = terrain_preset_to_internal(get_terrain_preset(soil))
        nn = load_nn_tire_model(str(CODE / 'nn_models' / 'tire_force_static'), t)
        b = DerivedBicycle(nn, VEH, plant)
        b.set_terrain(t)
        _models[key] = b
    return _models[key]


def split(runs):
    rng = np.random.default_rng(SEED)
    role = {}
    for _, g in runs.groupby(['powertrain', 'soil', 'family']):
        files = sorted(g.file)
        rng.shuffle(files)
        n_id = (len(files) + 1) // 2
        for i, f in enumerate(files):
            role[f] = 'ident' if i < n_id else 'valid'
    return role


def tightest(cols, y):
    """Smallest-mean nonnegative linear bound sum_i c_i cols_i >= y over all points (linear program)."""
    A = -np.stack(cols, axis=1)
    res = linprog(c=[float(np.mean(c)) for c in cols], A_ub=A, b_ub=-np.asarray(y, float),
                  bounds=[(0, None)] * len(cols), method='highs')
    return [float(v) for v in res.x]


def rollout(d, bike):
    x0 = np.array([d.x.iloc[0], d.y.iloc[0], d.psi.iloc[0], max(d.u.iloc[0], 0.0), d.v.iloc[0], d.r.iloc[0],
                   d.pedal_app.iloc[0]])
    t_end = float(d.t.iloc[-1])
    ts = np.arange(0.0, t_end, H)
    tq = d.t.to_numpy()
    steer = np.interp(ts, tq, d.steer_cmd.to_numpy())
    p_app = np.interp(np.append(ts, ts[-1] + H), tq, d.pedal_app.to_numpy())
    cmds = np.stack([steer, np.diff(p_app) / H / bike.pedal_rate], axis=1)
    xs = bike.rollout_rk4(x0, cmds, np.full(len(ts), H), H, t_tail=0.0)
    return np.append(ts, ts[-1] + H), xs


def cover_points(d, bike):
    tt, xs = rollout(d, bike)
    psi = np.unwrap(xs[:, 2])
    step = np.hypot(np.diff(xs[:, 0], prepend=xs[0, 0]), np.diff(xs[:, 1], prepend=xs[0, 1]))
    s_len = np.cumsum(step)
    S = np.cumsum(step * np.cumsum(np.abs(np.diff(psi, prepend=psi[0]))))
    out = []
    tq = d.t.to_numpy()
    for i in range(0, len(d), 5):
        j = min(int(np.searchsorted(tt, tq[i])), len(tt) - 1)
        worst = 0.0
        for off in DISCS:
            px = d.x.iloc[i] + off * np.cos(d.psi.iloc[i])
            py = d.y.iloc[i] + off * np.sin(d.psi.iloc[i])
            qx = xs[:, 0] + off * np.cos(xs[:, 2])
            qy = xs[:, 1] + off * np.sin(xs[:, 2])
            worst = max(worst, float(np.min(np.hypot(qx - px, qy - py))))
        out.append((tq[i], worst, s_len[j], S[j]))
    return out


def model_stop_distance(bike, u, p):
    """The straight backup's stopping distance on the model: pedal ramped to -1 at the limit, then held."""
    h, dist = 0.01, 0.0
    for _ in range(int(30.0 / h)):
        if u <= 0.05:
            break
        p = max(p - bike.pedal_rate * h, -1.0)
        u = max(u + h * bike.speed_rate(p, u), 0.0)
        dist += u * h
    return dist


def backup_point(d, bike):
    """(S_model, S_plant, L_plant, u at the start) for the run's final full-brake phase, or None."""
    full = np.flatnonzero(d.pedal_cmd.to_numpy() <= -0.999)
    if len(full) == 0:
        return None
    i0 = int(full[0])
    if not (d.pedal_cmd.iloc[i0:] <= -0.999).all() or d.u.iloc[i0] < 0.5 or d.u.iloc[-1] > 0.1:
        return None
    x0, y0, c, s = d.x.iloc[i0], d.y.iloc[i0], np.cos(d.psi.iloc[i0]), np.sin(d.psi.iloc[i0])
    rx, ry = d.x.iloc[i0:].to_numpy() - x0, d.y.iloc[i0:].to_numpy() - y0
    s_plant = float((rx * c + ry * s).max())
    l_plant = float(np.abs(-rx * s + ry * c).max())
    return model_stop_distance(bike, float(d.u.iloc[i0]), float(d.pedal_app.iloc[i0])), s_plant, l_plant, \
        float(d.u.iloc[i0])


def dob_series(d, bike):
    tq = d.t.to_numpy()
    tk = np.arange(0.0, tq[-1] - DT, DT)
    u = np.interp(tk, tq, d.u.to_numpy())
    u_next = np.interp(tk + DT, tq, d.u.to_numpy())
    sub = np.linspace(0.0, DT, 6)
    pa = d.pedal_app.to_numpy()
    a_mean = np.array([np.mean([bike.speed_rate(np.interp(t + s_, tq, pa), np.interp(t + s_, tq, d.u.to_numpy()))
                                for s_ in sub]) for t in tk])
    dist = (u_next - u) / DT - a_mean
    a_now = np.array([bike.speed_rate(np.interp(t, tq, pa), uu) for t, uu in zip(tk, u)])
    xi = u[0] - dist[0] / K_OBS
    err = np.zeros(len(tk))
    for k in range(len(tk)):
        d_hat = K_OBS * (u[k] - xi)
        err[k] = dist[k] - d_hat
        xi += DT * (a_now[k] + d_hat)
    v_lat = np.interp(tk, tq, d.v.to_numpy())
    delta = np.interp(tk, tq, d.steer_app.to_numpy()) * spec.MAX_ROAD_WHEEL
    r_err = np.interp(tk, tq, d.r.to_numpy()) - u * np.tan(delta) / bike.L
    return dist, err, v_lat, r_err


def main(out, dirs):
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.concat([pd.read_csv(Path(f) / 'results.csv').assign(dir=str(f)) for f in dirs], ignore_index=True)
    runs = runs[runs.file.notna()]
    role = split(runs)
    cov, dob, stops = [], [], []
    for r in runs.itertuples():
        d = pd.read_csv(Path(r.dir) / r.file)
        if len(d) < 30:
            continue
        bike = model(r.plant, r.soil)
        tag = dict(file=r.file, role=role[r.file], powertrain=r.powertrain, plant=r.plant, soil=r.soil,
                   family=r.family, maneuver=r.maneuver, speed=r.speed)
        for t, c, s, S in cover_points(d, bike):
            cov.append(dict(tag, t=t, cover=c, s=s, S=S))
        bp = backup_point(d, bike)
        if bp is not None:
            stops.append(dict(tag, S_model=bp[0], S_plant=bp[1], L_plant=bp[2], u_brake=bp[3]))
        dist, err, v_lat, r_err = dob_series(d, bike)
        for k in range(len(dist)):
            dob.append(dict(tag, k=k, d=dist[k], err=err[k], v_lat=v_lat[k], r_err=r_err[k]))
    cov, dob, stops = pd.DataFrame(cov), pd.DataFrame(dob), pd.DataFrame(stops)
    cov.to_csv(out / 'cover_points.csv', index=False)
    dob.to_csv(out / 'dob_series.csv', index=False)
    stops.to_csv(out / 'backup_stops.csv', index=False)

    report = dict(seed=SEED, step_s=H, period_s=DT, observer_gain=K_OBS, disc_offsets=list(DISCS),
                  disc_radius=DISC_R, runs=dict(ident=sorted(f for f, v in role.items() if v == 'ident'),
                                                valid=sorted(f for f, v in role.items() if v == 'valid')),
                  powertrains={})
    lines = [f"runs: {len(runs)}, ident {len(report['runs']['ident'])}, valid {len(report['runs']['valid'])}"]
    for plant in sorted(cov.plant.unique()):
        c, o, st = cov[cov.plant == plant], dob[dob.plant == plant], stops[stops.plant == plant]
        ci, cv = c[c.role == 'ident'], c[c.role == 'valid']
        c0, c1, eps = tightest([np.ones(len(ci)), ci.s.to_numpy(), ci.S.to_numpy()], ci.cover)
        tube_v = c0 + c1 * cv.s + eps * cv.S
        si, sv = st[st.role == 'ident'], st[st.role == 'valid']
        s0, s1 = tightest([np.ones(len(si)), si.S_model.to_numpy()], si.S_plant)
        l0, l1 = tightest([np.ones(len(si)), si.S_model.to_numpy()], si.L_plant)
        oi, ov = o[o.role == 'ident'], o[o.role == 'valid']
        b = dict(
            tube=dict(c0=c0, c1=c1, eps=eps),
            stop=dict(s0=s0, s1=s1), lateral=dict(l0=l0, l1=l1),
            observer=dict(sigma=float(oi.err.abs().max()), e0=float(oi.d.abs().max())),
            held_out=dict(
                tube_points=int(len(cv)), tube_missed=int((cv.cover > tube_v + 1e-9).sum()),
                tube_worst_excess=float(max((cv.cover - tube_v).max(), 0.0)),
                stop_runs=int(len(sv)), stop_missed=int((sv.S_plant > s0 + s1 * sv.S_model + 1e-9).sum()),
                stop_worst_excess=float(max((sv.S_plant - s0 - s1 * sv.S_model).max(), 0.0)) if len(sv) else 0.0,
                lateral_missed=int((sv.L_plant > l0 + l1 * sv.S_model + 1e-9).sum()),
                sigma_exceeded=int((ov.err.abs() > oi.err.abs().max()).sum()), dob_points=int(len(ov)),
                sigma_worst=float(ov.err.abs().max())),
            reported=dict(v_lat_max=float(o.v_lat.abs().max()), yaw_rate_error_max=float(o.r_err.abs().max())),
        )
        report['powertrains'][plant] = b
        lines += ['', f'== {plant}',
                  f"tube: {c0:.3f} + {c1:.4f} s + {eps:.4f} S",
                  f"backup stop: S_plant <= {s0:.3f} + {s1:.3f} S_model, off the line <= {l0:.3f} + {l1:.4f} S_model",
                  f"observer: sigma {b['observer']['sigma']:.2f}, e0 {b['observer']['e0']:.2f} m/s^2",
                  f"held out: {json.dumps(b['held_out'])}",
                  f"reported: {json.dumps(b['reported'])}",
                  '-- cover max (m) by family, ident / valid',
                  c.pivot_table(index='family', columns='role', values='cover', aggfunc='max').round(2).to_string(),
                  '-- backup stops by soil: median S_plant / S_model, worst L_plant',
                  st.assign(ratio=st.S_plant / st.S_model).groupby('soil').agg(
                      ratio=('ratio', 'median'), ratio_max=('ratio', 'max'), L_max=('L_plant', 'max'),
                      n=('ratio', 'size')).round(2).to_string(),
                  '-- observer |error| by family and soil (max, m/s^2)',
                  o.assign(ae=o.err.abs()).pivot_table(index='family', columns='soil', values='ae',
                                                       aggfunc='max').round(2).to_string(),
                  '-- observer |error| quantiles (m/s^2)',
                  o.err.abs().quantile([0.5, 0.9, 0.99, 0.999, 1.0]).round(2).to_string()]
    (out / 'ident.json').write_text(json.dumps(report, indent=1))
    (out / 'ident_tables.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]])
