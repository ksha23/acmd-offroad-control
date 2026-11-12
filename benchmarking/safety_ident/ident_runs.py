"""Plant manoeuvres for identifying the safety filters' error bounds (PLAN.md, "Identification").

Plant only, no filter, flat SCM. The HMMWV accelerates straight to a cruise speed under a speed controller,
holds it, then runs one open-loop manoeuvre (or, for a launch, starts the manoeuvre from rest). Commands pass
through the plant node's actuator rate limits (16/s steering, 8/s pedal). Every 0.02 s it logs the state the
filters receive (x, y, psi, u, v, omega of the chassis centre of mass), the commanded and applied inputs, the
four wheels' slip, and the gear and engine speed.

Both plant powertrains: `shafts` (engine map, torque converter, 3 gears: the battery) and `simple`
(ChEngineSimple with the SIMPLE_CVT transmission, or SIMPLE_MAP on Chrono 9.0.0: late detection, the static
matrix, the grade study).

    python ident_runs.py <out dir> --powertrain shafts|simple [--workers 14] [--certificate]

--certificate runs the shape the MPSF certifies: each manoeuvre for 1 or 2 s (its horizon is 2 s), then straight
full braking until stopped. The error tube is identified on these.

Derived from the journal workspace's experiments/filter_fixes/model_check.py.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('ACMD_CODE', str(Path(__file__).resolve().parents[2])))
SOILS = ('clay', 'dirt', 'sand')
SPEEDS = (3.0, 5.0, 7.0)
# The certificate-shaped set spans the speeds the studies reach (the battery's overspeed mode drives at 9 m/s).
CERT_SPEEDS = (3.0, 5.0, 7.0, 9.0, 11.0)
STEP = 3e-3
X_START = -92.0
STEER_RATE = 16.0
PEDAL_RATE = 8.0
LOG_DT = 0.02
SETTLE_S = 1.5
REST_S = 2.0
WARMUP_MAX = 40.0
MANEUVER_MAX = 6.0

CRUISE_MANEUVERS = ('brake030', 'brake060', 'brake100', 'steer030', 'steer060', 'steer100',
                    'swerve030', 'swerve060', 'swerve100', 'brakesteer60', 'halfbrakesteer60', 'lanechange60',
                    'thrsteer060p050', 'thrsteer060p100', 'thrsteer100p050', 'thrsteer100p100',
                    'thr050', 'thr100', 'coast')
LAUNCHES = ('launch050', 'launch100', 'launchsteer100p100')
# The certificate's shape (--certificate): up to the MPSF's 2 s horizon of manoeuvre, then braking to a stop.
CERT_HOLD_MS = (1000, 2000)
CERT_CRUISE = ('steer030', 'steer060', 'steer100', 'thrsteer060p050', 'thrsteer060p100', 'thrsteer100p050',
               'thrsteer100p100', 'thr050', 'thr100', 'coast', 'brakesteer60', 'halfbrakesteer60', 'brake030',
               'brake060')
CERT_ONCE = ('lanechange60_T2000', 'swerve030', 'swerve060', 'swerve100', 'brake100')
CERT_LAUNCH = ('launch050', 'launch100', 'launchsteer100p100')


def family(name):
    name = name.split('_T')[0]
    for f in ('launchsteer', 'launch', 'brakesteer', 'halfbrakesteer', 'brake', 'steer', 'swerve',
              'lanechange', 'thrsteer', 'thr', 'coast'):
        if name.startswith(f):
            return f
    raise ValueError(name)


def maneuver(name, tau, cruise):
    """(steer, pedal) command at time tau after the manoeuvre start. A name ending in _T<ms> is the
    certificate's shape: the manoeuvre for that many milliseconds, then straight full braking."""
    if '_T' in name:
        base, ms = name.rsplit('_T', 1)
        if tau >= float(ms) / 1000.0:
            return 0.0, -1.0
        name = base
    f = family(name)
    if f == 'brake':
        return 0.0, -float(name[5:]) / 100.0
    if f == 'steer':
        return float(name[5:]) / 100.0, cruise
    if f == 'swerve':
        s = float(name[6:9]) / 100.0
        return (s, cruise) if tau < 0.6 else (0.0, -1.0)
    if name == 'brakesteer60':
        return 0.6, -1.0
    if name == 'halfbrakesteer60':
        return 0.6, -0.4
    if f == 'thrsteer':
        return float(name[8:11]) / 100.0, float(name[12:15]) / 100.0
    if f == 'lanechange':
        if tau < 0.8:
            return 0.6, cruise
        if tau < 1.6:
            return -0.6, cruise
        return 0.0, cruise
    if f == 'thr':
        return 0.0, float(name[3:6]) / 100.0
    if f == 'coast':
        return 0.0, 0.0
    if f == 'launch':
        return 0.0, float(name[6:9]) / 100.0
    if f == 'launchsteer':
        return float(name[11:14]) / 100.0, float(name[15:18]) / 100.0
    raise ValueError(name)


def build_vehicle(chrono, veh, powertrain):
    chrono.SetChronoDataPath(chrono.GetChronoDataPath())
    set_data = getattr(veh, 'SetVehicleDataPath', None) or veh.SetDataPath
    set_data(chrono.GetChronoDataPath() + 'vehicle/')
    vehicle = veh.HMMWV_Full()
    vehicle.SetContactMethod(chrono.ChContactMethod_SMC)
    vehicle.SetChassisFixed(False)
    vehicle.SetChassisCollisionType(veh.CollisionType_PRIMITIVES)
    vehicle.SetInitPosition(chrono.ChCoordsysd(chrono.ChVector3d(X_START, 0, 1.5),
                                               chrono.ChQuaterniond(1, 0, 0, 0)))
    if powertrain == 'shafts':
        vehicle.SetEngineType(veh.EngineModelType_SHAFTS)
        vehicle.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SHAFTS)
        kind = 'shafts'
    else:
        # As chrono_setup.py does for --simple-powertrain.
        vehicle.SetEngineType(veh.EngineModelType_SIMPLE)
        cvt = getattr(veh, 'TransmissionModelType_AUTOMATIC_SIMPLE_CVT', None)
        vehicle.SetTransmissionType(cvt if cvt is not None else veh.TransmissionModelType_AUTOMATIC_SIMPLE_MAP)
        kind = 'simple_cvt' if cvt is not None else 'simple_map'
    vehicle.SetDriveType(veh.DrivelineTypeWV_AWD)
    vehicle.SetTireType(veh.TireModelType_RIGID)
    vehicle.Initialize()
    system = vehicle.GetSystem()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    vehicle.SetChassisVisualizationType(veh.VisualizationType_PRIMITIVES)
    vehicle.SetWheelVisualizationType(veh.VisualizationType_PRIMITIVES)
    vehicle.SetTireVisualizationType(veh.VisualizationType_PRIMITIVES)
    return system, vehicle, kind


def run(job):
    powertrain, soil, speed, name, out = job
    os.chdir(ROOT)
    for p in (ROOT, ROOT / 'simulation', ROOT / 'simulation' / 'runtime', ROOT / 'simulation' / 'shared'):
        sys.path.insert(0, str(p))
    import pychrono as chrono
    import pychrono.vehicle as veh
    from chrono_setup import setup_scm_terrain

    system, vehicle, kind = build_vehicle(chrono, veh, powertrain)
    terrain, _ = setup_scm_terrain(system, vehicle=vehicle, visualize=False, terrain_preset=soil,
                                   mesh_resolution=0.08)
    v_obj = vehicle.GetVehicle()
    chassis = vehicle.GetChassisBody()
    launch = family(name).startswith('launch')

    t = 0.0
    steer_app = 0.0
    pedal_app = 0.0
    integ = 0.0
    settled = 0.0
    t0 = None
    cruise = 0.0
    thr_hist = []
    rows = []
    next_log = 0.0
    stopped_for = 0.0
    while True:
        speed_now = v_obj.GetSpeed()
        if t0 is None:
            if launch:
                # Settle on the soil with the brakes on, then start from rest.
                steer_cmd, pedal_cmd = 0.0, -1.0
                if t >= REST_S:
                    t0 = t
            else:
                # Speed controller: full throttle until near the target, then PI.
                err = speed - speed_now
                if err > 1.0:
                    pedal_cmd = 1.0
                else:
                    integ = float(np.clip(integ + err * STEP, -2.0, 2.0))
                    pedal_cmd = float(np.clip(0.35 + 0.6 * err + 0.4 * integ, 0.0, 1.0))
                steer_cmd = 0.0
                thr_hist.append(pedal_app)
                settled = settled + STEP if abs(err) < 0.15 else 0.0
                if settled >= SETTLE_S or t > WARMUP_MAX:
                    t0 = t
                    cruise = float(np.mean(thr_hist[-int(0.5 / STEP):]))
        if t0 is not None:
            tau = t - t0
            steer_cmd, pedal_cmd = maneuver(name, tau, cruise)
            stopped_for = stopped_for + STEP if (speed_now < 0.05 and tau > 1.0) else 0.0
            t_max = MANEUVER_MAX + (6.0 if '_T' in name else 0.0)
            if tau > t_max or stopped_for > 0.3 or chassis.GetPos().x > 95.0:
                break
        d = STEER_RATE * STEP
        steer_app += float(np.clip(steer_cmd - steer_app, -d, d))
        d = PEDAL_RATE * STEP
        pedal_app += float(np.clip(pedal_cmd - pedal_app, -d, d))

        inputs = veh.DriverInputs()
        inputs.m_steering = steer_app
        inputs.m_throttle = max(pedal_app, 0.0)
        inputs.m_braking = max(-pedal_app, 0.0)
        terrain.Synchronize(t)
        vehicle.Synchronize(t, inputs, terrain)
        terrain.Advance(STEP)
        vehicle.Advance(STEP)
        t += STEP

        if t0 is not None and t >= next_log:
            next_log = t + LOG_DT - 1e-9
            pos = chassis.GetPos()
            rot = chassis.GetRot()
            psi = np.arctan2(2 * (rot.e0 * rot.e3 + rot.e1 * rot.e2), 1 - 2 * (rot.e2 ** 2 + rot.e3 ** 2))
            vel = rot.RotateBack(chassis.GetPosDt())
            u = v_obj.GetSpeed()
            row = dict(t=round(t - t0, 4), x=pos.x, y=pos.y, psi=psi, u=u, v=vel.y,
                       r=chassis.GetAngVelLocal().z, steer_cmd=steer_cmd, steer_app=steer_app,
                       pedal_cmd=pedal_cmd, pedal_app=pedal_app)
            try:
                row['delta_l'] = v_obj.GetSteeringAngle(0, veh.LEFT)
                row['delta_r'] = v_obj.GetSteeringAngle(0, veh.RIGHT)
            except Exception:  # noqa: BLE001
                pass
            try:
                pt = vehicle.GetVehicle().GetTransmission()
                row['gear'] = pt.GetCurrentGear()
                row['engine_w'] = vehicle.GetVehicle().GetEngine().GetMotorSpeed()
                row['engine_tq'] = vehicle.GetVehicle().GetEngine().GetOutputMotorshaftTorque()
            except Exception:  # noqa: BLE001
                pass
            for axle, side, tag in ((0, veh.LEFT, 'fl'), (0, veh.RIGHT, 'fr'), (1, veh.LEFT, 'rl'),
                                    (1, veh.RIGHT, 'rr')):
                wr = abs(v_obj.GetSpindleOmega(axle, side)) * v_obj.GetTire(axle, side).GetRadius()
                row['k_' + tag] = (wr - u) / max(wr, abs(u), 0.1)
            rows.append(row)
    d = pd.DataFrame(rows)
    path = out / f'{powertrain}_{soil}_v{speed:.0f}_{name}.csv'
    d.to_csv(path, index=False)
    return dict(powertrain=powertrain, plant=kind, soil=soil, speed=speed, maneuver=name, family=family(name),
                t0=round(t0 or -1, 2), u0=round(float(d.u.iloc[0]), 2), cruise=round(cruise, 3), rows=len(d),
                tau_end=round(float(d.t.iloc[-1]), 2), x_end=round(float(d.x.iloc[-1]), 1),
                y_end=round(float(d.y.iloc[-1]), 2), file=path.name)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--powertrain', choices=('shafts', 'simple'), required=True)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--only', default='')
    ap.add_argument('--certificate', action='store_true',
                    help='the certificate-shaped set: manoeuvre for 1 or 2 s, then straight full braking')
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.certificate:
        cruise = [f'{m}_T{ms}' for m in CERT_CRUISE for ms in CERT_HOLD_MS] + list(CERT_ONCE)
        launches = [f'{m}_T{ms}' for m in CERT_LAUNCH for ms in CERT_HOLD_MS]
    else:
        cruise, launches = CRUISE_MANEUVERS, LAUNCHES
    speeds = CERT_SPEEDS if args.certificate else SPEEDS
    jobs = [(args.powertrain, s, v, m, out) for s in SOILS for v in speeds for m in cruise]
    jobs += [(args.powertrain, s, 0.0, m, out) for s in SOILS for m in launches]
    if args.only:
        keep = args.only.split(',')
        jobs = [j for j in jobs if f'{j[1]}_v{j[2]:.0f}_{j[3]}' in keep]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run, j): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = dict(powertrain=j[0], soil=j[1], speed=j[2], maneuver=j[3], error=repr(e))
            results.append(r)
            print(r, flush=True)
    pd.DataFrame(results).to_csv(out / 'results.csv', index=False)
