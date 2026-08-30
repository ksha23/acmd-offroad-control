#!/usr/bin/env python3
"""
Launcher for the decoupled Chrono simulation and MPC controller
===============================================================

Starts and supervises two processes:
  1. chrono_sim_node.py — PyChrono HMMWV plant; publishes vehicle state and
     applies received commands.
  2. acados_mpc_controller_node.py — acados NMPC; consumes vehicle state and
     publishes commands.

Running plant and controller as separate processes means the controller
competes for wall-clock time exactly as it would on a vehicle: a solve that
overruns its period delays the command rather than pausing the plant. The
launcher owns the shared configuration, so the two processes cannot disagree
about the path, soil, speed, or transport a run uses, and it assigns each run
an isolated ROS domain so concurrent runs cannot exchange messages.

Usage:
    # Default (NN model, sand terrain, lane change, irrlicht visualization)
    python launch_decoupled.py

    # Sinusoidal path on clay, headless, 30s
    python launch_decoupled.py --path sinusoidal --terrain clay --no-vis --time 30

    # Sensor-only visualization (driver POV camera)
    python launch_decoupled.py --vis-mode sensor

    # Both irrlicht chase cam and sensor driver POV camera
    python launch_decoupled.py --vis-mode both

    # TMeasy MPC tire model
    python launch_decoupled.py --model tmeasy

    # Pacejka MPC tire model with sensor visualization
    python launch_decoupled.py --model pacejka --vis-mode sensor

    # Remote controller (sim on this machine, controller elsewhere)
    python launch_decoupled.py --ctrl-host 192.168.1.50
"""

import os as _os, sys as _sys  # flat-import bootstrap (simulation/flatpath.py)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import flatpath  # noqa: E402,F401
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


GRIT_BACKEND = "grit"
RIG_ACTIVE_ESTIMATOR_BACKEND = GRIT_BACKEND
TERRAIN_ESTIMATOR_BACKENDS = (
    "scalar_parent",
    GRIT_BACKEND,
    "bekker_ukf",
)


def main():
    p = argparse.ArgumentParser(
        description="Launch decoupled Chrono sim + MPC controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --path sinusoidal --terrain clay --time 20
  %(prog)s --model pacejka                   # Pacejka Magic Formula MPC (rigid-terrain params)
  %(prog)s --model pacejka-oracle --terrain clay  # Pacejka given ground-truth terrain identity
  %(prog)s --model tmeasy                    # TMeasy MPC tire model
  %(prog)s --vis-mode sensor                 # Driver POV via Chrono Sensor
  %(prog)s --vis-mode both                   # Irrlicht + Sensor simultaneously
  %(prog)s --sim-only          # Only start the sim node (controller started separately)
  %(prog)s --ctrl-only         # Only start the controller node
""",
    )

    # Shared args
    p.add_argument("--time", type=float, default=15.0, help="Simulation time (s)")
    p.add_argument("--speed", type=float, default=5.0, help="Target speed (m/s)")
    p.add_argument("--terrain", default="sand", choices=["sand", "clay", "dirt"])
    p.add_argument("--controller-prior-terrain", default=None,
                   choices=["sand", "clay", "dirt"],
                   help="Forwarded to the standard MPC controller: makes its "
                        "static terrain prior differ from --terrain (the plant). "
                        "Used by the wrong-prior estimator ablation.")
    p.add_argument(
        "--terrain-estimator-prior",
        default="dirt",
        choices=["sand", "clay", "dirt"],
        help="Shared blind prior for the controller and safety filter "
             "before the first accepted online terrain update.",
    )
    p.add_argument("--terrain-config", type=str, default=None)
    # Spatial soil transition, forwarded to the plant node: the soil changes
    # type partway along +x through a per-location SCM callback, so the
    # estimator meets a boundary it was not told about.
    p.add_argument("--terrain-transition", action="store_true",
                   help="Enable a spatial soil transition along +x "
                        "(--terrain-start blends into --terrain-end).")
    p.add_argument("--terrain-start", default=None,
                   help="Soil preset before the transition (defaults to --terrain).")
    p.add_argument("--terrain-end", default=None,
                   help="Soil preset after the transition.")
    p.add_argument("--transition-x", type=float, default=60.0,
                   help="Center of the soil transition, in terrain x (m).")
    p.add_argument("--transition-width", type=float, default=2.0,
                   help="Full width of the linear soil blend (m); 0 = hard step.")
    p.add_argument("--path", default="lane_change",
                   choices=["lane_change", "double_lane_change", "right_left", "sinusoidal", "straight"])
    p.add_argument("--sine-amplitude", type=float, default=2.0)
    p.add_argument("--sine-wavelength", type=float, default=30.0)
    p.add_argument("--lead-in", type=float, default=0.0,
                   help="Straight lead-in distance (m) before path starts")
    p.add_argument("--no-vis", action="store_true", help="Headless simulation (alias for --vis-mode none)")
    p.add_argument("--vis-mode", default=None,
                   choices=["irrlicht", "sensor", "both", "none"],
                   help="Visualization mode: irrlicht, sensor (driver POV), both, or none")
    p.add_argument("--irrlicht-window-size", type=int, nargs=2,
                   metavar=("WIDTH", "HEIGHT"), default=[4320, 720],
                   help="Irrlicht window size in pixels")
    p.add_argument("--no-rt",  action="store_true",
                   help="Disable real-time pacing (fast-forward; breaks MPC sync)")
    p.add_argument("--no-noise", action="store_true",
                   help="Disable sensor noise (noise ON by default)")
    p.add_argument(
        "--no-tire-forces",
        action="store_true",
        help="Do not publish Chrono tire-force truth diagnostics. The deployed "
             "IMU force adapter and terrain estimator must remain operational.",
    )
    p.add_argument("--sim-seed", type=int, default=None,
                   help="Seed all simulator measurement-noise channels.")
    p.add_argument("--torque-noise-std", type=float, default=5.0,
                   help="Per-wheel driveline/brake torque-sensor noise stdev (N m).")
    p.add_argument("--wheel-center-noise-std", type=float, default=0.01,
                   help="Per-wheel fused wheel-center elevation noise stdev (m).")
    p.add_argument(
        "--wheel-center-calibration-bias-std", type=float, default=0.0,
        help="Run-constant residual wheel-height calibration bias stdev (m).",
    )
    p.add_argument("--sim-diag-csv", default="",
                   help="Write sim-side state/control diagnostics to this CSV. "
                        "Useful for manual/HIL rounds where no controller diag exists.")
    p.add_argument("--latency-phase-s", type=float, default=0.0,
                   help="Forwarded to the sim: shift the latency-profile replay "
                        "window (s, mod trace length).")
    p.add_argument("--latency-profile-json", default="",
                   help="JSON profile for time-varying 5G-like one-way latency. "
                        "Forwarded to the sim for control/manual/camera channels.")
    p.add_argument("--latency-profile-log", default="",
                   help="Optional CSV path for logging active latency samples from the sim.")

    # IMU sensor (Chrono sensor module)
    p.add_argument("--no-imu", action="store_true",
                   help="Disable Chrono sensor-module IMU (use analytical ground-truth accel/gyro)")
    p.add_argument("--imu-rate", type=int, default=100,
                   help="IMU update rate in Hz (default 100)")
    p.add_argument("--imu-lag", type=float, default=0.0,
                   help="IMU sensor lag in seconds (default 0)")
    p.add_argument("--imu-acc-stdev", type=float, default=0.015,
                   help="Accelerometer noise stdev in m/s² (default 0.015)")
    p.add_argument("--imu-gyro-stdev", type=float, default=0.001,
                   help="Gyroscope noise stdev in rad/s (default 0.001)")

    # Controller-specific
    p.add_argument("--model", default="nn",
                   choices=["nn", "pacejka", "pacejka-oracle", "pacejka-rigfit", "tmeasy"],
                   help="MPC tire model: nn, pacejka (rigid-terrain defaults), "
                        "pacejka-oracle (ground-truth-terrain mu/B reference), "
                        "or tmeasy")
    p.add_argument("--speed-weight", type=float, default=70.0,
                   help="Standard-MPC speed tracking weight. Lower values reduce "
                        "reference-speed chasing in turns.")
    p.add_argument("--ay-safety", type=float, default=0.65,
                   help="Curvature-limited speed-profile lateral-accel budget, "
                        "as a fraction of the Coulomb limit. Higher = faster "
                        "cornering reference (forwarded to the standard MPC).")
    p.add_argument(
        "--reference-profile-friction-angle-deg",
        type=float,
        default=None,
        help="Fixed design-envelope friction angle for a shared static speed profile.",
    )
    p.add_argument(
        "--shared-ay-bound-friction-angle-deg",
        type=float,
        default=None,
        help="Fixed design-envelope friction angle for a shared NMPC ay bound.",
    )
    p.add_argument(
        "--terrain-independent-ay-bound",
        action="store_true",
        help="Use one generic NMPC lateral-acceleration bound across terrain "
             "conditioning arms.",
    )
    p.add_argument("--speed-cost-mode", choices=["symmetric", "overspeed"],
                   default="symmetric",
                   help="Standard-MPC speed cost: track v_ref symmetrically or "
                        "treat v_ref as an overspeed cap.")
    p.add_argument("--obstacle-weight", type=float, default=5e3,
                   help="Standard-MPC soft obstacle-barrier weight.")
    p.add_argument("--nn-model", default="tire_force_static")
    p.add_argument("--kappa", default="measured", choices=["zero", "approx", "measured"])
    p.add_argument("--no-lat-transfer", action="store_true")
    p.add_argument("--no-delay-comp", action="store_true")
    p.add_argument("--no-path-reindex", action="store_true")
    p.add_argument(
        "--symbolic-rates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute rate features symbolically in MPC dynamics (default: on). "
             "Use --no-symbolic-rates to disable.",
    )
    p.add_argument("--rms-time-start", type=float, default=2.0,
                   help="Start time for RMS calculation (s)")
    p.add_argument("--dob-ki", type=float, default=0.15,
                   help="Throttle DOB integrator gain [throttle/(m/s)/s]; 0 disables DOB")
    p.add_argument("--dob-max", type=float, default=0.35,
                   help="Asymmetric upper clip on the DOB throttle bias")
    p.add_argument("--dob-bleed", type=float, default=0.5,
                   help="Exponential bleed rate of DOB during MPC braking [1/s]")
    p.add_argument("--ff-drag", action="store_true",
                   help="Feedforward sinkage-drag term in NMPC longitudinal prediction")
    p.add_argument("--ff-drag-scale", type=float, default=1.0,
                   help="Scale on the calibrated feedforward drag (0 disables)")
    p.add_argument("--ff-throttle", action="store_true",
                   help="Feedforward terrain throttle offset replacing the integral DOB")
    p.add_argument("--ff-throttle-scale", type=float, default=1.0,
                   help="Scale on the calibrated feedforward throttle offset (0 disables)")
    p.add_argument("--ff-throttle-2d", action="store_true",
                   help="2-D (n_hat, u) feedforward throttle offset; reactive-DOB replacement")
    p.add_argument("--ff-drag-surrogate", action="store_true",
                   help="Live surrogate compaction-drag feedforward (Dallas drag term)")
    p.add_argument("--terrain-speed-profile", action="store_true",
                   help="Compatibility flag with no effect: the terrain- and "
                        "dynamics-aware g-g speed profile is the default.")
    p.add_argument("--speed-osc-amplitude", type=float, default=0.0,
                   help="Fractional sinusoidal modulation of the speed reference "
                        "(longitudinal excitation for terrain identification).")
    p.add_argument("--speed-osc-period-s", type=float, default=4.0,
                   help="Period in seconds of the speed-reference oscillation.")
    p.add_argument("--te-joint-model-dir", default=None,
                   help="Candidate-evaluation override for the joint estimator's force model.")
    p.add_argument("--te-joint-r-ay", type=float, default=None,
                   help="Candidate-evaluation override for the joint estimator's r_ay.")
    p.add_argument("--legacy-speed-ref", action="store_true",
                   help="Select the static curvature-only speed reference in "
                        "place of the default g-g profile. This is the "
                        "controlled arm of the speed-profile ablation.")
    p.add_argument("--longitudinal-force-balance", action="store_true",
                   help="Principled longitudinal force-balance NMPC (slip kappa as control, u_dot=SumFx(kappa)/M)")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip generating end-of-run plots")
    p.add_argument("--no-csv", action="store_true",
                   help="Skip diagnostic CSV output")
    p.add_argument("--plot-dir", default="plots",
                   help="Directory for output plots")

    # Terrain bumpiness
    p.add_argument("--bumpiness", type=int, default=0, choices=range(0, 11),
                   help="Terrain bumpiness level 0 (flat) to 10 (extreme)")

    # Unmodelled payload robustness test
    p.add_argument("--payload-mass", type=float, default=0.0,
                   help="Unmodelled cargo mass (kg) added to the chassis; "
                        "the controller keeps the nominal empty-vehicle mass.")
    p.add_argument("--simple-powertrain", action="store_true",
                   help="Plant drives via near-direct linear EngineSimple + CVT "
                        "(clean throttle->torque actuation map for force balance)")

    # Rock obstacles
    p.add_argument("--rocks", type=int, default=0,
                   help="Number of rock obstacles (0 = none)")
    p.add_argument("--rock-zone-x", type=float, nargs=2, default=[-15.0, 50.0])
    p.add_argument("--rock-zone-y", type=float, nargs=2, default=[-10.0, 10.0])
    p.add_argument("--rock-size", type=float, nargs=2, default=[0.5, 3.0])
    p.add_argument("--rock-seed", type=int, default=42)
    p.add_argument("--rock-min-spacing", type=float, default=0.0)
    p.add_argument("--rock-centerline-clear", type=float, default=0.0)
    p.add_argument("--rock-spawn-clear", type=float, default=12.0)

    # Safety filter
    p.add_argument("--safety-filter", action="store_true",
                   help="Enable safety filter (flavor selected via --safety-flavor)")
    p.add_argument("--mpsf-n-obstacles", type=int, default=None,
                   help="Obstacle constraints MPSF sizes its OCP for. Forwarded "
                        "to the sim; omitted here leaves the sim default.")
    p.add_argument("--safety-flavor", type=str, default="dob_cbf",
                   choices=["dob_cbf", "mpsf"],
                   help="dob_cbf -- the pointwise DOB-CBF-QP; mpsf -- the "
                        "predictive Model Predictive Safety Filter, which is "
                        "least-restrictive over a horizon and takes its "
                        "braking and cornering authority from the single-tire "
                        "rig force surrogate at the estimated soil.")
    p.add_argument("--no-safety-nn", action="store_true",
                   help="Run the plant-side safety filter without its neural "
                        "tire model, which falls back to kinematic steering "
                        "authority and fixed longitudinal limits. This is the "
                        "ablation arm isolating the tire model's contribution.")
    p.add_argument("--shield-no-sigma-gate", action="store_true",
                   help="Ablation arm: zero the friction-angle uncertainty as "
                        "it crosses into the plant node, which is equivalent "
                        "to --shield-sigma-mode off.")
    p.add_argument("--shield-sigma-mode", type=str, default="off",
                   choices=["tighten", "inflate", "both", "off"],
                   help="How the safety filter uses the estimator's friction-"
                        "angle uncertainty. The default, off, runs the filter "
                        "on its initial terrain belief. tighten, inflate, and "
                        "both are the arms of the uncertainty-gating ablation.")
    p.add_argument("--shield-sigma-buffer-gain", type=float, default=0.05,
                   help="Metres of extra obstacle buffer per degree of phi_sigma.")
    p.add_argument("--shield-horizon", type=int, default=12)
    p.add_argument("--mpc-blind-obstacles", action="store_true",
                   help="Make the MPC controller ignore obstacles — safety shield "
                        "becomes the sole collision-avoider.")
    p.add_argument("--cbf-alpha", type=float, default=1.0)
    p.add_argument("--safety-buffer", type=float, default=0.25)
    p.add_argument("--delay-steps", type=int, default=5)
    p.add_argument("--cbf-w-long", type=float, default=0.15)
    p.add_argument("--cbf-w-lat", type=float, default=0.50)
    p.add_argument("--cbf-forward-bias", type=float, default=1.5)
    p.add_argument("--dob-bandwidth", type=float, default=10.0)
    p.add_argument("--cbf-flavor", type=str, default="balance",
                   choices=["balance", "steer_priority", "throttle_priority"])
    p.add_argument("--teleop-delay", type=float, default=0.0,
                   help="Initial one-way teleop delay in seconds (0 = local)")
    p.add_argument("--stale-cmd-timeout", type=float, default=2.0,
                   help="Auto-brake if no command for this many seconds")

    # Network
    p.add_argument("--sim-port", type=int, default=5555)
    p.add_argument("--ctrl-port", type=int, default=5556)
    p.add_argument("--ctrl-host", default="localhost",
                   help="Host running the controller (for sim to subscribe)")
    p.add_argument("--transport", choices=["zmq", "ros"], default=os.environ.get("HIL_TRANSPORT", "ros"),
                   help="IPC transport for the sim<->controller link. 'ros' uses "
                        "rclpy/DDS and requires ROS 2 to be sourced. Chrono::ROS "
                        "body-state and /clock publishers are enabled by default.")
    p.add_argument("--no-chrono-ros", action="store_true",
                   help="Development-only opt-out from the default Chrono::ROS "
                        "body-state and /clock publishers.")

    # Mode
    p.add_argument("--sim-only", action="store_true",
                   help="Only launch the simulation node")
    p.add_argument("--ctrl-only", action="store_true",
                   help="Only launch the controller node")
    p.add_argument("--manual", action="store_true",
                   help="Manual control with G29 steering wheel (no MPC controller)")
    p.add_argument("--wasd", action="store_true",
                   help="Manual control with WASD keyboard (no MPC controller)")
    p.add_argument("--manual-honor-time", action="store_true",
                   help="In manual mode, stop automatically at --time instead of "
                        "requiring the driver to close the window.")
    p.add_argument("--manual-input-delay", type=float, default=0.0,
                   help="Apply a fixed actuation delay to manual steering/throttle/brake inputs.")
    p.add_argument("--camera-input-delay", type=float, default=0.0,
                   help="Apply a fixed lag to the driver POV camera feed (models "
                        "downlink video latency to the operator).")
    # Visualization sizing + SCM mesh (forwarded to the sim for real-time tuning).
    p.add_argument("--cam-width", type=int, default=1920,
                   help="Driver POV camera / window width (px).")
    p.add_argument("--cam-height", type=int, default=1080,
                   help="Driver POV camera / window height (px). Use 1200 for 16:10.")
    p.add_argument("--cam-fov", type=float, default=1.05,
                   help="Driver POV camera horizontal FOV (rad, ~1.05=60deg).")
    p.add_argument("--cam-rate", type=float, default=30.0,
                   help="Driver POV camera render rate (Hz); real-time lever.")
    p.add_argument("--cam-save-dir", type=str, default="",
                   help="Save driver-POV camera frames (PNG sequence) to this "
                        "directory for offline video encoding. Forwarded to the sim.")
    p.add_argument("--cam-no-window", action="store_true",
                   help="Capture POV frames without opening a live window "
                        "(headless video capture). Forwarded to the sim.")
    p.add_argument("--delayed-pov", action="store_true",
                   help="Show the driver POV through a software frame-delay buffer so "
                        "the operator sees the camera-channel latency (forwarded to sim).")
    p.add_argument("--pov-no-flip", action="store_true",
                   help="Disable the delayed POV's default vertical flip (forwarded to sim).")
    p.add_argument("--cam-fullscreen", action="store_true",
                   help="Display the driver POV fullscreen (renders at cam W x H).")
    p.add_argument("--convoy", type=str, default="",
                   help="Spawn PID-driven traffic for a convoy safety scenario "
                        "(lead_brake/cut_in/stalled/swerver/convoy/platoon/oncoming/"
                        "double_cut/stop_and_go/jam/overtake/gauntlet).")
    p.add_argument("--traffic-detail", choices=["auto", "mesh", "primitives"],
                   default="mesh", help="Traffic render detail (mesh|auto|primitives).")
    p.add_argument("--goal-distance", type=float, default=0.0,
                   help="If >0, place a visible goal gate this far ahead (m) and "
                        "end the round early once the ego reaches it.")
    p.add_argument("--replay-cmds", type=str, default="",
                   help="Counterfactual replay: re-drive the ego from a recorded "
                        "command trace CSV (no controller/manual driver).")
    p.add_argument("--mesh-resolution", type=float, default=None,
                   help="SCM mesh spacing (m). Default 0.08; 0.12 for real-time HIL.")
    p.add_argument("--grip-safety", type=float, default=None,
                   help="Forwarded to the controller: fraction of predicted "
                        "lateral capacity the speed profile will use.")
    p.add_argument("--step-size", type=float, default=None,
                   help="Physics step (s). Default 3e-3; used by the numerical "
                        "convergence study. Forwarded to the sim.")

    # Online terrain parameter estimator
    p.add_argument("--terrain-estimator", action="store_true",
                   help="Enable online terrain estimation from deployable vehicle-state, "
                        "wheel, steering, and sensor channels (no oracle data).")
    p.add_argument("--terrain-estimator-mode", choices=["n"], default="n",
                   help="Compatibility flag; the selected backend declares its "
                        "live output fields.")
    p.add_argument("--terrain-estimator-backend",
                   choices=TERRAIN_ESTIMATOR_BACKENDS,
                   default=RIG_ACTIVE_ESTIMATOR_BACKEND,
                   help="Runtime terrain estimator. scalar_parent "
                        "estimates the Bekker exponent alone under a matched "
                        "scalar profile; grit estimates the "
                        "exponent and the friction angle jointly and falls "
                        "back to a fixed, control-feasible low-grip point "
                        "whenever its freshness, observability, or boundary "
                        "gates reject a snapshot.")
    p.add_argument("--te-update-interval", type=int, default=10)
    p.add_argument("--te-filter-dt", type=float, default=0.1)
    p.add_argument("--nn-ukf-q-n", type=float, default=0.04)
    p.add_argument("--parent-grid-size", type=int, default=41)
    p.add_argument("--parent-student-dof", type=float, default=4.0)
    p.add_argument("--estimator-update-interval", type=int, default=1)
    p.add_argument("--estimator-block-dt", type=float, default=0.5)
    p.add_argument("--estimator-horizon", type=float, default=8.0)
    p.add_argument("--estimator-min-windows", type=int, default=12)
    p.add_argument("--estimator-min-window-samples", type=int, default=4)
    p.add_argument("--estimator-r-ax", type=float, default=0.35)
    p.add_argument("--estimator-r-ay", type=float, default=0.30)
    p.add_argument("--estimator-min-information", type=float, default=0.20)
    p.add_argument("--estimator-min-yaw-rate-rms", type=float, default=0.015)
    p.add_argument("--estimator-min-speed", type=float, default=2.5)
    p.add_argument("--estimator-max-abs-alpha", type=float, default=0.35)
    p.add_argument(
        "--estimator-enforce-feature-envelope",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--estimator-slip-mode",
        choices=("wheel", "average", "fixed"),
        default="average",
    )
    p.add_argument("--estimator-fixed-kappa", type=float, default=0.05)
    p.add_argument(
        "--estimator-rate-mode",
        choices=("signed", "zero", "legacy"),
        default="zero",
    )
    p.add_argument("--estimator-force-gain-std", type=float, default=0.04)
    p.add_argument("--estimator-ax-bias-std", type=float, default=0.10)
    p.add_argument("--estimator-ay-bias-std", type=float, default=0.05)
    p.add_argument("--estimator-force-gain-min", type=float, default=0.70)
    p.add_argument("--estimator-force-gain-max", type=float, default=1.30)
    p.add_argument(
        "--estimator-acceleration-bias-bound", type=float, default=0.30
    )
    p.add_argument("--estimator-profile-iterations", type=int, default=8)
    p.add_argument("--te-min-confidence", type=float, default=0.3)
    p.add_argument("--hedge-k", type=float, default=0.0,
                   help="Controller-side uncertainty-aware envelope: g-g grip "
                        "limits at n_hat - k*sigma_n (0 = off).")
    p.add_argument("--shield-hedge-k", type=float, default=0.0,
                   help="Sim-side belief-robust safety authority: filter "
                        "accel/brake limits at n_hat - k*sigma_n (0 = off).")
    p.add_argument("--shield-terrain-nn", action="store_true",
                   help="Forwarded: filter NN queries conditioned on live terrain.")
    p.add_argument("--shield-grip-scale", action="store_true",
                   help="Forwarded: filter authority scaled by the adapter's "
                        "measured grip ratio.")
    p.add_argument("--ukf-model-dir", default=None,
                   help="Override the single-tire rig checkpoint the force-map "
                        "estimators evaluate (default: "
                        "nn_models/tire_force_static_parent).")
    p.add_argument("--te-verbose", action="store_true",
                   help="Print verbose terrain-estimator predictions in the "
                        "controller (useful for offline log parsing)")
    p.add_argument("--terrain-id-probe", action="store_true")
    p.add_argument("--terrain-id-probe-target-alpha", type=float, default=0.10)
    p.add_argument("--terrain-id-probe-slew-rate", type=float, default=0.40)
    p.add_argument("--terrain-id-probe-signed-dwell", type=float, default=0.15)
    p.add_argument("--terrain-id-probe-clearance", type=float, default=35.0)
    p.add_argument("--terrain-id-probe-max-latency", type=float, default=0.30)
    p.add_argument("--excitation-steer-amp", type=float, default=0.0,
                   help=argparse.SUPPRESS)
    p.add_argument("--excitation-steer-period", type=float, default=1.0,
                   help=argparse.SUPPRESS)

    p.add_argument("--ax-filter-tau", type=float, default=0.5,
                   help="Complementary filter time constant (s) for IMU ax (0 = no filter)")
    p.add_argument("--vel-filter-tau", type=float, default=0.05,
                   help="EMA time constant (s) for smoothing noisy [u, v, omega] (0 = off)")

    args = p.parse_args()
    if args.excitation_steer_amp > 0.0:
        p.error("--excitation-steer-* is not supported; use --terrain-id-probe")
    if args.terrain_estimator and args.model != "nn":
        p.error("--terrain-estimator requires --model nn")
    script_dir = Path(__file__).resolve().parent   # simulation/runtime (holds chrono_sim_node.py)
    sim_root = script_dir.parent                    # simulation/ (holds control/, runtime/)
    project_root = sim_root.parent                  # repo root
    if args.latency_profile_json:
        profile_path = Path(args.latency_profile_json).expanduser()
        args.latency_profile_json = str(profile_path.resolve())
    # Resolve vis mode: --no-vis is shorthand for --vis-mode none
    vis_mode = args.vis_mode
    if vis_mode is None:
        vis_mode = 'none' if args.no_vis else 'irrlicht'

    # ---- Build command lines ----
    sim_cmd = [
        sys.executable, str(script_dir / "chrono_sim_node.py"),
        "--time", str(args.time),
        "--speed", str(args.speed),
        "--terrain", args.terrain,
        "--path", args.path,
        "--sine-amplitude", str(args.sine_amplitude),
        "--sine-wavelength", str(args.sine_wavelength),
        "--lead-in", str(args.lead_in),
        "--sim-port", str(args.sim_port),
        "--ctrl-host", args.ctrl_host,
        "--ctrl-port", str(args.ctrl_port),
        "--transport", args.transport,
        "--bumpiness", str(args.bumpiness),
        "--vis-mode", vis_mode,
        "--irrlicht-window-size", str(args.irrlicht_window_size[0]),
        str(args.irrlicht_window_size[1]),
    ]
    if args.no_chrono_ros:
        sim_cmd.append("--no-chrono-ros")
    if getattr(args, "simple_powertrain", False):
        sim_cmd.append("--simple-powertrain")
    if args.terrain_transition:
        sim_cmd.append("--terrain-transition")
        if args.terrain_start:
            sim_cmd.extend(["--terrain-start", args.terrain_start])
        if args.terrain_end:
            sim_cmd.extend(["--terrain-end", args.terrain_end])
        sim_cmd.extend(["--transition-x", str(args.transition_x)])
        sim_cmd.extend(["--transition-width", str(args.transition_width)])
    if args.no_rt:
        sim_cmd.append("--no-rt")
    if args.no_noise:
        sim_cmd.append("--no-noise")
    if args.no_tire_forces:
        sim_cmd.append("--no-tire-forces")
    if args.terrain_estimator and args.model == "nn":
        _belief_prior = (
            args.controller_prior_terrain or args.terrain_estimator_prior
        )
        sim_cmd.extend(["--terrain-belief-prior", _belief_prior])
    if args.sim_seed is not None:
        sim_cmd.extend(["--sim-seed", str(args.sim_seed)])
    sim_cmd.extend(["--torque-noise-std", str(args.torque_noise_std)])
    sim_cmd.extend([
        "--wheel-center-noise-std", str(args.wheel_center_noise_std)
    ])
    sim_cmd.extend([
        "--wheel-center-calibration-bias-std",
        str(args.wheel_center_calibration_bias_std),
    ])
    if args.sim_diag_csv:
        sim_cmd.extend(["--sim-diag-csv", args.sim_diag_csv])
    if args.latency_profile_json:
        sim_cmd.extend(["--latency-profile-json", args.latency_profile_json])
        if getattr(args, "latency_phase_s", 0.0):
            sim_cmd.extend(["--latency-phase-s", str(args.latency_phase_s)])
    if args.latency_profile_log:
        sim_cmd.extend(["--latency-profile-log", args.latency_profile_log])
    sim_cmd.extend(["--cam-width", str(args.cam_width),
                    "--cam-height", str(args.cam_height),
                    "--cam-fov", str(args.cam_fov),
                    "--cam-rate", str(args.cam_rate)])
    if args.cam_save_dir:
        sim_cmd.extend(["--cam-save-dir", args.cam_save_dir])
    if args.cam_no_window:
        sim_cmd.append("--cam-no-window")
    if args.cam_fullscreen:
        sim_cmd.append("--cam-fullscreen")
    if args.delayed_pov:
        sim_cmd.append("--delayed-pov")
    if args.pov_no_flip:
        sim_cmd.append("--pov-no-flip")
    if args.convoy:
        sim_cmd.extend(["--convoy", args.convoy, "--traffic-detail", args.traffic_detail])
    if args.goal_distance > 0:
        sim_cmd.extend(["--goal-distance", str(args.goal_distance)])
    if args.replay_cmds:
        sim_cmd.extend(["--replay-cmds", args.replay_cmds])
    if args.mesh_resolution is not None:
        sim_cmd.extend(["--mesh-resolution", str(args.mesh_resolution)])
    if args.step_size is not None:
        sim_cmd.extend(["--step-size", str(args.step_size)])
    if args.manual:
        sim_cmd.append("--manual")
    if args.wasd:
        sim_cmd.append("--wasd")
    if args.manual_honor_time:
        sim_cmd.append("--manual-honor-time")
    if args.manual_input_delay > 0:
        sim_cmd.extend(["--manual-input-delay", str(args.manual_input_delay)])
    if args.camera_input_delay > 0:
        sim_cmd.extend(["--camera-input-delay", str(args.camera_input_delay)])
    if args.teleop_delay > 0:
        sim_cmd.extend(["--teleop-delay", str(args.teleop_delay)])
        sim_cmd.extend(["--stale-cmd-timeout", str(args.stale_cmd_timeout)])
    if args.terrain_config:
        sim_cmd.extend(["--terrain-config", args.terrain_config])
    if args.payload_mass and args.payload_mass > 0:
        sim_cmd.extend(["--payload-mass", str(args.payload_mass)])
    # Rock obstacles
    if args.rocks > 0:
        sim_cmd.extend(["--rocks", str(args.rocks)])
        sim_cmd.extend(["--rock-zone-x"] + [str(v) for v in args.rock_zone_x])
        sim_cmd.extend(["--rock-zone-y"] + [str(v) for v in args.rock_zone_y])
        sim_cmd.extend(["--rock-size"] + [str(v) for v in args.rock_size])
        sim_cmd.extend(["--rock-seed", str(args.rock_seed)])
        sim_cmd.extend(["--rock-min-spacing", str(args.rock_min_spacing)])
        sim_cmd.extend(["--rock-centerline-clear", str(args.rock_centerline_clear)])
        sim_cmd.extend(["--rock-spawn-clear", str(args.rock_spawn_clear)])
    # Safety filter
    if args.safety_filter:
        sim_cmd.append("--safety-filter")
        sim_cmd.extend(["--safety-flavor", args.safety_flavor])
        if args.mpsf_n_obstacles is not None:
            sim_cmd.extend(["--mpsf-n-obstacles", str(args.mpsf_n_obstacles)])
        if float(args.shield_hedge_k) > 0.0:
            sim_cmd.extend(["--shield-hedge-k", str(args.shield_hedge_k)])
        if args.shield_terrain_nn:
            sim_cmd.append("--shield-terrain-nn")
        if args.shield_grip_scale:
            sim_cmd.append("--shield-grip-scale")
        if args.no_safety_nn:
            sim_cmd.append("--no-safety-nn")
        sim_cmd.extend(["--safety-buffer", str(args.safety_buffer)])
        # Barrier-QP parameters. They are forwarded unconditionally; the
        # predictive filter ignores the ones it does not define.
        sim_cmd.extend(["--cbf-alpha", str(args.cbf_alpha)])
        sim_cmd.extend(["--delay-steps", str(args.delay_steps)])
        sim_cmd.extend(["--cbf-w-long", str(args.cbf_w_long)])
        sim_cmd.extend(["--cbf-w-lat", str(args.cbf_w_lat)])
        sim_cmd.extend(["--cbf-forward-bias", str(args.cbf_forward_bias)])
        sim_cmd.extend(["--dob-bandwidth", str(args.dob_bandwidth)])
        sim_cmd.extend(["--cbf-flavor", args.cbf_flavor])

    # IMU sensor args
    if args.no_imu:
        sim_cmd.append("--no-imu")
    if args.imu_rate != 100:
        sim_cmd.extend(["--imu-rate", str(args.imu_rate)])
    if args.imu_lag > 0:
        sim_cmd.extend(["--imu-lag", str(args.imu_lag)])
    if args.imu_acc_stdev != 0.015:
        sim_cmd.extend(["--imu-acc-stdev", str(args.imu_acc_stdev)])
    if args.imu_gyro_stdev != 0.001:
        sim_cmd.extend(["--imu-gyro-stdev", str(args.imu_gyro_stdev)])

    # Controller process: the reference-tracking acados NMPC.
    ctrl_cmd = [
        sys.executable, str(sim_root / "control" / "acados_mpc_controller_node.py"),
        "--model", args.model,
        "--nn-model", args.nn_model,
        "--kappa", args.kappa,
        "--path", args.path,
        "--speed", str(args.speed),
        "--terrain", args.terrain,
        "--terrain-estimator-prior", args.terrain_estimator_prior,
        "--time", str(args.time),
        "--sine-amplitude", str(args.sine_amplitude),
        "--sine-wavelength", str(args.sine_wavelength),
        "--lead-in", str(args.lead_in),
        "--sim-host", "localhost",
        "--sim-port", str(args.sim_port),
        "--ctrl-port", str(args.ctrl_port),
        "--transport", args.transport,
        "--rms-time-start", str(args.rms_time_start),
        "--plot-dir", args.plot_dir,
        "--dob-ki", str(args.dob_ki),
        "--dob-max", str(args.dob_max),
        "--dob-bleed", str(args.dob_bleed),
        "--ff-drag-scale", str(args.ff_drag_scale),
        "--ff-throttle-scale", str(args.ff_throttle_scale),
    ]
    if getattr(args, "ff_drag", False):
        ctrl_cmd.append("--ff-drag")
    if getattr(args, "ff_throttle", False):
        ctrl_cmd.append("--ff-throttle")
    if getattr(args, "ff_throttle_2d", False):
        ctrl_cmd.append("--ff-throttle-2d")
    if getattr(args, "ff_drag_surrogate", False):
        ctrl_cmd.append("--ff-drag-surrogate")
    if getattr(args, "terrain_speed_profile", False):
        ctrl_cmd.append("--terrain-speed-profile")
    if getattr(args, "grip_safety", None) is not None:
        ctrl_cmd.extend(["--grip-safety", str(args.grip_safety)])
    if getattr(args, "legacy_speed_ref", False):
        ctrl_cmd.append("--legacy-speed-ref")
    if getattr(args, "te_joint_model_dir", None):
        ctrl_cmd += ["--te-joint-model-dir", str(args.te_joint_model_dir)]
    if getattr(args, "te_joint_r_ay", None) is not None:
        ctrl_cmd += ["--te-joint-r-ay", str(args.te_joint_r_ay)]
    if float(getattr(args, "speed_osc_amplitude", 0.0)) > 0.0:
        ctrl_cmd += ["--speed-osc-amplitude", str(args.speed_osc_amplitude),
                     "--speed-osc-period-s", str(args.speed_osc_period_s)]
    if getattr(args, "longitudinal_force_balance", False):
        ctrl_cmd.append("--longitudinal-force-balance")
    if getattr(args, "controller_prior_terrain", None):
        ctrl_cmd.extend(["--controller-prior-terrain",
                         args.controller_prior_terrain])
    if args.no_delay_comp:
        ctrl_cmd.append("--no-delay-comp")
    if args.no_lat_transfer:
        ctrl_cmd.append("--no-lat-transfer")
    if args.no_path_reindex:
        ctrl_cmd.append("--no-path-reindex")
    ctrl_cmd.extend(["--speed-weight", str(args.speed_weight)])
    ctrl_cmd.extend(["--ay-safety", str(args.ay_safety)])
    if args.reference_profile_friction_angle_deg is not None:
        ctrl_cmd.extend([
            "--reference-profile-friction-angle-deg",
            str(args.reference_profile_friction_angle_deg),
        ])
    if args.shared_ay_bound_friction_angle_deg is not None:
        ctrl_cmd.extend([
            "--shared-ay-bound-friction-angle-deg",
            str(args.shared_ay_bound_friction_angle_deg),
        ])
    if args.terrain_independent_ay_bound:
        ctrl_cmd.append("--terrain-independent-ay-bound")
    ctrl_cmd.extend(["--speed-cost-mode", args.speed_cost_mode])
    ctrl_cmd.extend(["--obstacle-weight", str(args.obstacle_weight)])
    if args.symbolic_rates:
        ctrl_cmd.append("--symbolic-rates")
    else:
        ctrl_cmd.append("--no-symbolic-rates")
    if args.no_plot:
        ctrl_cmd.append("--no-plot")
    if args.no_csv:
        ctrl_cmd.append("--no-csv")
    if args.mpc_blind_obstacles:
        ctrl_cmd.append("--mpc-blind-obstacles")
    if args.terrain_estimator:
        ctrl_cmd.append("--terrain-estimator")
        ctrl_cmd.extend(["--terrain-estimator-mode", str(args.terrain_estimator_mode)])
        ctrl_cmd.extend(["--terrain-estimator-backend", str(args.terrain_estimator_backend)])
        ctrl_cmd.extend(["--te-update-interval", str(args.te_update_interval)])
        ctrl_cmd.extend(["--te-filter-dt", str(args.te_filter_dt)])
        ctrl_cmd.extend(["--te-min-confidence", str(args.te_min_confidence)])
        if float(args.hedge_k) > 0.0:
            ctrl_cmd.extend(["--hedge-k", str(args.hedge_k)])
        ctrl_cmd.extend(["--nn-ukf-q-n", str(args.nn_ukf_q_n)])
        ctrl_cmd.extend(["--parent-grid-size", str(args.parent_grid_size)])
        ctrl_cmd.extend(["--parent-student-dof", str(args.parent_student_dof)])
        ctrl_cmd.extend([
            "--estimator-update-interval",
            str(args.estimator_update_interval),
            "--estimator-block-dt", str(args.estimator_block_dt),
            "--estimator-horizon", str(args.estimator_horizon),
            "--estimator-min-windows", str(args.estimator_min_windows),
            "--estimator-min-window-samples",
            str(args.estimator_min_window_samples),
            "--estimator-r-ax", str(args.estimator_r_ax),
            "--estimator-r-ay", str(args.estimator_r_ay),
            "--estimator-min-information",
            str(args.estimator_min_information),
            "--estimator-min-yaw-rate-rms",
            str(args.estimator_min_yaw_rate_rms),
            "--estimator-min-speed", str(args.estimator_min_speed),
            "--estimator-max-abs-alpha",
            str(args.estimator_max_abs_alpha),
            "--estimator-slip-mode", str(args.estimator_slip_mode),
            "--estimator-fixed-kappa", str(args.estimator_fixed_kappa),
            "--estimator-rate-mode", str(args.estimator_rate_mode),
            "--estimator-force-gain-std",
            str(args.estimator_force_gain_std),
            "--estimator-ax-bias-std", str(args.estimator_ax_bias_std),
            "--estimator-ay-bias-std", str(args.estimator_ay_bias_std),
            "--estimator-force-gain-min",
            str(args.estimator_force_gain_min),
            "--estimator-force-gain-max",
            str(args.estimator_force_gain_max),
            "--estimator-acceleration-bias-bound",
            str(args.estimator_acceleration_bias_bound),
            "--estimator-profile-iterations",
            str(args.estimator_profile_iterations),
        ])
        ctrl_cmd.append(
            "--estimator-enforce-feature-envelope"
            if args.estimator_enforce_feature_envelope
            else "--no-rig-dynamics-enforce-feature-envelope"
        )
        if args.ukf_model_dir:
            ctrl_cmd.extend(["--ukf-model-dir", str(args.ukf_model_dir)])
        if args.te_verbose:
            ctrl_cmd.append("--te-verbose")

    ctrl_cmd.extend(["--ax-filter-tau", str(args.ax_filter_tau)])
    ctrl_cmd.extend(["--vel-filter-tau", str(args.vel_filter_tau)])
    if args.terrain_id_probe:
        ctrl_cmd.extend([
            "--terrain-id-probe",
            "--terrain-id-probe-target-alpha", str(args.terrain_id_probe_target_alpha),
            "--terrain-id-probe-slew-rate", str(args.terrain_id_probe_slew_rate),
            "--terrain-id-probe-signed-dwell", str(args.terrain_id_probe_signed_dwell),
            "--terrain-id-probe-clearance", str(args.terrain_id_probe_clearance),
            "--terrain-id-probe-max-latency", str(args.terrain_id_probe_max_latency),
        ])

    # ---- Launch ----
    # The acados-generated solvers, for both the NMPC and the predictive safety
    # filter, link libqpOASES_e.so at load time. That library lives under
    # ACADOS_SOURCE_DIR/lib, which is not always on the shell's
    # LD_LIBRARY_PATH, so the directory is added here, before any Popen, and
    # the spawned plant and controller inherit it.
    _acados_src = os.environ.get("ACADOS_SOURCE_DIR")
    if _acados_src:
        _acados_lib = os.path.join(_acados_src, "lib")
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        if os.path.isdir(_acados_lib) and _acados_lib not in _ld.split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = (
                _acados_lib + (os.pathsep + _ld if _ld else ""))

    # Isolate this run's DDS graph so parallel sweep workers cannot exchange
    # messages: the ROS domain is derived from the simulation port, and each
    # concurrent run is assigned a distinct port block and therefore a distinct
    # domain. The child processes inherit this launcher's environment, so plant
    # and controller share the one domain.
    if args.transport == "ros":
        os.environ["ROS_DOMAIN_ID"] = str(args.sim_port % 101)  # valid range 0-101
        print(f"[launch] ROS_DOMAIN_ID={os.environ['ROS_DOMAIN_ID']} (from sim-port {args.sim_port})")

    procs = []

    def cleanup():
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def handle_signal(signum, frame):
        cleanup()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    exit_code = 0
    try:
        if args.ctrl_only:
            print(f"[launch] Starting controller only")
            print(f"  cmd: {' '.join(ctrl_cmd)}")
            proc = subprocess.Popen(ctrl_cmd)
            procs.append(proc)
            exit_code = proc.wait()
        elif args.sim_only or args.manual or args.wasd:
            if args.wasd:
                mode = "manual (WASD keyboard)"
            elif args.manual:
                mode = "manual (G29)"
            else:
                mode = "simulation only"
            print(f"[launch] Starting {mode}")
            print(f"  cmd: {' '.join(sim_cmd)}")

            proc = subprocess.Popen(sim_cmd)
            procs.append(proc)
            exit_code = proc.wait()
        else:
            # The controller starts first and waits for the plant's config
            # message.
            print(f"[launch] Starting controller...")
            ctrl_proc = subprocess.Popen(ctrl_cmd)
            procs.append(ctrl_proc)

            # The plant then waits for the controller's first command before
            # advancing, so acados code generation and compilation do not
            # consume simulated time. The controller emits ready pings until
            # vehicle state arrives (see acados_mpc_controller_node).
            time.sleep(0.5)
            print(f"[launch] Starting simulation...")
            sim_cmd_both = sim_cmd + ["--wait-for-controller", "300"]
            completion_marker = None
            if os.environ.get("HIL_RUN_LOG_DIR"):
                completion_marker = (Path(os.environ["HIL_RUN_LOG_DIR"])
                                     / ".sim_complete")
                try:
                    completion_marker.unlink()
                except FileNotFoundError:
                    pass
            sim_proc = subprocess.Popen(sim_cmd_both)
            procs.append(sim_proc)

            # Wait on the simulation's output being complete rather than on DDS
            # participant destruction. Under heavily parallel ROS-domain sweeps
            # rclpy and Fast DDS can linger after the plant has closed every
            # metric file and printed its summary. The plant writes a durable
            # completion marker immediately before closing its transport, so a
            # normal exit is given ten seconds and anything beyond that is
            # teardown alone and is reaped.
            while sim_proc.poll() is None:
                if completion_marker is not None and completion_marker.exists():
                    try:
                        sim_code = sim_proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        print("[launch] Simulator completed but ROS teardown "
                              "lingered; terminating the completed process.")
                        sim_proc.terminate()
                        try:
                            sim_proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            sim_proc.kill()
                            sim_proc.wait()
                        sim_code = 0
                    break
                time.sleep(0.1)
            else:
                sim_code = sim_proc.returncode
            print("[launch] Simulation finished. Waiting for controller...")

            # Allow the controller time to process the repeated stop signal.
            # Should a best-effort terminal sample still be dropped under a
            # heavily parallel sweep, the completed simulator is authoritative
            # and the idle controller is reaped, rather than valid metrics
            # being recorded as a failed run because a shutdown notification
            # was lost.
            try:
                ctrl_code = ctrl_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("[launch] Controller missed ROS stop; terminating after "
                      "successful simulator completion.")
                ctrl_proc.terminate()
                try:
                    ctrl_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    ctrl_proc.kill()
                    ctrl_proc.wait()
                ctrl_code = 0 if sim_code == 0 else ctrl_proc.returncode
            exit_code = sim_code or ctrl_code
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        cleanup()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
