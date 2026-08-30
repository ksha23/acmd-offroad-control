#!/usr/bin/env bash
# Single-operator human-in-the-loop DEMONSTRATION for the ACMD paper.
# Scope and claims are fixed by the preregistration in the paper's evidence
# bundle: this run demonstrates that the full human path (physical wheel ->
# uplink delay -> safety filter -> plant -> delayed POV) composes end to end;
# it supports no human-behavior generalization.
#
#   ./collect_hil.sh                              # operator id defaults to op01
#   HIL_OPERATOR_ID=pilot01 ./collect_hil.sh      # anonymized id override
#   ./collect_hil.sh --dry-run                    # write/inspect plans only
#
# Design: three excluded familiarization rounds, then four reproducibly
# randomized complete blocks. Each block contains every cell in
#   {none, DOB-CBF, MPSF} x {lead_brake, cut_in, stalled, oncoming}
# once, for 48 recorded study trials. Corresponding filter-off/on cells share
# the operator, scenario, statistical latency trace, terrain, and Chrono seed.
# Single operator: repetitions quantify within-operator variability and must
# not be presented as independent participants.
#
# The sensor-camera POV receives the profile's camera/downlink latency while
# manual commands receive its control/uplink latency; the plant is the
# deployed torque-command powertrain (--simple-powertrain, injected by the
# runner) so trials are comparable to the published arms. Native Chrono body
# contact is collision truth; geometric clearance stays a diagnostic and, at
# the 0.12 m interactive grid, supports no ordering claims.
# Familiarization logs live under practice/ and are excluded from results.csv.
# Recorded study traces live under raw/ and can also be replayed with
# benchmarking/convoy_counterfactual_eval.py for identical-intent analysis.
#
# Results land in benchmarking/results/human_delay_compensation_rounds_<ts>/.
set -o pipefail   # NOT -u: conda's (de)activate scripts reference unset vars

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- environment (scm-terrain + the PyChrono build + acados; see SETUP.md) ---
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate scm-terrain 2>/dev/null || true   # already-active env can trip a re-activate
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ERROR: ROS 2 Jazzy setup not found at /opt/ros/jazzy/setup.bash" >&2
  exit 2
fi
source /opt/ros/jazzy/setup.bash
PYTHON_EXE="$HOME/miniconda3/envs/scm-terrain/bin/python"
CHRONO_ROOT="$HOME/Documents/sbel/chrono_fork"
CHRONO_BUILD="$CHRONO_ROOT/build"
export PYTHONPATH="$CHRONO_BUILD/bin${PYTHONPATH:+:$PYTHONPATH}"
export CHRONO_DATA_DIR="$CHRONO_ROOT/chrono/data/"
export ACADOS_SOURCE_DIR="$HOME/Documents/sbel/acados"
export LD_LIBRARY_PATH="$CHRONO_BUILD/lib:$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DISPLAY="${DISPLAY:-:0}"   # use the desktop's display (G29 + window)

# --- preflight: do not silently record a session of zero-input trials ---
DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  fi
done
if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "ERROR: scm-terrain Python not found: $PYTHON_EXE" >&2
  exit 2
fi
if [[ ! -d "$CHRONO_BUILD/bin/pychrono" ]]; then
  echo "ERROR: PyChrono build not found: $CHRONO_BUILD/bin/pychrono" >&2
  exit 2
fi
if ! "$PYTHON_EXE" -c 'import rclpy, pychrono.ros' >/dev/null 2>&1; then
  echo "ERROR: default paper transport requires rclpy and pychrono.ros." >&2
  echo "Rebuild PyChrono with -DCH_ENABLE_MODULE_ROS=ON (see SETUP.md)." >&2
  exit 2
fi
if [[ ! -f "data/latency_profiles/5g_nhits_geforce.json" ]]; then
  echo "ERROR: paper latency profile is missing." >&2
  exit 2
fi
JOYSTICK_DEV="$(compgen -G '/dev/input/js*' | head -n 1)"
if [[ $DRY_RUN -eq 0 ]] && [[ -z "$JOYSTICK_DEV" ]]; then
  echo "ERROR: no joystick device (/dev/input/js*) detected." >&2
  echo "Plug in and power the Logitech wheel (G29/G923) before collecting." >&2
  exit 2
fi
# The POV camera needs a working NVIDIA stack (OptiX + GLX). A driver/
# library mismatch (e.g., unattended upgrade under a running session)
# kills every round instantly with a GLX BadValue.
if ! nvidia-smi -L > /dev/null 2>&1; then
  echo "ERROR: NVIDIA driver not usable ($(nvidia-smi 2>&1 | head -1))." >&2
  echo "Most common cause: driver updated under the running kernel module." >&2
  echo "Fix: REBOOT, then rerun this script." >&2
  if [[ $DRY_RUN -eq 0 ]]; then exit 2; fi
  echo "(dry run continues despite the driver state)" >&2
fi
# Existence is not readability: SDL reads /dev/input/event*, which is
# root:input. Without membership or a logind ACL the wheel is invisible to
# pygame and every trial would record zero input. Probe DEFINITIVELY.
if [[ $DRY_RUN -eq 0 ]]; then
  if ! "$PYTHON_EXE" - << 'PYCHECK'
import sys, time
import pygame
pygame.joystick.init(); pygame.init()
n = pygame.joystick.get_count()
if n == 0:
    print("ERROR: SDL sees no joysticks (permission problem, not absence).", file=sys.stderr)
    print("Fix once:  sudo usermod -aG input $USER   then log out and back in", file=sys.stderr)
    print("(or run this script via:  sg input -c ./collect_hil.sh )", file=sys.stderr)
    sys.exit(1)
js = None
for i in range(n):
    j = pygame.joystick.Joystick(i); j.init()
    print(f"  joystick [{i}]: {j.get_name()} (axes={j.get_numaxes()})")
    if any(k in j.get_name().upper() for k in ("G29", "G27", "G920", "G923", "LOGITECH")):
        js = j
if js is None:
    print("ERROR: no Logitech wheel among visible joysticks.", file=sys.stderr)
    sys.exit(1)
print("\n  TURN THE WHEEL a quarter turn within 8 seconds to confirm input...")
t0 = time.time(); base = js.get_axis(0); moved = False
while time.time() - t0 < 8.0:
    pygame.event.pump()
    if abs(js.get_axis(0) - base) > 0.10:
        moved = True; break
    time.sleep(0.02)
if not moved:
    print("ERROR: wheel visible but no steering motion detected.", file=sys.stderr)
    sys.exit(1)
print("  Wheel input confirmed.")
PYCHECK
  then
    echo "ERROR: wheel liveness check failed; not starting the session." >&2
    exit 2
  fi
fi
CONTROLLER_MODEL="not_connected_dry_run"
if [[ -n "$JOYSTICK_DEV" ]]; then
  CONTROLLER_MODEL="$(udevadm info --query=property --name="$JOYSTICK_DEV" 2>/dev/null \
    | sed -n 's/^ID_MODEL=//p' | head -n 1)"
  CONTROLLER_MODEL="${CONTROLLER_MODEL:-unknown_logitech_wheel}"
fi

echo "Starting ACMD single-operator HIL demonstration (Logitech wheel + Chrono::Sensor + live HUD)."
echo "Input device: $CONTROLLER_MODEL${JOYSTICK_DEV:+ ($JOYSTICK_DEV)}"
echo "3 excluded practice rounds + 48 study rounds in 4 randomized blocks."
echo "Take the prompted breaks. Do not use a personal name as the operator id."
echo "Confirm any required PI/institutional human-subjects determination before collection."
echo

LATENCY_PROFILE="data/latency_profiles/5g_nhits_geforce.json"

# Trials are short enough to avoid operator fatigue while retaining the
# configuration of the paper's recorded manual traces: a 40 m / 25 s clay course
# with five rocks at a 5 m/s reference, which is the tire matrix's lowest
# setpoint and the estimator evidence's maneuver speed. The single-traffic-
# vehicle scenarios (lead_brake, cut_in, stalled) hold real time with the sensor
# POV and match the recorded reckless-trace set. They are not the paper's convoy
# counterfactual set (double_cut, jam, oncoming), whose two-vehicle scenes do
# not hold real time with rendering.
exec "$PYTHON_EXE" benchmarking/human_delay_compensation_rounds.py \
    --study-id acmd_single_operator_hil_demo_v2 \
    --operator-id "${HIL_OPERATOR_ID:-op01}" \
    --input-device-model "$CONTROLLER_MODEL" \
    --manual-mode g29 \
    --vis-mode sensor \
    --live-hud \
    --latency-profile-json "$LATENCY_PROFILE" \
    --convoy lead_brake cut_in stalled oncoming \
    --filters none dob_cbf mpsf \
    --terrains clay \
    --paths straight \
    --goal-distance 40 \
    --time 25 \
    --rocks 5 \
    --rock-min-spacing 6.0 \
    --rock-centerline-clear 3.0 \
    --rock-spawn-clear 8.0 \
    --rock-size 0.5 1.4 \
    --speeds 5 \
    --bumpiness 0 \
    --practice-rounds 3 \
    --order randomized-blocks \
    --order-seed 20260719 \
    --base-seed 910 \
    --rounds 4 \
    "$@"
