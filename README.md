# Terrain- and latency-aware control on deformable terrain

This repository is the reproducible implementation and benchmark suite for
*Terrain- and Latency-Aware Shared and Autonomous Control on Deformable
Terrain* (ACMD 2026, `my_paper/acmd_fullpaper.tex`).

Every learned component of the control and estimation stack is supervised by
the controlled single-tire Chrono SCM rig, which records per-wheel SCM forces
against a commanded operating point and known soil parameters. The stack has
four parts behind fixed interfaces:

- **Control**: an acados NMPC whose internal tire model is the rig-trained
  static surrogate `nn_models/tire_force_static`, driving a torque-command
  plant (`simulation/control/`).
- **Estimation**: GRIT (backend identifier `grit`, contract
  `independent_n_phi_joint_profile`) infers the Bekker exponent `n` and the
  friction angle `phi` jointly from vehicle and inertial signals alone, by
  evaluating the `nn_models/tire_force_rate` force surrogate over gridded soil
  hypotheses — no pose datum, no terrain truth, no force or torque sensing. It
  publishes gated snapshots and otherwise holds a labelled low-grip prior
  (`simulation/estimators/`).
- **Safety**: a swappable filter layer — DOB-CBF, the deployed reactive
  default, and MPSF, the predictive alternative — with explicit one-way
  command-delay compensation (`simulation/safety/`).
- **Latency environment**: replayed traffic from the measurement-grounded
  N-HiTS 5G generator (`data/latency_profiles/5g_nhits_geforce.json`, traces in
  `data/5g_generated/`), applied to the command uplink and the camera
  downlink.

## Repository layout

| Path | Role |
| --- | --- |
| `simulation/` | The runtime, one package per swappable role: `control/` (NMPC, solver, speed profile), `estimators/` (GRIT and its comparison backends), `safety/` (DOB-CBF, MPSF), `runtime/` (launchers, plant node, transport), `scenarios/`, `sensors/`, `teleop/`, `tire_models/`, `shared/`. |
| `benchmarking/` | Everything that produces a number in the paper: the studies `run.py` orchestrates, the estimator-evidence chain, figure makers, the fail-closed publisher, and the provenance verifier. |
| `data/` | Tracked inputs the runtime reads: the rig training corpora, reference paths, the 5G traffic traces, and the latency profile that replays them. |
| `data_collection/` | How `data/` is produced: the C++ single-tire rig collectors, their build script, and the hash-pinned binaries that collected the deployed corpus. |
| `nn_training/` | How `nn_models/` is produced: the trainer and the wrappers that reproduce each checkpoint. |
| `nn_models/` | The three checkpoints the paper uses. `simulation/` loads them directly at runtime and the published evidence records their hashes, so they are inputs rather than build products. |
| `tests/` | Unit and contract tests, mirroring the package layout. |
| `data_sync/` | Off-machine snapshot tooling for the multi-GB raw result generations (see `DATA.md`). |
| `my_paper/` | The manuscript, kept in its own repository and checked out here (see below). |

The three stages read left to right: `data_collection/` produces `data/`,
`nn_training/` turns that into `nn_models/`, and `simulation/` consumes the
checkpoints while `benchmarking/` measures the result.

## Paper

The manuscript lives in its own private repository
(`ksha23/acmd-fullpaper`), checked out at `my_paper/` (this repository ignores
it). The figure pipeline and the verifiers expect that checkout to be present:

```bash
git clone git@github.com:ksha23/acmd-fullpaper.git my_paper
tectonic my_paper/acmd_fullpaper.tex   # 10 A4 pages including references
```

## Environment

The framework is Chrono-based: Chrono::Vehicle supplies the vehicle and SCM
plant, Chrono::Sensor the driver camera and IMU, and Chrono::ROS plus ROS 2
(`rclpy`/DDS) the paper process boundary. Paper launchers select ROS
explicitly; no alternate transport appears in the benchmark path. ROS Jazzy
needs Python 3.12, so use the `scm-terrain` conda environment:

```bash
source /opt/ros/jazzy/setup.bash
export PYTHONPATH="$PWD/third_party/chrono/build/bin:$PYTHONPATH"
export ACADOS_SOURCE_DIR="$PWD/third_party/acados"
export LD_LIBRARY_PATH="/opt/ros/jazzy/lib:$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH"
conda activate scm-terrain
```

`SETUP.md` covers the source builds (PyChrono from the vendored
`third_party/chrono` submodule, acados, ROS); `DATA.md` covers large-data
restoration. The estimator's preregistered design, its promotion evidence, and
the claim-by-claim provenance registry are maintained in the private
development repository and accompany the paper as supplementary material; the
scoring tools that default to a preregistration path take it explicitly with
`--preregistration`.

## Reproducing the paper

```bash
conda run -n scm-terrain python benchmarking/run.py --tier smoke
conda run -n scm-terrain python benchmarking/run.py --tier paper
conda run -n scm-terrain python benchmarking/make_paper_figures.py
conda run -n scm-terrain python benchmarking/verify_provenance_chain.py
```

The paper tier consists of 9 commands; its dry-run manifest
(`--tier paper --dry-run`) is the authority for the run count and the exact
command lines:

1. `tire_models` — neural surrogate versus calibrated Pacejka and TMeasy
   tracking (405 paired runs per arm);
2. `tire_estimator` — matched-terrain, GRIT, scalar-parent, and fixed
   low-grip-fallback conditioning of the same neural tire model;
3. `terrain_estimator` — the locked joint (n, phi) promotion evidence
   (`joint_n_phi_evidence.py`, replayed against SHA-256-identified frozen
   sensor traces, with truth joined only at scoring);
4. `speed_profile` — the static curvature reference versus the terrain-aware
   friction-circle (g–g) speed profile;
5. `grit_speed` — the GRIT terrain-adaptive speed matrix (Figure 3);
6. `safety` — the planner-blind and planner-aware native-contact safety matrix;
7. `convoy_cf` — paired convoy counterfactual replays;
8. `latency_awareness` — the delay-awareness dose response;
9. `teleop_battery` — the teleoperation failure-mode battery (Table 4,
   Figure 4).

Shorter forms:

```bash
python benchmarking/run.py --tier pilot
python benchmarking/run.py --tier paper --only terrain_estimator
python benchmarking/run.py --tier paper --dry-run
```

`make_paper_figures.py` regenerates the four manuscript figures and holds an
exact contract with every `\includegraphics` in the tex: it fails if either
side contains an unaccounted-for figure. Published CSV and figure provenance is
recorded in `my_paper/paper_figures/publish_manifest.json` (sources, hashes,
and transformations for every canonical artifact).

The paper tier, the publishers, and the statistics validation read the raw
result generations, which are restored from the data snapshot (`DATA.md`)
rather than tracked in git. Without that restore, a bare clone still rebuilds
the manuscript figures from the committed evidence with
`python benchmarking/make_paper_figures.py --from-published`, and
`verify_provenance_chain.py` remains fully self-contained.

Reproduction starts from the rig corpus, which is tracked in-repo:
`data/tire_rig_commanded` is DERIVED, collected end to end by the tracked
`data_collection/build_rig_cmd/` binaries and hash-matched by
`verify_provenance_chain.py`. Paper matrices use parallel workers with unique
ROS domain IDs; run one top-level ROS benchmark at a time.

## Human-in-the-loop demonstration

`./collect_hil.sh` runs the single-operator demonstration station: a Logitech
wheel, the Chrono::Sensor delayed point of view, a live HUD showing commanded
versus applied channels, uplink-delayed commands, and the same DOB-CBF and
MPSF filters as the paper studies. The runner is
`benchmarking/human_delay_compensation_rounds.py`; its scope and claims are
fixed by the preregistration that ships with the paper's evidence bundle.

## Learned models

| Checkpoint | Purpose | Training source |
| --- | --- | --- |
| `tire_force_static` | NMPC internal model, DOB-CBF, MPSF, speed profile (Table 1 neural arm) | `data/tire_rig_commanded` at the commanded operating point (DERIVED; `nn_training/train_deployed_surrogates.sh`) |
| `tire_force_rate` | Force surrogate GRIT evaluates over the (n, phi) grid (Table 2 joint row) | `data/tire_rig_commanded` at the commanded operating point (DERIVED; `nn_training/train_deployed_surrogates.sh`) |
| `tire_force_static_parent` | Scalar-parent estimator's own force model for live scalar-arm runs and UKF references (the Table 2 parent *replay* evaluates `tire_force_rate`, matched to GRIT) | `data/tire_rig_static/train.csv` (PRIMARY; `nn_training/train_scalar_parent.sh`) |

## Collision metric

`collisions` counts unique logical obstacles for which Chrono reports an
ego-body–obstacle-body contact. `min_clearance_m` and `near_misses` are
geometric diagnostics and never synthesize contact truth.
