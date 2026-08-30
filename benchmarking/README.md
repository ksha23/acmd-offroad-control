# Benchmarking

`run.py` is the suite orchestrator. Its 9-command paper tier contains only
experiments that produce a figure, table, or number in
`my_paper/acmd_fullpaper.tex`, and every learned force query resolves to a
checkpoint under `nn_models/tire_force_*`. Paper launchers select ROS 2 through
Chrono::ROS explicitly; run them in the Python 3.12 `scm-terrain` environment.

```bash
python benchmarking/run.py --tier smoke
python benchmarking/run.py --tier pilot
python benchmarking/run.py --tier paper
python benchmarking/run.py --tier paper --dry-run   # authoritative plan
```

Each Chrono-driving sweep writes a timestamped directory under
`benchmarking/results/` and updates its CSV incrementally. The files tracked
under `results/` are the frozen evidence behind quoted numbers: the
teleoperation failure-mode battery, the tire-model sweep results and summaries,
the convoy per-cell summary, the joint-estimator scope decision with its
collection and replay tables, the Pacejka fit, the mesh and step refinement
sessions, the numerical-convergence, drag-feedforward, and predictive-filter
probes, and the human-in-the-loop demonstration session. Everything else under
`results/` is untracked and regenerable.

## Script map

- **Paper-tier studies** (one per `run.py` command): `mpc_tire_model_sweep`,
  `tire_model_with_estimator_ablation`, `joint_n_phi_evidence`,
  `speed_profile_ablation`, `grit_adaptive_speed_matrix`,
  `safety_filter_sweep`, `convoy_counterfactual_eval`,
  `latency_awareness_ablation`, `teleop_failure_modes`.
- **Estimator evidence chain**: `terrain_estimator_trace` (the sensor-only
  trace contract), `collect_terrain_estimator_traces`,
  `terrain_estimator_replay`, `develop_joint_estimator`,
  `score_joint_estimator`, `score_joint_evidence`,
  `active_estimator_diagnostics`, `profile_live_statistics`. Traces exclude
  plant truth by construction; truth joins only at scoring.
- **Preregistered scorers** (pinned by name in the paper's evidence bundle):
  `score_promotion_endpoints`, `score_roughness_scope`,
  `score_latency_ablation`.
- **Figures and publication**: `make_paper_figures` (the four-figure tex
  contract), `publish_paper_figures` (the fail-closed publisher, which holds to
  the committed manifest selection), `publish_teleop_scenario_traces`,
  `make_fig_*`, `render_topdown`, `paper_style`.
- **Calibration for the analytical arms** (Table 1):
  `calibrate_analytical_tires`, `calibrate_motion_resistance`,
  `collect_drag_calib_logs`, `fit_rigfit_pacejka`.
- **Section 5 reproduction**: `dallas_twin_repro`,
  `dallas_protocol_cross_eval`, over
  `simulation/estimators/ukf_reference_models.py`.
- **Framework-claim probes** (numbers quoted in the tex, run outside the tier):
  `numerical_convergence` (the 2.7 and 0.5 m refinement deviations),
  `mpsf_solve_scaling` (predictive-filter solve cost), `ff_drag_ablation` (the
  +0.067 drag-feedforward credit).
- **Environment and human-in-the-loop**: `train_5g_nhits` (provenance of the
  `data/5g_generated` traffic), `human_delay_compensation_rounds` (the
  demonstration runner behind `collect_hil.sh`).
- **Verifiers and shared contracts**: `verify_provenance_chain`,
  `paper_provenance`, `common`.

Tests live in `tests/benchmarking/`.

## Terrain-estimator evidence and ROS isolation

The deployed estimator is GRIT (backend identifier `grit`,
contract `independent_n_phi_joint_profile`), which evaluates the
`nn_models/tire_force_rate` surrogate over independent (n, phi) hypotheses.
Evidence discipline: collect estimator-disabled traces, replay them against
SHA-256-identified sensor streams, and join plant truth only in the scoring
step. `joint_n_phi_evidence.py` validates the locked promotion result end to
end. The scalar parent (`scalar_parent`) is replayed on the same traces as
the Table 2 comparison row; in that replay it evaluates the same
`nn_models/tire_force_rate` surrogate GRIT uses -- the matched-model,
fairer comparison -- while its own frozen checkpoint
`nn_models/tire_force_static_parent` serves the *live* scalar-arm runs and
the UKF reference models, not the Table 2 replay.

Run only one top-level ROS benchmark at a time. The affected scripts hold an
exclusive lease, batch workers before DDS domain IDs can wrap, and reject rows
whose logged path, speed, seed, domain, or ports differ from the request.

## Publication contract

```bash
python benchmarking/make_paper_figures.py
python benchmarking/verify_provenance_chain.py
```

`make_paper_figures.py` regenerates exactly the four figures the ACMD source
includes and fails on any mismatch with the tex. The publisher accepts only
generations meeting the row-count, expected-variant, estimator-backend,
truth-isolation, and collision-source contracts, and records hashes plus the
selected source directories in
`my_paper/paper_figures/publish_manifest.json`.
