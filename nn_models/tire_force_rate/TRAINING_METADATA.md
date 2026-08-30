# tire_force_rate — training metadata

| | |
| --- | --- |
| Architecture | rate-mode MLP, hidden = [64, 32], 3106 params, tanh |
| Inputs (14) | `slip_ratio`, `slip_angle`, `velocity`, `vertical_load`, `steering_rate`, `d_slip_ratio`, `d_slip_angle`, `d_velocity`, `bekker_Kphi`, `bekker_Kc`, `bekker_n`, `mohr_cohesion`, `mohr_friction`, `janosi_shear` |
| Outputs | per-wheel tire-frame `(Fx, Fy)` |
| Operating point | commanded (rig command at the measurement instant) |
| Normal load | commanded |
| Random-row test | R2 Fx = 0.9913, R2 Fy = 0.9884, MAE Fx = 86.5 N, MAE Fy = 48.7 N |

## Training data

- **Source**: Chrono SCM single-tire rig, `data_collection/collect_rate_data.cpp`.
- **CSV**: `data/tire_rig_commanded/train.csv` (31,495 rows after validation).
- **CSV SHA-256**: `17bd30b5464e49460c7d9a149fc7cec6014342f23477e6a5dcffca420feae31e`.
- **Seed**: 42.

## Independent holdout

- **CSV**: `/home/ksha/Documents/sbel/offroad-control/data/tire_rig_commanded/holdout.csv` (6,223 rows).
- **SHA-256**: `f7267be00019d3e38731aee96c313d31ffd933b8dfb0fc0b7af421b3d8827266`.
- **Scores**: R2 Fx = 0.9623, R2 Fy = 0.4691, MAE Fx = 212.1 N, MAE Fy = 526.6 N.
