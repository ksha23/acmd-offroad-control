"""Contracts for sensor-noise reproducibility and truth isolation in the plant.

A seeded noise stream must replay identically for a benchmark cell to be
comparable across arms, so these tests establish that the IMU noise depends
only on its seed and accumulates bias deterministically. They also establish
that the no-tire-force plant mode cannot reconstruct the truth payload it is
meant to withhold, even under noise.
"""

from types import SimpleNamespace

import numpy as np

from chrono_sim_node import (
    ReproducibleIMUNoise,
    _attach_truth_pose_to_force_diagnostics,
    update_safety_terrain_from_command,
)


NOISE_KWARGS = dict(
    update_rate=100.0,
    acc_stdev=0.015,
    acc_bias_drift=1e-4,
    acc_tau_drift=100.0,
    gyro_stdev=0.001,
    gyro_bias_drift=5e-6,
    gyro_tau_drift=500.0,
)


def _sequence(seed: int, samples: int = 100):
    noise = ReproducibleIMUNoise(np.random.default_rng(seed), **NOISE_KWARGS)
    return np.asarray([
        np.concatenate((noise.add_accel((0.0, 0.0, 0.0)),
                        noise.add_gyro((0.0, 0.0, 0.0))))
        for _ in range(samples)
    ])


def test_reproducible_imu_noise_replays_same_seed_exactly():
    assert np.array_equal(_sequence(42), _sequence(42))


def test_reproducible_imu_noise_changes_with_seed():
    assert not np.array_equal(_sequence(42), _sequence(43))


def test_reproducible_imu_noise_accumulates_bias_state():
    noise = ReproducibleIMUNoise(
        np.random.default_rng(7),
        **{**NOISE_KWARGS, "acc_stdev": 0.0, "gyro_stdev": 0.0},
    )
    first_acc = noise.add_accel((0.0, 0.0, 0.0))
    second_acc = noise.add_accel((0.0, 0.0, 0.0))
    assert np.any(first_acc != 0.0)
    assert not np.array_equal(first_acc, second_acc)


def test_no_tire_force_mode_cannot_recreate_truth_payload_under_noise():
    pose = SimpleNamespace(x=1.0, y=2.0)
    rotation = SimpleNamespace(e0=1.0, e1=0.0, e2=0.0, e3=0.0)
    velocity = SimpleNamespace(x=3.0)

    payload = _attach_truth_pose_to_force_diagnostics(
        None, noise={"x": 0.1}, pos=pose, rot=rotation, vel_loc=velocity
    )

    assert payload is None


class _SafetySpy:
    def __init__(self):
        self.updates = []

    def update_terrain(self, terrain):
        self.updates.append(terrain)


def test_safety_terrain_metadata_advances_once_when_command_is_accepted():
    safety = _SafetySpy()
    command = SimpleNamespace(
        terrain_n=0.93,
        terrain_update_seq=7,
        terrain_Kphi=2.1e6,
        terrain_Kc=3.2e4,
        terrain_c=1200.0,
        terrain_phi_deg=25.0,
        terrain_k=0.02,
    )

    sequence, applied = update_safety_terrain_from_command(safety, command, 6)
    duplicate_sequence, duplicate_applied = update_safety_terrain_from_command(
        safety, command, sequence
    )

    assert (sequence, applied) == (7, True)
    assert (duplicate_sequence, duplicate_applied) == (7, False)
    assert len(safety.updates) == 1
    terrain = safety.updates[0]
    assert terrain["n"] == 0.93
    assert terrain["phi"] == 25.0
