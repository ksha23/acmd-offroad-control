"""Contracts binding a benchmark row to the launch it requested.

Parallel workers share a finite pool of DDS domains, so a row could in
principle record a simulation that ran under another cell's configuration.
These tests establish that the launch-identity check reads the child's own
log and accepts a row only when path, speed, seed, and ports match the
request, rejecting both mismatched and ambiguous receipts.
"""

from pathlib import Path

from benchmarking.paper_provenance import launch_identity_contract


def _write_log(directory: Path, *, path: str = "lane_change", speed: float = 7.0):
    directory.mkdir()
    (directory / "run.log").write_text(
        "\n".join([
            "[launch] ROS_DOMAIN_ID=78 (from sim-port 22904)",
            "  Subscribing to state from localhost:22904 (ros)",
            f"  Path: {path}, v_target: {speed} m/s",
            "  Publishing controls on port 22905 (ros)",
            "  Simulation seed: 402 (state + IMU streams)",
            "  Publishing state on port 22904 (ros)",
            "  Subscribing to controls from localhost:22905 (ros)",
        ]) + "\n"
    )


def test_launch_identity_accepts_exact_child_receipt(tmp_path):
    run = tmp_path / "exact"
    _write_log(run)
    result = launch_identity_contract(
        run,
        expected_path="lane_change",
        expected_speed_mps=7.0,
        expected_seed=402,
        expected_sim_port=22904,
        expected_ctrl_port=22905,
    )
    assert result["launch_identity_match"] is True


def test_launch_identity_rejects_cross_talk_configuration(tmp_path):
    run = tmp_path / "cross_talk"
    _write_log(run, path="sinusoidal", speed=9.0)
    result = launch_identity_contract(
        run,
        expected_path="lane_change",
        expected_speed_mps=7.0,
        expected_seed=402,
        expected_sim_port=22904,
        expected_ctrl_port=22905,
    )
    assert result["launch_identity_match"] is False
    assert result["observed_path"] == "sinusoidal"
    assert result["observed_speed_mps"] == 9.0


def test_launch_identity_rejects_ambiguous_receipts(tmp_path):
    run = tmp_path / "ambiguous"
    _write_log(run)
    with (run / "run.log").open("a") as stream:
        stream.write("  Path: right_left, v_target: 5.0 m/s\n")
    result = launch_identity_contract(
        run,
        expected_path="lane_change",
        expected_speed_mps=7.0,
        expected_seed=402,
        expected_sim_port=22904,
        expected_ctrl_port=22905,
    )
    assert result["launch_identity_match"] is False

