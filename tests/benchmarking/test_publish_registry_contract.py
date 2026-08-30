"""Contracts binding the publish registry to the orchestrator that feeds it.

The publisher fails closed when the orchestrator's source map does not bind a
registered prefix, so a prefix added to the registry without a command that
produces it turns ``run.py --tier paper`` into a matrix that runs for hours and
then fails at the publish step. Nothing else in the suite exercises that pair,
which is how it happened once.

The simulator-flag contract is the same shape of problem. ``HIL_SIM_EXTRA`` is
substituted wholesale rather than extended, so a command that names its own
flags silently drops the deployed stack -- a different powertrain, a different
disturbance observer -- while every result column continues to look identical.
A study collected that way is not comparable with the ones it is printed
beside.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarking"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # dataclass decorators resolve their module through sys.modules, so the
    # module must be registered before it executes.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run = _load("run")
publish = _load("publish_paper_figures")


class TestEveryPublishedPrefixHasAProducer(unittest.TestCase):
    def test_registry_prefixes_are_all_produced_by_run_py(self):
        produced = set(run.RESULT_PREFIX.values())
        registered = set(publish.SPECS)
        orphaned = sorted(registered - produced)
        self.assertEqual(
            orphaned, [],
            "publish_paper_figures.SPECS registers a prefix that no run.py "
            "command produces, so --tier paper fails at the publish step "
            "after running its whole matrix: " + ", ".join(orphaned))

    def test_every_registered_prefix_has_a_row_count(self):
        missing = sorted(set(publish.SPECS) - set(publish.PAPER_ROWS))
        self.assertEqual(missing, [], f"PAPER_ROWS is missing {missing}")

    def test_paper_tier_emits_every_producing_command(self):
        arguments = run.parse_args.__globals__["argparse"].Namespace(
            tier="paper", only=[], workers=6, base_port=7800,
            port_stride=40, dry_run=True)
        emitted = {command.name for command in run.build_commands(arguments)}
        needed = {name for name, prefix in run.RESULT_PREFIX.items()
                  if prefix in publish.SPECS}
        self.assertEqual(
            sorted(needed - emitted), [],
            "a command that produces a published prefix is absent from the "
            "paper tier")


class TestDeployedSimulatorFlags(unittest.TestCase):
    """Commands that override HIL_SIM_EXTRA must still carry the deployed stack."""

    #: Commands whose plant deliberately differs, with the reason.
    EXEMPT = {
        "tire_models": "compares tire models with the DOB active on every arm",
        "tire_models_calibrated":
            "same tire-comparison plant as tire_models, by design",
        "terrain_estimator": "hash-verified evidence matrix",
        "teleop_battery": "declares its own 5G profile",
    }

    def test_overrides_begin_with_the_deployed_flags_verbatim(self):
        """Token containment is not enough: `--dob-ki 5` passes a token check
        because `--dob-ki` and a `0` borrowed from another pair are both
        present, and argparse is last-wins so a trailing override also
        passes. The deployed prefix must appear verbatim, in order, at the
        start, so no value can be changed and no trailing flag can override
        one inside it."""
        for name, extra in run._SIM_EXTRA_BY_COMMAND.items():
            if name in self.EXEMPT or extra is None:
                continue
            with self.subTest(command=name):
                self.assertTrue(
                    extra.startswith(run._DEPLOYED_EXTRA),
                    f"{name} overrides HIL_SIM_EXTRA without the deployed "
                    f"stack as its verbatim prefix; the substitution is "
                    f"wholesale, so the study would run a different plant "
                    f"from the ones its numbers are read against")
                tail = extra[len(run._DEPLOYED_EXTRA):]
                for flag in ("--dob-ki", "--dob-max", "--simple-powertrain",
                             "--ff-drag-surrogate"):
                    self.assertNotIn(
                        flag, tail,
                        f"{name} repeats {flag} after the deployed prefix; "
                        f"argparse is last-wins, so the repeat overrides the "
                        f"deployed value")

    def test_exemptions_are_declared_not_implicit(self):
        for name in self.EXEMPT:
            self.assertIn(name, run._SIM_EXTRA_BY_COMMAND,
                          f"{name} is exempted but not configured")

    def test_none_is_not_a_silent_exemption_for_published_commands(self):
        """`None` deletes HIL_SIM_EXTRA entirely -- the full nonlinear
        powertrain, not the deployed stack -- which is the most consequential
        silent change available. Any command that produces a published prefix
        and carries `None` must be a declared exemption."""
        publishing = {name for name, prefix in run.RESULT_PREFIX.items()
                      if prefix in publish.SPECS}
        for name, extra in run._SIM_EXTRA_BY_COMMAND.items():
            if extra is None and name in publishing:
                self.assertIn(
                    name, self.EXEMPT,
                    f"{name} produces a published prefix and deletes "
                    f"HIL_SIM_EXTRA without a declared exemption")

    def test_live_grit_integration_carries_the_deployed_stack(self):
        extra = run._SIM_EXTRA_BY_COMMAND["live_grit_integration"]
        self.assertTrue(extra.startswith(run._DEPLOYED_EXTRA))
        for flag in ("--terrain-estimator", "--teleop-delay"):
            self.assertIn(flag, extra)



class TestManifestKeysAreUnique(unittest.TestCase):
    """A repeated key makes the whole manifest unreadable, not just shadowed.

    ``publish_paper_figures.read_key_value_manifest`` returns None for any
    manifest with a duplicated key, so a study that records provenance of its
    own and then has the same keys appended becomes unpublishable -- with a
    row-count error that says nothing about the real cause.
    """

    def test_write_manifest_yields_no_duplicate_keys(self):
        import argparse
        import csv as _csv
        import collections
        import tempfile

        common = _load("common")
        # Arguments deliberately carrying the provenance key names, which is
        # how the estimator ablation records its own contract.
        args = argparse.Namespace(
            code_git_head="deadbeef", tracked_worktree_dirty=False,
            uncommitted_source_files=[], paper_evidence_eligible=True,
            some_other_option=3,
        )
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            common.write_manifest(out, args, "unit test")
            keys = [row[0] for row in _csv.reader((out / "manifest.csv").open())
                    if len(row) > 1]
        repeated = [k for k, n in collections.Counter(keys).items() if n > 1]
        self.assertEqual(repeated, [], f"duplicate manifest keys: {repeated}")

    def test_measured_provenance_beats_caller_supplied_values(self):
        """An argparse dest sharing a provenance key name must not be able to
        forge eligibility; the measured value wins."""
        import argparse
        import ast as _ast
        import csv as _csv
        import tempfile

        common = _load("common")
        args = argparse.Namespace(paper_evidence_eligible="caller-value",
                                  code_git_head="deadbeef")
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            common.write_manifest(out, args, "unit test")
            rows = {r[0]: r[1] for r in _csv.reader((out / "manifest.csv").open())
                    if len(r) > 1}
        self.assertNotEqual(rows["paper_evidence_eligible"], "'caller-value'")
        self.assertIn(_ast.literal_eval(rows["paper_evidence_eligible"]),
                      (True, False))
        self.assertNotEqual(rows["code_git_head"], "'deadbeef'")

if __name__ == "__main__":
    unittest.main()
