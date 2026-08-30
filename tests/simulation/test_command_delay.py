"""Contract of the simulation-time command delay queue.

These are the first tests the delay-application machinery has had: the paper's
"the delay under test is exact and pacing-independent" claim rests on the
buffer semantics pinned here.
"""
from __future__ import annotations

import unittest

from simulation.shared.command_delay import CommandDelayBuffer


class CommandDelayBufferTest(unittest.TestCase):
    def test_exact_delay_in_the_callers_clock(self):
        buf = CommandDelayBuffer()
        buf.push(0.30, "a")                     # issued at t=0, delay 0.30
        self.assertIsNone(buf.pop_latest(0.299))
        self.assertEqual(buf.pop_latest(0.300), "a")

    def test_nothing_new_returns_none_for_hold_last(self):
        buf = CommandDelayBuffer()
        buf.push(0.1, "a")
        self.assertEqual(buf.pop_latest(0.1), "a")
        self.assertIsNone(buf.pop_latest(0.2))  # caller holds "a"

    def test_downward_jitter_does_not_rewind_to_older_command(self):
        # Old command with a 0.4 s delay, newer command with 0.1 s: the newer
        # one comes due first, and when the older one's due time passes it
        # must be discarded, not applied over the newer one.
        buf = CommandDelayBuffer()
        buf.push(0.40, "old")    # issued t=0.00, delay 0.40
        buf.push(0.15, "new")    # issued t=0.05, delay 0.10
        self.assertEqual(buf.pop_latest(0.15), "new")
        self.assertIsNone(buf.pop_latest(0.45))  # "old" dropped, hold "new"

    def test_batch_release_yields_newest_issued(self):
        # After an outage every queued command comes due at once; the receiver
        # applies only the newest-issued one.
        buf = CommandDelayBuffer()
        buf.push(2.0, "first")
        buf.push(2.1, "second")
        buf.push(2.2, "third")
        self.assertEqual(buf.pop_latest(5.0), "third")
        self.assertEqual(len(buf), 0)

    def test_upward_jitter_preserves_order(self):
        buf = CommandDelayBuffer()
        buf.push(0.10, "a")
        buf.push(0.20, "b")
        self.assertEqual(buf.pop_latest(0.10), "a")
        self.assertEqual(buf.pop_latest(0.20), "b")

    def test_same_due_time_resolves_by_issue_order(self):
        buf = CommandDelayBuffer()
        buf.push(0.10, "a")
        buf.push(0.10, "b")
        self.assertEqual(buf.pop_latest(0.10), "b")


if __name__ == "__main__":
    unittest.main()
