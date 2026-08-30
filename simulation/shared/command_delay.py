"""Simulation-time command delay queue with latest-issue-wins release.

The teleoperation studies delay the command channel by a per-sample latency
drawn from the 5G profile. Two properties define a faithful receiver, and both
previously failed in the inline buffer implementations:

* **Simulation-time indexing.** Due times were wall-clock. Real-time pacing
  only sleeps when the simulation is ahead, so whenever the simulation fell
  behind real time the effective delay in simulation time shrank below the
  labelled value -- a "fixed 0.30 s" cell delivered roughly half that in the
  cells whose scenario ran slowest. Due times here are in the caller's clock,
  and the caller passes simulation time, so the delay under test is exact
  regardless of pacing and stays synchronized with the simulation-time-indexed
  camera channel.

* **Latest-issue-wins.** Under per-sample jitter, an older command with a
  larger delay can come due after a newer command with a smaller one. A real
  receiver that keys on sequence numbers never applies the older command; the
  previous sort-by-due-time buffers did, rewinding the actuated command and
  biasing the realized delay toward the upper envelope of the jitter. Here an
  entry is dropped, not applied, when a newer-issued entry has already been
  released.

The queue never drops the newest entry: across a gap (an outage), the last
command issued before the gap is eventually released and then held by the
caller (hold-last semantics are the caller's).
"""
from __future__ import annotations

import heapq
from typing import Any, Optional


class CommandDelayBuffer:
    """Delay queue: push(now + delay, payload); pop_latest(now) to actuate."""

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, Any]] = []
        self._seq = 0
        self._last_released_seq = -1

    def __len__(self) -> int:
        return len(self._heap)

    def push(self, due_time: float, payload: Any) -> None:
        """Enqueue a command issued now, deliverable at ``due_time``."""
        self._seq += 1
        heapq.heappush(self._heap, (float(due_time), self._seq, payload))

    def pop_latest(self, now: float) -> Optional[Any]:
        """Release every entry due by ``now``; return the newest-issued one.

        Entries issued before an already-released entry are discarded
        (latest-issue-wins). Returns None when nothing new is deliverable,
        in which case the caller holds the last applied command.
        """
        released: Optional[Any] = None
        while self._heap and self._heap[0][0] <= now:
            _, seq, payload = heapq.heappop(self._heap)
            if seq > self._last_released_seq:
                self._last_released_seq = seq
                released = payload
        return released
