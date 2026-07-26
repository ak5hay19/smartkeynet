"""
env/deferral_queue.py

Priority deferral queue for hybrid-mandatory requests the pool can't
currently cover (PLAN.md Addition C; implements Hard Rule 9). Owned by
Person B (split.md §1).

Hard Rule 9: pool exhaustion never causes a downgrade. A hybrid-
mandatory request that can't be served is queued here -- priority by
sensitivity class, FIFO within class -- until the pool refills enough
to cover it. It is never served below its floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from env.contracts import DeferredCriticalStep, RegretEvent, Request


@dataclass
class QueuedRequest:
    """A `Request` waiting in the deferral queue, plus queueing metadata."""

    request: Request
    bits_required: float
    step_enqueued: int
    steps_waited: int = 0


@dataclass
class DeferralQueue:
    """Priority-by-sensitivity-class, FIFO-within-class deferral queue.

    Each queued request accrues latency per step (Addition C) and is
    served automatically once `env/pool_sim.py` can cover its draw.
    Never downgraded (Hard Rule 9).
    """

    _queued: list[QueuedRequest] = field(default_factory=list)

    def enqueue(self, request: Request, bits_required: float, step: int) -> RegretEvent:
        """Queue a request the pool can't currently cover.

        Returns the `RegretEvent` marking this deferral's onset
        (Addition C event log).
        """
        raise NotImplementedError

    def tick(self, step: int) -> list[DeferredCriticalStep]:
        """Advance one step: age every queued request by one step.

        Returns one `DeferredCriticalStep` log entry per still-queued
        request (Addition C).
        """
        raise NotImplementedError

    def pop_servable(self, can_draw: Callable[[float], bool]) -> list[QueuedRequest]:
        """Pop and return all queued requests that `can_draw(bits_required)`
        now covers.

        Priority order: highest sensitivity class first, FIFO within
        class (Addition C unit test: "queue priority/FIFO ordering").
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._queued)
