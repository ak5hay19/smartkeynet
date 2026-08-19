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

from collections.abc import Callable
from dataclasses import dataclass, field

from env.contracts import Action, DeferredCriticalStep, RegretEvent, Request


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

    def enqueue(
        self,
        request: Request,
        bits_required: float,
        step: int,
        pool_fill_at_onset: float,
    ) -> RegretEvent:
        """Queue a request the pool can't currently cover.

        Returns the `RegretEvent` marking this deferral's onset
        (Addition C event log). Called once per request, at the moment
        it first can't be covered -- never re-emitted by `tick()`.

        Only hybrid-mandatory requests land here (Hard Rule 9), so the
        floor recorded on the `RegretEvent` is always
        `Action.SERVE_HYBRID` -- the request's `sensitivity_class` and
        floor are carried through unchanged, never lowered.
        """
        queued = QueuedRequest(
            request=request,
            bits_required=bits_required,
            step_enqueued=step,
        )
        self._queued.append(queued)
        return RegretEvent(
            step=step,
            request_id=request["request_id"],
            tenant=request["tenant"],
            sensitivity_class=request["sensitivity_class"],
            policy_floor=int(Action.SERVE_HYBRID),
            pool_fill_at_onset=pool_fill_at_onset,
        )

    def sla_breaches(self, sla_max_steps: int) -> list[QueuedRequest]:
        """Queued requests that have waited past the SLA.

        §S2: "if r.wait_steps > sla_max_steps: emit sla_breach  # still NEVER
        downgraded". A breach is a reportable availability failure, not a
        licence to serve below the floor -- there is deliberately no code path
        here that changes a request's floor or removes it from the queue.
        """
        return [queued for queued in self._queued if queued.steps_waited > sla_max_steps]

    @property
    def max_wait_steps(self) -> int:
        """Longest wait currently in the queue (§S2 metrics list)."""
        return max((queued.steps_waited for queued in self._queued), default=0)

    def tick(self, step: int) -> list[DeferredCriticalStep]:
        """Advance one step: age every queued request by one step.

        Returns one `DeferredCriticalStep` log entry per still-queued
        request (Addition C).
        """
        entries: list[DeferredCriticalStep] = []
        for queued in self._queued:
            queued.steps_waited += 1
            entries.append(
                DeferredCriticalStep(
                    step=step,
                    request_id=queued.request["request_id"],
                    steps_waited=queued.steps_waited,
                )
            )
        return entries

    def pop_servable(self, can_draw: Callable[[float], bool]) -> list[QueuedRequest]:
        """Pop and return all queued requests that `can_draw(bits_required)`
        now covers.

        Priority order: highest sensitivity class first, FIFO within
        class (Addition C unit test: "queue priority/FIFO ordering").

        `can_draw` is a pure feasibility predicate (mirrors
        `PoolSim.can_draw`), not a draw -- it doesn't account for bits
        this call already "spent" on earlier, higher-priority requests
        in the same pass. To avoid over-committing the pool within one
        `pop_servable` call, each candidate is checked against the
        *cumulative* bits already committed to requests popped earlier
        in this same pass, not just its own `bits_required` in
        isolation. A candidate that doesn't fit is skipped (left
        queued) rather than blocking lower-priority requests behind it
        that might still fit in the remaining headroom.
        """
        ordered = sorted(
            self._queued,
            key=lambda queued: (-queued.request["sensitivity_class"], queued.step_enqueued),
        )

        servable: list[QueuedRequest] = []
        committed_bits = 0.0
        for queued in ordered:
            candidate_total = committed_bits + queued.bits_required
            if can_draw(candidate_total):
                servable.append(queued)
                committed_bits = candidate_total

        popped_ids = {id(queued) for queued in servable}
        self._queued = [queued for queued in self._queued if id(queued) not in popped_ids]

        return servable

    def __len__(self) -> int:
        return len(self._queued)
