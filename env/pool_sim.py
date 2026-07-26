"""
env/pool_sim.py

QKD pool simulator: trace-driven refill, draw-down, and exhaustion
(PLAN.md §4 architecture diagram "QKD POOL SIM"; PLAN.md §10 kickoff
step 2). Owned by Person B (split.md §1).

This is the scarcity engine the whole thesis rests on: the pool level
computed here is what `env/masking.py` checks for SERVE_HYBRID
feasibility, and what `env/deferral_queue.py` queues against when a
hybrid-mandatory request can't be covered (Hard Rule 9).

Scaffolding only -- no real refill/drain arithmetic yet. PLAN.md §10
step 2: "Build the pool simulator first (trace-driven refill,
draw-down, exhaustion events) with unit tests."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass
class PoolState:
    """Snapshot of pool physics at one step."""

    fill: float  # current pool level, bits
    capacity: float  # max pool capacity, bits
    skr: float  # instantaneous secret-key rate driving refill
    qber: float  # current quantum bit error rate


class SKRQBERTrace(Protocol):
    """Interface for the CV-QKD trace source driving pool refill.

    Implemented by Person A's trace loader (PLAN.md "Datasets &
    Provenance" -> "QKD pool refill (SKR/QBER)" row) or a documented
    synthetic generator. `PoolSim` only ever consumes this interface.
    """

    def __iter__(self) -> Iterator[tuple[float, float]]:
        """Yield (skr, qber) pairs, one per simulator step."""
        ...


class PoolExhaustedError(Exception):
    """Raised by `PoolSim.draw` if called without a prior `can_draw` check.

    Exhaustion itself is not an error condition for the environment --
    it is handled by `env/deferral_queue.py` (Hard Rule 9) -- but a
    caller that draws without checking feasibility first is a bug.
    """


class PoolSim:
    """Trace-driven QKD pool simulator (refill + drain + exhaustion).

    Implements the "QKD POOL SIM" box in PLAN.md §4's architecture
    diagram. Refill is driven by an `SKRQBERTrace`; drain happens when
    `env/environment.py` calls `draw()` for a SERVE_HYBRID action.
    """

    def __init__(self, capacity: float, trace: SKRQBERTrace) -> None:
        raise NotImplementedError

    def reset(self) -> PoolState:
        """Reset pool to its initial fill level and rewind the trace."""
        raise NotImplementedError

    def step(self) -> PoolState:
        """Advance one timestep: pull the next (skr, qber) from the trace
        and refill the pool accordingly."""
        raise NotImplementedError

    def can_draw(self, bits: float) -> bool:
        """Feasibility check used by `env/masking.py` before allowing
        SERVE_HYBRID, and by `env/deferral_queue.py` to decide when a
        queued request can finally be served."""
        raise NotImplementedError

    def draw(self, bits: float) -> None:
        """Consume `bits` from the pool for a SERVE_HYBRID action.

        Raises `PoolExhaustedError` if `bits` exceeds current fill --
        callers must check `can_draw()` first. Pool exhaustion is
        handled by `env/deferral_queue.py`, never by silently
        under-drawing (Hard Rule 9).
        """
        raise NotImplementedError
