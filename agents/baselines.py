"""
agents/baselines.py

Tuned non-RL baselines: always-PQC, always-hybrid, static-threshold
(grid-searched), random (PLAN.md Hard Rule 7). Owned by Person C
(split.md §1).

Build these *before* tuning the DQN (Hard Rule 7; split.md Week 3).
Disqualification rule (PLAN.md §2): if a simple threshold policy
matches the DQN in evaluation, the project premise fails.
"""

from __future__ import annotations

from typing import Callable, Protocol

from env.contracts import Action, ActionMask, StateDict


class Policy(Protocol):
    """Common shape for every baseline plus the DQN, so
    `experiments/harness.py` can run any of them identically."""

    def act(self, state: StateDict, mask: ActionMask) -> Action: ...


class AlwaysPQCPolicy:
    """Serves SERVE_PQC whenever legal, else the lowest legal tier."""

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        raise NotImplementedError


class AlwaysHybridPolicy:
    """Serves SERVE_HYBRID whenever legal, else the lowest legal tier.

    The "drains the pool" baseline (PLAN.md §6 Demo Beat 2).
    """

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        raise NotImplementedError


class StaticThresholdPolicy:
    """Serves SERVE_HYBRID iff `pool_fill` exceeds a fixed threshold,
    else SERVE_PQC. Threshold is grid-searched, not hand-picked (Hard
    Rule 7)."""

    def __init__(self, pool_fill_threshold: float) -> None:
        raise NotImplementedError

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        raise NotImplementedError

    @staticmethod
    def grid_search(
        candidate_thresholds: list[float], eval_fn: Callable[["StaticThresholdPolicy"], float]
    ) -> "StaticThresholdPolicy":
        """Pick the threshold maximizing `eval_fn(StaticThresholdPolicy(t))`."""
        raise NotImplementedError


class RandomPolicy:
    """Uniform-random choice among currently-legal actions."""

    def __init__(self, seed: int | None = None) -> None:
        raise NotImplementedError

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        raise NotImplementedError
