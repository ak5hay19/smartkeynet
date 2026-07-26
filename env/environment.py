"""
env/environment.py

Gymnasium-style environment: `reset()`/`step()`/state/mask (PLAN.md §4
architecture diagram; §10 kickoff step 5). Owned by Person B
(split.md §1) -- this is the spine; everyone else's work is
meaningless if this is wrong.

Wires together `pool_sim`, `deferral_queue`, `masking`, a
`ForecastProvider`, and the full reward formula (PLAN.md §4):

    r = - w_lat*latency - w_en*energy + w_fr*freshness
        - w_qkd*(pool bits consumed)
        - R_starve*(deferred_critical_steps)
        - c_rekey(load)*1[rekey]

    where c_rekey(load) = c0 * (1 + beta * load)

Hard Rule 1: no security term anywhere in this formula, ever -- not
even temporarily "to help training".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from env.contracts import Action, ActionMask, StateDict

# NOTE: environment.py wires these together once implemented. Imported
# here (rather than deferred) so a broken constructor signature in any
# of them fails loudly at import time -- see tests/test_environment.py.
from env.deferral_queue import DeferralQueue  # noqa: F401
from env.forecast_provider import MovingAverageForecaster  # noqa: F401
from env.masking import PolicyTable, compute_mask  # noqa: F401
from env.pool_sim import PoolSim  # noqa: F401


@dataclass
class StepResult:
    """Named return shape mirroring Gymnasium's 5-tuple, so B/C don't
    have to remember positional order when passing results around."""

    state: StateDict
    mask: ActionMask
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class SmartKeyNetEnv(gym.Env):
    """The MDP (PLAN.md §4). One agent, one MDP (Hard Rule 3).

    `config` selects the scenario (S1-S6, PLAN.md §5) and the
    `use_foresight` flag (Addition A) that determines which
    `ForecastProvider` is constructed and how long the flattened state
    vector is.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        raise NotImplementedError

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[StateDict, dict[str, Any]]:
        """Gymnasium-standard reset. Also resets `pool_sim`,
        `deferral_queue`, the request stream, and the forecast
        provider's window."""
        raise NotImplementedError

    def step(self, action: Action) -> tuple[StateDict, float, bool, bool, dict[str, Any]]:
        """Gymnasium-standard step.

        `info` must include the current `action_mask` (masked-env
        convention: the agent needs it for the *next* decision) and
        any event-log entries emitted this step (`RegretEvent`,
        `DeferredCriticalStep`, `ForcedRekey` -- see
        `env/contracts.py`).
        """
        raise NotImplementedError

    def action_mask(self) -> ActionMask:
        """Current legal-action mask, per `env/masking.py`. Exposed
        separately so the agent can query it without stepping."""
        raise NotImplementedError
