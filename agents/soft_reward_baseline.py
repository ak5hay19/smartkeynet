"""
agents/soft_reward_baseline.py

Reproduction of the Noetzold-style soft-reward agent, where a threat
score is a *reward* term rather than a masking constraint (PLAN.md §2
research problem; §5 scenario S5). Owned by Person C (split.md §1).

This agent is the thesis's cautionary tale, not a design to emulate
elsewhere in this codebase: it exists solely so
`attack/steering_trace.py` has something to steer (PLAN.md §6 Demo
Beat 3). Never let its reward shape leak into `agents/dqn.py`
(Hard Rule 1).

Trained/evaluated only via `experiments/harness.py`'s S5
steering-attack path -- never used to select real KMS actions.
"""

from __future__ import annotations

from env.contracts import Action, ActionMask, StateDict


def soft_reward(state: StateDict, action: Action, threat_score: float) -> float:
    """Reproduces the soft-reward formula that folds `threat_score`
    directly into the reward -- the design PLAN.md §2 critiques.

    Contrast with the real reward wired in `env/environment.py`, which
    never takes a threat term (Hard Rule 1). This function exists only
    to reproduce the baseline being attacked.
    """
    raise NotImplementedError


class SoftRewardAgent:
    """Q-learning-style agent trained against `soft_reward` instead of
    action masking.

    Its policy is expected to be steerable by an adversarially shaped
    threat trace (PLAN.md §5 S5).
    """

    def __init__(self, state_dim: int) -> None:
        raise NotImplementedError

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        raise NotImplementedError

    def learn(self, *args: object, **kwargs: object) -> dict[str, float]:
        raise NotImplementedError
