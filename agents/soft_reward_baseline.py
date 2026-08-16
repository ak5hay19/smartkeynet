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

Trained/evaluated only via the S5 steering-attack path -- never used to
select real KMS actions.

---------------------------------------------------------------------
Why this file is allowed to contain a security reward
---------------------------------------------------------------------
Hard Rule 1 says the reward must never contain a security term. That
rule governs **our** agent. This module deliberately implements the
opposite design, because the project's headline claim is a comparison:

    "Security as constraint (action masking), not reward -- demonstrated
     against a live steering attack on a reproduced soft-reward baseline."

You cannot show that a steering attack fails against masking without
also showing that it succeeds against the thing masking replaces.

Three safeguards keep the exception from leaking:

  1. `SoftRewardAgent` computes its own reward internally and **ignores
     the environment's reward entirely**. Nothing under `env/` is
     modified or parameterised to emit a security-flavoured reward, so
     the Hard Rule 1 guarantees about the environment hold unchanged.
  2. Nothing in `env/` imports this module, and neither does
     `agents/dqn.py`. The dependency only points this way.
  3. `SOFT_REWARD_IS_THE_CRITIQUED_DESIGN = True` is a module-level
     marker the tests assert on, so this file cannot be quietly
     repurposed as a "better reward" for our own agent.

---------------------------------------------------------------------
The reproduced design and its attack surface
---------------------------------------------------------------------
The critiqued family shares one structure: the threat score enters the
reward as a soft term, so stronger crypto is *bought* rather than
*required*.

    r_soft = w_security * security_score(tier) * threat_score
             - w_cost * cost(tier)

When the reported threat is low the first term shrinks, the cost term
dominates, and the agent's preferred tier slides *down* the ladder. An
adversary who can shape the threat signal therefore moves the agent to
weaker keys without touching any cryptography. That gradient is exactly
what scenario S5 exploits.

Our masked agent has no such gradient: it never sees the threat score
in its reward at all, and the signal only reaches the policy table,
where it can only raise floors.

What is deliberately NOT claimed: this agent still acts through the same
`ActionMask` as ours, so it cannot literally serve below a floor here.
The comparison isolates the reward design rather than confounding it
with a second difference, and the attack is therefore measured on the
**served-tier distribution** -- how far down the ladder the agent slides
within what it is allowed -- which is precisely what PLAN.md §6 Beat 3
puts on screen ("the served-tier histogram slides downward").
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from env.contracts import Action, ActionMask, StateDict

SOFT_REWARD_IS_THE_CRITIQUED_DESIGN = True
"""Marker asserted by the tests. This module reproduces the design the
project argues against; it must never become our agent's reward."""


_SECURITY_SCORE: dict[Action, float] = {
    Action.SERVE_CLASSICAL: 0.0,  # quantum-vulnerable
    Action.SERVE_PQC: 0.6,  # quantum-resistant
    Action.SERVE_HYBRID: 1.0,  # PQC + QKD key material
    Action.REUSE: 0.3,  # no fresh key material
    Action.REKEY_NOW: 0.6,  # refreshes at the session's current tier
}
"""The "how secure does this feel" table the critiqued designs require.

These are **not** citable security constants and are not claimed to be
-- they reproduce a modelling choice this project argues against. Hard
Rule 4 governs the floor table in `env/masking.py`, which is what the
project actually stands behind. That the numbers here are essentially
arbitrary is itself part of the critique: a soft security reward forces
somebody to invent a cardinal "amount of security" per tier, and the
agent's behaviour then depends on those invented numbers.
"""

_COST: dict[Action, float] = {
    Action.SERVE_CLASSICAL: 0.3,
    Action.SERVE_PQC: 0.5,
    Action.SERVE_HYBRID: 1.0,
    Action.REUSE: 0.05,
    Action.REKEY_NOW: 0.5,
}
"""Operating cost per action, ordered like `env/environment.py`'s
latency/energy tables: hybrid dearest, reuse cheapest."""


def soft_reward(
    state: StateDict,
    action: Action,
    threat_score: float,
    w_security: float = 2.0,
    w_cost: float = 1.0,
) -> float:
    """Reproduces the soft-reward formula that folds `threat_score`
    directly into the reward -- the design PLAN.md §2 critiques.

    Contrast with the real reward wired in `env/environment.py`, which
    never takes a threat term (Hard Rule 1). This function exists only
    to reproduce the baseline being attacked.

    `state` is accepted to match the published designs' signature shape
    and is otherwise unused: the whole point of the critique is that
    these designs key their security decision on a single scalar threat
    score.
    """
    return w_security * _SECURITY_SCORE[action] * float(threat_score) - w_cost * _COST[action]


@dataclass
class SoftRewardConfig:
    w_security: float = 2.0
    w_cost: float = 1.0
    learning_rate: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    threat_bins: int = 10
    """Tabular Q-learning over a discretised threat score, matching the
    published designs' tabular Q-learning rather than deep RL. Staying
    tabular makes the steering effect legible -- the learned policy can
    be read straight out of the table, one row per threat bin."""


class SoftRewardAgent:
    """Q-learning agent trained against `soft_reward` instead of relying
    on action masking for its security guarantee.

    Its policy is expected to be steerable by an adversarially shaped
    threat trace (PLAN.md §5 S5). Its state is just the discretised
    threat score, because that is what the critiqued designs key on and
    what the attack manipulates.
    """

    def __init__(self, config: SoftRewardConfig | None = None, seed: int | None = None) -> None:
        self.config = config if config is not None else SoftRewardConfig()
        self._rng = random.Random(seed)
        self.q_table = np.zeros((self.config.threat_bins, len(Action)), dtype=np.float64)
        self._act_calls = 0

    def soft_reward(self, action: Action, threat_score: float) -> float:
        """This agent's own reward. THIS IS THE DESIGN THE PROJECT
        ARGUES AGAINST -- see the module docstring."""
        empty_state: Any = {}
        return soft_reward(
            state=empty_state,  # deliberately unused; see soft_reward's docstring
            action=action,
            threat_score=threat_score,
            w_security=self.config.w_security,
            w_cost=self.config.w_cost,
        )

    def _bin_for(self, threat_score: float) -> int:
        clamped = min(max(float(threat_score), 0.0), 1.0)
        return min(int(clamped * self.config.threat_bins), self.config.threat_bins - 1)

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self._act_calls / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        """Epsilon-greedy over the Q-table, restricted to the mask.

        The mask still applies -- this agent is compared against ours
        *inside the same environment*, so the comparison isolates the
        reward design. See the module docstring on why the attack is
        therefore measured on served tiers.
        """
        self._act_calls += 1
        legal = [action for action in Action if mask[int(action)]]
        if not legal:
            raise ValueError("no legal action in mask")

        if self._rng.random() < self.epsilon:
            return self._rng.choice(legal)
        return self.act_greedy(state, mask)

    def act_greedy(self, state: StateDict, mask: ActionMask) -> Action:
        """Deterministic evaluation policy (epsilon = 0), so a trained
        agent can be scored without burning its exploration schedule."""
        legal = [action for action in Action if mask[int(action)]]
        if not legal:
            raise ValueError("no legal action in mask")
        threat_bin = self._bin_for(state["threat_score"])
        return max(legal, key=lambda action: self.q_table[threat_bin, int(action)])

    def learn(
        self,
        state: StateDict,
        action: Action,
        next_state: StateDict,
        next_mask: ActionMask,
    ) -> dict[str, float]:
        """One tabular Q-update using this agent's OWN soft reward.

        The environment's reward is deliberately not a parameter: this
        agent has to be driven by the critiqued reward for the
        comparison to mean anything, and `env/` must not be modified to
        supply one.
        """
        reward = self.soft_reward(action, state["threat_score"])

        threat_bin = self._bin_for(state["threat_score"])
        next_bin = self._bin_for(next_state["threat_score"])
        next_legal = [a for a in Action if next_mask[int(a)]]
        best_next = max((self.q_table[next_bin, int(a)] for a in next_legal), default=0.0)

        current = self.q_table[threat_bin, int(action)]
        self.q_table[threat_bin, int(action)] = current + self.config.learning_rate * (
            reward + self.config.gamma * best_next - current
        )
        return {
            "soft_reward": float(reward),
            "q_value": float(self.q_table[threat_bin, int(action)]),
        }


@dataclass
class GreedySoftRewardPolicy:
    """Adapter so a trained `SoftRewardAgent` drops into
    `experiments/harness.py` like any other `Policy`."""

    agent: SoftRewardAgent

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        return self.agent.act_greedy(state, mask)
