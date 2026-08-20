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

Why the reproduction has to be faithful
---------------------------------------
The project's headline claim is a *comparative* one: soft-reward
designs can be steered, and this architecture cannot. That claim is
worth nothing if the soft-reward baseline is a strawman built to lose.
So this reproduces the published design's actual structure --

    reward = w_sec * threat_score * tier_strength(action)   <-- security IS the reward
             - w_lat * latency - w_en * energy - w_qkd * bits

-- in which a higher observed threat score makes stronger key material
more attractive, and a lower one makes it less attractive. That is a
*sensible* design on its face: it does select stronger crypto under
threat, it does economize under calm, and with an honest threat signal
it behaves well. The vulnerability is not incompetence; it is that the
protection level is a **preference**, and preferences can be bid down
by whoever controls the signal they are priced against.

Two structural differences from `agents/dqn.py`, both deliberate and
both essential to the comparison:

  1. **No action masking.** This agent chooses freely among all five
     actions. That is the point -- there is no floor, only a gradient.
  2. **Security in the reward.** Hard Rule 1 forbids this everywhere
     else in the repo; this module is the single, quarantined exception
     that makes the rule's value demonstrable rather than asserted.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from env.contracts import N_ACTIONS, Action, ActionMask, StateDict

# ---------------------------------------------------------------------------
# The soft reward
# ---------------------------------------------------------------------------

_TIER_STRENGTH: dict[Action, float] = {
    Action.SERVE_CLASSICAL: 0.0,
    Action.SERVE_PQC: 0.5,
    Action.SERVE_HYBRID: 1.0,
    Action.REUSE: 0.0,      # delivers whatever the session already had; no new protection
    Action.REKEY_NOW: 0.5,  # refreshes in place; mid-strength by construction
}
"""How much "security" the soft-reward design credits each action with.

Ordinal only, and deliberately so: the point of reproducing this design
is its *shape* (protection is a scalar you can trade against cost), not
any particular published constant. Hard Rule 4 does not apply -- this
is not a security floor, it is a critique target."""

_LATENCY_UNITS: dict[Action, float] = {
    Action.REUSE: 0.2,
    Action.SERVE_CLASSICAL: 1.0,
    Action.SERVE_PQC: 1.2,
    Action.SERVE_HYBRID: 1.5,
    Action.REKEY_NOW: 1.2,
}
_ENERGY_UNITS: dict[Action, float] = {
    Action.REUSE: 0.1,
    Action.SERVE_CLASSICAL: 1.0,
    Action.SERVE_PQC: 1.3,
    Action.SERVE_HYBRID: 1.6,
    Action.REKEY_NOW: 1.3,
}
"""Mirrors `env/environment.py`'s cost tables so the two agents face the
same operational economics and the only difference between them is the
security term and the mask."""


@dataclass(frozen=True)
class SoftRewardConfig:
    w_sec: float = 3.0
    """Weight on the security term. Large enough that an honest HIGH
    threat signal genuinely buys hybrid over PQC against their cost
    difference -- i.e. the design works as intended when the signal is
    honest, which is what makes the steering result interesting."""

    w_lat: float = 1.0
    w_en: float = 0.1
    w_qkd: float = 0.004
    epsilon: float = 0.1
    lr: float = 0.1
    gamma: float = 0.9
    n_pool_bins: int = 5
    n_threat_bins: int = 5


def soft_reward(
    state: StateDict,
    action: Action,
    threat_score: float,
    config: SoftRewardConfig | None = None,
) -> float:
    """Reproduces the soft-reward formula that folds `threat_score`
    directly into the reward -- the design PLAN.md §2 critiques.

    Contrast with the real reward wired in `env/environment.py`, which
    never takes a threat term (Hard Rule 1). This function exists only
    to reproduce the baseline being attacked.

    The steerability is visible right here in the algebra: the security
    term is `w_sec * threat_score * tier_strength`, so driving
    `threat_score` toward 0 drives the entire security contribution to
    0 regardless of tier, leaving only costs -- which are minimized by
    the *weakest* action. An adversary who can shape the threat signal
    does not need to attack the agent, the network, or the crypto; it
    only has to make the agent believe things are calm.
    """
    config = config if config is not None else SoftRewardConfig()
    bits = 256.0 if action is Action.SERVE_HYBRID else 0.0
    return (
        config.w_sec * float(threat_score) * _TIER_STRENGTH[action]
        - config.w_lat * _LATENCY_UNITS[action]
        - config.w_en * _ENERGY_UNITS[action]
        - config.w_qkd * bits
    )


class SoftRewardAgent:
    """Tabular Q-learning agent trained against `soft_reward` instead of
    action masking.

    Its policy is expected to be steerable by an adversarially shaped
    threat trace (PLAN.md §5 S5).

    Tabular, not a second DQN, on purpose: the published designs this
    reproduces are Q-learning-scale, the state discretization below
    (sensitivity class x threat bin x pool bin) is the same context the
    soft-reward literature conditions on, and a tabular agent converges
    reliably enough that the steering result is a property of the
    *reward design* rather than of a training run's luck. Making the
    critique target a well-converged agent is the conservative choice.

    `act()` accepts a mask for interface compatibility with
    `agents.baselines.Policy` (so `experiments/harness.py` can run it)
    but **ignores it by default** -- `respect_mask=False` is what makes
    this the unmasked comparison point. The masked variant exists only
    so the attack harness can run it inside the real environment, which
    raises `IllegalActionError` on an illegal action; see
    `attack/run_attack.py` for how the two are kept distinct in the
    reported numbers.
    """

    def __init__(
        self,
        config: SoftRewardConfig | None = None,
        seed: int | None = None,
        respect_mask: bool = False,
    ) -> None:
        self.config = config if config is not None else SoftRewardConfig()
        self.respect_mask = respect_mask
        self._rng = random.Random(seed)
        self._q: dict[tuple[int, int, int], np.ndarray] = {}

    # -- state discretization -------------------------------------------

    def _key(self, state: StateDict, threat_score: float) -> tuple[int, int, int]:
        sensitivity = int(state["sensitivity_class"])
        threat_bin = min(
            self.config.n_threat_bins - 1,
            int(float(threat_score) * self.config.n_threat_bins),
        )
        pool_bin = min(
            self.config.n_pool_bins - 1,
            int(float(state["pool_fill"]) * self.config.n_pool_bins),
        )
        return sensitivity, threat_bin, pool_bin

    def _row(
        self,
        key: tuple[int, int, int],
        state: StateDict | None = None,
        threat_score: float | None = None,
    ) -> np.ndarray:
        """Q-row for `key`, created on first visit.

        Initialization is the **myopic value** -- each action's immediate
        `soft_reward` in the state that first reached this cell -- not
        zeros. Zeros are actively wrong here: every real soft-reward
        value in this design is negative (costs outweigh the security
        term except under strong threat), so a zero-initialized row makes
        every *unvisited* action strictly preferred to every learned one,
        and a greedy policy ends up selecting whatever it has never
        tried. That produced a dose-response curve that was flat at 44%
        across every dose -- the agent was reporting its initialization,
        not its learning. Myopic initialization is a sensible prior (it
        is the correct value at gamma = 0) and leaves the greedy policy
        well-defined in thinly-visited cells.
        """
        if key not in self._q:
            if state is None or threat_score is None:
                self._q[key] = np.full(N_ACTIONS, -1e3, dtype=np.float64)
            else:
                self._q[key] = np.array(
                    [soft_reward(state, Action(i), threat_score, self.config) for i in range(N_ACTIONS)],
                    dtype=np.float64,
                )
        return self._q[key]

    # -- Policy ----------------------------------------------------------

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        """Greedy over the learned Q-row.

        `threat_score` is read straight off the state, which is exactly
        the channel the steering attack controls.
        """
        row = self._row(self._key(state, state["threat_score"]), state, float(state["threat_score"]))
        if self.respect_mask:
            candidates = [i for i in range(N_ACTIONS) if mask[i]]
            if not candidates:
                raise ValueError("no legal action in mask")
            return Action(max(candidates, key=lambda i: row[i]))
        return Action(int(np.argmax(row)))

    def act_exploring(self, state: StateDict, mask: ActionMask) -> Action:
        if self._rng.random() < self.config.epsilon:
            if self.respect_mask:
                legal = [i for i in range(N_ACTIONS) if mask[i]]
                return Action(self._rng.choice(legal))
            return Action(self._rng.randrange(N_ACTIONS))
        return self.act(state, mask)

    def learn(
        self,
        state: StateDict,
        action: Action,
        threat_score: float,
        next_state: StateDict,
        next_threat_score: float,
    ) -> dict[str, float]:
        """One tabular Q-learning update against `soft_reward`."""
        key = self._key(state, threat_score)
        next_key = self._key(next_state, next_threat_score)
        reward = soft_reward(state, action, threat_score, self.config)

        row = self._row(key, state, threat_score)
        bootstrap = float(np.max(self._row(next_key, next_state, next_threat_score)))
        target = reward + self.config.gamma * bootstrap
        td_error = target - row[int(action)]
        row[int(action)] += self.config.lr * td_error

        return {"reward": reward, "td_error": float(td_error)}
