"""
agents/dqn.py

Masked DQN agent (PLAN.md §4 architecture diagram "DQN AGENT"; §10
kickoff step 5). Owned by Person C (split.md §1).

Consumes `env.contracts.StateDict` + `ActionMask` from
`env/environment.py`. Reward has no security term (Hard Rule 1) --
this module must never add one, even temporarily "to stabilize
training" (split.md §4 anti-patterns).

Start vanilla; upgrade to Double/Dueling DQN only if needed (PLAN.md
tech stack).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from env.contracts import Action, ActionMask, N_ACTIONS, StateDict


def flatten_state(state: StateDict) -> torch.Tensor:
    """Flatten a `StateDict` into the fixed-order tensor the Q-network
    consumes.

    Vector length depends on `use_foresight` (Addition A) -- cover
    both lengths with a unit test.
    """
    raise NotImplementedError


class QNetwork(nn.Module):
    """Feedforward Q-network: state vector -> one Q-value per action."""

    def __init__(self, state_dim: int, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class DQNConfig:
    """Hyperparameters for `DQNAgent` (values live in
    `configs/default.yaml`)."""

    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    target_update_every: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000


class DQNAgent:
    """Masked DQN: argmax over Q-values restricted to the legal action set.

    The mask is applied by setting illegal actions' Q-values to -inf
    before argmax/softmax -- the network itself is never trained to
    label masked actions good or bad. Masking is structural, not
    learned (Hard Rule 2).
    """

    def __init__(self, state_dim: int, config: DQNConfig | None = None) -> None:
        raise NotImplementedError

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        """Epsilon-greedy action selection restricted to `mask`."""
        raise NotImplementedError

    def observe(
        self,
        state: StateDict,
        action: Action,
        reward: float,
        next_state: StateDict,
        next_mask: ActionMask,
        done: bool,
    ) -> None:
        """Push a transition to the replay buffer."""
        raise NotImplementedError

    def learn(self) -> dict[str, float]:
        """One gradient step on a sampled batch. Returns a loss/metrics dict."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
