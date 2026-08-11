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

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

from env.contracts import Action, ActionMask, N_ACTIONS, StateDict

# ---------------------------------------------------------------------------
# State flattening (Addition A: vector length depends on use_foresight)
# ---------------------------------------------------------------------------

_OFF_STATE_DIM = 13
_FORECAST_STATE_DIM = 28
"""`flatten_state`'s two possible output lengths, spelled out here so
tests don't need to hardcode them independently of the implementation
below. See `flatten_state`'s docstring for the field-by-field
breakdown that produces each number."""


def flatten_state(state: StateDict, has_forecast: bool) -> torch.Tensor:
    """Flatten a `StateDict` into the fixed-order tensor the Q-network
    consumes.

    `has_forecast` must be supplied explicitly by the caller -- it is
    never inferred from `state`'s own contents. An earlier version of
    this function guessed the mode via `state["threat_score"] != 0.0`
    (relying on `MovingAverageForecaster`'s sigmoid output never being
    exactly `0.0`). That guess was only correct by accident: it held
    solely because today's placeholder `threat_features`
    (`[qber, load]`, both always non-negative -- see
    `env/environment.py`'s `_threat_features_placeholder`) happen to
    keep the sigmoid input away from wherever it might round to
    exactly zero. Nothing guarantees that once real (possibly
    negative/normalized) threat data replaces the placeholder, so the
    inference was removed rather than hardened -- the real answer
    ("is this episode running with foresight on?") is a config-time
    fact, not something derivable from a single state observation.
    The natural source at any call site is `config["use_foresight"] !=
    "off"` -- the exact same condition `env/environment.py`'s
    `_build_forecaster` uses to decide whether forecast fields get
    populated at all.

    Field order matches `env/contracts.py`'s `StateDict` declaration
    order exactly; forecast-derived fields (`threat_score`,
    `threat_forecast`, `pool_level_hat`, `skr_mean_hat`,
    `hybrid_demand_hat`) are included only when `has_forecast` is
    True -- genuinely omitted, not zero-padded, when False. This is
    what makes the eventual E-A ablation (off vs. ewma vs. lstm) a
    real input-dimensionality difference rather than a cosmetic one.
    `regret_event_recent` (Addition C bookkeeping, not forecast-derived)
    is always included regardless of `has_forecast`.

    Two possible lengths, both fixed and documented:
      - `has_forecast=False` (`use_foresight: off`): 13 -- qber, skr,
                pool_fill, arrival_rate, load, avg_latency, key_age,
                key_type_onehot(3), sensitivity_class, policy_floor,
                regret_event_recent.
      - `has_forecast=True` (`use_foresight: ewma`/`lstm`): 28 --
                threat_score, threat_forecast(5), [the 13 fields
                above], pool_level_hat(3), skr_mean_hat(3),
                hybrid_demand_hat(3).
    """
    fields: list[float] = []
    if has_forecast:
        fields.append(float(state["threat_score"]))
        fields.extend(float(v) for v in state["threat_forecast"])

    fields.append(float(state["qber"]))
    fields.append(float(state["skr"]))
    fields.append(float(state["pool_fill"]))
    fields.append(float(state["arrival_rate"]))
    fields.append(float(state["load"]))
    fields.append(float(state["avg_latency"]))
    fields.append(float(state["key_age"]))
    fields.extend(float(v) for v in state["key_type_onehot"])
    fields.append(float(state["sensitivity_class"]))
    fields.append(float(state["policy_floor"]))

    if has_forecast:
        fields.extend(float(v) for v in state["pool_level_hat"])
        fields.extend(float(v) for v in state["skr_mean_hat"])
        fields.extend(float(v) for v in state["hybrid_demand_hat"])

    fields.append(float(state["regret_event_recent"]))

    return torch.tensor(fields, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Q-network
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    """Feedforward Q-network: state vector -> one Q-value per action.

    Plain two-hidden-layer MLP -- this isn't the research contribution
    (PLAN.md tech stack: "start vanilla, upgrade to Double/Dueling if
    needed").
    """

    def __init__(self, state_dim: int, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


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


def load_dqn_config(path: str | Path | None = None) -> DQNConfig:
    """Read the `dqn:` block out of `configs/default.yaml` into a
    `DQNConfig` -- mirrors `env.pool_sim.load_pool_config` /
    `env.masking.load_key_lifetime_config`'s existing convention, so
    hyperparameters live in one YAML instead of being duplicated as
    Python literals wherever a `DQNAgent` is constructed."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return DQNConfig(**config["dqn"])


# ---------------------------------------------------------------------------
# Replay buffer (no separate replay_buffer.py in this repo's layout --
# kept internal to this module)
# ---------------------------------------------------------------------------


@dataclass
class _Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    next_mask: np.ndarray
    done: bool


@dataclass
class _ReplayBuffer:
    """Fixed-capacity circular buffer. Backed by a plain list with a
    write pointer (not `collections.deque`) so `random.sample` gets
    O(1) indexed access per draw instead of `deque`'s O(n) -- matters
    here since `learn()` samples a fresh batch every call, potentially
    thousands of times per training run."""

    capacity: int
    _storage: list[_Transition] = field(default_factory=list)
    _write_idx: int = 0

    def push(self, transition: _Transition) -> None:
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._write_idx] = transition
            self._write_idx = (self._write_idx + 1) % self.capacity

    def sample(self, batch_size: int) -> list[_Transition]:
        return random.sample(self._storage, batch_size)

    def __len__(self) -> int:
        return len(self._storage)


_REPLAY_BUFFER_CAPACITY = 50_000
"""Not in `configs/default.yaml`'s `dqn:` block (only the hyperparameters
PLAN.md/split.md already named there exist as config), so this is a
plain documented internal default rather than something to "load, not
hardcode" -- there's nothing in config to load it from."""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class DQNAgent:
    """Masked DQN: argmax over Q-values restricted to the legal action set.

    The mask is applied by setting illegal actions' Q-values to -inf
    before argmax/softmax -- the network itself is never trained to
    label masked actions good or bad. Masking is structural, not
    learned (Hard Rule 2): both the acting policy (`act`) and the
    bootstrap target (`learn`) mask illegal *next*-state actions to
    -inf before taking a max over them, so the network is never even
    implicitly trained to assign a high value to an action the
    environment would never have allowed.

    Reward is consumed exactly as `env/environment.py` computes it --
    this agent never adds, reshapes, or substitutes any term of its
    own (Hard Rule 1: no security term in the reward, ever).
    """

    def __init__(
        self,
        state_dim: int,
        has_forecast: bool,
        config: DQNConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """`has_forecast` must match whatever `use_foresight` mode the
        environment this agent will run against was built with (True
        for `ewma`/`lstm`, False for `off`) -- a given training run is
        in one mode for its whole lifetime, so this is fixed once here
        rather than passed to every `act`/`observe` call. The natural
        way to derive it: `config["use_foresight"] != "off"`, using
        the same `config` dict passed to `SmartKeyNetEnv`. Every
        internal `flatten_state` call uses this value; nothing in this
        class infers the mode from a `StateDict`'s contents.

        `seed`, if given, makes this agent's own randomness --
        `QNetwork` weight init, epsilon-greedy exploration (`act`'s
        `random.random()`/`random.choice()`), and replay-buffer
        sampling (`random.sample()`) -- reproducible: it reseeds
        Python's global `random` module and PyTorch's global RNG
        immediately below, *before* `QNetwork` is constructed, so
        initialization itself is covered, not just what happens
        afterward. Found and flagged, not fixed, in the 2026-08-10
        10-seed load-spike sweep session (see SESSION_LOG.md): none of
        this randomness was seedable before, so `experiments/train.py`'s
        `training.seed` only ever reached the environment's request
        stream (`SmartKeyNetEnv`/`random_request_generator` use their
        own local `np.random.default_rng(seed)` instances, genuinely
        independent of this agent's RNGs -- see env/pool_sim.py,
        env/request_generator.py -- so reusing the same integer for
        both is safe, not a collision). `seed=None` (the default)
        leaves both RNGs at whatever ambient state the process already
        has -- unseeded, exactly the pre-existing behavior -- so no
        existing caller changes behavior by omission.

        Caveat: this reseeds *global* RNG state, not a private
        per-instance generator (mirrors this module's own test suite's
        pre-existing `torch.manual_seed(0)`-before-construction
        convention in `test_dqn_agent_loss_trends_down_training_against_
        real_env_s1`) -- constructing a second seeded agent, or any
        other global `random`/`torch` call, between this agent's
        construction and the calls you want reproduced will perturb
        the shared stream. Fine for this repo's actual use (one agent
        per training run); documented so it's not a surprise later.
        """
        self.state_dim = state_dim
        self.has_forecast = has_forecast
        self.config = config if config is not None else DQNConfig()
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.q_network = QNetwork(state_dim)
        self.target_network = QNetwork(state_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.config.lr)
        self._replay_buffer = _ReplayBuffer(capacity=_REPLAY_BUFFER_CAPACITY)

        self._act_calls = 0
        self._learn_calls = 0

    def _current_epsilon(self) -> float:
        cfg = self.config
        if cfg.epsilon_decay_steps <= 0:
            return cfg.epsilon_end
        fraction = min(1.0, self._act_calls / cfg.epsilon_decay_steps)
        return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        """Epsilon-greedy action selection restricted to `mask`.

        Illegal actions can never be returned at any epsilon: the
        random-explore branch samples only from the mask's legal
        indices directly, and the greedy branch masks Q-values to
        -inf before argmax -- epsilon only ever chooses *between*
        those two already-legal-only paths, it never widens what's
        choosable (Hard Rule 2).
        """
        legal_indices = [i for i in range(N_ACTIONS) if mask[i]]
        if not legal_indices:
            raise ValueError("no legal action in mask -- a valid mask must have at least one True entry")

        epsilon = self._current_epsilon()
        self._act_calls += 1

        if random.random() < epsilon:
            return Action(random.choice(legal_indices))

        with torch.no_grad():
            q_values = self.q_network(flatten_state(state, self.has_forecast).unsqueeze(0)).squeeze(0)
        masked_q_values = q_values.clone()
        illegal = ~torch.as_tensor(np.asarray(mask, dtype=bool))
        masked_q_values[illegal] = float("-inf")
        best_action = int(torch.argmax(masked_q_values).item())
        return Action(best_action)

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
        self._replay_buffer.push(
            _Transition(
                state=flatten_state(state, self.has_forecast),
                action=int(action),
                reward=float(reward),
                next_state=flatten_state(next_state, self.has_forecast),
                next_mask=np.asarray(next_mask, dtype=bool).copy(),
                done=bool(done),
            )
        )

    def learn(self) -> dict[str, float]:
        """One gradient step on a sampled batch. Returns a loss/metrics dict.

        A no-op (returns `{"loss": 0.0, ...}`) until the buffer holds
        at least `batch_size` transitions -- there's nothing meaningful
        to sample yet.
        """
        if len(self._replay_buffer) < self.config.batch_size:
            return {"loss": 0.0, "buffer_size": float(len(self._replay_buffer))}

        batch = self._replay_buffer.sample(self.config.batch_size)

        states = torch.stack([t.state for t in batch])
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.stack([t.next_state for t in batch])
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)
        next_masks = torch.as_tensor(np.stack([t.next_mask for t in batch]), dtype=torch.bool)

        q_values = self.q_network(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            # Hard Rule 2 applies at bootstrap time too: an action the
            # mask would have forbidden next step must never contribute
            # to the target, or the network would be implicitly taught
            # that an illegal action was a good future to bootstrap
            # from.
            next_q_values = next_q_values.masked_fill(~next_masks, float("-inf"))
            next_q_max = next_q_values.max(dim=1).values
            # Defensive only: env/masking.py guarantees at least one
            # legal action always exists, so this should never actually
            # fire -- but a mask with nothing legal would otherwise
            # poison the whole batch's loss with -inf.
            next_q_max = torch.nan_to_num(next_q_max, neginf=0.0, posinf=0.0)
            targets = rewards + self.config.gamma * next_q_max * (1.0 - dones)

        loss = nn.functional.mse_loss(q_selected, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._learn_calls += 1
        if self._learn_calls % self.config.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {"loss": float(loss.item()), "buffer_size": float(len(self._replay_buffer))}

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dim": self.state_dim,
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "act_calls": self._act_calls,
                "learn_calls": self._learn_calls,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._act_calls = checkpoint["act_calls"]
        self._learn_calls = checkpoint["learn_calls"]
