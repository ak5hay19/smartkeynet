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
from collections import deque
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
    """Q-network with an optional dueling head (spec §S8, ladder rung 3).

    Dueling splits the estimate into a state value and per-action advantages.
    It helps most when many actions have similar value, which action masking
    makes common here: on a typical step two or three of the five actions are
    illegal and the rest are near-substitutes.

    THE CENTRING MUST BE OVER LEGAL ACTIONS ONLY. The spec calls this out as
    "a real and rarely-mentioned trap", and it is worth restating why:
    subtracting a mean taken over all five actions lets the advantages of
    *masked* actions -- which the network is never trained on, so they drift
    arbitrarily -- leak into `V(s)`. The value head then tracks noise from
    actions that could not be taken. Centring over the legal set keeps `V`
    estimating what it is supposed to estimate.

    `forward` without a mask falls back to centring over all actions, which is
    only correct when every action is legal; callers inside this module always
    pass the mask.
    """

    def __init__(self, state_dim: int, n_actions: int = N_ACTIONS, dueling: bool = True) -> None:
        super().__init__()
        self.dueling = dueling
        self.n_actions = n_actions

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )
        if dueling:
            self.value_head = nn.Linear(128, 1)
            self.advantage_head = nn.Linear(128, n_actions)
        else:
            self.head = nn.Linear(128, n_actions)

    def forward(self, state: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        features = self.trunk(state)
        if not self.dueling:
            return self.head(features)

        value = self.value_head(features)
        advantage = self.advantage_head(features)

        if mask is None:
            advantage_mean = advantage.mean(dim=1, keepdim=True)
        else:
            legal = mask.to(advantage.dtype)
            legal_count = legal.sum(dim=1, keepdim=True).clamp(min=1.0)
            advantage_mean = (advantage * legal).sum(dim=1, keepdim=True) / legal_count

        return value + advantage - advantage_mean


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DQNConfig:
    """Hyperparameters for `DQNAgent` (values live in
    `configs/default.yaml`)."""

    gamma: float = 0.995
    lr: float = 1e-3
    batch_size: int = 64
    target_update_every: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000

    # --- upgrade-ladder switches (SMARTKEYNET_BUILD_SPEC.md §8) ---
    dueling: bool = True
    """Dueling architecture with legal-only advantage centring (spec §S8,
    ladder rung 3). Helps when many actions are near-substitutes, which
    masking makes common."""

    double: bool = True
    """Double DQN: select the bootstrap action with the online network,
    evaluate it with the target network. Spec §8 ranks this second on
    the ladder and calls it "always worth it" -- a plain `max` over a
    noisy target network systematically overestimates, and action
    masking makes that worse rather than better, since the max is taken
    over a legal set whose size varies step to step."""

    n_step: int = 3
    """n-step returns. Spec §8 calls this "the highest-value rung for
    this problem", because credit has to travel through the pool's slow
    dynamics: a discretionary hybrid serve and the deferral it
    eventually causes can be a hundred steps apart, and 1-step backups
    move that signal only one transition per gradient update."""

    huber_delta: float = 1.0
    """Huber (smooth L1) loss instead of MSE. With `r_starve` an order of
    magnitude larger than the other reward terms, a single starvation
    step produces a large TD error that MSE squares into a gradient
    spike."""

    grad_clip_norm: float = 10.0
    """Global gradient-norm clip. 0 disables clipping."""


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
    mask: np.ndarray
    """Mask of `state` itself. Needed by dueling's legal-only advantage
    centring; `next_mask` alone is not enough."""

    done: bool
    n_steps: int = 1
    """How many environment steps of reward `reward` accumulates, and
    therefore the power gamma is raised to when bootstrapping. 1 for a
    plain one-step transition; see `DQNAgent._collapse_n_step_window`."""


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


class RunningMeanStd:
    """Welford running mean/variance per state feature.

    WHY THIS EXISTS. `flatten_state` emits raw environment units, and
    they are not remotely on the same scale: `key_age` runs to 500 (the
    SP 800-57 lifetime cap) while every other feature sits at or below
    3. Measured over 600 real states, feature 12 had max |x| = 500.0
    against a next-largest of 3.0 -- a 150x disparity feeding an
    unnormalised MLP.

    The consequence was not subtle. The first layer was dominated by
    key age, so pool level, policy floor and threat score were
    effectively noise, and the learned policy was degenerate: it chose
    `REKEY_NOW` on 900 of 1,000 decisions and chose `REUSE` **never**,
    despite REUSE being legal on 836 of them and strictly cheaper on
    every one (0.2 vs 1.0+ latency, and no rekey cost). A policy that
    systematically avoids a strictly dominant action does not have a
    hard exploration problem; it has broken inputs.

    SMARTKEYNET_BUILD_SPEC.md §3.2 specifies exactly this component --
    `obs_norm: running_mean_std  # frozen at eval` -- and it had simply
    never been implemented.

    **Frozen at eval** is load-bearing and not a detail: if the
    statistics kept updating during evaluation, two policies would see
    differently-normalised inputs depending on the states they happened
    to visit, and the comparison would not be like-for-like.
    """

    def __init__(self, n_features: int, epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.var = np.ones(n_features, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Standardise, then clip. Clipping bounds the damage a single
        outlier state can do to a gradient step -- standard practice,
        and cheap insurance given how skewed these features are."""
        normalized = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -10.0, 10.0)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


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

        self.q_network = QNetwork(state_dim, dueling=self.config.dueling)
        self.target_network = QNetwork(state_dim, dueling=self.config.dueling)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.config.lr)
        self._replay_buffer = _ReplayBuffer(capacity=_REPLAY_BUFFER_CAPACITY)
        # observation normalisation (spec §3.2 `obs_norm: running_mean_std`)
        self.obs_rms = RunningMeanStd(state_dim)
        self.normalizer_frozen = False
        self._last_act_mask: np.ndarray = np.ones(N_ACTIONS, dtype=bool)
        # pending n-step accumulation window (see `observe`)
        self._n_step_window: deque[_Transition] = deque(maxlen=max(1, self.config.n_step))

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
        # remembered so `observe` can store the mask this decision was made
        # under -- dueling needs it to centre advantages over legal actions
        self._last_act_mask = np.asarray(mask, dtype=bool).copy()

        if random.random() < epsilon:
            return Action(random.choice(legal_indices))

        with torch.no_grad():
            mask_tensor = torch.as_tensor(np.asarray(mask, dtype=bool)).unsqueeze(0)
            q_values = self.q_network(
                self.normalized_state(state).unsqueeze(0), mask_tensor
            ).squeeze(0)
        masked_q_values = q_values.clone()
        illegal = ~torch.as_tensor(np.asarray(mask, dtype=bool))
        masked_q_values[illegal] = float("-inf")
        best_action = int(torch.argmax(masked_q_values).item())
        return Action(best_action)

    def normalized_state(self, state: StateDict, update: bool = False) -> torch.Tensor:
        """Flatten and normalise one state.

        `update` is True only on the training path. Evaluation must not
        move the statistics, or two policies scored on the same seeds
        would see different input distributions.
        """
        flat = flatten_state(state, self.has_forecast).numpy().astype(np.float64)
        if update and not self.normalizer_frozen:
            self.obs_rms.update(flat.reshape(1, -1))
        return torch.tensor(self.obs_rms.normalize(flat), dtype=torch.float32)

    def freeze_normalizer(self) -> None:
        """Stop updating observation statistics -- call before evaluation."""
        self.normalizer_frozen = True

    def observe(
        self,
        state: StateDict,
        action: Action,
        reward: float,
        next_state: StateDict,
        next_mask: ActionMask,
        done: bool,
    ) -> None:
        """Accumulate an n-step return, then push to the replay buffer.

        With `config.n_step == 1` this is exactly the old behaviour: the
        transition goes straight in. For `n > 1` the transition is held
        in a short FIFO window and only emitted once `n` steps of reward
        have accumulated behind it, so the stored transition carries

            R = sum_{k=0}^{n-1} gamma^k * r_{t+k}

        and bootstraps from `s_{t+n}` instead of `s_{t+1}` (with
        `discount = gamma^n` applied in `learn`). Spec §8: this is the
        highest-value rung of the upgrade ladder here, because the pool's
        dynamics are slow -- a discretionary hybrid serve and the
        deferral it causes can be a hundred steps apart.

        On `done` the window is flushed, emitting every partially
        accumulated transition, so no experience is silently dropped at
        an episode boundary.
        """
        self._n_step_window.append(
            _Transition(
                state=self.normalized_state(state, update=True),
                action=int(action),
                reward=float(reward),
                next_state=self.normalized_state(next_state, update=False),
                next_mask=np.asarray(next_mask, dtype=bool).copy(),
                mask=self._last_act_mask.copy(),
                done=bool(done),
            )
        )

        if len(self._n_step_window) >= self.config.n_step:
            self._replay_buffer.push(self._collapse_n_step_window())

        if done:
            while self._n_step_window:
                self._replay_buffer.push(self._collapse_n_step_window())

    def _collapse_n_step_window(self) -> _Transition:
        """Fold the pending window into one n-step transition and drop
        its oldest entry.

        The emitted transition keeps the *first* entry's state and
        action (that is the decision being credited), accumulates the
        discounted reward across the window, and takes its bootstrap
        state/mask from the *last* entry. `n_steps` records how many
        rewards were actually accumulated, which is what `learn` raises
        gamma to -- it is shorter than `config.n_step` only for the
        tail transitions flushed at an episode boundary.
        """
        first = self._n_step_window[0]

        accumulated_reward = 0.0
        discount = 1.0
        last = first
        for n_accumulated, transition in enumerate(self._n_step_window, start=1):
            accumulated_reward += discount * transition.reward
            discount *= self.config.gamma
            last = transition
            if transition.done:
                break
        else:
            n_accumulated = len(self._n_step_window)

        collapsed = _Transition(
            state=first.state,
            action=first.action,
            reward=accumulated_reward,
            next_state=last.next_state,
            next_mask=last.next_mask,
            mask=first.mask,
            done=last.done,
            n_steps=n_accumulated,
        )
        self._n_step_window.popleft()
        return collapsed

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
        # the mask of s itself, for dueling's legal-only advantage centring
        masks = torch.as_tensor(np.stack([t.mask for t in batch]), dtype=torch.bool)
        n_steps = torch.tensor([t.n_steps for t in batch], dtype=torch.float32)

        q_values = self.q_network(states, masks)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Hard Rule 2 applies at bootstrap time too: an action the
            # mask would have forbidden next step must never contribute
            # to the target, or the network would be implicitly taught
            # that an illegal action was a good future to bootstrap
            # from. Under Double DQN the mask has to be applied to BOTH
            # networks -- the online net that selects the action and the
            # target net that evaluates it -- or the selection step
            # reintroduces exactly the illegal-action leak the masking
            # is there to prevent.
            target_q_values = self.target_network(next_states, next_masks)
            if self.config.double:
                online_next_q = self.q_network(next_states, next_masks)
                online_next_q = online_next_q.masked_fill(~next_masks, float("-inf"))
                best_actions = online_next_q.argmax(dim=1, keepdim=True)
                next_q_max = target_q_values.gather(1, best_actions).squeeze(1)
            else:
                masked_target_q = target_q_values.masked_fill(~next_masks, float("-inf"))
                next_q_max = masked_target_q.max(dim=1).values

            # Defensive only: env/masking.py guarantees at least one
            # legal action always exists, so this should never actually
            # fire -- but a mask with nothing legal would otherwise
            # poison the whole batch's loss with -inf.
            next_q_max = torch.nan_to_num(next_q_max, neginf=0.0, posinf=0.0)
            # gamma^n, because `rewards` already accumulates n steps of
            # discounted reward (see `_collapse_n_step_window`).
            discounts = torch.pow(
                torch.tensor(self.config.gamma, dtype=torch.float32), n_steps.float()
            )
            targets = rewards + discounts * next_q_max * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_selected, targets, beta=self.config.huber_delta)

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        self._learn_calls += 1
        if self._learn_calls % self.config.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {"loss": float(loss.item()), "buffer_size": float(len(self._replay_buffer))}

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dim": self.state_dim,
                "obs_rms": self.obs_rms.state_dict(),
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "act_calls": self._act_calls,
                "learn_calls": self._learn_calls,
            },
            path,
        )

    def load(self, path: str) -> None:
        # weights_only=False: the checkpoint also carries the observation
        # normaliser's numpy statistics, which are not tensors.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._act_calls = checkpoint["act_calls"]
        self._learn_calls = checkpoint["learn_calls"]
        if "obs_rms" in checkpoint:
            # Restoring the observation statistics is not optional: a
            # checkpoint evaluated against different normalisation than
            # it was trained under is a different function.
            self.obs_rms.load_state_dict(checkpoint["obs_rms"])
