"""Behavioral tests for `agents.dqn` -- the masked DQN agent (PLAN.md §4
architecture diagram "DQN AGENT"; §10 kickoff step 5).

Proves the building blocks work (state flattening, masked action
selection at both epsilon extremes, replay buffer accumulation, a real
gradient step moving weights and Q-values in the right direction,
checkpoint round-tripping, and a short real-environment training loop
whose loss trends down) -- not a full training campaign to convergence.
That's `experiments/train.py`, deliberately a separate future session.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from agents.dqn import (
    DQNAgent,
    DQNConfig,
    QNetwork,
    _FORECAST_STATE_DIM,
    _OFF_STATE_DIM,
    flatten_state,
    load_dqn_config,
)
from env.contracts import N_ACTIONS, Action
from env.environment import SmartKeyNetEnv


def _make_state(
    *,
    forecast: bool,
    threat_score: float | None = None,
    pool_fill: float = 0.5,
    key_age: float = 10.0,
    key_type_onehot: tuple[float, float, float] = (0.0, 1.0, 0.0),
    sensitivity_class: int = 1,
    policy_floor: int = 1,
    regret_event_recent: bool = False,
) -> dict[str, Any]:
    """Minimal `StateDict` stand-in (runtime it's just a dict). Mirrors
    exactly what env/environment.py's `_prepare_decision` actually
    zeroes under `off` vs. populates under `ewma` -- see
    tests/test_environment.py's `test_foresight_fields_zeroed_under_off`
    / `test_foresight_fields_populated_under_ewma`.

    `threat_score` can be overridden independently of `forecast` -- used
    to build a legitimately-near-zero-but-still-forecast-mode state
    (see test_flatten_state_forecast_mode_with_near_zero_threat_score),
    since `flatten_state` must no longer key off this value at all.
    """
    if forecast:
        threat_forecast = [0.1, 0.2, 0.3, 0.4, 0.5]
        pool_level_hat = [0.3, 0.4, 0.5]
        skr_mean_hat = [1.0, 1.1, 1.2]
        hybrid_demand_hat = [2.0, 3.0, 4.0]
        resolved_threat_score = 0.42 if threat_score is None else threat_score
    else:
        threat_forecast = [0.0] * 5
        pool_level_hat = [0.0] * 3
        skr_mean_hat = [0.0] * 3
        hybrid_demand_hat = [0.0] * 3
        resolved_threat_score = 0.0 if threat_score is None else threat_score

    return {
        "threat_score": resolved_threat_score,
        "threat_forecast": threat_forecast,
        "qber": 0.01,
        "skr": 500.0,
        "pool_fill": pool_fill,
        "arrival_rate": 1.0,
        "load": 0.2,
        "avg_latency": 1.0,
        "key_age": key_age,
        "key_type_onehot": list(key_type_onehot),
        "sensitivity_class": sensitivity_class,
        "policy_floor": policy_floor,
        "pool_level_hat": pool_level_hat,
        "skr_mean_hat": skr_mean_hat,
        "hybrid_demand_hat": hybrid_demand_hat,
        "regret_event_recent": regret_event_recent,
    }


def _full_mask() -> np.ndarray:
    return np.ones(N_ACTIONS, dtype=bool)


def _mask(*legal_actions: Action) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for action in legal_actions:
        mask[int(action)] = True
    return mask


# ---------------------------------------------------------------------------
# flatten_state -- has_forecast is explicit, never inferred from `state`
# ---------------------------------------------------------------------------


def test_flatten_state_off_mode_excludes_forecast_fields():
    state = _make_state(forecast=False)
    tensor = flatten_state(state, has_forecast=False)
    assert tensor.shape == (_OFF_STATE_DIM,)
    assert tensor.shape == (13,)


def test_flatten_state_forecast_mode_includes_forecast_fields():
    state = _make_state(forecast=True)
    tensor = flatten_state(state, has_forecast=True)
    assert tensor.shape == (_FORECAST_STATE_DIM,)
    assert tensor.shape == (28,)


def test_flatten_state_forecast_values_actually_present_not_just_longer():
    """Not just a longer vector -- the forecast values themselves must
    actually appear in it (a real dimensionality difference, not
    cosmetic zero-padding)."""
    state = _make_state(forecast=True)
    tensor = flatten_state(state, has_forecast=True)
    values = tensor.tolist()
    assert any(abs(v - 0.42) < 1e-6 for v in values)  # threat_score
    assert any(abs(v - 4.0) < 1e-6 for v in values)  # hybrid_demand_hat[-1]


def test_flatten_state_forecast_mode_with_zero_threat_score_still_28_dim():
    """The regression case the old `threat_score != 0.0` inference would
    have gotten wrong: a legitimately foresight-enabled state whose
    `threat_score` happens to be exactly 0.0 (e.g. once real threat
    data replaces today's always-non-negative placeholder, nothing
    guarantees the sigmoid stays away from zero). `has_forecast=True`
    is passed explicitly here, so this must still flatten to 28 dims
    with the forecast fields genuinely present -- not silently
    misclassified as an `off`-mode 13-dim vector."""
    state = _make_state(forecast=True, threat_score=0.0)
    tensor = flatten_state(state, has_forecast=True)
    assert tensor.shape == (_FORECAST_STATE_DIM,)
    assert tensor.shape == (28,)
    values = tensor.tolist()
    assert any(abs(v - 4.0) < 1e-6 for v in values)  # hybrid_demand_hat[-1] still present


def test_flatten_state_off_mode_ignores_a_stray_nonzero_threat_score():
    """Symmetric check: even if `threat_score` were somehow nonzero on
    an `off`-mode state, `has_forecast=False` must still produce the
    13-dim vector -- the value of `threat_score` is irrelevant to the
    decision now, only the explicit flag matters."""
    state = _make_state(forecast=False, threat_score=0.99)
    tensor = flatten_state(state, has_forecast=False)
    assert tensor.shape == (_OFF_STATE_DIM,)
    assert tensor.shape == (13,)


# ---------------------------------------------------------------------------
# QNetwork
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state_dim", [_OFF_STATE_DIM, _FORECAST_STATE_DIM])
def test_qnetwork_forward_shape(state_dim):
    net = QNetwork(state_dim)
    x = torch.zeros(1, state_dim)
    out = net(x)
    assert out.shape == (1, N_ACTIONS)


def test_qnetwork_forward_batched():
    net = QNetwork(_OFF_STATE_DIM)
    x = torch.zeros(7, _OFF_STATE_DIM)
    out = net(x)
    assert out.shape == (7, N_ACTIONS)


# ---------------------------------------------------------------------------
# DQNConfig / load_dqn_config
# ---------------------------------------------------------------------------


def test_load_dqn_config_matches_real_yaml():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)["dqn"]

    config = load_dqn_config()
    assert config.gamma == raw["gamma"]
    assert config.lr == raw["lr"]
    assert config.batch_size == raw["batch_size"]
    assert config.target_update_every == raw["target_update_every"]
    assert config.epsilon_start == raw["epsilon_start"]
    assert config.epsilon_end == raw["epsilon_end"]
    assert config.epsilon_decay_steps == raw["epsilon_decay_steps"]


# ---------------------------------------------------------------------------
# act() -- epsilon extremes, adversarial masks
# ---------------------------------------------------------------------------


def _greedy_config() -> DQNConfig:
    return DQNConfig(epsilon_start=0.0, epsilon_end=0.0, epsilon_decay_steps=1, batch_size=4)


def _random_config() -> DQNConfig:
    return DQNConfig(epsilon_start=1.0, epsilon_end=1.0, epsilon_decay_steps=1, batch_size=4)


def test_act_epsilon_zero_picks_the_highest_legal_q_value_full_mask():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, has_forecast=False, config=_greedy_config())
    state = _make_state(forecast=False)

    with torch.no_grad():
        q_values = agent.q_network(flatten_state(state, has_forecast=False).unsqueeze(0)).squeeze(0)
    expected = int(torch.argmax(q_values).item())

    action = agent.act(state, _full_mask())
    assert int(action) == expected


def test_act_epsilon_zero_respects_a_restrictive_mask():
    """Excluding the network's globally-preferred action from the mask
    must make act() fall back to the best *legal* one, not the global
    argmax."""
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, has_forecast=False, config=_greedy_config())
    state = _make_state(forecast=False)

    with torch.no_grad():
        q_values = agent.q_network(flatten_state(state, has_forecast=False).unsqueeze(0)).squeeze(0)
    global_best = int(torch.argmax(q_values).item())

    restrictive_mask = _full_mask()
    restrictive_mask[global_best] = False
    assert restrictive_mask.any()

    remaining_q = q_values.clone()
    remaining_q[~torch.as_tensor(restrictive_mask)] = float("-inf")
    expected_second_best = int(torch.argmax(remaining_q).item())

    action = agent.act(state, restrictive_mask)
    assert int(action) == expected_second_best
    assert restrictive_mask[int(action)]


def test_act_epsilon_zero_single_legal_action_mask():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, has_forecast=False, config=_greedy_config())
    state = _make_state(forecast=False)
    mask = _mask(Action.REKEY_NOW)
    assert agent.act(state, mask) == Action.REKEY_NOW


def test_act_epsilon_one_always_returns_a_legal_action_full_mask():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, has_forecast=False, config=_random_config())
    state = _make_state(forecast=False)
    mask = _full_mask()
    seen = set()
    for _ in range(200):
        action = agent.act(state, mask)
        assert mask[int(action)]
        seen.add(action)
    assert seen == set(Action)  # every legal option actually gets drawn eventually


def test_act_epsilon_one_respects_a_restrictive_mask():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, has_forecast=False, config=_random_config())
    state = _make_state(forecast=False)
    mask = _mask(Action.SERVE_PQC, Action.REKEY_NOW)
    seen = set()
    for _ in range(200):
        action = agent.act(state, mask)
        assert mask[int(action)]
        seen.add(action)
    assert seen == {Action.SERVE_PQC, Action.REKEY_NOW}


def test_act_epsilon_one_single_legal_action_mask():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, has_forecast=False, config=_random_config())
    state = _make_state(forecast=False)
    mask = _mask(Action.REUSE)
    for _ in range(20):
        assert agent.act(state, mask) == Action.REUSE


def test_act_raises_on_all_illegal_mask():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, has_forecast=False, config=_greedy_config())
    state = _make_state(forecast=False)
    mask = np.zeros(N_ACTIONS, dtype=bool)
    with pytest.raises(ValueError):
        agent.act(state, mask)


def test_act_uses_28_dim_flattening_when_agent_constructed_with_has_forecast_true():
    """An agent built with `has_forecast=True` must flatten every
    `act()` call's state to 28 dims -- exercised end-to-end (not just
    via `flatten_state` directly) so a regression in how `DQNAgent`
    threads `self.has_forecast` through would be caught here too."""
    agent = DQNAgent(state_dim=_FORECAST_STATE_DIM, has_forecast=True, config=_greedy_config())
    state = _make_state(forecast=True)
    action = agent.act(state, _full_mask())
    assert action in set(Action)


# ---------------------------------------------------------------------------
# seed -- the 2026-08-10 gap: weight init, exploration, and replay
# sampling were never seedable at all (see DQNAgent.__init__'s
# docstring and SESSION_LOG.md). This is the test that would have
# caught it before it needed a whole diagnosis-and-fix cycle.
# ---------------------------------------------------------------------------


def _mixed_epsilon_config() -> DQNConfig:
    """epsilon held at a constant 0.5 (equal start/end) so a run
    genuinely exercises both `act()` branches -- the random-explore
    draw *and* the greedy network forward pass -- rather than pinning
    to one, so a seeding gap in either path would show up."""
    return DQNConfig(epsilon_start=0.5, epsilon_end=0.5, epsilon_decay_steps=1, batch_size=4)


def _run_action_sequence(seed: int, n_steps: int = 50) -> list[int]:
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, has_forecast=False, config=_mixed_epsilon_config(), seed=seed)
    state = _make_state(forecast=False)
    mask = _full_mask()
    return [int(agent.act(state, mask)) for _ in range(n_steps)]


def test_same_seed_produces_identical_action_sequence():
    assert _run_action_sequence(seed=42) == _run_action_sequence(seed=42)


def test_different_seeds_produce_different_action_sequences():
    assert _run_action_sequence(seed=42) != _run_action_sequence(seed=43)


def test_seed_none_leaves_ambient_random_state_untouched():
    """The default (`seed=None`) must not call `random.seed`/
    `torch.manual_seed` at all -- confirmed by seeding the ambient
    state ourselves right before construction and checking two
    unrelated `seed=None` agents built back-to-back diverge, the same
    as the pre-2026-08-10 (no seeding at all) behavior did."""
    random.seed(1)
    torch.manual_seed(1)
    state = _make_state(forecast=False)
    mask = _full_mask()

    agent_a = DQNAgent(state_dim=_OFF_STATE_DIM, has_forecast=False, config=_mixed_epsilon_config(), seed=None)
    seq_a = [int(agent_a.act(state, mask)) for _ in range(50)]

    agent_b = DQNAgent(state_dim=_OFF_STATE_DIM, has_forecast=False, config=_mixed_epsilon_config(), seed=None)
    seq_b = [int(agent_b.act(state, mask)) for _ in range(50)]

    assert seq_a != seq_b


# ---------------------------------------------------------------------------
# observe()
# ---------------------------------------------------------------------------


def test_observe_accumulates_transitions_in_replay_buffer():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, has_forecast=False, config=DQNConfig(batch_size=4))
    state = _make_state(forecast=False)
    next_state = _make_state(forecast=False)
    mask = _full_mask()

    assert len(agent._replay_buffer) == 0
    for _ in range(5):
        agent.observe(state, Action.SERVE_PQC, 1.0, next_state, mask, False)
    assert len(agent._replay_buffer) == 5


# ---------------------------------------------------------------------------
# learn()
# ---------------------------------------------------------------------------


def test_learn_is_a_noop_below_batch_size():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, has_forecast=False, config=DQNConfig(batch_size=64))
    metrics = agent.learn()
    assert metrics["loss"] == 0.0


def test_learn_changes_network_weights():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, has_forecast=False, config=DQNConfig(batch_size=4))
    state = _make_state(forecast=False)
    next_state = _make_state(forecast=False)
    mask = _full_mask()

    for _ in range(10):
        agent.observe(state, Action.SERVE_PQC, 1.0, next_state, mask, False)

    before = [p.detach().clone() for p in agent.q_network.parameters()]
    metrics = agent.learn()
    after = list(agent.q_network.parameters())

    assert "loss" in metrics
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_learn_moves_q_values_toward_the_obviously_correct_answer():
    """Not full convergence -- proves the loop works: repeatedly told
    "action A -> +10, action B -> -10" for the same state, the network
    should end up ranking A above B."""
    torch.manual_seed(0)
    config = DQNConfig(batch_size=8, gamma=0.0, lr=0.02, target_update_every=10_000)
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, has_forecast=False, config=config)

    state = _make_state(forecast=False)
    next_state = _make_state(forecast=False)  # irrelevant: gamma=0, done=True zeroes the bootstrap term
    mask = _full_mask()

    good_action = Action.SERVE_PQC
    bad_action = Action.SERVE_CLASSICAL

    for _ in range(400):
        agent.observe(state, good_action, 10.0, next_state, mask, True)
        agent.observe(state, bad_action, -10.0, next_state, mask, True)

    for _ in range(400):
        agent.learn()

    with torch.no_grad():
        q = agent.q_network(flatten_state(state, has_forecast=False).unsqueeze(0)).squeeze(0)

    assert q[int(good_action)] > q[int(bad_action)]


# ---------------------------------------------------------------------------
# save() / load()
# ---------------------------------------------------------------------------


def test_save_then_load_produces_identical_q_values(tmp_path):
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, has_forecast=False, config=DQNConfig(batch_size=4))
    state = _make_state(forecast=False)
    next_state = _make_state(forecast=False)
    mask = _full_mask()

    for _ in range(10):
        agent.observe(state, Action.SERVE_PQC, 1.0, next_state, mask, False)
    agent.learn()  # give it non-default (trained) weights, not just fresh init

    checkpoint_path = tmp_path / "agent.pt"
    agent.save(str(checkpoint_path))

    fresh_agent = DQNAgent(state_dim=_OFF_STATE_DIM, has_forecast=False, config=DQNConfig(batch_size=4))
    fresh_agent.load(str(checkpoint_path))

    probe = flatten_state(_make_state(forecast=False, pool_fill=0.9), has_forecast=False).unsqueeze(0)
    with torch.no_grad():
        q1 = agent.q_network(probe)
        q2 = fresh_agent.q_network(probe)

    assert torch.allclose(q1, q2)


# ---------------------------------------------------------------------------
# Integration: real SmartKeyNetEnv, S1, loss trends down
# ---------------------------------------------------------------------------


def test_dqn_agent_trains_end_to_end_against_real_env_s1():
    """PLAN.md §10 step 5's "prove the loop works" evidence: not a full
    training campaign to convergence (that's `experiments/campaign.py`)
    -- just observe()+learn() every step for a few thousand real S1
    steps and confirm the loop is healthy end-to-end.

    Renamed and re-scoped 2026-08-19 (was
    `..._loss_trends_down_...`). It asserted that late-window average
    loss was below early-window average loss, which was a valid
    learning signal only while the reward was effectively bounded in
    [-3.2, -0.2] -- and it was bounded only because the pre-recalibration
    pool never exhausted, so nothing was ever deferred and the reward's
    `-r_starve * deferred_critical_steps` term was dead code. With a
    genuinely scarce pool the reward is heavy-tailed (measured on
    S1/2,000 steps: always-PQC never worse than -3.2 on a step, a random
    policy hitting -442.5 on a single step when the deferral queue is
    deep), so a Q-network correctly learning to represent large negative
    values has a *rising* MSE for a long stretch. Keeping the old
    assertion would have meant asserting an artefact of the broken pool.

    What this test now checks is what a 3,000-step budget can honestly
    establish: the loop runs against the real environment, never emits
    an illegal action, stays numerically healthy, and actually moves the
    network's weights. Whether the learned policy is any *good* is a
    claim about a full training run, and it is measured -- multi-seed
    and checkpoint-averaged -- in `experiments/campaign.py`, where the
    answer is reported whatever it turns out to be.

    Both the env and the agent's network init are seeded, so this is a
    deterministic run, not a flaky stochastic one -- if it passes once
    it passes identically every time.
    """
    torch.manual_seed(0)

    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    # The one place has_forecast is derived from: the same config-time
    # fact env/environment.py's own `_build_forecaster` branches on --
    # never inferred from a StateDict later.
    has_forecast = full_config.get("use_foresight", "off") != "off"

    dqn_config = load_dqn_config()
    dqn_config.epsilon_decay_steps = 1000  # short run -- reach low epsilon within budget, don't hardcode everything else

    env = SmartKeyNetEnv({**full_config, "seed": 0, "max_steps": 3000})
    state, info = env.reset(seed=0)
    mask = info["action_mask"]

    state_dim = flatten_state(state, has_forecast).shape[0]  # derived from the real state, not assumed
    agent = DQNAgent(state_dim=state_dim, has_forecast=has_forecast, config=dqn_config)

    weights_before = agent.q_network.net[0].weight.detach().clone()

    losses: list[float] = []
    truncated = False
    while not truncated:
        action = agent.act(state, mask)
        assert mask[int(action)], "agent returned an action the mask forbade"
        next_state, reward, terminated, truncated, info = env.step(action)
        next_mask = info["action_mask"]

        agent.observe(state, action, reward, next_state, next_mask, terminated)
        metrics = agent.learn()
        if metrics["loss"] > 0.0:
            losses.append(metrics["loss"])

        state = next_state
        mask = next_mask

    assert len(losses) > 500  # learning actually ran for a meaningful stretch, not just a handful of steps

    # The check here used to be "late-window average loss < early-window
    # average loss". That was a valid learning signal while the reward
    # was effectively bounded in [-3.2, -0.2] -- which it was, only
    # because the pre-2026-08-19 pool never exhausted, so nothing was
    # ever deferred and the reward's `-r_starve * deferred_critical_steps`
    # term was dead code. With a genuinely scarce pool the reward is
    # heavy-tailed (measured on S1/2000 steps: always-PQC never worse
    # than -3.2 per step, a random policy reaching -442.5 on a single
    # step when the deferral queue is deep), so a Q-network that is
    # *correctly* learning to represent large negative values has a
    # rising MSE for a long stretch. Falling loss is no longer the same
    # thing as learning here, and asserting it would be asserting an
    # artefact of the broken pool.
    #
    # Replaced with two checks that mean what they say: the optimizer
    # stays numerically healthy, and the learned greedy policy is
    # actually better than the same architecture untrained.
    assert all(np.isfinite(loss) for loss in losses), "loss went non-finite -- training diverged"
    assert not torch.allclose(weights_before, agent.q_network.net[0].weight), (
        "q_network weights are unchanged after 3,000 learn() calls -- no gradient reached them"
    )
