"""
experiments/train.py

Real training campaign for `agents.dqn.DQNAgent`, per PLAN.md §10 step
5. Owned by Person C (split.md §1).

`train()`'s `scenario` parameter (2026-08-24, defaulting to `"S1"`)
follows `experiments/harness.py`'s `run_scenario`/`run_grid`
convention -- see `train()`'s own docstring. Before this, `train()`
hardcoded `scenario: "S1"` regardless of what config it was handed,
which is what made the real Gate W3 S3 attempt (SESSION_LOG.md
2026-08-24) have to reimplement this module's training loop in a
throwaway scratchpad script rather than calling `train()` directly;
that workaround is no longer necessary for any future multi-scenario
training campaign. `evaluate_against_baseline` below is unchanged and
still evaluates against S1 only -- out of this session's scope (see
PROGRESS.md/SESSION_LOG.md for the fix's exact boundaries).

Hard Rule 1 (no security term in the reward, ever): this module trains
against exactly the reward `env/environment.py` computes via the
transitions `DQNAgent.observe()` is given -- it adds, reshapes, or
substitutes nothing of its own, same as `agents/dqn.py` itself.

Design note -- greedy evaluation without touching agents/dqn.py:
`DQNAgent.act()` always uses its own internal epsilon-greedy schedule
(tied to `self._act_calls`, which also drives its epsilon *decay* for
training), so calling it directly during evaluation would both (a) not
reflect what the agent actually learned, since it could still explore
randomly, and (b) burn through the training epsilon-decay budget on
eval steps that were never real training experience. Since
`agents/dqn.py` is out of scope this session (flag changes there
first, per instructions), the resolution lives entirely in this
module: `GreedyDQNPolicy` below wraps a trained agent and calls its
`q_network` directly, replicating `act()`'s greedy branch exactly
(illegal actions masked to `-inf` before `argmax`) without ever calling
`agent.act()` or touching `agent._act_calls` -- a read-only forward
pass, zero side effects on the agent's training state. It also
satisfies `agents.baselines.Policy`'s `act(state, mask) -> Action`
shape, so it drops straight into `experiments/harness.py`'s
`run_scenario`/`run_grid` exactly like any other policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from agents.baselines import StaticThresholdPolicy
from agents.dqn import DQNAgent, flatten_state, load_dqn_config
from agents.soft_reward_baseline import SoftRewardConfig, compute_soft_reward
from env.contracts import Action, ActionMask, StateDict
from env.environment import SmartKeyNetEnv
from experiments.harness import ScenarioResult, run_scenario

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
_SOFT_REWARD_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "soft_reward_baseline.yaml"


def load_full_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the real `configs/default.yaml` (or an explicit override
    path) -- mirrors the `load_test_config`-style helpers other test
    modules already use, but lives here since `train()`/`main()` need
    it too, not just tests."""
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class GreedyDQNPolicy:
    """Wraps a trained `DQNAgent` for greedy (epsilon=0) evaluation --
    see the module docstring's design note. Read-only: never mutates
    `agent`."""

    agent: DQNAgent

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        with torch.no_grad():
            q_values = self.agent.q_network(
                flatten_state(state, self.agent.has_forecast).unsqueeze(0)
            ).squeeze(0)
        masked_q_values = q_values.clone()
        illegal = ~torch.as_tensor(np.asarray(mask, dtype=bool))
        masked_q_values[illegal] = float("-inf")
        return Action(int(torch.argmax(masked_q_values).item()))


@dataclass
class TrainingRecord:
    """Tracked curve from one `train()` call: per-learn-step loss (only
    logged once the replay buffer is full enough for `learn()` to take
    a real gradient step -- see `agents.dqn.DQNAgent.learn`), a
    rolling average of raw training reward per `eval_every` window (the
    cheap, always-available overfitting signal -- no extra env needed),
    and periodic greedy-mode `ScenarioResult` snapshots via
    `experiments/harness.py` (the real "what would this checkpoint
    score" evidence, per PLAN.md §10 step 5)."""

    loss_steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    reward_window_avgs: list[float] = field(default_factory=list)
    eval_snapshots: list[tuple[int, ScenarioResult]] = field(default_factory=list)
    checkpoint_path: str | None = None


def train(
    full_config: dict[str, Any] | None = None,
    training_overrides: dict[str, Any] | None = None,
    scenario: str = "S1",
) -> tuple[DQNAgent, TrainingRecord]:
    """Run one continuous training episode against `scenario` (the env
    has no natural terminal state -- see env/environment.py -- so
    training never needs to reset mid-run) for `training.total_steps`
    steps, `observe()`+`learn()` every step, with a periodic
    greedy-mode evaluation snapshot (also run against `scenario`) every
    `training.eval_every` steps. Saves a final checkpoint via
    `DQNAgent.save`.

    `scenario` is a real, explicit parameter -- not read from
    `full_config["scenario"]` -- matching `experiments/harness.py`'s
    `run_scenario`/`run_grid` convention exactly (`scenario` is threaded
    into the env config, overriding whatever `full_config` itself says,
    so the same `full_config` can in principle be pointed at different
    scenarios by varying this argument alone). Defaults to `"S1"` so
    every pre-existing call site that doesn't pass it behaves
    byte-for-byte identically to before this parameter existed.

    The Hard Rule 8 `train_eligible` guard below is keyed on
    `full_config["train_eligible"]`, not on this `scenario` argument --
    see the guard's own comment for why (it must survive regardless of
    what scenario is requested, e.g. a config that sets `train_eligible:
    false` for a scenario other than S6 in the future).

    `training_overrides` shallow-merges over the real `configs/
    default.yaml`'s `training:` block (itself flat, no nested dicts) --
    tests use this to shrink `total_steps`/`eval_every`/`eval_max_steps`
    down to a CI-fast smoke run without touching this file's real
    defaults.
    """
    full_config = full_config if full_config is not None else load_full_config()

    # Hard Rule 8 guard (2026-08-24): "train on stationary scenarios;
    # the migration-wave scenario is held-out evaluation only." A
    # config-level `train_eligible` flag (default True -- every
    # existing config is silently unaffected) is checked before any
    # training work happens, not merely documented -- `configs/
    # scenarios/s6_migration.yaml` is the one config that sets this
    # `False`. Deliberately keyed on the flag itself, not on a hardcoded
    # `scenario == "S6"` string check, nor on the `scenario` parameter
    # above: this stays correct regardless of what scenario is
    # requested (e.g. a config that sets `train_eligible: false` for
    # some other scenario in the future) -- see this session's fix
    # (2026-08-24, `scenario` param) that generalized `train()` past its
    # former hardcoded `scenario: "S1"` override without touching this
    # guard's own logic or location, per instruction.
    if not full_config.get("train_eligible", True):
        raise ValueError(
            f"refusing to train: scenario {full_config.get('scenario')!r} has "
            "train_eligible: false (Hard Rule 8 -- the migration-wave scenario is "
            "held-out evaluation only; training on its schedule would mean "
            "memorizing the timeline, proving nothing). Use experiments/harness.py's "
            "run_scenario/run_grid to evaluate a policy against it instead."
        )

    training_cfg = {**full_config["training"], **(training_overrides or {})}

    # The one place has_forecast is derived from -- config-time, never
    # inferred from a StateDict's contents (2026-08-08 fix; see
    # agents/dqn.py's flatten_state docstring).
    has_forecast = full_config.get("use_foresight", "off") != "off"

    env_config = {**full_config, "scenario": scenario, "seed": training_cfg["seed"]}
    env = SmartKeyNetEnv(env_config)
    state, info = env.reset(seed=training_cfg["seed"])
    mask = info["action_mask"]

    state_dim = flatten_state(state, has_forecast).shape[0]  # derived from the real state, not assumed
    dqn_config = load_dqn_config()
    # Same integer as env_config["seed"]/env.reset(seed=...) above --
    # safe to reuse: DQNAgent's seed only touches Python's/torch's
    # global RNG, while SmartKeyNetEnv/random_request_generator use
    # their own local np.random.default_rng(seed) instances (see
    # env/pool_sim.py, env/request_generator.py). Genuinely
    # independent RNG systems, not a collision. Before this, only the
    # environment side was seeded -- see agents/dqn.py's DQNAgent.__init__
    # docstring and SESSION_LOG.md 2026-08-10 for why that mattered.
    agent = DQNAgent(
        state_dim=state_dim, has_forecast=has_forecast, config=dqn_config, seed=training_cfg["seed"]
    )

    total_steps = training_cfg["total_steps"]
    eval_every = training_cfg["eval_every"]
    eval_seed = training_cfg["eval_seed"]
    eval_max_steps = training_cfg["eval_max_steps"]

    record = TrainingRecord()
    reward_window: list[float] = []

    for step in range(1, total_steps + 1):
        action = agent.act(state, mask)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_mask = info["action_mask"]

        agent.observe(state, action, reward, next_state, next_mask, terminated)
        metrics = agent.learn()

        reward_window.append(reward)
        if metrics["loss"] > 0.0:
            record.loss_steps.append(step)
            record.losses.append(metrics["loss"])

        state, mask = next_state, next_mask

        if step % eval_every == 0 or step == total_steps:
            record.reward_window_avgs.append(sum(reward_window) / len(reward_window))
            reward_window = []

            eval_config = {**full_config, "max_steps": eval_max_steps}
            eval_result = run_scenario(GreedyDQNPolicy(agent), scenario, eval_config, seed=eval_seed)
            record.eval_snapshots.append((step, eval_result))

    checkpoint_path = training_cfg["checkpoint_path"]
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(checkpoint_path)
    record.checkpoint_path = checkpoint_path

    return agent, record


def train_soft_reward_baseline(
    full_config: dict[str, Any] | None = None,
    training_overrides: dict[str, Any] | None = None,
    scenario: str = "S1",
) -> tuple[DQNAgent, TrainingRecord]:
    """`agents.soft_reward_baseline`'s training entry point -- additive,
    does not touch `train()` above (Hard Rule 1: the masked agent's own
    training loop must remain byte-for-byte unaffected). A parallel
    function, not a generalization of `train()`, because the one thing
    that genuinely differs -- which reward value gets passed to
    `agent.observe()` -- isn't a parameterizable variation of `train()`'s
    existing shape; `train()` is tightly coupled to using `env.step()`'s
    own returned reward, which is exactly what this agent must NOT train
    against (see `agents/soft_reward_baseline.py`'s module docstring).
    Everything else is deliberately identical to `train()`'s own loop
    (same conventions: `total_steps`/`eval_every`/checkpointing), reusing
    `DQNAgent`/`GreedyDQNPolicy`/`run_scenario` completely unmodified --
    see this repo's own architecture-reuse convention (`agents/dqn.py`
    is never subclassed or modified for this agent; only the reward
    source differs, and that source lives entirely in this function plus
    `agents/soft_reward_baseline.py`, not in `DQNAgent` itself).

    Defaults to loading `configs/soft_reward_baseline.yaml` (not
    `configs/default.yaml`) -- that config's own `security_masking:
    false` is what gives this agent "no action masking" at the
    environment level (see `env/environment.py`'s design decision 16);
    a caller who passes a `full_config` without that key gets the
    ordinary masked-floor environment behavior instead, matching this
    function's "trust the config" convention (same as `train()` itself
    never re-validates `full_config["reward"]`'s contents).

    `env.step()`'s own returned reward is computed every step (`SmartKeyNetEnv`
    always computes its own internal, Hard-Rule-1-clean reward -- it has
    no notion of which agent is training against it) but is discarded
    here, never passed to `agent.observe()` -- `compute_soft_reward` is
    the only reward this agent ever learns from, computed independently
    from the pre-decision `state` and the chosen `action` alone.
    """
    full_config = full_config if full_config is not None else load_full_config(_SOFT_REWARD_CONFIG_PATH)

    # Same Hard Rule 8 guard as train() (mirrored, not shared code --
    # see that function's own comment for why it's keyed on the flag
    # itself). No committed soft-reward config sets this False today, but
    # the guard is a real, structural check, not conditional on that.
    if not full_config.get("train_eligible", True):
        raise ValueError(
            f"refusing to train: scenario {full_config.get('scenario')!r} has "
            "train_eligible: false (Hard Rule 8 -- the migration-wave scenario is "
            "held-out evaluation only). Use experiments/harness.py's "
            "run_scenario/run_grid to evaluate a policy against it instead."
        )

    training_cfg = {**full_config["training"], **(training_overrides or {})}
    has_forecast = full_config.get("use_foresight", "off") != "off"
    soft_reward_cfg = SoftRewardConfig(**full_config["soft_reward"])

    env_config = {**full_config, "scenario": scenario, "seed": training_cfg["seed"]}
    env = SmartKeyNetEnv(env_config)
    state, info = env.reset(seed=training_cfg["seed"])
    mask = info["action_mask"]

    state_dim = flatten_state(state, has_forecast).shape[0]
    dqn_config = load_dqn_config()
    agent = DQNAgent(
        state_dim=state_dim, has_forecast=has_forecast, config=dqn_config, seed=training_cfg["seed"]
    )

    total_steps = training_cfg["total_steps"]
    eval_every = training_cfg["eval_every"]
    eval_seed = training_cfg["eval_seed"]
    eval_max_steps = training_cfg["eval_max_steps"]

    record = TrainingRecord()
    reward_window: list[float] = []

    for step in range(1, total_steps + 1):
        action = agent.act(state, mask)
        soft_reward = compute_soft_reward(state, action, soft_reward_cfg)
        next_state, _env_reward, terminated, truncated, info = env.step(action)
        next_mask = info["action_mask"]

        agent.observe(state, action, soft_reward, next_state, next_mask, terminated)
        metrics = agent.learn()

        reward_window.append(soft_reward)
        if metrics["loss"] > 0.0:
            record.loss_steps.append(step)
            record.losses.append(metrics["loss"])

        state, mask = next_state, next_mask

        if step % eval_every == 0 or step == total_steps:
            record.reward_window_avgs.append(sum(reward_window) / len(reward_window))
            reward_window = []

            eval_config = {**full_config, "max_steps": eval_max_steps}
            # ScenarioResult.total_reward here is env.step()'s OWN
            # (masked-agent-style) reward summed over the eval episode --
            # NOT this agent's soft reward. Still genuinely useful: it's
            # an apples-to-apples "what would the Hard-Rule-1 formula have
            # scored this trajectory" number, and (more importantly for
            # this agent's own purpose) ScenarioResult.floor_violations
            # is real, direct evidence of whether this checkpoint's greedy
            # policy actually served any request below its real floor --
            # exactly the property this agent exists to demonstrate. See
            # tests/test_train.py's soft-reward-baseline tests.
            eval_result = run_scenario(GreedyDQNPolicy(agent), scenario, eval_config, seed=eval_seed)
            record.eval_snapshots.append((step, eval_result))

    checkpoint_path = training_cfg["checkpoint_path"]
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(checkpoint_path)
    record.checkpoint_path = checkpoint_path

    return agent, record


def evaluate_against_baseline(
    agent: DQNAgent,
    full_config: dict[str, Any],
    eval_seed: int,
    eval_max_steps: int,
) -> tuple[ScenarioResult, ScenarioResult]:
    """Compare the trained agent (greedy, via `GreedyDQNPolicy`) against
    a `StaticThresholdPolicy` grid-searched on S1 -- both through
    `experiments/harness.run_scenario` on the same fixed eval seed, so
    it's an apples-to-apples comparison.

    Hard Rule 7 is already satisfied (the four baselines exist and are
    tuned -- see agents/baselines.py): this only *calls*
    `StaticThresholdPolicy.grid_search`, it doesn't re-derive or widen
    the grid itself (`configs/default.yaml`'s `baselines.
    static_threshold_grid`, not hardcoded here).
    """
    eval_config = {**full_config, "max_steps": eval_max_steps}

    def threshold_eval_fn(policy: StaticThresholdPolicy) -> float:
        # grid_search maximizes eval_fn; lower p99 latency is better,
        # so score it negatively -- mirrors tests/test_harness.py's
        # own existing grid-search-then-run convention.
        result = run_scenario(policy, "S1", eval_config, seed=eval_seed)
        return -result.p99_latency

    grid = full_config["baselines"]["static_threshold_grid"]
    tuned_threshold = StaticThresholdPolicy.grid_search(grid, threshold_eval_fn)

    dqn_result = run_scenario(GreedyDQNPolicy(agent), "S1", eval_config, seed=eval_seed)
    threshold_result = run_scenario(tuned_threshold, "S1", eval_config, seed=eval_seed)

    return dqn_result, threshold_result


def _format_result(label: str, result: ScenarioResult) -> str:
    m = result.episode_metrics
    return (
        f"{label}: p99_latency={result.p99_latency:.4f}  "
        f"total_reward={result.total_reward:.2f}  "
        f"regret_events={m.regret_events}  "
        f"pool_exhaustion_events={result.pool_exhaustion_events}  "
        f"deferred_critical_steps={m.deferred_critical_steps}  "
        f"rekeys_per_100_requests={m.rekeys_per_100_requests:.2f}  "
        f"forced_rekey_ratio={m.forced_rekey_ratio:.3f}  "
        f"floor_violations={result.floor_violations}"
    )


def main() -> None:
    full_config = load_full_config()
    training_cfg = full_config["training"]

    print(f"Training DQNAgent on S1 for {training_cfg['total_steps']} steps...")
    agent, record = train(full_config)
    print(f"Checkpoint saved to {record.checkpoint_path}")
    print(f"Reward window averages over the run: {[round(r, 4) for r in record.reward_window_avgs]}")
    print("Greedy-mode eval snapshots during training:")
    for step, result in record.eval_snapshots:
        print(f"  step {step}: {_format_result('DQN (greedy)', result)}")

    dqn_result, threshold_result = evaluate_against_baseline(
        agent, full_config, training_cfg["eval_seed"], training_cfg["eval_max_steps"]
    )
    print("\nFinal S1 comparison (fixed eval seed, same episode length):")
    print(f"  {_format_result('DQN (greedy, trained)', dqn_result)}")
    print(f"  {_format_result('StaticThresholdPolicy (grid-searched)', threshold_result)}")


if __name__ == "__main__":
    main()
