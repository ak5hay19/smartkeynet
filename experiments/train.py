"""
experiments/train.py

Real training campaign for `agents.dqn.DQNAgent` on S1 -- overfit S1 on
purpose to prove the training loop works, per PLAN.md §10 step 5. Owned
by Person C (split.md §1).

This is a genuine checkpoint toward split.md's Gate W3 ("DQN beats the
tuned threshold baseline on S1 and S3"), not the gate itself: S3
scenario dispatch isn't wired into `env/environment.py` yet (still
S1-only -- see PROGRESS.md), so everything here is S1-only. Attempting
the real gate is separate future work once S2-S4 dispatch exists.

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
from env.contracts import Action, ActionMask, StateDict
from env.environment import SmartKeyNetEnv
from experiments.harness import ScenarioResult, run_scenario

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


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
) -> tuple[DQNAgent, TrainingRecord]:
    """Run one continuous S1 training episode (the env has no natural
    terminal state -- see env/environment.py -- so training never
    needs to reset mid-run) for `training.total_steps` steps,
    `observe()`+`learn()` every step, with a periodic greedy-mode
    evaluation snapshot every `training.eval_every` steps. Saves a
    final checkpoint via `DQNAgent.save`.

    `training_overrides` shallow-merges over the real `configs/
    default.yaml`'s `training:` block (itself flat, no nested dicts) --
    tests use this to shrink `total_steps`/`eval_every`/`eval_max_steps`
    down to a CI-fast smoke run without touching this file's real
    defaults.
    """
    full_config = full_config if full_config is not None else load_full_config()
    training_cfg = {**full_config["training"], **(training_overrides or {})}

    # The one place has_forecast is derived from -- config-time, never
    # inferred from a StateDict's contents (2026-08-08 fix; see
    # agents/dqn.py's flatten_state docstring).
    has_forecast = full_config.get("use_foresight", "off") != "off"

    env_config = {**full_config, "scenario": "S1", "seed": training_cfg["seed"]}
    env = SmartKeyNetEnv(env_config)
    state, info = env.reset(seed=training_cfg["seed"])
    mask = info["action_mask"]

    state_dim = flatten_state(state, has_forecast).shape[0]  # derived from the real state, not assumed
    dqn_config = load_dqn_config()
    agent = DQNAgent(state_dim=state_dim, has_forecast=has_forecast, config=dqn_config)

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
            eval_result = run_scenario(GreedyDQNPolicy(agent), "S1", eval_config, seed=eval_seed)
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
