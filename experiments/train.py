"""
experiments/train.py

Real training campaign for `agents.dqn.DQNAgent`, per PLAN.md §10 step
5. Owned by Person C (split.md §1).

Scenario-parameterized as of 2026-08-19 (`train(..., scenario=...)`):
S1-S4 dispatch now exists in `env/environment.py`, so Gate W3 ("DQN
beats the tuned threshold baseline on S1 and S3") is finally
attemptable for real. The campaign runner that actually attempts it --
multi-seed, checkpoint-averaged, per PROGRESS.md's standing
instruction about the training-stability finding -- is
`experiments/campaign.py`; this module stays the single-run trainer it
has always been.

Hard Rule 8: training scenarios only. S6 (migration wave) is held-out
evaluation and must never be passed here -- `train()` rejects it.

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
    eval_snapshot_metrics: list[tuple[int, dict[str, float]]] = field(default_factory=list)
    """Per-snapshot metrics averaged over *all* of `training.eval_seeds`,
    parallel to `eval_snapshots` (which keeps one representative
    `ScenarioResult` at the primary `training.eval_seed`, unchanged).

    Averaging several fixed eval episodes per checkpoint removes
    eval-episode randomness from the comparison. PROGRESS.md item 6
    established that this does NOT tame the checkpoint-to-checkpoint
    oscillation -- the mean of 8 eval draws still swings nearly as hard
    as a single draw -- so this is not a fix for that; it just stops
    eval noise from being confounded with it. Taming the oscillation is
    `experiments/campaign.py`'s checkpoint averaging, and the residual
    is reported as spread rather than hidden."""
    checkpoint_path: str | None = None


_METRIC_KEYS: tuple[str, ...] = (
    "total_reward",
    "p99_latency",
    "pool_exhaustion_events",
    "regret_events",
    "deferred_critical_steps",
    "forced_rekey_ratio",
    "rekeys_per_100_requests",
    "discretionary_hybrid_serves",
    "floor_violations",
)
"""The metrics every comparison in this project reports, in the order
PLAN2 §7.7's closing table wants them. Defined once here so the
trainer, the campaign runner, the API facade and the dashboard cannot
drift apart on what a "result" contains."""


def result_metrics(result: ScenarioResult) -> dict[str, float]:
    """Flatten one `ScenarioResult` into the `_METRIC_KEYS` dict."""
    m = result.episode_metrics
    return {
        "total_reward": float(result.total_reward),
        "p99_latency": float(result.p99_latency),
        "pool_exhaustion_events": float(result.pool_exhaustion_events),
        "regret_events": float(m.regret_events),
        "deferred_critical_steps": float(m.deferred_critical_steps),
        "forced_rekey_ratio": float(m.forced_rekey_ratio),
        "rekeys_per_100_requests": float(m.rekeys_per_100_requests),
        "discretionary_hybrid_serves": float(m.discretionary_hybrid_serves),
        "floor_violations": float(result.floor_violations),
    }


def mean_result_metrics(results: list[ScenarioResult]) -> dict[str, float]:
    """Element-wise mean of several `ScenarioResult`s' metrics."""
    if not results:
        raise ValueError("mean_result_metrics() needs at least one result")
    per_result = [result_metrics(r) for r in results]
    return {key: float(np.mean([m[key] for m in per_result])) for key in _METRIC_KEYS}


_HELD_OUT_SCENARIOS = frozenset({"S6"})
"""Hard Rule 8: "train on stationary scenarios; the migration-wave
scenario is held-out evaluation only." Enforced in code rather than
left to discipline -- training on S6 would invalidate the only
robustness claim the project makes about non-stationarity, and it is
a one-character mistake to make."""


def train(
    full_config: dict[str, Any] | None = None,
    training_overrides: dict[str, Any] | None = None,
    scenario: str = "S1",
) -> tuple[DQNAgent, TrainingRecord]:
    """Run one continuous training episode (the env has no natural
    terminal state -- see env/environment.py -- so training never
    needs to reset mid-run) for `training.total_steps` steps,
    `observe()`+`learn()` every step, with a periodic greedy-mode
    evaluation snapshot every `training.eval_every` steps. Saves a
    final checkpoint via `DQNAgent.save`.

    `scenario` selects the training scenario (S1-S4). Evaluation
    snapshots run on the *same* scenario, so a snapshot measures what
    this checkpoint scores on the distribution it is being trained on.
    `S6` is rejected outright (Hard Rule 8).

    `training_overrides` shallow-merges over the real `configs/
    default.yaml`'s `training:` block (itself flat, no nested dicts) --
    tests use this to shrink `total_steps`/`eval_every`/`eval_max_steps`
    down to a CI-fast smoke run without touching this file's real
    defaults.
    """
    if scenario in _HELD_OUT_SCENARIOS:
        raise ValueError(
            f"scenario {scenario!r} is held-out evaluation only (Hard Rule 8) -- never train on it"
        )

    full_config = full_config if full_config is not None else load_full_config()
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
    eval_seeds = list(training_cfg.get("eval_seeds") or [eval_seed])
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
            policy = GreedyDQNPolicy(agent)
            seed_results = [
                run_scenario(policy, scenario, eval_config, seed=s) for s in eval_seeds
            ]
            primary = (
                seed_results[eval_seeds.index(eval_seed)]
                if eval_seed in eval_seeds
                else run_scenario(policy, scenario, eval_config, seed=eval_seed)
            )
            record.eval_snapshots.append((step, primary))
            record.eval_snapshot_metrics.append((step, mean_result_metrics(seed_results)))

    checkpoint_path = str(training_cfg["checkpoint_path"]).format(
        scenario=scenario, seed=training_cfg["seed"]
    )
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(checkpoint_path)
    record.checkpoint_path = checkpoint_path

    return agent, record


def tune_threshold_policy(
    full_config: dict[str, Any],
    scenario: str,
    eval_seeds: list[int],
    eval_max_steps: int,
) -> StaticThresholdPolicy:
    """Grid-search `StaticThresholdPolicy` on `scenario` (Hard Rule 7).

    Two deliberate choices about what "tuned" means here, both in the
    baseline's favour:

    1. **Tuned on `total_reward`, not `p99_latency`.** The agent
       optimizes episode reward; scoring its mandatory competitor on a
       coarse 4-valued latency proxy instead would be stacking the
       deck. The strongest honest baseline is one tuned on exactly the
       objective the agent is being credited for.
    2. **Tuned on multiple seeds**, averaged, so the chosen threshold
       isn't an artefact of one lucky episode.

    This only *calls* `StaticThresholdPolicy.grid_search`; the grid
    itself stays `configs/default.yaml`'s `baselines.
    static_threshold_grid`, never widened or re-derived here.
    """
    eval_config = {**full_config, "max_steps": eval_max_steps}

    def threshold_eval_fn(policy: StaticThresholdPolicy) -> float:
        # grid_search maximizes eval_fn, and higher reward is better.
        return float(
            np.mean(
                [
                    run_scenario(policy, scenario, eval_config, seed=seed).total_reward
                    for seed in eval_seeds
                ]
            )
        )

    grid = full_config["baselines"]["static_threshold_grid"]
    return StaticThresholdPolicy.grid_search(grid, threshold_eval_fn)


def evaluate_against_baseline(
    agent: DQNAgent,
    full_config: dict[str, Any],
    eval_seed: int,
    eval_max_steps: int,
    scenario: str = "S1",
) -> tuple[ScenarioResult, ScenarioResult]:
    """Single-checkpoint comparison of the trained agent (greedy, via
    `GreedyDQNPolicy`) against a tuned `StaticThresholdPolicy` -- both
    through `experiments/harness.run_scenario` on the same fixed eval
    seed, so it's an apples-to-apples comparison.

    NOTE: single-checkpoint. PROGRESS.md documents that this training
    setup's greedy policy oscillates checkpoint-to-checkpoint by a
    large margin with an unidentified mechanism, so a single-checkpoint
    number is not evidence about the agent. Use
    `experiments/campaign.py` for any comparison that is meant to
    support a claim; this stays for `main()`'s human-readable run
    summary and for tests.
    """
    eval_config = {**full_config, "max_steps": eval_max_steps}
    tuned_threshold = tune_threshold_policy(full_config, scenario, [eval_seed], eval_max_steps)

    dqn_result = run_scenario(GreedyDQNPolicy(agent), scenario, eval_config, seed=eval_seed)
    threshold_result = run_scenario(tuned_threshold, scenario, eval_config, seed=eval_seed)

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
