"""
experiments/ablation.py

E-A, the foresight ablation (PLAN2 §9's E-A row): the same agent
architecture with the threat/pool forecast varied
`off / ewma / lstm`, on S3 (QKD degradation) and then on S6 (the
held-out migration wave).

    python -m experiments.ablation

What the ablation is actually asking
------------------------------------
PLAN2 §12 states the claim this is meant to quantify: "the tuned
threshold baseline reacts to the *current* pool level; the agent (once
the forecaster is real) acts on the *forecast* -- pool trajectory and
incoming demand. The foresight ablation quantifies exactly how much
anticipation is worth."

So the measured quantity is the *delta between foresight modes*, not
any single mode's absolute score. Three modes, identical in every other
respect:

  off   13-dim state, no forecast fields at all (genuinely omitted, not
        zero-padded -- see agents/dqn.py's flatten_state).
  ewma  28-dim state, EWMA fallback: no learned parameters, and every
        horizon flat-holds the current estimate, so it is "reactive
        dressed as a forecast".
  lstm  28-dim state, the trained frozen dual-head forecaster, whose
        pool head genuinely extrapolates.

All three arms run with `threat_input.source: rt_iot2022`, i.e. real
flow-feature windows. This is not optional dressing: the LSTM was
trained on 16-dimensional windows, and the `scenario` source emits a
single standardized scalar, so evaluating the LSTM arm under `scenario`
feeds it out-of-distribution input and measures a distribution mismatch
rather than the value of foresight. (Observed directly on a smoke run
before this was fixed: the `lstm` arm scored -341,702 against `off`'s
-6,506 on S6, with 34,075 deferred critical steps -- a garbage threat
score ratcheting floors to HIGH and holding them there.) Holding the
threat source fixed across all three arms is also what makes the
comparison an ablation of the *forecaster* rather than of its input.

Because `off` changes the state dimensionality, each mode trains its
own agents from scratch -- a checkpoint is not portable across modes.

Every number here comes through `experiments/campaign.py`, so it is
checkpoint-averaged and multi-seed with the spread reported, for the
reason recorded there: this training setup's greedy policy oscillates
checkpoint-to-checkpoint by a wide margin with an unidentified
mechanism (PROGRESS.md), and a single-checkpoint number is one draw
from a wide distribution rather than a result.

Hard Rule 8: agents are trained on S3 only. The S6 arm evaluates those
same S3-trained agents on the held-out schedule -- it never trains on
it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.campaign import (
    _DEFAULT_N_CHECKPOINT_WINDOWS,
    SeedSpread,
    checkpoint_average,
    evaluate_policy_across_seeds,
)
from experiments.train import _METRIC_KEYS, GreedyDQNPolicy, load_full_config, train

FORESIGHT_MODES: tuple[str, ...] = ("off", "ewma", "lstm")

_ABLATION_METRICS: tuple[str, ...] = (
    "total_reward",
    "pool_exhaustion_events",
    "regret_events",
    "deferred_critical_steps",
    "p99_latency",
    "forced_rekey_ratio",
)
"""The columns PLAN2's E-A row asks for: "report regret/exhaustion/p99
deltas"."""


@dataclass
class AblationArm:
    """One (foresight mode, evaluation scenario) cell."""

    foresight: str
    scenario: str
    train_scenario: str
    metrics: dict[str, SeedSpread]


@dataclass
class AblationResult:
    train_scenario: str
    eval_scenarios: tuple[str, ...]
    training_seeds: tuple[int, ...]
    arms: list[AblationArm]

    def arm(self, foresight: str, scenario: str) -> AblationArm:
        for candidate in self.arms:
            if candidate.foresight == foresight and candidate.scenario == scenario:
                return candidate
        raise KeyError(f"no arm for foresight={foresight!r} scenario={scenario!r}")

    def to_json(self) -> str:
        return json.dumps(
            {
                "train_scenario": self.train_scenario,
                "eval_scenarios": list(self.eval_scenarios),
                "training_seeds": list(self.training_seeds),
                "arms": [
                    {
                        "foresight": arm.foresight,
                        "scenario": arm.scenario,
                        "metrics": {
                            key: {
                                "mean": spread.mean,
                                "stdev": spread.stdev,
                                "values": list(spread.values),
                            }
                            for key, spread in arm.metrics.items()
                        },
                    }
                    for arm in self.arms
                ],
            },
            indent=2,
        )


def run_ablation(
    config: dict[str, Any] | None = None,
    train_scenario: str = "S3",
    eval_scenarios: tuple[str, ...] = ("S3", "S6"),
    training_seeds: list[int] | None = None,
    n_checkpoint_windows: int = _DEFAULT_N_CHECKPOINT_WINDOWS,
    training_overrides: dict[str, Any] | None = None,
) -> AblationResult:
    """Train one set of agents per foresight mode, evaluate each on
    every scenario in `eval_scenarios`."""
    config = config if config is not None else load_full_config()
    training_seeds = (
        training_seeds
        if training_seeds is not None
        else list(config["training"].get("campaign_seeds", [0, 1, 2]))
    )
    training_cfg = {**config["training"], **(training_overrides or {})}
    eval_seeds = list(training_cfg.get("eval_seeds") or [training_cfg["eval_seed"]])
    eval_max_steps = int(training_cfg["eval_max_steps"])

    arms: list[AblationArm] = []
    for foresight in FORESIGHT_MODES:
        mode_config = {
            **config,
            "use_foresight": foresight,
            # Same threat source for every arm -- see the module docstring.
            "threat_input": {**(config.get("threat_input") or {}), "source": "rt_iot2022"},
        }

        trained: list[Any] = []
        train_averages = []
        for seed in training_seeds:
            overrides = {**(training_overrides or {}), "seed": seed}
            agent, record = train(
                mode_config, training_overrides=overrides, scenario=train_scenario
            )
            trained.append(agent)
            train_averages.append(
                checkpoint_average(
                    record.eval_snapshot_metrics,
                    scenario=train_scenario,
                    training_seed=seed,
                    n_windows=n_checkpoint_windows,
                )
            )

        for scenario in eval_scenarios:
            if scenario == train_scenario:
                # reuse the in-training checkpoint averages -- same
                # scenario, same eval seeds, and they already span
                # several checkpoints rather than one
                metrics = {
                    key: SeedSpread(
                        metric=key,
                        values=tuple(a.mean[key] for a in train_averages),
                        seeds=tuple(training_seeds),
                    )
                    for key in _METRIC_KEYS
                }
            else:
                # Held-out evaluation (S6): the trained agents are run on
                # a scenario none of them has ever seen (Hard Rule 8).
                per_seed: list[dict[str, SeedSpread]] = [
                    evaluate_policy_across_seeds(
                        lambda _s, a=agent: GreedyDQNPolicy(a),
                        scenario,
                        mode_config,
                        eval_seeds,
                        eval_max_steps,
                    )
                    for agent in trained
                ]
                metrics = {
                    key: SeedSpread(
                        metric=key,
                        values=tuple(p[key].mean for p in per_seed),
                        seeds=tuple(training_seeds),
                    )
                    for key in _METRIC_KEYS
                }

            arms.append(
                AblationArm(
                    foresight=foresight,
                    scenario=scenario,
                    train_scenario=train_scenario,
                    metrics=metrics,
                )
            )

    return AblationResult(
        train_scenario=train_scenario,
        eval_scenarios=tuple(eval_scenarios),
        training_seeds=tuple(training_seeds),
        arms=arms,
    )


def format_ablation(result: AblationResult) -> str:
    lines = [
        "=== E-A foresight ablation ===",
        f"trained on {result.train_scenario} "
        f"(seeds {list(result.training_seeds)}, checkpoint-averaged); "
        f"evaluated on {list(result.eval_scenarios)}",
        f"S6 is HELD-OUT -- no agent in this table has ever trained on it (Hard Rule 8).",
        "",
    ]
    for scenario in result.eval_scenarios:
        lines.append(f"-- evaluated on {scenario} --")
        for metric in _ABLATION_METRICS:
            lines.append(f"   {metric}")
            for foresight in FORESIGHT_MODES:
                spread = result.arm(foresight, scenario).metrics[metric]
                lines.append(f"      {foresight:<6} {spread.render()}")
        lines.append("")

    lines.append("-- deltas vs. `off` (the quantity E-A exists to measure) --")
    for scenario in result.eval_scenarios:
        baseline = result.arm("off", scenario)
        for foresight in ("ewma", "lstm"):
            arm = result.arm(foresight, scenario)
            deltas = ", ".join(
                f"{m}={arm.metrics[m].mean - baseline.metrics[m].mean:+.2f}"
                for m in ("total_reward", "regret_events", "p99_latency")
            )
            lines.append(f"   {scenario} {foresight:<5} {deltas}")
    return "\n".join(lines)


def main() -> None:
    config = load_full_config()
    result = run_ablation(config)
    print(format_ablation(result))

    out = Path("results/foresight_ablation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.to_json())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
