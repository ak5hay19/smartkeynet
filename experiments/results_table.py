"""
experiments/results_table.py

The closing comparison table (PLAN2 §7.7, Panel 7): one row per policy,
one column per metric, across S1-S4 plus the held-out S6.

    python -m experiments.results_table

Columns are exactly PLAN2 §7.7's: p99 latency, pool-exhaustion events,
regret events, forced-rekey ratio, floor violations. The masked agent's
floor-violations column reads 0 with an explicit "structural" label --
"not a result that happened to come out well, a guarantee the
architecture makes impossible to violate" (PLAN2 §7.7).

That label is earned rather than asserted: `env/masking.py`'s five
legality rules are what make it true, `experiments/harness.py` counts
*delivered tier* rather than chosen action so the count would notice if
they were removed, and `tests/test_harness.py` asserts it across every
scenario and policy.

Every DQN number here is checkpoint-averaged across multiple training
seeds with the spread reported (see `experiments/campaign.py` for why
that is not optional in this setup). Baselines are deterministic given
a seed, so their spread is across evaluation seeds.

Hard Rule 8: the agents evaluated on S6 are trained on S1 only. S6 is
never trained on, by any arm, ever.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.baselines import AlwaysHybridPolicy, AlwaysPQCPolicy, RandomPolicy
from experiments.campaign import (
    _DEFAULT_N_CHECKPOINT_WINDOWS,
    SeedSpread,
    checkpoint_average,
    evaluate_policy_across_seeds,
)
from experiments.train import (
    GreedyDQNPolicy,
    load_full_config,
    train,
    tune_threshold_policy,
)

TABLE_METRICS: tuple[str, ...] = (
    "p99_latency",
    "pool_exhaustion_events",
    "regret_events",
    "forced_rekey_ratio",
    "floor_violations",
)
"""PLAN2 §7.7's columns, in its order."""

TABLE_SCENARIOS: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S6")
"""PLAN2 §7.7's rows. S5 is excluded on purpose: it is a different
experiment with a different denominator (share of establishments below
the class floor under an adversarial trace), reported by
`attack/run_attack.py`, and folding it into an operational-metrics
table would misrepresent both."""

_TRAIN_SCENARIO = "S1"
"""Agents are trained on the benign stationary scenario and evaluated
everywhere. S2/S3/S4 are in-distribution perturbations they did not
train on either; S6 is the formally held-out one (Hard Rule 8)."""


@dataclass
class ResultsTable:
    scenarios: tuple[str, ...]
    policies: tuple[str, ...]
    training_seeds: tuple[int, ...]
    eval_seeds: tuple[int, ...]
    cells: dict[tuple[str, str], dict[str, SeedSpread]]

    def to_json(self) -> str:
        return json.dumps(
            {
                "scenarios": list(self.scenarios),
                "policies": list(self.policies),
                "training_seeds": list(self.training_seeds),
                "eval_seeds": list(self.eval_seeds),
                "cells": {
                    f"{scenario}|{policy}": {
                        metric: {"mean": spread.mean, "stdev": spread.stdev}
                        for metric, spread in metrics.items()
                    }
                    for (scenario, policy), metrics in self.cells.items()
                },
            },
            indent=2,
        )


def build_results_table(
    config: dict[str, Any] | None = None,
    training_seeds: list[int] | None = None,
    n_checkpoint_windows: int = _DEFAULT_N_CHECKPOINT_WINDOWS,
    training_overrides: dict[str, Any] | None = None,
) -> ResultsTable:
    config = config if config is not None else load_full_config()
    training_seeds = (
        training_seeds
        if training_seeds is not None
        else list(config["training"].get("campaign_seeds", [0, 1, 2, 3, 4]))
    )
    training_cfg = {**config["training"], **(training_overrides or {})}
    eval_seeds = list(training_cfg.get("eval_seeds") or [training_cfg["eval_seed"]])
    eval_max_steps = int(training_cfg["eval_max_steps"])

    # One set of agents, trained once on S1, evaluated on every scenario.
    agents = []
    train_averages = []
    for seed in training_seeds:
        overrides = {**(training_overrides or {}), "seed": seed}
        agent, record = train(config, training_overrides=overrides, scenario=_TRAIN_SCENARIO)
        agents.append(agent)
        train_averages.append(
            checkpoint_average(
                record.eval_snapshot_metrics,
                scenario=_TRAIN_SCENARIO,
                training_seed=seed,
                n_windows=n_checkpoint_windows,
            )
        )

    policies = (
        "masked DQN",
        "static-threshold (tuned)",
        "always-hybrid",
        "always-PQC",
        "random",
    )
    cells: dict[tuple[str, str], dict[str, SeedSpread]] = {}

    for scenario in TABLE_SCENARIOS:
        tuned = tune_threshold_policy(config, scenario, eval_seeds, eval_max_steps)
        baselines = {
            "static-threshold (tuned)": lambda _s, t=tuned: t,
            "always-hybrid": lambda _s: AlwaysHybridPolicy(),
            "always-PQC": lambda _s: AlwaysPQCPolicy(),
            "random": lambda s: RandomPolicy(seed=s),
        }
        for name, factory in baselines.items():
            cells[(scenario, name)] = evaluate_policy_across_seeds(
                factory, scenario, config, eval_seeds, eval_max_steps
            )

        if scenario == _TRAIN_SCENARIO:
            cells[(scenario, "masked DQN")] = {
                metric: SeedSpread(
                    metric=metric,
                    values=tuple(a.mean[metric] for a in train_averages),
                    seeds=tuple(training_seeds),
                )
                for metric in TABLE_METRICS
            }
        else:
            per_seed = [
                evaluate_policy_across_seeds(
                    lambda _s, a=agent: GreedyDQNPolicy(a),
                    scenario,
                    config,
                    eval_seeds,
                    eval_max_steps,
                )
                for agent in agents
            ]
            cells[(scenario, "masked DQN")] = {
                metric: SeedSpread(
                    metric=metric,
                    values=tuple(p[metric].mean for p in per_seed),
                    seeds=tuple(training_seeds),
                )
                for metric in TABLE_METRICS
            }

    return ResultsTable(
        scenarios=TABLE_SCENARIOS,
        policies=policies,
        training_seeds=tuple(training_seeds),
        eval_seeds=tuple(eval_seeds),
        cells=cells,
    )


def format_results_table(table: ResultsTable) -> str:
    lines = [
        "=== Closing comparison (PLAN2 §7.7, Panel 7) ===",
        f"agents trained on {_TRAIN_SCENARIO} only, seeds {list(table.training_seeds)}, "
        f"checkpoint-averaged; eval seeds {list(table.eval_seeds)}",
        "S6 is HELD-OUT: no agent in this table trained on the migration schedule "
        "(Hard Rule 8).",
        "",
    ]
    header = f"{'scenario':<4} {'policy':<26}" + "".join(f"{m[:18]:>20}" for m in TABLE_METRICS)
    for scenario in table.scenarios:
        lines.append(header)
        for policy in table.policies:
            metrics = table.cells[(scenario, policy)]
            row = f"{scenario:<4} {policy:<26}"
            for metric in TABLE_METRICS:
                spread = metrics[metric]
                if metric == "floor_violations":
                    label = "0  (structural)" if spread.mean == 0.0 else f"{spread.mean:.2f}  !!"
                    row += f"{label:>20}"
                else:
                    row += f"{spread.mean:>13.2f}+/-{spread.stdev:<5.1f}"
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    config = load_full_config()
    table = build_results_table(config)
    print(format_results_table(table))

    out = Path("results/closing_table.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table.to_json())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
