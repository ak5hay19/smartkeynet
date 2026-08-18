"""
experiments/gate_w3.py

Gate W3 (split.md; PLAN.md §7): **does the masked DQN beat a tuned
threshold baseline on S1 and S3?**

This is the make-or-break gate. PLAN.md states the disqualification rule
plainly: *"if a simple threshold policy matches the DQN in evaluation,
the project premise fails"*. SMARTKEYNET_BUILD_SPEC.md §7.1 fix C is
equally clear about what an honest partial result looks like -- a tuned
threshold tying on stationary S1 while the DQN wins on the degraded and
non-stationary cases is a publishable, *scoped* claim, and far better
than an over-claimed one.

Protocol (SMARTKEYNET_BUILD_SPEC.md §9):
  - **Train seeds and eval seeds are disjoint.** Training uses seeds
    0..n-1; evaluation uses 1000.. -- so no agent is ever scored on the
    arrival stream or SKR trace it trained against.
  - **The threshold baseline is grid-searched per scenario on the
    training seeds**, never on the eval seeds. Reporting a lazily-tuned
    threshold is the fastest way for a reviewer to dismiss the result
    (spec §S7).
  - **Common random numbers**: every policy in a given (scenario, eval
    seed) cell sees the identical arrival stream and SKR trace, which is
    the cheapest variance reduction available here (spec §9.2).
  - **Multi-seed reporting, never a single point estimate.** The
    2026-08-10 sessions found a genuine bimodal learn/don't-learn split
    across training seeds; whether or not it survives recalibration, a
    single seed cannot be trusted to represent this agent.

Run it directly:

    .venv/bin/python -m experiments.gate_w3 --train-seeds 5 --steps 25000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agents.baselines import (
    AlwaysHybridPolicy,
    AlwaysPQCPolicy,
    GreedyRecommenderPolicy,
    RandomPolicy,
    StaticThresholdPolicy,
)
from metrics.aggregate import bootstrap_ci, paired_difference
from experiments.harness import run_scenario
from experiments.train import GreedyDQNPolicy, load_full_config, train

GATE_SCENARIOS: tuple[str, ...] = ("S1", "S3")
"""The two scenarios Gate W3 is defined over (split.md)."""

_EVAL_SEED_OFFSET = 1000
"""Eval seeds start here so they cannot collide with training seeds.
Sharing a seed between training and evaluation would let an agent be
scored on the exact arrival stream it memorised."""


@dataclass
class PolicyScore:
    """Aggregated scores for one policy on one scenario, across seeds."""

    policy: str
    scenario: str
    rewards: list[float] = field(default_factory=list)
    exhaustion_events: list[int] = field(default_factory=list)
    floor_violations: list[int] = field(default_factory=list)
    p99_latencies: list[float] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards))

    @property
    def std_reward(self) -> float:
        return float(np.std(self.rewards))

    @property
    def median_reward(self) -> float:
        return float(np.median(self.rewards))

    @property
    def iqm(self):
        """IQM with a 95% bootstrap CI over seeds -- spec §9 rules 2 and 3.

        The mean is not used as the headline because a single diverged seed
        moves it by orders of magnitude; this project measured per-seed
        results spanning -1,326 to -3,015,813 on one configuration."""
        return bootstrap_ci(self.rewards)

    def summary_row(self) -> str:
        estimate = self.iqm
        return (
            f"{self.policy:22s} {estimate.point:12.1f} "
            f"[{estimate.low:11.1f}, {estimate.high:11.1f}] "
            f"{float(np.mean(self.exhaustion_events)):9.1f} "
            f"{int(sum(self.floor_violations)):6d}"
        )


def evaluate_policy(
    policy: Any, scenario: str, config: dict[str, Any], eval_seeds: list[int]
) -> PolicyScore:
    """Score one policy across the eval seeds. Every policy is given the
    same seed list, so all policies in a cell share arrival streams and
    SKR traces (common random numbers)."""
    score = PolicyScore(policy=type(policy).__name__, scenario=scenario)
    for seed in eval_seeds:
        result = run_scenario(policy, scenario, config, seed=seed)
        score.rewards.append(result.total_reward)
        score.exhaustion_events.append(result.pool_exhaustion_events)
        score.floor_violations.append(result.floor_violations)
        score.p99_latencies.append(result.p99_latency)
    return score


def tuned_threshold_for(
    scenario: str, config: dict[str, Any], train_seeds: list[int]
) -> StaticThresholdPolicy:
    """Grid-search the threshold baseline **on the training seeds only**.

    Hard Rule 7 and spec §S7: the threshold must be tuned as carefully
    as the agent, or the comparison is worthless. Tuning it on the eval
    seeds instead would be the mirror-image cheat.
    """
    tau_grid = config["baselines"]["static_threshold_grid"]
    c_min_grid = config["baselines"]["static_threshold_class_grid"]
    rho_grid = config["baselines"]["static_threshold_rekey_frac_grid"]
    max_key_age = float(config["key_lifetime"]["max_key_age_steps"])

    best_policy: StaticThresholdPolicy | None = None
    best_reward = -np.inf

    for tau in tau_grid:
        for c_min in c_min_grid:
            for rho in rho_grid:
                candidate = StaticThresholdPolicy(
                    pool_fill_threshold=tau,
                    min_hybrid_class=c_min,
                    rekey_age_frac=rho,
                    max_key_age=max_key_age,
                )
                rewards = [
                    run_scenario(candidate, scenario, config, seed=seed).total_reward
                    for seed in train_seeds
                ]
                mean_reward = float(np.mean(rewards))
                if mean_reward > best_reward:
                    best_reward, best_policy = mean_reward, candidate

    assert best_policy is not None
    return best_policy


def run_gate(
    n_train_seeds: int = 5,
    n_eval_seeds: int = 5,
    total_steps: int = 25_000,
    eval_max_steps: int = 2_000,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full gate and return a JSON-serialisable report."""
    config = config if config is not None else load_full_config()
    config = {**config, "scenario_steps": eval_max_steps}

    train_seeds = list(range(n_train_seeds))
    eval_seeds = [_EVAL_SEED_OFFSET + i for i in range(n_eval_seeds)]
    eval_config = {**config, "max_steps": eval_max_steps}

    report: dict[str, Any] = {
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "total_steps": total_steps,
        "eval_max_steps": eval_max_steps,
        "scenarios": {},
    }

    for scenario in GATE_SCENARIOS:
        print(f"\n{'=' * 78}\nGATE W3 — scenario {scenario}\n{'=' * 78}")

        # --- baselines (Hard Rule 7: all four, threshold tuned) ---
        print(f"  grid-searching the threshold baseline on train seeds {train_seeds} ...")
        tuned = tuned_threshold_for(scenario, eval_config, train_seeds)
        print(f"  tuned threshold: {tuned!r}")

        baselines = {
            "always_pqc": AlwaysPQCPolicy(),
            "always_hybrid": AlwaysHybridPolicy(),
            "static_threshold_tuned": tuned,
            "random": RandomPolicy(seed=0),
            # not one of Hard Rule 7's mandatory four -- this is spec §S7's
            # diagnostic 6, the "isn't this just a recommender system?"
            # objection expressed as a number. The DQN's margin over it is
            # the measured value of coupling decisions through the pool.
            "greedy_recommender": GreedyRecommenderPolicy(),
        }
        scores = {
            name: evaluate_policy(policy, scenario, eval_config, eval_seeds)
            for name, policy in baselines.items()
        }

        # --- the DQN, one training run per train seed ---
        dqn_rewards: list[float] = []
        dqn_score = PolicyScore(policy="dqn", scenario=scenario)
        for seed in train_seeds:
            print(f"  training DQN on {scenario}, seed {seed} ({total_steps} steps) ...", flush=True)
            agent, _record = train(
                full_config=config,
                training_overrides={
                    "seed": seed,
                    "total_steps": total_steps,
                    "eval_every": max(1, total_steps),  # skip mid-run snapshots; we score at the end
                    "eval_max_steps": eval_max_steps,
                    "checkpoint_path": f"checkpoints/gate_w3_{scenario}_seed{seed}.pt",
                },
                scenario=scenario,
            )
            per_seed = evaluate_policy(
                GreedyDQNPolicy(agent), scenario, eval_config, eval_seeds
            )
            # one number per *training* seed: the mean over eval seeds
            dqn_rewards.append(per_seed.mean_reward)
            dqn_score.rewards.extend(per_seed.rewards)
            dqn_score.exhaustion_events.extend(per_seed.exhaustion_events)
            dqn_score.floor_violations.extend(per_seed.floor_violations)
            dqn_score.p99_latencies.extend(per_seed.p99_latencies)
            print(f"    seed {seed}: mean eval reward {per_seed.mean_reward:.1f}")

        scores["dqn"] = dqn_score

        print(
            f"\n  {'policy':22s} {'IQM':>12s} {'95% CI':>26s} {'exhaust':>9s} {'viol':>6s}"
        )
        for name in ("dqn", "static_threshold_tuned", "greedy_recommender", "always_pqc", "always_hybrid", "random"):
            print(f"  {scores[name].summary_row()}")

        # The claim is a PAIRED difference over shared seeds (spec §9 rule 4):
        # policies see identical arrival streams and SKR traces, so the
        # per-seed difference has far less variance than either policy's own
        # spread. A CI on the difference that excludes zero is the claim.
        threshold_score = scores["static_threshold_tuned"]
        n_paired = min(len(dqn_score.rewards), len(threshold_score.rewards))
        difference = paired_difference(
            dqn_score.rewards[:n_paired], threshold_score.rewards[:n_paired]
        )
        beats = difference.point > 0 and difference.excludes(0.0)

        print(
            f"\n  DQN - threshold (paired, IQM): {difference.point:.1f} "
            f"[{difference.low:.1f}, {difference.high:.1f}]"
        )
        print(
            f"  DQN {'BEATS' if beats else 'DOES NOT BEAT'} the tuned threshold on "
            f"{scenario} (CI on the difference "
            f"{'excludes' if difference.excludes(0.0) else 'includes'} zero)"
        )
        report.setdefault("paired_difference", {})[scenario] = {
            "point": difference.point,
            "low": difference.low,
            "high": difference.high,
            "excludes_zero": bool(difference.excludes(0.0)),
        }
        print(f"  per-training-seed DQN means: {[round(r, 1) for r in dqn_rewards]}")

        report["scenarios"][scenario] = {
            "tuned_threshold": tuned.pool_fill_threshold,
            "dqn_per_train_seed_mean": dqn_rewards,
            "beats_threshold": bool(beats),
            "policies": {
                name: {
                    "iqm_reward": score.iqm.point,
                    "iqm_ci_low": score.iqm.low,
                    "iqm_ci_high": score.iqm.high,
                    "mean_reward": score.mean_reward,
                    "std_reward": score.std_reward,
                    "median_reward": score.median_reward,
                    "mean_exhaustion_events": float(np.mean(score.exhaustion_events)),
                    "total_floor_violations": int(sum(score.floor_violations)),
                    "mean_p99_latency": float(np.mean(score.p99_latencies)),
                }
                for name, score in scores.items()
            },
        }

    report["gate_passed"] = all(
        report["scenarios"][scenario]["beats_threshold"] for scenario in GATE_SCENARIOS
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gate W3.")
    parser.add_argument("--train-seeds", type=int, default=5)
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=25_000)
    parser.add_argument("--eval-max-steps", type=int, default=2_000)
    parser.add_argument("--out", type=str, default="results/gate_w3.json")
    args = parser.parse_args()

    report = run_gate(
        n_train_seeds=args.train_seeds,
        n_eval_seeds=args.eval_seeds,
        total_steps=args.steps,
        eval_max_steps=args.eval_max_steps,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 78}")
    print(f"GATE W3: {'PASSED' if report['gate_passed'] else 'NOT PASSED'}")
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
