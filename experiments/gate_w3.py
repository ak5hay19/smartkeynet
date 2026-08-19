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
from experiments.harness import run_scenario
from experiments.train import GreedyDQNPolicy, load_full_config, train
from metrics.aggregate import bootstrap_ci, paired_difference

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
    regret_events: list[int] = field(default_factory=list)
    exhaustion_events: list[int] = field(default_factory=list)
    floor_violations: list[int] = field(default_factory=list)
    p99_latencies: list[float] = field(default_factory=list)
    overflow_keys: list[int] = field(default_factory=list)

    @property
    def mean_regret(self) -> float:
        """The primary Gate W3 metric (§6). Lower is better."""
        return float(np.mean(self.regret_events)) if self.regret_events else 0.0

    @property
    def mean_overflow(self) -> float:
        """Wasted key material. Reported alongside regret because a policy can
        drive regret to zero simply by never spending, and overflow is what
        exposes that (§S1: "a good agent should show near-zero overflow *and*
        near-zero regret")."""
        return float(np.mean(self.overflow_keys)) if self.overflow_keys else 0.0

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
        score.regret_events.append(result.episode_metrics.regret_events)
        score.exhaustion_events.append(result.pool_exhaustion_events)
        score.floor_violations.append(result.floor_violations)
        score.p99_latencies.append(result.p99_latency)
        score.overflow_keys.append(result.pool_overflow_keys)
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
    best_key: tuple[float, float] | None = None
    primary_scores: list[float] = []

    for tau in tau_grid:
        for c_min in c_min_grid:
            for rho in rho_grid:
                candidate = StaticThresholdPolicy(
                    pool_fill_threshold=tau,
                    min_hybrid_class=c_min,
                    rekey_age_frac=rho,
                    max_key_age=max_key_age,
                )
                results = [
                    run_scenario(candidate, scenario, config, seed=seed) for seed in train_seeds
                ]
                # Selected on REGRET EVENTS, the primary metric §6 names for
                # Gate W3, with reward only as a tiebreak. This selected on
                # total reward until 2026-08-19, which tuned the baseline for a
                # different objective than the gate then judged it on -- and
                # §9.7 is explicit that a scalar return must never be the
                # headline, because it is reward-function-specific.
                mean_regret = float(np.mean([r.episode_metrics.regret_events for r in results]))
                mean_reward = float(np.mean([r.total_reward for r in results]))
                candidate_key = (-mean_regret, mean_reward)
                primary_scores.append(mean_regret)
                if best_key is None or candidate_key > best_key:
                    best_key, best_policy = candidate_key, candidate

    assert best_policy is not None

    # How many grid points tie on the primary metric? If most of them do, the
    # parameter is UNIDENTIFIED rather than badly bounded, and "extend the
    # grid" is the wrong remedy -- the metric simply cannot discriminate here.
    # Distinguishing the two matters: an edge optimum among ties is not the
    # HR7 problem §S7 test 4 is warning about, and reporting it as one would
    # be misleading in the opposite direction.
    best_primary = min(primary_scores) if primary_scores else 0.0
    n_tied = sum(1 for score in primary_scores if score == best_primary)
    print(
        f"  grid: {n_tied}/{len(primary_scores)} configurations tie at the best "
        f"regret score ({best_primary:.1f})"
    )
    if n_tied > len(primary_scores) // 2:
        print(
            "  NOTE: the primary metric does not discriminate between most grid points "
            "on this scenario, so the selected parameters are effectively arbitrary "
            "among the tied set (chosen by the reward tiebreak). Report the tie, not a "
            "tuned value."
        )
    else:
        _warn_if_grid_optimum_is_on_an_edge(best_policy, tau_grid, c_min_grid, rho_grid, scenario)
    return best_policy


def _warn_if_grid_optimum_is_on_an_edge(
    policy: StaticThresholdPolicy,
    tau_grid: list[float],
    c_min_grid: list[int],
    rho_grid: list[float],
    scenario: str,
) -> None:
    """§S7 test 4: "the selected `tau` should not sit at the edge of the grid.
    If it does, **extend the grid** -- an edge optimum means you have not
    actually tuned the baseline, and HR7 is violated in spirit."

    Beating a baseline that is pinned to its grid boundary is a weak claim,
    because the boundary, not the search, chose the parameter. Reported loudly
    rather than raised: the run should still complete and say so.
    """
    edges: list[str] = []
    if policy.pool_fill_threshold in (min(tau_grid), max(tau_grid)):
        edges.append(f"tau={policy.pool_fill_threshold} (grid {min(tau_grid)}..{max(tau_grid)})")
    if policy.min_hybrid_class in (min(c_min_grid), max(c_min_grid)):
        edges.append(f"c_min={policy.min_hybrid_class}")
    if policy.rekey_age_frac in (min(rho_grid), max(rho_grid)):
        edges.append(f"rho={policy.rekey_age_frac}")
    if edges:
        print(
            f"  WARNING [{scenario}]: threshold grid optimum sits on an edge: "
            f"{', '.join(edges)}. §S7 test 4 says extend the grid -- an edge optimum "
            "means the baseline is not actually tuned."
        )


def run_gate(
    n_train_seeds: int = 5,
    n_eval_seeds: int = 10,  # §9.6: 10 seeds for every number in the final table
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
            print(
                f"  training DQN on {scenario}, seed {seed} ({total_steps} steps) ...", flush=True
            )
            agent, _record = train(
                full_config=config,
                training_overrides={
                    "seed": seed,
                    "total_steps": total_steps,
                    "eval_every": max(
                        1, total_steps
                    ),  # skip mid-run snapshots; we score at the end
                    "eval_max_steps": eval_max_steps,
                    "checkpoint_path": f"checkpoints/gate_w3_{scenario}_seed{seed}.pt",
                },
                scenario=scenario,
            )
            per_seed = evaluate_policy(GreedyDQNPolicy(agent), scenario, eval_config, eval_seeds)
            # one number per *training* seed: the mean over eval seeds
            dqn_rewards.append(per_seed.mean_reward)
            dqn_score.rewards.extend(per_seed.rewards)
            dqn_score.regret_events.extend(per_seed.regret_events)
            dqn_score.exhaustion_events.extend(per_seed.exhaustion_events)
            dqn_score.floor_violations.extend(per_seed.floor_violations)
            dqn_score.p99_latencies.extend(per_seed.p99_latencies)
            dqn_score.overflow_keys.extend(per_seed.overflow_keys)
            print(f"    seed {seed}: mean eval reward {per_seed.mean_reward:.1f}")

        scores["dqn"] = dqn_score

        print(
            f"\n  {'policy':22s} {'regret':>8s} {'overflow':>9s} {'p99 ms':>8s} "
            f"{'IQM reward':>12s} {'exhaust':>9s} {'viol':>6s}"
        )
        for name in (
            "dqn",
            "static_threshold_tuned",
            "greedy_recommender",
            "always_pqc",
            "always_hybrid",
            "random",
        ):
            score = scores[name]
            print(
                f"  {score.policy:22s} {score.mean_regret:8.1f} {score.mean_overflow:9.1f} "
                f"{float(np.mean(score.p99_latencies)):8.1f} {score.iqm.point:12.1f} "
                f"{float(np.mean(score.exhaustion_events)):9.1f} "
                f"{int(sum(score.floor_violations)):6d}"
            )
        print(
            "  (primary metric is REGRET EVENTS, lower better -- §6. Reward is "
            "descriptive only: §9.7 forbids a scalar return as the headline.)"
        )

        # The claim is a PAIRED difference over shared seeds (spec §9 rule 4):
        # policies see identical arrival streams and SKR traces, so the
        # per-seed difference has far less variance than either policy's own
        # spread. A CI on the difference that excludes zero is the claim.
        threshold_score = scores["static_threshold_tuned"]
        n_paired = min(len(dqn_score.regret_events), len(threshold_score.regret_events))
        # Primary metric is regret events and LOWER is better, so the paired
        # difference is taken as (threshold - dqn): a positive difference whose
        # CI excludes zero means the DQN caused fewer regret events.
        difference = paired_difference(
            [float(v) for v in threshold_score.regret_events[:n_paired]],
            [float(v) for v in dqn_score.regret_events[:n_paired]],
        )
        reward_difference = paired_difference(
            dqn_score.rewards[:n_paired], threshold_score.rewards[:n_paired]
        )
        beats_on_regret = difference.point > 0 and difference.excludes(0.0)

        # §6 secondary constraints: p99 latency and pool exhaustion must not
        # regress by more than 10%. A win on regret bought by a large latency
        # regression is not a win.
        threshold_p99 = float(np.mean(threshold_score.p99_latencies)) or 1.0
        threshold_exhaust = float(np.mean(threshold_score.exhaustion_events))
        dqn_p99 = float(np.mean(dqn_score.p99_latencies))
        dqn_exhaust = float(np.mean(dqn_score.exhaustion_events))
        p99_regression = (dqn_p99 - threshold_p99) / threshold_p99
        exhaust_regression = (
            (dqn_exhaust - threshold_exhaust) / threshold_exhaust
            if threshold_exhaust > 0
            else (0.0 if dqn_exhaust == 0 else float("inf"))
        )
        secondary_ok = p99_regression <= 0.10 and exhaust_regression <= 0.10
        beats = beats_on_regret and secondary_ok

        print(
            f"\n  threshold - DQN regret events (paired, IQM): {difference.point:.1f} "
            f"[{difference.low:.1f}, {difference.high:.1f}]   (positive = DQN better)"
        )
        print(
            f"  reward difference (descriptive): {reward_difference.point:.1f} "
            f"[{reward_difference.low:.1f}, {reward_difference.high:.1f}]"
        )
        print(
            f"  secondary: p99 latency {p99_regression:+.1%}, pool exhaustion "
            f"{exhaust_regression:+.1%} (each must be <= +10%) -> "
            f"{'OK' if secondary_ok else 'REGRESSED'}"
        )
        if threshold_score.mean_regret == 0.0:
            print(
                "  NOTE: the tuned threshold caused ZERO regret events, so the primary\n"
                "        metric cannot discriminate -- it achieves that by hoarding\n"
                f"        ({threshold_score.mean_overflow:.0f} keys of overflow wasted vs "
                f"{dqn_score.mean_overflow:.0f} for the DQN).\n"
                "        Compare overflow and reward instead, and say so in the report."
            )
        print(
            f"  DQN {'BEATS' if beats else 'DOES NOT BEAT'} the tuned threshold on "
            f"{scenario} (regret CI "
            f"{'excludes' if difference.excludes(0.0) else 'includes'} zero"
            f"{'' if secondary_ok else '; secondary metrics regressed'})"
        )
        report.setdefault("paired_difference", {})[scenario] = {
            "primary_metric": "regret_events",
            "point": difference.point,
            "low": difference.low,
            "high": difference.high,
            "excludes_zero": bool(difference.excludes(0.0)),
            "reward_difference_descriptive": {
                "point": reward_difference.point,
                "low": reward_difference.low,
                "high": reward_difference.high,
            },
            "secondary_p99_regression": p99_regression,
            "secondary_exhaustion_regression": exhaust_regression,
            "secondary_within_10pct": bool(secondary_ok),
            "threshold_regret_is_zero": bool(threshold_score.mean_regret == 0.0),
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
                    "mean_regret_events": score.mean_regret,
                    "mean_overflow_keys": score.mean_overflow,
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
    parser.add_argument("--eval-seeds", type=int, default=10)
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
