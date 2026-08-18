"""
experiments/campaign.py

Multi-seed, checkpoint-averaged campaign runner -- the module that
actually attempts Gate W3 ("DQN beats the tuned threshold baseline on
S1 and S3") and produces every DQN-vs-baseline number this project
reports. Owned by Person C (split.md §1).

Why this module exists at all
-----------------------------
PROGRESS.md documents an unresolved training-stability finding, five
diagnostic sessions deep: this setup's greedy policy oscillates
checkpoint-to-checkpoint by large margins (`forced_rekey_ratio` swings
of 0.5+ between adjacent 1,000-step snapshots, roughly 1 in 3 of the
time, continuously across an entire 75,000-step run, for every seed
tested). Two candidate explanations -- single-episode eval noise, and
eval-cadence/target-sync aliasing -- were tested and both disfavoured.
The mechanism is still unidentified.

That is treated here as a **documented property of this training
setup**, not as something to chase. The consequence for reporting is
concrete and non-negotiable: a single-checkpoint DQN number is not
evidence about the agent, it is one draw from a wide distribution.
Every number this module produces is therefore

  1. **checkpoint-averaged** -- the mean over the last
     `n_checkpoint_windows` `eval_every` snapshots of a run, not the
     final checkpoint;
  2. **eval-seed-averaged within each snapshot** -- so eval-episode
     randomness is not confounded with the oscillation (this does not
     tame the oscillation; PROGRESS.md item 6 established that
     directly);
  3. **repeated across multiple training seeds**, with the across-seed
     spread reported alongside the mean, never collapsed into it.

`SeedSpread` carries the spread as a first-class field precisely so
that a caller cannot accidentally report a mean without it.

Hard Rule 7
-----------
The threshold baseline this compares against is grid-searched on the
same scenario, on the same objective the agent is credited for
(`total_reward`), averaged over the same eval seeds -- see
`experiments.train.tune_threshold_policy` for why each of those
choices is deliberately in the baseline's favour. If the tuned
threshold ties or beats the agent, that is the result, and
`format_campaign` prints it as such.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from agents.baselines import (
    AlwaysHybridPolicy,
    AlwaysPQCPolicy,
    Policy,
    RandomPolicy,
)
from experiments.harness import run_scenario
from experiments.train import (
    _METRIC_KEYS,
    GreedyDQNPolicy,
    load_full_config,
    mean_result_metrics,
    result_metrics,
    train,
    tune_threshold_policy,
)

_DEFAULT_N_CHECKPOINT_WINDOWS = 4
"""How many trailing `eval_every` snapshots a checkpoint average spans.

Four windows at the default `eval_every: 2500` covers the last 10,000
of 25,000 training steps -- late enough that epsilon has been at
`epsilon_end` throughout (decay completes at 12,500), wide enough to
average over several independent draws of the documented oscillation,
and narrow enough that it is still "the end of training" rather than a
whole-run average."""

_HIGHER_IS_BETTER: frozenset[str] = frozenset({"total_reward"})
"""Metric direction, so `format_campaign` can mark wins without a
caller having to remember which way each column points. Everything else
in `_METRIC_KEYS` is a cost, a failure count, or a ratio where lower is
better -- except `floor_violations`, which is structurally 0 for every
masked policy and is reported rather than compared."""


@dataclass(frozen=True)
class SeedSpread:
    """One metric, aggregated across seeds, with its spread kept.

    The spread is a required field, not an optional extra: given this
    setup's documented instability, a mean reported without it is
    misleading, and making it structural is the cheapest way to stop
    that happening by accident.
    """

    metric: str
    values: tuple[float, ...]
    seeds: tuple[int, ...]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def minimum(self) -> float:
        return min(self.values)

    @property
    def maximum(self) -> float:
        return max(self.values)

    def render(self) -> str:
        return f"{self.mean:.3f} +/- {self.stdev:.3f} [{self.minimum:.3f}, {self.maximum:.3f}]"


@dataclass(frozen=True)
class CheckpointAverage:
    """One training run's result: each metric averaged over the last N
    eval snapshots, plus the within-run spread across those snapshots
    (i.e. the magnitude of the documented oscillation for this seed)."""

    scenario: str
    training_seed: int
    steps_averaged: tuple[int, ...]
    mean: dict[str, float]
    stdev: dict[str, float]


def checkpoint_average(
    snapshot_metrics: list[tuple[int, dict[str, float]]],
    scenario: str,
    training_seed: int,
    n_windows: int = _DEFAULT_N_CHECKPOINT_WINDOWS,
) -> CheckpointAverage:
    """Average the last `n_windows` snapshots of one training run.

    Raises if the run produced fewer snapshots than requested rather
    than quietly averaging over whatever it has -- a checkpoint average
    over one window is a single-checkpoint number wearing a disguise,
    which is the exact thing this module exists to prevent.
    """
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    if len(snapshot_metrics) < n_windows:
        raise ValueError(
            f"run produced {len(snapshot_metrics)} eval snapshots but {n_windows} were requested "
            "for checkpoint averaging -- lower training.eval_every or raise total_steps"
        )

    window = snapshot_metrics[-n_windows:]
    steps = tuple(step for step, _ in window)
    per_metric = {key: [metrics[key] for _, metrics in window] for key in _METRIC_KEYS}

    return CheckpointAverage(
        scenario=scenario,
        training_seed=training_seed,
        steps_averaged=steps,
        mean={k: statistics.fmean(v) for k, v in per_metric.items()},
        stdev={k: (statistics.stdev(v) if len(v) > 1 else 0.0) for k, v in per_metric.items()},
    )


@dataclass
class CampaignResult:
    """One scenario's full comparison: the agent against every
    mandatory baseline (Hard Rule 7), all on the same eval seeds."""

    scenario: str
    training_seeds: tuple[int, ...]
    eval_seeds: tuple[int, ...]
    n_checkpoint_windows: int
    tuned_threshold: float
    dqn: dict[str, SeedSpread]
    baselines: dict[str, dict[str, SeedSpread]]
    per_seed_checkpoint_averages: list[CheckpointAverage] = field(default_factory=list)
    within_run_oscillation: dict[str, SeedSpread] = field(default_factory=dict)
    """For each metric, the *within-run* checkpoint-to-checkpoint stdev,
    itself aggregated across training seeds. This is the documented
    instability, measured and reported rather than smoothed away."""

    def verdict(self, metric: str = "total_reward") -> str:
        """Plain-language comparison of the agent against the tuned
        threshold on `metric`, honest about overlap.

        "Ties" is a real, reportable outcome (Hard Rule 7 explicitly
        anticipates it), so it is a first-class verdict here rather
        than something a caller has to infer from two numbers.
        """
        agent = self.dqn[metric]
        threshold = self.baselines["static-threshold (tuned)"][metric]
        better = (
            agent.mean > threshold.mean
            if metric in _HIGHER_IS_BETTER
            else agent.mean < threshold.mean
        )
        # Overlapping +/-1 stdev bands mean the difference is not
        # separated by the seed spread, whichever way the means point.
        overlap = abs(agent.mean - threshold.mean) < (agent.stdev + threshold.stdev)
        if overlap:
            return "TIE (means differ by less than the combined seed spread)"
        return "DQN WINS" if better else "TUNED THRESHOLD WINS"


def _baseline_policies(seed: int) -> dict[str, Policy]:
    """The three seed-independent mandatory baselines plus a freshly
    seeded random policy (Hard Rule 7). The tuned threshold is added by
    the caller, since tuning depends on the scenario."""
    return {
        "always-pqc": AlwaysPQCPolicy(),
        "always-hybrid": AlwaysHybridPolicy(),
        "random": RandomPolicy(seed=seed),
    }


def evaluate_policy_across_seeds(
    policy_factory,
    scenario: str,
    config: dict[str, Any],
    eval_seeds: list[int],
    max_steps: int,
) -> dict[str, SeedSpread]:
    """Run one policy across `eval_seeds`, returning per-metric spreads."""
    eval_config = {**config, "max_steps": max_steps}
    per_seed = [
        result_metrics(run_scenario(policy_factory(seed), scenario, eval_config, seed=seed))
        for seed in eval_seeds
    ]
    return {
        key: SeedSpread(
            metric=key,
            values=tuple(m[key] for m in per_seed),
            seeds=tuple(eval_seeds),
        )
        for key in _METRIC_KEYS
    }


def run_campaign(
    scenario: str,
    training_seeds: list[int],
    config: dict[str, Any] | None = None,
    n_checkpoint_windows: int = _DEFAULT_N_CHECKPOINT_WINDOWS,
    training_overrides: dict[str, Any] | None = None,
) -> CampaignResult:
    """Train `len(training_seeds)` agents on `scenario` and compare the
    checkpoint-averaged result against every mandatory baseline.

    This is the only supported way to produce a DQN-vs-baseline number
    for this project. Everything about it is deliberately conservative
    toward the agent: the threshold baseline is tuned on the agent's
    own objective, the agent's number is an average over several
    checkpoints rather than its best one, and the across-seed spread is
    reported next to the mean.
    """
    config = config if config is not None else load_full_config()
    training_cfg = {**config["training"], **(training_overrides or {})}
    eval_seeds = list(training_cfg.get("eval_seeds") or [training_cfg["eval_seed"]])
    eval_max_steps = int(training_cfg["eval_max_steps"])

    averages: list[CheckpointAverage] = []
    for seed in training_seeds:
        overrides = {**(training_overrides or {}), "seed": seed}
        _agent, record = train(config, training_overrides=overrides, scenario=scenario)
        averages.append(
            checkpoint_average(
                record.eval_snapshot_metrics,
                scenario=scenario,
                training_seed=seed,
                n_windows=n_checkpoint_windows,
            )
        )

    dqn = {
        key: SeedSpread(
            metric=key,
            values=tuple(a.mean[key] for a in averages),
            seeds=tuple(training_seeds),
        )
        for key in _METRIC_KEYS
    }
    within_run = {
        key: SeedSpread(
            metric=key,
            values=tuple(a.stdev[key] for a in averages),
            seeds=tuple(training_seeds),
        )
        for key in _METRIC_KEYS
    }

    tuned = tune_threshold_policy(config, scenario, eval_seeds, eval_max_steps)
    baselines: dict[str, dict[str, SeedSpread]] = {
        "static-threshold (tuned)": evaluate_policy_across_seeds(
            lambda _s: tuned, scenario, config, eval_seeds, eval_max_steps
        )
    }
    for name in _baseline_policies(0):
        baselines[name] = evaluate_policy_across_seeds(
            lambda s, n=name: _baseline_policies(s)[n],
            scenario,
            config,
            eval_seeds,
            eval_max_steps,
        )

    return CampaignResult(
        scenario=scenario,
        training_seeds=tuple(training_seeds),
        eval_seeds=tuple(eval_seeds),
        n_checkpoint_windows=n_checkpoint_windows,
        tuned_threshold=float(tuned.pool_fill_threshold),
        dqn=dqn,
        baselines=baselines,
        per_seed_checkpoint_averages=averages,
        within_run_oscillation=within_run,
    )


def format_campaign(result: CampaignResult, metrics: tuple[str, ...] = _METRIC_KEYS) -> str:
    """Human-readable campaign report, spread always shown."""
    lines: list[str] = []
    lines.append(f"=== Scenario {result.scenario} ===")
    lines.append(
        f"training seeds: {list(result.training_seeds)} | eval seeds: {list(result.eval_seeds)} | "
        f"checkpoint windows averaged: {result.n_checkpoint_windows} "
        f"(steps {list(result.per_seed_checkpoint_averages[0].steps_averaged)})"
    )
    lines.append(f"tuned threshold: pool_fill > {result.tuned_threshold}")
    lines.append("")

    rows = [("masked DQN", result.dqn)] + list(result.baselines.items())
    for metric in metrics:
        lines.append(f"-- {metric} --")
        for name, spreads in rows:
            lines.append(f"   {name:<26} {spreads[metric].render()}")
        lines.append("")

    lines.append(f"VERDICT (total_reward): {result.verdict('total_reward')}")
    lines.append(
        "within-run checkpoint oscillation (stdev across the averaged windows, "
        "itself averaged over training seeds):"
    )
    for metric in ("total_reward", "forced_rekey_ratio"):
        lines.append(f"   {metric:<26} {result.within_run_oscillation[metric].render()}")
    return "\n".join(lines)


def main() -> None:
    """Gate W3: masked DQN vs the tuned threshold on S1 and S3."""
    config = load_full_config()
    seeds = list(config["training"].get("campaign_seeds", [0, 1, 2, 3, 4]))
    for scenario in ("S1", "S3"):
        result = run_campaign(scenario, training_seeds=seeds, config=config)
        print(format_campaign(result))
        print()


if __name__ == "__main__":
    main()
