"""
dashboard/render_results_demo.py

Real driver for `dashboard/render_dose_response.py` and
`dashboard/render_comparison_table.py` -- regenerates both headline
demo visuals from a fresh real eval run, not from SESSION_LOG.md prose.

Data provenance (see this session's SESSION_LOG.md entry for the full
reasoning): no saved raw-results file existed anywhere in the repo for
either the S5 dose-response sweep (2026-08-26) or the masked-vs-soft-
reward S3 comparison (2026-08-25) -- both sessions' real numbers lived
only in SESSION_LOG.md prose. But every checkpoint both sessions
trained is still on disk (gitignored, per `.gitignore`'s `*.pt` rule,
but never deleted): `checkpoints/dqn_s2.pt`, `checkpoints/
soft_reward_baseline_s2.pt` (S5 sweep) and `checkpoints/
s3_{masked,soft_reward}_seed{1,4,7}.pt` (S3 comparison). Re-running
`evaluate_multi_seed`/`evaluate_attack_multi_seed` against already-
trained checkpoints is eval-only (no training), calibrated at well
under a minute total for everything this module needs -- so this
driver reloads those real checkpoints and re-runs the real evals fresh
rather than transcribing SESSION_LOG.md's numbers. A spot check before
building this driver reproduced SESSION_LOG.md's own per-seed numbers
exactly (masked S3 seed=1: below_floor_rate 0.0000, total_reward
-8373.50, forced_rekey_ratio 0.196, matching byte-for-byte; masked S2
alpha=0.9: below_floor_rate_true 0.3000 +/- 0.0076, matching exactly)
-- confirming these checkpoints are the same trained weights, not a
drifted or different set.

The fresh results are saved to `dashboard/samples/results_data.json`
(this session's saved raw-results file, so future sessions have a real
file to prefer over re-running or re-transcribing) before being handed
directly to the renderers -- so the rendered HTML and the saved JSON
are guaranteed to describe the same real run, not two different ones.

Run directly: `python -m dashboard.render_results_demo`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from agents.dqn import DQNAgent, flatten_state, load_dqn_config
from dashboard.render_comparison_table import (
    AgentMetrics,
    ComparisonTableData,
    write_comparison_table_html,
)
from dashboard.render_dose_response import (
    DoseResponsePoint,
    DoseResponseSeries,
    write_dose_response_html,
)
from env.environment import SmartKeyNetEnv
from experiments.harness import (
    MultiSeedAttackEvalResult,
    MultiSeedEvalResult,
    evaluate_attack_multi_seed,
    evaluate_multi_seed,
)
from experiments.train import GreedyDQNPolicy, load_full_config

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINTS_DIR = _REPO_ROOT / "checkpoints"
_SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

_S3_EVAL_SEEDS = [900, 901, 902, 903, 904, 905, 906, 907]
_S3_TRAINING_SEEDS = [1, 4, 7]
_S5_EVAL_SEEDS = [900, 901, 902, 903, 904]
_S5_ALPHAS = [round(0.1 * i, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0


def _load_greedy_policy(checkpoint_path: Path, config: dict[str, Any]) -> GreedyDQNPolicy:
    """Reconstruct a `DQNAgent` with the same `state_dim`/`has_forecast`
    `experiments/train.py::train()` used, then load real trained
    weights from `checkpoint_path` -- mirrors that module's own
    construction exactly (`load_dqn_config()`'s default path -- the
    `dqn:` hyperparameters, which govern network *shape*, are copied
    verbatim across every config in this repo's convention, so this is
    the same architecture every checkpoint was trained with)."""
    has_forecast = config.get("use_foresight", "off") != "off"
    env = SmartKeyNetEnv({**config, "seed": 0})
    state, _info = env.reset(seed=0)
    state_dim = flatten_state(state, has_forecast).shape[0]
    agent = DQNAgent(state_dim=state_dim, has_forecast=has_forecast, config=load_dqn_config())
    agent.load(str(checkpoint_path))
    return GreedyDQNPolicy(agent)


# ---------------------------------------------------------------------------
# S3 comparison
# ---------------------------------------------------------------------------


def _aggregate_across_training_seeds(label: str, per_seed: list[MultiSeedEvalResult]) -> AgentMetrics:
    """Checkpoint-averaged summary: mean/std, across the real
    per-training-seed checkpoints, of each checkpoint's own real
    mean-over-eval-seeds -- the same "mean +/- training-seed std"
    methodology the masked-vs-soft-reward S3 comparison session (and
    Gate W3 before it) used. Every input value is a real field off a
    real `MultiSeedEvalResult` -- nothing here is invented."""
    below = [r.below_floor_rate_mean for r in per_seed]
    reward = [r.total_reward_mean for r in per_seed]
    frr = [r.forced_rekey_ratio_mean for r in per_seed]
    regret = [r.regret_events_mean for r in per_seed]
    p99 = [r.p99_latency_mean for r in per_seed]

    return AgentMetrics(
        label=label,
        below_floor_rate_mean=float(np.mean(below)),
        below_floor_rate_std=float(np.std(below)),
        total_reward_mean=float(np.mean(reward)),
        total_reward_std=float(np.std(reward)),
        forced_rekey_ratio_mean=float(np.mean(frr)),
        forced_rekey_ratio_std=float(np.std(frr)),
        regret_events_mean=float(np.mean(regret)),
        regret_events_std=float(np.std(regret)),
        p99_latency_mean=float(np.mean(p99)),
        p99_latency_std=float(np.std(p99)),
        floor_violations_total=sum(r.floor_violations_total for r in per_seed),
        n_training_seeds=len(per_seed),
        n_eval_seeds_per_checkpoint=len(per_seed[0].eval_seeds) if per_seed else 0,
    )


def collect_real_s3_comparison() -> ComparisonTableData:
    """Reload the 6 real S3 checkpoints (3 training seeds x 2 agents)
    and re-run `evaluate_multi_seed` fresh against each -- eval only,
    the checkpoints already hold the trained weights."""
    masked_config = load_full_config(_REPO_ROOT / "configs" / "scenarios" / "s3_degradation.yaml")
    masked_config["max_steps"] = 250
    soft_config = load_full_config(_REPO_ROOT / "configs" / "soft_reward_baseline_s3.yaml")
    soft_config["max_steps"] = 250

    masked_results: list[MultiSeedEvalResult] = []
    soft_results: list[MultiSeedEvalResult] = []
    for seed in _S3_TRAINING_SEEDS:
        masked_policy = _load_greedy_policy(_CHECKPOINTS_DIR / f"s3_masked_seed{seed}.pt", masked_config)
        masked_results.append(evaluate_multi_seed(masked_policy, "S3", masked_config, _S3_EVAL_SEEDS))

        soft_policy = _load_greedy_policy(_CHECKPOINTS_DIR / f"s3_soft_reward_seed{seed}.pt", soft_config)
        soft_results.append(evaluate_multi_seed(soft_policy, "S3", soft_config, _S3_EVAL_SEEDS))

    masked_metrics = _aggregate_across_training_seeds("Masked DQN", masked_results)
    soft_metrics = _aggregate_across_training_seeds("Soft-reward baseline", soft_results)

    return ComparisonTableData(scenario="S3", masked=masked_metrics, soft_reward=soft_metrics, include_p99=True)


# ---------------------------------------------------------------------------
# S5 dose-response sweep
# ---------------------------------------------------------------------------


def collect_real_dose_response() -> list[DoseResponseSeries]:
    """Reload the 2 real S2 checkpoints and re-run
    `evaluate_attack_multi_seed` fresh across all 11 real alpha steps
    -- eval only, the checkpoints already hold the trained weights."""
    masked_config = load_full_config(_REPO_ROOT / "configs" / "scenarios" / "s2_hndl.yaml")
    masked_config["max_steps"] = 250
    soft_config = load_full_config(_REPO_ROOT / "configs" / "soft_reward_baseline_s2.yaml")
    soft_config["max_steps"] = 250

    masked_policy = _load_greedy_policy(_CHECKPOINTS_DIR / "dqn_s2.pt", masked_config)
    soft_policy = _load_greedy_policy(_CHECKPOINTS_DIR / "soft_reward_baseline_s2.pt", soft_config)

    masked_attack_results: list[MultiSeedAttackEvalResult] = []
    soft_attack_results: list[MultiSeedAttackEvalResult] = []
    for alpha in _S5_ALPHAS:
        masked_attack_results.append(
            evaluate_attack_multi_seed(masked_policy, "S2", masked_config, _S5_EVAL_SEEDS, alpha)
        )
        soft_attack_results.append(
            evaluate_attack_multi_seed(soft_policy, "S2", soft_config, _S5_EVAL_SEEDS, alpha)
        )

    masked_series = DoseResponseSeries(
        label="Masked DQN",
        series_key="masked",
        points=[
            DoseResponsePoint(alpha=r.alpha, mean=r.below_floor_rate_true_mean, std=r.below_floor_rate_true_std)
            for r in masked_attack_results
        ],
    )
    soft_series = DoseResponseSeries(
        label="Soft-reward baseline",
        series_key="soft_reward",
        points=[
            DoseResponsePoint(alpha=r.alpha, mean=r.below_floor_rate_true_mean, std=r.below_floor_rate_true_std)
            for r in soft_attack_results
        ],
    )
    return [masked_series, soft_series]


# ---------------------------------------------------------------------------
# Provenance JSON + entry point
# ---------------------------------------------------------------------------


def _save_provenance_json(
    comparison: ComparisonTableData, dose_series: list[DoseResponseSeries], path: Path
) -> None:
    payload = {
        "provenance": (
            "Re-run this session (see SESSION_LOG.md) from real, already-trained "
            "checkpoints on disk (checkpoints/dqn_s2.pt, checkpoints/soft_reward_baseline_s2.pt, "
            "checkpoints/s3_{masked,soft_reward}_seed{1,4,7}.pt) via "
            "experiments.harness.evaluate_multi_seed/evaluate_attack_multi_seed -- eval only, "
            "no retraining. Spot-checked against SESSION_LOG.md's own 2026-08-25/2026-08-26 "
            "figures and reproduced byte-for-byte before this driver was built."
        ),
        "s3_comparison": {
            "scenario": comparison.scenario,
            "eval_seeds": _S3_EVAL_SEEDS,
            "training_seeds": _S3_TRAINING_SEEDS,
            "masked": dataclasses.asdict(comparison.masked),
            "soft_reward": dataclasses.asdict(comparison.soft_reward),
        },
        "s5_dose_response": {
            "scenario": "S2",
            "eval_seeds": _S5_EVAL_SEEDS,
            "alphas": _S5_ALPHAS,
            "series": [dataclasses.asdict(s) for s in dose_series],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    print("Re-running real S3 comparison eval (6 checkpoints, 8 eval seeds each)...")
    comparison = collect_real_s3_comparison()
    print(
        f"  masked below_floor_rate={comparison.masked.below_floor_rate_mean:.4f} "
        f"soft={comparison.soft_reward.below_floor_rate_mean:.4f}"
    )

    print("Re-running real S5 dose-response sweep (2 checkpoints, 11 alphas x 5 eval seeds each)...")
    dose_series = collect_real_dose_response()
    for s in dose_series:
        nonzero = [p for p in s.points if p.mean > 0.0]
        print(f"  {s.label}: {len(s.points)} alpha points, {len(nonzero)} with nonzero V(pi)")

    _SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    json_path = _SAMPLES_DIR / "results_data.json"
    _save_provenance_json(comparison, dose_series, json_path)
    print(f"wrote {json_path}")

    table_path = _SAMPLES_DIR / "s3_comparison_table.html"
    write_comparison_table_html(comparison, table_path)
    print(f"wrote {table_path}")

    dose_path = _SAMPLES_DIR / "dose_response_chart.html"
    write_dose_response_html(dose_series, dose_path)
    print(f"wrote {dose_path}")


if __name__ == "__main__":
    main()
