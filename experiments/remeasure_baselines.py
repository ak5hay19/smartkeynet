"""
experiments/remeasure_baselines.py

Camera-ready baselines session (2026-11-16): re-measure every
non-learned baseline on S1 and S3 under the CURRENT configuration --
the same setup Addition E Phase 4b's Gate W3 used. NO training happens
here and no checkpoint is read for the DQN: the masked-DQN reference
numbers are the already-published Gate W3 figures, quoted as constants.

Setup being reproduced (SESSION_LOG.md/PROGRESS.md, Phase 4b, Gate W3):
  * S1 -> configs/default.yaml, S3 -> configs/scenarios/s3_degradation.yaml
  * forecaster: `RTIoT2022ReplayProvider` over the combined real held-out
    RT-IoT2022 rows (benign span then attack span, CSV order), via a
    per-episode `forecast_provider_factory` -- the helper that lived in
    the gitignored `scratchpad/phase4b_common.py` is re-stated below,
    unchanged in logic, because committed code cannot import scratchpad
  * eval seeds 900-907, `experiments.harness.evaluate_multi_seed`
  * recalibrated floor table (it is simply the table in env/masking.py)

STEP 1 (hard gate, runs first): `StaticThresholdPolicy(0.9)` must
reproduce the logged Gate W3 reference total reward on S1 and S3. If it
does not match to the published precision (2 decimals) this script exits
with status 2 BEFORE measuring anything else, because a wiring mismatch
would make every later number untrustworthy.

Needs (all gitignored, not in the repo): checkpoints/lstm_threat_forecaster.pt
and data/raw/rt_iot2022/RT_IOT2022.csv (see README.md "How to reproduce").

Run (from the repo root):
    python -m experiments.remeasure_baselines
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from agents.baselines import (
    AlwaysHybridPolicy,
    AlwaysPQCPolicy,
    RandomPolicy,
    StaticThresholdPolicy,
)
from agents.reuse_first import ReuseFirstPolicy
from agents.dqn import DQNAgent, flatten_state, load_dqn_config
from env.contracts import Action
from env.environment import SmartKeyNetEnv
from experiments.harness import MultiSeedEvalResult, evaluate_multi_seed
from experiments.train import GreedyDQNPolicy, load_full_config
from forecaster.rt_iot_replay import RTIoT2022ReplayProvider, held_out_replay_rows

_REPO_ROOT = Path(__file__).resolve().parent.parent
LSTM_CHECKPOINT = str(_REPO_ROOT / "checkpoints" / "lstm_threat_forecaster.pt")
RESULTS_PATH = _REPO_ROOT / "docs" / "baselines_remeasured_results.json"

EVAL_SEEDS = list(range(900, 908))
EPISODE_STEPS = 250  # experiments/harness.py::_DEFAULT_MAX_STEPS (neither config overrides it)

SCENARIOS: dict[str, str | None] = {"S1": None, "S3": "configs/scenarios/s3_degradation.yaml"}

# Published masked-DQN Gate W3 pooled figures (PROGRESS.md Phase 4b /
# SESSION_LOG.md; dashboard/samples/results_data.json for S3). Quoted,
# not re-measured -- this session does not touch checkpoints.
DQN_PUBLISHED = {"S1": -11524.57, "S3": -10784.76}

# Step 1 reference: StaticThresholdPolicy(0.9) total_reward_mean from the
# Gate W3 log (scratchpad/phase4b_gate_w3_results.json, "grid_scores"
# "0.9"), full precision. The paper quotes them to 2 decimals.
STEP1_REFERENCE = {"S1": -64648.447499999944, "S3": -38566.87000000001}
STEP1_PUBLISHED = {"S1": -64648.45, "S3": -38566.87}
STEP1_TOLERANCE = 0.005  # half a unit in the last published decimal

THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # configs' baselines.static_threshold_grid
TUNED_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# Forecaster wiring -- logic copied unchanged from scratchpad/phase4b_common.py
# ---------------------------------------------------------------------------


def combined_replay_features() -> np.ndarray:
    rows = held_out_replay_rows()
    return np.concatenate([rows.benign_features, rows.attack_features], axis=0)


def replay_provider_factory(start_offset_base: int = 0):
    features = combined_replay_features()

    def factory(episode_seed: int | None):
        offset = start_offset_base + (episode_seed or 0)
        return RTIoT2022ReplayProvider(LSTM_CHECKPOINT, features, start_offset=offset)

    return factory


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


class _CountingPolicy:
    """Transparent wrapper: counts every decision's action so the
    fresh-key-issuance statistic can be cross-checked against the
    harness's own `rekeys_per_100_requests`. Changes no decision."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.counts: Counter[Action] = Counter()

    def act(self, state, mask):  # type: ignore[no-untyped-def]
        action = self.inner.act(state, mask)
        self.counts[action] += 1
        return action


def _summarize(result: MultiSeedEvalResult, counter: _CountingPolicy) -> dict[str, Any]:
    per_episode = result.results
    n_eps = len(per_episode)
    total_decisions = sum(counter.counts.values())
    if total_decisions != n_eps * EPISODE_STEPS:
        raise RuntimeError(
            f"expected {n_eps * EPISODE_STEPS} decisions across {n_eps} episodes, observed {total_decisions} -- "
            "EPISODE_STEPS no longer matches the harness"
        )
    # Fresh-key issuances per episode: every non-REUSE decision. Derived
    # per episode from the harness's own rekeys_per_100_requests.
    fresh = [r.episode_metrics.rekeys_per_100_requests * EPISODE_STEPS / 100.0 for r in per_episode]
    fresh_from_counts = (total_decisions - counter.counts[Action.REUSE]) / n_eps
    if abs(float(np.mean(fresh)) - fresh_from_counts) > 1e-9:
        raise RuntimeError("harness rekeys_per_100_requests disagrees with directly counted non-REUSE decisions")
    waiting = [r.episode_metrics.deferred_critical_steps for r in per_episode]
    disc_hybrid = [r.episode_metrics.discretionary_hybrid_serves for r in per_episode]
    regret = [r.episode_metrics.regret_events for r in per_episode]
    rewards = [r.total_reward for r in per_episode]
    return {
        "total_reward_mean": result.total_reward_mean,
        "total_reward_std": result.total_reward_std,
        "total_reward_per_seed": dict(zip(result.eval_seeds, rewards)),
        "below_floor_rate_mean": result.below_floor_rate_mean,
        "below_floor_rate_std": result.below_floor_rate_std,
        "floor_violations_total": result.floor_violations_total,
        "regret_events_mean": result.regret_events_mean,
        "regret_events_std": float(np.std(regret)),
        "pool_exhaustion_events_mean": result.pool_exhaustion_events_mean,
        "forced_rekey_ratio_mean": result.forced_rekey_ratio_mean,
        "forced_rekey_ratio_std": result.forced_rekey_ratio_std,
        "fresh_key_issuances_per_episode_mean": float(np.mean(fresh)),
        "fresh_key_issuances_per_episode_std": float(np.std(fresh)),
        "reuse_decisions_per_episode_mean": counter.counts[Action.REUSE] / n_eps,
        "deferred_critical_steps_mean": float(np.mean(waiting)),
        "deferred_critical_steps_std": float(np.std(waiting)),
        "discretionary_hybrid_serves_mean": float(np.mean(disc_hybrid)),
        "p99_latency_mean": result.p99_latency_mean,
        "action_counts_per_episode_mean": {a.name: counter.counts[a] / n_eps for a in Action},
        "n_eval_seeds": n_eps,
    }


def _evaluate(policy: Any, scenario: str, config: dict[str, Any], factory) -> dict[str, Any]:
    counter = _CountingPolicy(policy)
    result = evaluate_multi_seed(counter, scenario, config, EVAL_SEEDS, forecast_provider_factory=factory)
    return _summarize(result, counter)


DQN_CHECKPOINT_FMT = {"S1": "checkpoints/dqn_s1_seed{seed}.pt", "S3": "checkpoints/s3_masked_seed{seed}.pt"}
DQN_TRAIN_SEEDS = [1, 4, 7]


def dqn_profile(scenario: str, config: dict[str, Any], factory) -> dict[str, Any] | None:
    """SUPPLEMENTARY, read-only: re-evaluate the already-trained masked
    DQN checkpoints (loaded, never written or retrained) with the same
    seeds/forecaster, to (a) cross-check the published pooled figure
    quoted in DQN_PUBLISHED and (b) show the DQN's own action mix next to
    the baselines'. Returns None if any checkpoint is missing."""
    paths = [_REPO_ROOT / DQN_CHECKPOINT_FMT[scenario].format(seed=s) for s in DQN_TRAIN_SEEDS]
    if not all(p.exists() for p in paths):
        return None
    per_seed = []
    for seed, path in zip(DQN_TRAIN_SEEDS, paths):
        env = SmartKeyNetEnv({**config, "scenario": scenario, "seed": 0}, forecast_provider_factory=factory)
        state, _ = env.reset(seed=0)
        has_forecast = config.get("use_foresight", "off") != "off"
        agent = DQNAgent(
            state_dim=flatten_state(state, has_forecast).shape[0],
            has_forecast=has_forecast,
            config=load_dqn_config(),
            seed=0,
        )
        agent.load(str(path))
        per_seed.append(_evaluate(GreedyDQNPolicy(agent), scenario, config, factory))
    pooled = float(np.mean([r["total_reward_mean"] for r in per_seed]))
    return {
        "training_seeds": DQN_TRAIN_SEEDS,
        "pooled_total_reward_mean": pooled,
        "pooled_total_reward_std_across_training_seeds": float(np.std([r["total_reward_mean"] for r in per_seed])),
        "per_training_seed": {str(s): r for s, r in zip(DQN_TRAIN_SEEDS, per_seed)},
        "matches_published_to_2dp": round(pooled, 2) == DQN_PUBLISHED[scenario],
        "action_counts_per_episode_mean": {
            a.name: float(np.mean([r["action_counts_per_episode_mean"][a.name] for r in per_seed])) for a in Action
        },
        "fresh_key_issuances_per_episode_mean": float(np.mean([r["fresh_key_issuances_per_episode_mean"] for r in per_seed])),
        "forced_rekey_ratio_mean": float(np.mean([r["forced_rekey_ratio_mean"] for r in per_seed])),
        "below_floor_rate_mean": float(np.mean([r["below_floor_rate_mean"] for r in per_seed])),
    }


def step1_reference_check(configs: dict[str, dict[str, Any]], factory) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ok = True
    for scenario, config in configs.items():
        measured = _evaluate(StaticThresholdPolicy(TUNED_THRESHOLD), scenario, config, factory)
        got = measured["total_reward_mean"]
        diff_published = abs(round(got, 2) - STEP1_PUBLISHED[scenario])
        diff_logged = abs(got - STEP1_REFERENCE[scenario])
        match = diff_published <= STEP1_TOLERANCE and diff_logged <= STEP1_TOLERANCE
        ok = ok and match
        out[scenario] = {
            "measured_total_reward_mean": got,
            "published_total_reward": STEP1_PUBLISHED[scenario],
            "logged_total_reward_full_precision": STEP1_REFERENCE[scenario],
            "abs_diff_vs_logged": diff_logged,
            "match": bool(match),
        }
        print(
            f"[STEP 1] {scenario}: StaticThreshold(t={TUNED_THRESHOLD}) total_reward_mean={got:.4f}  "
            f"published={STEP1_PUBLISHED[scenario]:.2f}  logged={STEP1_REFERENCE[scenario]:.6f}  "
            f"|diff|={diff_logged:.2e}  -> {'MATCH' if match else 'MISMATCH'}"
        )
    out["all_match"] = ok
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-grid", action="store_true", help="skip re-running the threshold grid (tuned t only)")
    parser.add_argument("--skip-reuse-first", action="store_true")
    parser.add_argument("--skip-dqn-profile", action="store_true", help="skip the read-only re-evaluation of the DQN checkpoints")
    args = parser.parse_args()

    started = time.time()
    for needed in (Path(LSTM_CHECKPOINT), _REPO_ROOT / "data" / "raw" / "rt_iot2022" / "RT_IOT2022.csv"):
        if not needed.exists():
            print(f"MISSING required file: {needed}", file=sys.stderr)
            return 3

    configs = {s: (load_full_config(p) if p else load_full_config()) for s, p in SCENARIOS.items()}
    factory = replay_provider_factory()

    step1 = step1_reference_check(configs, factory)
    if not step1["all_match"]:
        print("STEP 1 FAILED: reference not reproduced; refusing to measure anything else.", file=sys.stderr)
        RESULTS_PATH.write_text(json.dumps({"step1": step1, "status": "STEP1_FAILED"}, indent=2))
        return 2

    results: dict[str, Any] = {}
    for scenario, config in configs.items():
        sc: dict[str, Any] = {}
        sc["always_pqc"] = _evaluate(AlwaysPQCPolicy(), scenario, config, factory)
        sc["always_hybrid"] = _evaluate(AlwaysHybridPolicy(), scenario, config, factory)
        # RandomPolicy is stateful: ONE instance is shared across the 8
        # eval episodes (as evaluate_multi_seed requires), so the result
        # depends on its RNG seed. Primary = seed 0; seeds 1,2 reported
        # as a sensitivity check.
        sc["random"] = _evaluate(RandomPolicy(seed=0), scenario, config, factory)
        sc["random_rng_seed_sensitivity"] = {
            str(s): _evaluate(RandomPolicy(seed=s), scenario, config, factory) for s in (1, 2)
        }
        sc["static_threshold_tuned"] = _evaluate(StaticThresholdPolicy(TUNED_THRESHOLD), scenario, config, factory)
        if not args.skip_grid:
            sc["static_threshold_grid_total_reward_mean"] = {
                str(t): _evaluate(StaticThresholdPolicy(t), scenario, config, factory)["total_reward_mean"]
                for t in THRESHOLD_GRID
            }
        if not args.skip_reuse_first:
            sc["reuse_first"] = _evaluate(ReuseFirstPolicy(), scenario, config, factory)

        if not args.skip_dqn_profile:
            prof = dqn_profile(scenario, config, factory)
            if prof is None:
                print(f"[{scenario}] DQN checkpoints missing -- supplementary DQN profile skipped")
            else:
                sc["masked_dqn_reevaluated_supplementary"] = prof
                print(
                    f"[{scenario}] DQN re-eval pooled={prof['pooled_total_reward_mean']:.2f} "
                    f"(published {DQN_PUBLISHED[scenario]}; match={prof['matches_published_to_2dp']}) "
                    f"actions/ep={ {k: round(v, 1) for k, v in prof['action_counts_per_episode_mean'].items()} }"
                )

        dqn = DQN_PUBLISHED[scenario]
        for name, row in sc.items():
            if isinstance(row, dict) and "total_reward_mean" in row:
                row["cost_ratio_vs_published_masked_dqn"] = row["total_reward_mean"] / dqn
        results[scenario] = sc
        print(f"[{scenario}] done (elapsed {time.time() - started:.0f}s)")
        for name, row in sc.items():
            if isinstance(row, dict) and "total_reward_mean" in row:
                print(
                    f"  {name:24s} reward={row['total_reward_mean']:>11.2f} +/- {row['total_reward_std']:>8.2f}  "
                    f"below_floor={row['below_floor_rate_mean']:.4f}  regret={row['regret_events_mean']:.3f}  "
                    f"forced_rekey={row['forced_rekey_ratio_mean']:.4f}  fresh_keys/ep={row['fresh_key_issuances_per_episode_mean']:.1f}  "
                    f"x_dqn={row['cost_ratio_vs_published_masked_dqn']:.2f}"
                )

    payload = {
        "provenance": (
            "experiments/remeasure_baselines.py; no training. Same setup as Phase 4b Gate W3: "
            "RTIoT2022ReplayProvider forecaster, eval seeds 900-907, 250 decisions/episode, current floor table. "
            "cost_ratio_vs_published_masked_dqn = policy total_reward_mean / published masked-DQN pooled "
            "total_reward (both negative; >1 means costlier than the DQN)."
        ),
        "eval_seeds": EVAL_SEEDS,
        "episode_steps": EPISODE_STEPS,
        "published_masked_dqn_total_reward": DQN_PUBLISHED,
        "step1": step1,
        "scenarios": results,
        "elapsed_seconds": time.time() - started,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {RESULTS_PATH}  (elapsed {time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
