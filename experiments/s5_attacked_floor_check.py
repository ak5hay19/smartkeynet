"""
experiments/s5_attacked_floor_check.py

Read-only verification (2026-11-16): on the CURRENT S5 steering-attack
dose-response sweep (Addition E Phase 4b / 2026-09-14 cold-start-fix
re-run -- the sweep behind the paper's Fig. 4), did every masked-agent
decision clear the floor computed from the ATTACKED posture, at every
alpha and every seed?

No training, no checkpoint written, no edits to env/, agents/, attack/,
experiments/harness.py, the reward, or the floor table. It calls the
harness's own `run_scenario_under_attack` (one call per alpha x seed x
policy -- the same function `evaluate_attack_multi_seed` loops over), with
the Phase 4b setup:

  * scenario S2, masked DQN = checkpoints/dqn_s2.pt with configs/scenarios/s2_hndl.yaml;
    soft-reward baseline = checkpoints/soft_reward_baseline_s2.pt with
    configs/soft_reward_baseline_s2.yaml
  * real `LSTMForecastProvider` (checkpoints/lstm_threat_forecaster.pt) as
    BOTH the attacked base provider and the true-posture shadow provider
  * 11 alphas 0.0..1.0 step 0.1, eval seeds 900-904, `GreedyDQNPolicy`

Two different counts per episode, both read from the harness:
  * `scenario_result.floor_violations` -- decisions whose delivered tier
    was below the floor the AGENT WAS SHOWN (the attacked floor,
    `state["policy_floor"]`). Expected 0 for the masked agent by
    construction.
  * `below_floor_true_count` -- decisions whose delivered tier was below
    the floor the TRUE (unshaped) posture would have required: V(pi),
    equation 4, the quantity plotted in Fig. 4.

Cross-checks added here (no harness change): a transparent wrapper
records each decision's shown (attacked) floor and whether the chosen
action was legal under the mask; both harness counts are then recomputed
independently from the harness's own `delivered_tier_log` /
`true_floor_log` and compared.

STEP 1 (hard gate): V(pi) per alpha must reproduce the published sweep
(PROGRESS.md 2026-09-14: masked 0.0000 at alpha<=0.1, 0.0008/0.0024/0.0048/
0.0072 at 0.2-0.5, 0.2984 at alpha>=0.6; and the committed
dashboard/samples/results_data.json). Mismatch -> exit 2, nothing else run.

Needs the gitignored checkpoints named above. Run from the repo root:
    python -m experiments.s5_attacked_floor_check
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from agents.dqn import DQNAgent, flatten_state, load_dqn_config
from env.contracts import Action
from env.environment import SmartKeyNetEnv
from experiments.harness import run_scenario_under_attack
from experiments.train import GreedyDQNPolicy, load_full_config
from forecaster.model import LSTMForecastProvider

_REPO_ROOT = Path(__file__).resolve().parent.parent
LSTM_CHECKPOINT = str(_REPO_ROOT / "checkpoints" / "lstm_threat_forecaster.pt")
RESULTS_PATH = _REPO_ROOT / "docs" / "s5_attacked_floor_check_results.json"
PUBLISHED_JSON = _REPO_ROOT / "dashboard" / "samples" / "results_data.json"

ALPHAS = [round(i * 0.1, 1) for i in range(11)]
EVAL_SEEDS = list(range(900, 905))

AGENTS = {
    "masked": ("checkpoints/dqn_s2.pt", "configs/scenarios/s2_hndl.yaml"),
    "soft_reward": ("checkpoints/soft_reward_baseline_s2.pt", "configs/soft_reward_baseline_s2.yaml"),
}

# Published V(pi) means (PROGRESS.md 2026-09-14 table; identical to the
# Phase 4b scratchpad log and the committed results_data.json s5 series).
PUBLISHED_MASKED = {0.0: 0.0, 0.1: 0.0, 0.2: 0.0008, 0.3: 0.0024, 0.4: 0.0048, 0.5: 0.0072,
                    0.6: 0.2984, 0.7: 0.2984, 0.8: 0.2984, 0.9: 0.2984, 1.0: 0.2984}
PUBLISHED_SOFT = {0.0: 0.3912, 0.1: 0.3912, 0.2: 0.3912, 0.3: 0.3912, 0.4: 0.3912, 0.5: 0.3912,
                  0.6: 0.4272, 0.7: 0.4272, 0.8: 0.4272, 0.9: 0.4272, 1.0: 0.4272}
PUBLISHED = {"masked": PUBLISHED_MASKED, "soft_reward": PUBLISHED_SOFT}
STEP1_TOLERANCE = 5e-5  # published to 4 decimals


def _load_agent(checkpoint: str, config: dict[str, Any]) -> DQNAgent:
    env = SmartKeyNetEnv({**config, "scenario": "S2", "seed": 0})
    state, _ = env.reset(seed=0)
    has_forecast = config.get("use_foresight", "off") != "off"
    agent = DQNAgent(
        state_dim=flatten_state(state, has_forecast).shape[0],
        has_forecast=has_forecast,
        config=load_dqn_config(),
        seed=0,
    )
    agent.load(str(_REPO_ROOT / checkpoint))
    return agent


class _RecordingPolicy:
    """Transparent wrapper: records, per decision, the floor the agent was
    SHOWN (`state["policy_floor"]`, the attacked floor) and whether the
    chosen action was legal under the mask it was handed. Changes no
    decision."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.reset()

    def reset(self) -> None:
        self.shown_floor: list[int] = []
        self.illegal_choices = 0

    def act(self, state, mask):  # type: ignore[no-untyped-def]
        action = self.inner.act(state, mask)
        self.shown_floor.append(int(state["policy_floor"]))
        if not mask[int(action)]:
            self.illegal_choices += 1
        return action


def _provider_factory():
    return LSTMForecastProvider(LSTM_CHECKPOINT)


def _episode(policy: _RecordingPolicy, config: dict[str, Any], alpha: float, seed: int) -> dict[str, Any]:
    policy.reset()
    r = run_scenario_under_attack(
        policy, "S2", config, seed, alpha,
        base_provider_factory=_provider_factory, shadow_provider_factory=_provider_factory,
    )
    delivered = [int(t) for t in r.delivered_tier_log]
    true_floor = [int(t) for t in r.true_floor_log]
    shown = policy.shown_floor
    n = r.total_requests
    if not (len(delivered) == len(true_floor) == len(shown) == n):
        raise RuntimeError("per-decision logs have inconsistent lengths")
    # Independent recounts from the logs.
    attacked_below_recount = sum(d < f for d, f in zip(delivered, shown))
    true_below_recount = sum(d < f for d, f in zip(delivered, true_floor))
    if attacked_below_recount != r.scenario_result.floor_violations:
        raise RuntimeError(
            f"attacked-floor recount {attacked_below_recount} != harness floor_violations "
            f"{r.scenario_result.floor_violations} (alpha={alpha}, seed={seed})"
        )
    if true_below_recount != r.below_floor_true_count:
        raise RuntimeError(f"true-floor recount {true_below_recount} != harness {r.below_floor_true_count}")
    return {
        "alpha": alpha,
        "seed": seed,
        "decisions": n,
        "attacked_floor_violations": r.scenario_result.floor_violations,
        "illegal_action_choices": policy.illegal_choices,
        "below_true_floor_count": r.below_floor_true_count,
        "v_pi": r.below_floor_rate_true,
        # Decisions where the attack actually lowered the floor the agent
        # was shown below what the true posture demanded.
        "shown_floor_below_true_floor": sum(f < t for f, t in zip(shown, true_floor)),
        "total_reward": r.scenario_result.total_reward,
    }


def main() -> int:
    started = time.time()
    for needed in [Path(LSTM_CHECKPOINT)] + [_REPO_ROOT / c for c, _ in AGENTS.values()]:
        if not needed.exists():
            print(f"MISSING required file: {needed}", file=sys.stderr)
            return 3

    configs = {k: load_full_config(cfg) for k, (_, cfg) in AGENTS.items()}
    policies = {k: _RecordingPolicy(GreedyDQNPolicy(_load_agent(ck, configs[k]))) for k, (ck, _) in AGENTS.items()}

    episodes: dict[str, list[dict[str, Any]]] = {k: [] for k in AGENTS}
    for alpha in ALPHAS:
        for name in AGENTS:
            for seed in EVAL_SEEDS:
                episodes[name].append(_episode(policies[name], configs[name], alpha, seed))
        print(f"alpha={alpha:.1f} done (elapsed {time.time() - started:.0f}s)", flush=True)

    per_alpha: dict[str, dict[str, Any]] = {}
    mismatches: list[str] = []
    committed = json.loads(PUBLISHED_JSON.read_text())["s5_dose_response"]["series"]
    committed_by_key = {s["series_key"]: {round(p["alpha"], 1): p["mean"] for p in s["points"]} for s in committed}
    for name in AGENTS:
        per_alpha[name] = {}
        for alpha in ALPHAS:
            eps = [e for e in episodes[name] if e["alpha"] == alpha]
            v = [e["v_pi"] for e in eps]
            row = {
                "v_pi_mean": float(np.mean(v)),
                "v_pi_std": float(np.std(v)),
                "attacked_floor_violations_total": sum(e["attacked_floor_violations"] for e in eps),
                "attacked_floor_violations_per_seed": {str(e["seed"]): e["attacked_floor_violations"] for e in eps},
                "illegal_action_choices_total": sum(e["illegal_action_choices"] for e in eps),
                "below_true_floor_total": sum(e["below_true_floor_count"] for e in eps),
                "below_true_floor_per_seed": {str(e["seed"]): e["below_true_floor_count"] for e in eps},
                "decisions_total": sum(e["decisions"] for e in eps),
                "shown_floor_below_true_floor_total": sum(e["shown_floor_below_true_floor"] for e in eps),
                "total_reward_mean": float(np.mean([e["total_reward"] for e in eps])),
                "published_v_pi": PUBLISHED[name][alpha],
                "committed_results_data_json_v_pi": committed_by_key[name][alpha],
            }
            row["step1_match"] = bool(
                abs(row["v_pi_mean"] - PUBLISHED[name][alpha]) <= STEP1_TOLERANCE
                and abs(row["v_pi_mean"] - committed_by_key[name][alpha]) <= STEP1_TOLERANCE
            )
            if not row["step1_match"]:
                mismatches.append(f"{name} alpha={alpha}: measured {row['v_pi_mean']:.6f} vs published {PUBLISHED[name][alpha]}")
            per_alpha[name][f"{alpha:.1f}"] = row
            print(
                f"[{name:11s}] alpha={alpha:.1f} V(pi)={row['v_pi_mean']:.4f} (pub {PUBLISHED[name][alpha]:.4f})  "
                f"attacked-floor violations={row['attacked_floor_violations_total']}  "
                f"illegal={row['illegal_action_choices_total']}  shown<true={row['shown_floor_below_true_floor_total']}/{row['decisions_total']}"
            )

    payload = {
        "provenance": (
            "experiments/s5_attacked_floor_check.py; read-only; no training. S2, real LSTMForecastProvider as base+shadow, "
            "11 alphas, eval seeds 900-904, GreedyDQNPolicy. attacked_floor_violations = decisions delivering a tier below the "
            "floor the agent was shown (state['policy_floor']); below_true_floor = V(pi) numerator vs the TRUE posture's floor."
        ),
        "alphas": ALPHAS,
        "eval_seeds": EVAL_SEEDS,
        "step1_all_match": not mismatches,
        "step1_mismatches": mismatches,
        "per_alpha": per_alpha,
        "episodes": episodes,
        "elapsed_seconds": time.time() - started,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {RESULTS_PATH}")
    if mismatches:
        print("STEP 1 FAILED: V(pi) does not reproduce the published sweep:\n  " + "\n  ".join(mismatches), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
