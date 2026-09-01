"""
forecaster/replay_campaign.py

Step 5 of the RT-IoT2022 replay-wiring session: run a MEANINGFUL number
of real episodes (not 2-3) against `SmartKeyNetEnv`, driven by
`RTIoT2022ReplayProvider` (real, held-out RT-IoT2022 replay features),
and report the real resulting posture/floor distribution -- honestly,
including if it saturates given the real 90/10 benign/attack imbalance.

Run separately for the benign-only and attack-only held-out spans
(rather than one artificially interleaved stream) because that is what
the real data actually is: RT-IoT2022 ships with each Attack_type as
one contiguous block (see `train_threat_lstm.py`'s module docstring),
so there is no real interleaved "mixed" ground truth to replay --
reporting the two classes separately is the honest, non-fabricated
choice, and it directly answers Step 5's question ("does posture
saturate on the majority class"). Different seeds use a different
`start_offset` into the real held-out span (see
`RTIoT2022ReplayProvider`'s docstring), so a many-episode run covers a
genuinely broad real slice of the held-out data, not the same first 64
rows repeated.

Does NOT edit `env/environment.py` -- `RTIoT2022ReplayProvider` is
injected via the existing `forecast_provider_factory` seam, same as
Addition E Phase 4a.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from agents.baselines import StaticThresholdPolicy
from env.contracts import Action
from env.environment import SmartKeyNetEnv
from experiments.train import load_full_config
from forecaster.rt_iot_replay import RTIoT2022ReplayProvider, held_out_replay_rows

REAL_CHECKPOINT = Path("checkpoints/lstm_threat_forecaster.pt")


def _run_one_class(features, checkpoint_path: str, config: dict, n_episodes: int, max_steps: int) -> dict[str, Any]:
    policy = StaticThresholdPolicy(pool_fill_threshold=0.5)
    floor_counts: Counter = Counter()
    all_scores: list[float] = []
    per_episode: list[dict[str, Any]] = []

    for seed in range(n_episodes):
        env = SmartKeyNetEnv(
            config,
            forecast_provider_factory=lambda _seed, feats=features, off=seed * max_steps: RTIoT2022ReplayProvider(
                checkpoint_path, feats, start_offset=off
            ),
        )
        state, info = env.reset(seed=seed)
        episode_scores: list[float] = []
        episode_floor_counts: Counter = Counter()
        truncated = False
        while not truncated:
            mask = info["action_mask"]
            action = policy.act(state, mask)
            state, reward, terminated, truncated, info = env.step(action)
            episode_scores.append(state["threat_score"])
            floor_name = Action(state["policy_floor"]).name
            floor_counts[floor_name] += 1
            episode_floor_counts[floor_name] += 1

        all_scores.extend(episode_scores)
        per_episode.append(
            {
                "seed": seed,
                "n_decisions": len(episode_scores),
                "threat_score_mean": sum(episode_scores) / len(episode_scores),
                "floor_distribution": dict(episode_floor_counts),
            }
        )

    return {
        "n_episodes": n_episodes,
        "n_decisions_total": sum(floor_counts.values()),
        "floor_distribution": dict(floor_counts),
        "threat_score_min": min(all_scores),
        "threat_score_max": max(all_scores),
        "threat_score_mean": sum(all_scores) / len(all_scores),
        "per_episode": per_episode,
    }


def run_replay_campaign(
    n_episodes_per_class: int = 15,
    max_steps: int = 60,
    checkpoint_path: str = str(REAL_CHECKPOINT),
) -> dict[str, Any]:
    holdout = held_out_replay_rows()
    config = {**load_full_config(), "max_steps": max_steps}

    results = {
        "n_episodes_per_class": n_episodes_per_class,
        "max_steps_per_episode": max_steps,
        "held_out_benign_rows": int(holdout.benign_features.shape[0]),
        "held_out_attack_rows": int(holdout.attack_features.shape[0]),
        "benign": _run_one_class(holdout.benign_features, checkpoint_path, config, n_episodes_per_class, max_steps),
        "attack": _run_one_class(holdout.attack_features, checkpoint_path, config, n_episodes_per_class, max_steps),
    }
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-episodes-per-class", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=60)
    args = parser.parse_args()

    result = run_replay_campaign(n_episodes_per_class=args.n_episodes_per_class, max_steps=args.max_steps)
    summary = {
        "n_episodes_per_class": result["n_episodes_per_class"],
        "max_steps_per_episode": result["max_steps_per_episode"],
        "held_out_benign_rows": result["held_out_benign_rows"],
        "held_out_attack_rows": result["held_out_attack_rows"],
        "benign": {k: v for k, v in result["benign"].items() if k != "per_episode"},
        "attack": {k: v for k, v in result["attack"].items() if k != "per_episode"},
    }
    print(json.dumps(summary, indent=2))
