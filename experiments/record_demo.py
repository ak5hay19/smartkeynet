"""
experiments/record_demo.py

Records demo episodes to `events.jsonl.gz` so the dashboard can replay them.

This is the *recording* half of the §S13 split. The dashboard renders and must
never touch the environment; something has to run the environment to produce a
log, and that something lives here, on the experiments side of the line. Before
2026-08-19 the dashboard did both, which is why it could not replay a recorded
run and why the §4.4 event schema went unexercised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.baselines import AlwaysHybridPolicy, Policy, StaticThresholdPolicy
from env.environment import SmartKeyNetEnv
from experiments.train import load_full_config

DEFAULT_LOG_DIR = Path("results/demo_logs")


def record_episode(
    policy: Policy,
    scenario: str,
    out_path: str | Path,
    n_steps: int = 800,
    seed: int = 0,
    config: dict[str, Any] | None = None,
) -> Path:
    """Run one episode and write its §4.4 event log."""
    config = config if config is not None else load_full_config()
    env = SmartKeyNetEnv(
        {
            **config,
            "scenario": scenario,
            "seed": seed,
            "max_steps": n_steps,
            "scenario_steps": n_steps + 200,
        }
    )
    state, info = env.reset(seed=seed)
    for _ in range(n_steps):
        state, _reward, _terminated, truncated, info = env.step(
            policy.act(state, info["action_mask"])
        )
        if truncated:
            break
    return env.write_event_log(out_path)


def record_beat_two(
    log_dir: str | Path = DEFAULT_LOG_DIR, n_steps: int = 800, seed: int = 0
) -> tuple[Path, Path]:
    """Beat 2: the frugal threshold policy against the always-hybrid villain
    on S3 -- the comparison that produces two diverging pool curves and two
    very different regret counters."""
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config = load_full_config()
    frugal = StaticThresholdPolicy(pool_fill_threshold=0.5, min_hybrid_class=2, rekey_age_frac=0.9)
    frugal_path = record_episode(
        frugal, "S3", directory / "beat2_frugal.jsonl.gz", n_steps, seed, config
    )
    villain_path = record_episode(
        AlwaysHybridPolicy(), "S3", directory / "beat2_villain.jsonl.gz", n_steps, seed, config
    )
    return frugal_path, villain_path


def main() -> None:
    frugal_path, villain_path = record_beat_two()
    print(f"recorded {frugal_path}")
    print(f"recorded {villain_path}")


if __name__ == "__main__":
    main()
