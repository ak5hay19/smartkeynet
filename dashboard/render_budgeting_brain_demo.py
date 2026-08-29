"""
dashboard/render_budgeting_brain_demo.py

Real driver for `dashboard/render_budgeting_brain.py` -- runs one real,
same-seed S3 (QKD degradation) episode under two policies:

- the trained masked DQN, reloaded from `checkpoints/s3_masked_seed1.pt`
  (the real checkpoint the 2026-08-25 S3-comparison session trained;
  the 2026-08-29 demo-visuals session's own reproducibility check
  already confirmed this checkpoint reloads byte-identical trained
  weights), evaluated on a fresh held-out seed (900, the first of the
  established `_S3_EVAL_SEEDS` list `dashboard/render_results_demo.py`
  already uses) -- never the checkpoint's own training seed, so this
  is a genuine held-out episode, not training-seed leakage.
- `agents.baselines.AlwaysHybridPolicy` -- that module's own
  documented "drains the pool" baseline (PLAN.md Demo Beat 2). No
  checkpoint needed: it's a deterministic rule ("serve SERVE_HYBRID
  whenever legal, else the lowest legal tier"), run fresh.

Both policies run on the SAME seed (900) -- `env/environment.py::
reset()`'s `episode_seed` drives both the QKD refill/QBER trace
(`SyntheticSKRQBERTrace(seed=episode_seed, ...)`) and the request
arrival stream (`random_request_generator(seed=episode_seed, ...)`)
independently of any policy choice, so running the same seed against
two different policies gives both the IDENTICAL exogenous conditions
-- the fairest possible same-conditions comparison: one real episode
"script", two policies, purely policy-driven divergence. (This
same-seed approach already has real precedent: `configs/scenarios/
s3_degradation.yaml`'s own header comment, 2026-08-24 finding, reports
a same-seed AlwaysHybridPolicy run on S3 showing "near-total
exhaustion... with 21-33 real regret (deferral) events per 250-step
episode across seeds 0/1/4/7" -- so this baseline's real
pool-draining behavior on S3 was already independently observed
before this session, not assumed here for the first time.)

Data provenance: re-derived from a real eval run this session (per
instruction's strong preference), not transcribed from SESSION_LOG.md
prose -- no prior session ran exactly this same-seed head-to-head
trajectory comparison. The raw per-step trajectory + event data is
saved to `dashboard/samples/budgeting_data.json` before rendering, so
a future session has a real file to prefer over re-running.

This driver deliberately does NOT call `experiments/harness.py::
run_scenario` (it only returns final aggregate `ScenarioResult`
stats, no per-step pool trajectory) -- it mirrors that function's own
step loop instead (same public `state`/`info` fields, same
floor/mask/action/cost-resolution sequence; it even imports and reuses
`run_scenario`'s own private `_delivered_tier`/`_resolved_cost_action`
helpers directly, rather than re-deriving that logic, so
`floor_violations`/`below_floor_rate`/`p99_latency` can never silently
drift from harness.py's own real definitions), adding only the
per-step `(env._step_count, state["pool_fill"])` capture and real
`RegretEvent` collection `run_scenario`'s aggregate-only contract
doesn't need. `experiments/harness.py` itself is read-only here, per
instruction -- not modified.

`env._step_count` (the real internal simulator tick counter) is read
directly off the env instance, mirroring the established precedent
(`dashboard/render_living_system_demo.py`, `dashboard/explain.py`) of
reaching into env internals read-only when the public API doesn't
surface needed observability -- here, a real per-tick x-axis that
lines up exactly with `RegretEvent["step"]`'s own real tick numbering
(both come from the same `_step_count` counter), which a purely
decision-indexed axis could not guarantee, since multiple internal
ticks -- and multiple real regret events -- can occur inside a single
external `env.step()` call (see `env/environment.py::
_advance_to_next_decision`).

Run directly: `python -m dashboard.render_budgeting_brain_demo`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from agents.baselines import AlwaysHybridPolicy, Policy
from agents.dqn import DQNAgent, flatten_state, load_dqn_config
from dashboard.render_budgeting_brain import (
    BudgetingBrainData,
    ExhaustionEvent,
    PolicyEpisode,
    PoolTrajectoryPoint,
    write_budgeting_brain_html,
)
from env.contracts import Action
from env.environment import _LATENCY_UNITS, SmartKeyNetEnv
from experiments.harness import _delivered_tier, _resolved_cost_action
from experiments.train import GreedyDQNPolicy, load_full_config

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINTS_DIR = _REPO_ROOT / "checkpoints"
_SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

_SCENARIO = "S3"
_SEED = 900  # held-out eval seed, NOT the checkpoint's own training seed (1)
_MAX_STEPS = 250


def _load_masked_policy(config: dict[str, Any]) -> GreedyDQNPolicy:
    """Mirrors `dashboard/render_results_demo.py::_load_greedy_policy`
    exactly (same `state_dim`/`has_forecast` derivation, same
    `load_dqn_config()` network-shape source) -- reused here as its
    own real function is private to that module's own driver, so this
    is a byte-for-byte-identical re-implementation of that established
    reload pattern, not a divergent one."""
    has_forecast = config.get("use_foresight", "off") != "off"
    env = SmartKeyNetEnv({**config, "seed": 0})
    state, _info = env.reset(seed=0)
    state_dim = flatten_state(state, has_forecast).shape[0]
    agent = DQNAgent(state_dim=state_dim, has_forecast=has_forecast, config=load_dqn_config())
    agent.load(str(_CHECKPOINTS_DIR / "s3_masked_seed1.pt"))
    return GreedyDQNPolicy(agent)


def _run_episode_with_trajectory(
    policy: Policy,
    config: dict[str, Any],
    seed: int,
    *,
    label: str,
    series_key: str,
    tag: str,
    pool_capacity_bits: float,
) -> PolicyEpisode:
    """Run one real S3 episode with `policy`, collecting the real
    per-decision `(env._step_count, state["pool_fill"])` trajectory and
    every real `RegretEvent`, alongside the same aggregate stats
    `experiments/harness.py::run_scenario` computes (via its own
    `_delivered_tier`/`_resolved_cost_action` helpers, imported
    directly -- not re-derived)."""
    env_config = {**config, "scenario": _SCENARIO, "seed": seed, "max_steps": _MAX_STEPS}
    env = SmartKeyNetEnv(env_config)
    state, info = env.reset(seed=seed)

    trajectory: list[PoolTrajectoryPoint] = [
        PoolTrajectoryPoint(step=env._step_count, pool_fill=state["pool_fill"])
    ]
    regret_events_raw: list[Any] = list(info["regret_events"])

    latencies: list[float] = []
    floor_violations = 0
    total_rekeys = 0
    total_requests = 0
    forced_rekeys: list[Any] = []

    truncated = False
    while not truncated:
        mask = info["action_mask"]
        floor = Action(state["policy_floor"])
        key_type_onehot = state["key_type_onehot"]

        action = policy.act(state, mask)

        if int(_delivered_tier(action, key_type_onehot, floor)) < int(floor):
            floor_violations += 1  # should never fire for a masked policy

        cost_action = _resolved_cost_action(action, key_type_onehot, floor)
        latencies.append(_LATENCY_UNITS[cost_action])

        if action is not Action.REUSE:
            total_rekeys += 1
        total_requests += 1

        state, _reward, _terminated, truncated, info = env.step(action)
        trajectory.append(PoolTrajectoryPoint(step=env._step_count, pool_fill=state["pool_fill"]))

        regret_events_raw.extend(info["regret_events"])
        if "forced_rekey" in info:
            forced_rekeys.append(info["forced_rekey"])

    exhaustion_events = [
        ExhaustionEvent(
            step=ev["step"],
            pool_fill_normalized=ev["pool_fill_at_onset"] / pool_capacity_bits,
            tenant=ev["tenant"],
            sensitivity_class=int(ev["sensitivity_class"]),
        )
        for ev in regret_events_raw
    ]

    forced_rekey_ratio = len(forced_rekeys) / total_rekeys if total_rekeys > 0 else 0.0
    below_floor_rate = floor_violations / total_requests if total_requests > 0 else 0.0
    p99_latency = float(np.percentile(latencies, 99)) if latencies else 0.0

    return PolicyEpisode(
        label=label,
        series_key=series_key,
        tag=tag,
        trajectory=trajectory,
        exhaustion_events=exhaustion_events,
        regret_events=len(exhaustion_events),
        pool_exhaustion_events=len(exhaustion_events),
        below_floor_rate=below_floor_rate,
        forced_rekey_ratio=forced_rekey_ratio,
        p99_latency=p99_latency,
    )


def collect_real_budgeting_brain_data() -> BudgetingBrainData:
    """Run both real policies on the same real S3 episode (seed=900)
    and return the real `BudgetingBrainData` this session's renderer
    consumes -- eval only for the masked DQN (reloads an already-
    trained checkpoint), no training performed by this driver at all
    (`AlwaysHybridPolicy` needs none)."""
    config = load_full_config(_REPO_ROOT / "configs" / "scenarios" / "s3_degradation.yaml")
    pool_capacity_bits = float(config["pool"]["capacity_bits"])

    masked_policy = _load_masked_policy(config)
    agent_episode = _run_episode_with_trajectory(
        masked_policy,
        config,
        _SEED,
        label="Masked DQN (agent)",
        series_key="agent",
        tag=f"foresight: {config.get('use_foresight', 'off')}",
        pool_capacity_bits=pool_capacity_bits,
    )

    baseline_policy = AlwaysHybridPolicy()
    baseline_episode = _run_episode_with_trajectory(
        baseline_policy,
        config,
        _SEED,
        label="Always-Hybrid (baseline)",
        series_key="baseline",
        tag="no budgeting",
        pool_capacity_bits=pool_capacity_bits,
    )

    return BudgetingBrainData(
        scenario=_SCENARIO,
        seed=_SEED,
        agent=agent_episode,
        baseline=baseline_episode,
        include_p99=True,
    )


def _save_provenance_json(data: BudgetingBrainData, path: Path) -> None:
    payload = {
        "provenance": (
            "Real S3 (QKD degradation) episode, seed=900 (held-out, not the checkpoint's own "
            "training seed 1), run fresh this session via dashboard/render_budgeting_brain_demo.py "
            "-- masked DQN reloaded from checkpoints/s3_masked_seed1.pt (eval only, no retraining); "
            "AlwaysHybridPolicy run directly (no checkpoint, deterministic rule). Both policies see "
            "the identical QKD refill/QBER trace and request-arrival stream (same env seed)."
        ),
        "scenario": data.scenario,
        "seed": data.seed,
        "agent": dataclasses.asdict(data.agent),
        "baseline": dataclasses.asdict(data.baseline),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    print("Running real S3 episode (seed=900) under the masked DQN (reloaded checkpoint)...")
    data = collect_real_budgeting_brain_data()
    print(
        f"  agent: regret_events={data.agent.regret_events} "
        f"below_floor_rate={data.agent.below_floor_rate:.4f} "
        f"trajectory points={len(data.agent.trajectory)}"
    )
    print(
        f"  baseline: regret_events={data.baseline.regret_events} "
        f"below_floor_rate={data.baseline.below_floor_rate:.4f} "
        f"trajectory points={len(data.baseline.trajectory)}"
    )

    _SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    json_path = _SAMPLES_DIR / "budgeting_data.json"
    _save_provenance_json(data, json_path)
    print(f"wrote {json_path}")

    html_path = _SAMPLES_DIR / "budgeting_brain.html"
    write_budgeting_brain_html(data, html_path)
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
