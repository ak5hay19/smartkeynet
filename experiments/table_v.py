"""Generate Table V: S3, ten evaluation seeds, median. Real runs only.

Every cell here comes from an actual run. The victim row is evaluated with
floors ADVISORY (spec S10 ), which is the only
configuration in which its V(pi) column can be anything but a structural zero
-- see agents/soft_reward_baseline.py and tests/test_soft_reward_baseline.py.

    .venv/bin/python -m experiments.table_v
"""

import json

import numpy as np
import yaml

from agents.baselines import (
    AlwaysHybridPolicy,
    AlwaysPQCPolicy,
    RandomPolicy,
    StaticThresholdPolicy,
)
from agents.soft_reward_baseline import GreedySoftRewardPolicy
from experiments.harness import run_scenario
from experiments.steering_attack import train_soft_reward_agent
from experiments.train import GreedyDQNPolicy, train

base = yaml.safe_load(open("configs/default.yaml"))
EVAL_SEEDS = [1000 + i for i in range(10)]
TRAIN_SEEDS = [0, 1, 2]
EVAL_CFG = {**base, "max_steps": 1500, "scenario_steps": 1700}
# The victim runs with floors ADVISORY (spec §S10), which is the only way its
# V(pi) column can be anything but a structural zero.
VICTIM_CFG = {**EVAL_CFG, "masking": {"enabled": False}}


def row(name, policy, cfg):
    rs = [run_scenario(policy, "S3", cfg, seed=s) for s in EVAL_SEEDS]
    med = lambda f: float(np.median([f(r) for r in rs]))
    return {
        "policy": name,
        "p99_latency_ms": med(lambda r: r.p99_latency_ms),
        "pool_exhaustion_events": med(lambda r: r.pool_exhaustion_events),
        "regret_events": med(lambda r: r.episode_metrics.regret_events),
        "rekeys_per_100_requests": med(lambda r: r.episode_metrics.rekeys_per_100_requests),
        "floor_violations_total": int(sum(r.floor_violations for r in rs)),
    }


rows = []

print("training masked DQN (ours) ...", flush=True)
dqn_rows = []
for seed in TRAIN_SEEDS:
    agent, _ = train(
        full_config=base,
        training_overrides={
            "seed": seed,
            "total_steps": 250_000,
            "eval_every": 250_000,
            "eval_max_steps": 1500,
            "checkpoint_path": f"checkpoints/tablev_dqn_s{seed}.pt",
        },
        scenario="S3",
    )
    dqn_rows.append(row(f"dqn_seed{seed}", GreedyDQNPolicy(agent), EVAL_CFG))
    print(f"  seed {seed}: regret {dqn_rows[-1]['regret_events']}", flush=True)
# median across training seeds of each column
merged = {"policy": "Masked DQN (ours)"}
for key in dqn_rows[0]:
    if key == "policy":
        continue
    merged[key] = float(np.median([r[key] for r in dqn_rows]))
merged["floor_violations_total"] = int(sum(r["floor_violations_total"] for r in dqn_rows))
rows.append(merged)

print("training soft-reward DQN (victim, floors advisory) ...", flush=True)
victim_rows = []
for seed in TRAIN_SEEDS:
    victim = train_soft_reward_agent(VICTIM_CFG, n_steps=60_000, seed=seed)
    victim_rows.append(row(f"victim_seed{seed}", GreedySoftRewardPolicy(victim), VICTIM_CFG))
    print(f"  seed {seed}: violations {victim_rows[-1]['floor_violations_total']}", flush=True)
merged = {"policy": "Soft-reward DQN"}
for key in victim_rows[0]:
    if key == "policy":
        continue
    merged[key] = float(np.median([r[key] for r in victim_rows]))
merged["floor_violations_total"] = int(sum(r["floor_violations_total"] for r in victim_rows))
rows.append(merged)

for name, mk in [
    ("Static threshold", lambda: StaticThresholdPolicy(0.95, 0, 0.9)),
    ("Always-hybrid", AlwaysHybridPolicy),
    ("Always-PQC", AlwaysPQCPolicy),
    ("Random (safe set)", lambda: RandomPolicy(seed=0)),
]:
    rows.append(row(name, mk(), EVAL_CFG))

json.dump(
    {
        "scenario": "S3",
        "eval_seeds": EVAL_SEEDS,
        "train_seeds": TRAIN_SEEDS,
        "statistic": "median",
        "rows": rows,
    },
    open("results/table_v.json", "w"),
    indent=2,
)

print()
print("TABLE V -- S3, ten evaluation seeds, MEDIAN")
print(f"{'Policy':22s} {'p99 lat':>8s} {'Exhaust':>8s} {'Regret':>8s} {'Rekey':>7s} {'V(pi)':>7s}")
for r in rows:
    print(
        f"{r['policy']:22s} {r['p99_latency_ms']:8.1f} {r['pool_exhaustion_events']:8.1f} "
        f"{r['regret_events']:8.1f} {r['rekeys_per_100_requests']:7.1f} "
        f"{r['floor_violations_total']:7d}"
    )
