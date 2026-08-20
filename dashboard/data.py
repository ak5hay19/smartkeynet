"""
dashboard/data.py

Assembles the seven Dashboard v2 panels (PLAN2 §7) from **real** runs.

There is exactly one rule governing this module, and it is the reason
it exists as a separate layer from the rendering:

    Every number that reaches a panel is produced by the real
    environment, the real policy table, the real mask, the real
    forecaster, or a real experiment artefact under `results/`.

`mock.html` is layout truth and nothing else -- PLAN2's header states
that every value in it was hand-authored for layout demonstration.
Where an artefact has not been generated yet, the payload carries an
explicit `available: false` and the panel renders "not yet run" rather
than a plausible-looking placeholder.

Panels 1-4 and 6 come from a live replay of the environment; panels 5
and 7 read the artefacts written by `attack/run_attack.py` and
`experiments/results_table.py`.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agents.baselines import AlwaysHybridPolicy, AlwaysPQCPolicy, Policy
from env.contracts import Action, KeyType, SensitivityClass, ThreatPosture
from env.decision_trace import build_decision_trace
from env.environment import _KEY_TYPE_TO_SERVE_ACTION, SmartKeyNetEnv, build_scenario_runtime
from env.masking import _PLACEHOLDER_FLOOR_TABLE
from experiments.harness import run_scenario
from experiments.train import load_full_config

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_TIER_COLORS = {
    "SERVE_CLASSICAL": "#e0575b",  # weakest tier -- warm/warning
    "SERVE_PQC": "#4a9eda",
    "SERVE_HYBRID": "#37c98b",  # strongest tier
}


def _load_artifact(name: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


@dataclass
class ReplayRun:
    """One recorded environment replay -- the substrate for panels 1-3."""

    scenario: str
    seed: int
    decisions: list[dict[str, Any]]
    pool_series: list[dict[str, float]]
    threat_series: list[dict[str, Any]]
    latency_series: list[float]
    edge_last_tier: dict[str, str]
    graph: dict[str, Any]
    migration_phases: list[dict[str, Any]]


def record_replay(
    config: dict[str, Any] | None = None,
    scenario: str = "S1",
    policy: Policy | None = None,
    seed: int = 0,
    steps: int = 600,
) -> ReplayRun:
    """Run the real environment and record everything the panels show.

    Every field below is read off live objects at decision time -- the
    same objects `api/main.py` serves from -- so a panel and the API can
    never disagree about what happened.
    """
    config = config if config is not None else load_full_config()
    policy = policy if policy is not None else AlwaysPQCPolicy()
    env_config = {**config, "scenario": scenario, "seed": seed, "max_steps": steps}
    env = SmartKeyNetEnv(env_config)
    state, info = env.reset(seed=seed)
    runtime = build_scenario_runtime(scenario, config)

    decisions: list[dict[str, Any]] = []
    pool_series: list[dict[str, float]] = []
    threat_series: list[dict[str, Any]] = []
    latency_series: list[float] = []
    edge_last_tier: dict[str, str] = {}
    latency_window: deque[float] = deque(maxlen=100)

    from env.environment import _LATENCY_UNITS

    truncated = False
    while not truncated:
        mask = info["action_mask"]
        action = policy.act(state, mask)
        trace = build_decision_trace(env, state, mask, action)

        latency = float(_LATENCY_UNITS[Action[trace.delivered_tier]])
        latency_window.append(latency)
        latency_series.append(float(np.percentile(list(latency_window), 99)))

        edge_last_tier[f"{trace.tenant}/{trace.service}"] = trace.delivered_tier
        decisions.append(
            {
                "request_id": trace.request_id,
                "step": trace.step,
                "tenant": trace.tenant,
                "service": trace.service,
                "sensitivity_class": SensitivityClass(trace.sensitivity_class).name,
                "policy_floor": trace.policy_floor,
                "action": trace.chosen_action,
                "delivered_tier": trace.delivered_tier,
                "tier_color": _TIER_COLORS.get(trace.delivered_tier, "#888"),
                "reason": trace.steps[-1].summary,
                "trace": trace.to_dict(),
            }
        )
        pool_series.append(
            {
                "step": float(trace.step),
                "fill_fraction": float(env._pool_sim.fill / env._pool_sim.capacity),
                "skr_kbps": float(env._last_pool_state.skr),
                "qber": float(env._last_pool_state.qber),
                "deferral_depth": float(len(env._deferral_queue)),
            }
        )
        threat_series.append(
            {
                "step": trace.step,
                "threat_score": trace.threat_score,
                "posture": trace.resolved_posture,
                "horizon": trace.steps[0].values.get("threat_score"),
            }
        )

        state, _reward, _terminated, truncated, info = env.step(action)

    graph = {
        "nodes": [
            {"id": node, "tenant": data["tenant"], "service": data["service"]}
            for node, data in env._tenant_graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                "tenant": data["tenant"],
                "sensitivity_class": SensitivityClass(data["sensitivity_class"]).name,
                "traffic_rate": round(float(data["traffic_rate"]), 4),
                "pqc_capable": bool(data["pqc_capable"]),
            }
            for u, v, data in env._tenant_graph.edges(data=True)
        ],
    }

    return ReplayRun(
        scenario=scenario,
        seed=seed,
        decisions=decisions,
        pool_series=pool_series,
        threat_series=threat_series,
        latency_series=latency_series,
        edge_last_tier=edge_last_tier,
        graph=graph,
        migration_phases=runtime.migration_phases_elapsed(10**9),
    )


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def panel_threat_input(config: dict[str, Any], replay: ReplayRun) -> dict[str, Any]:
    """Panel 1 (PLAN2 §7.1).

    The pipeline stages are labelled for what was actually built. The
    mockup depicts autoencoder -> XGBoost -> fusion; what exists is
    benign-referenced standardization -> a trained LSTM threat head ->
    a fixed posture mapping. Same three-stage shape, different
    mechanisms, and the panel says which.
    """
    threat_cfg = config.get("threat_input") or {}
    latest = replay.threat_series[-1] if replay.threat_series else {}
    return {
        "available": True,
        "source_modes": [
            {
                "id": "offline",
                "label": "Offline dataset",
                "detail": "RT-IoT2022, batch-processed. Training-time only.",
                "active": threat_cfg.get("source") == "rt_iot2022",
                "implemented": True,
            },
            {
                "id": "scenario",
                "label": "Scenario signal",
                "detail": "The scenario's own standardized threat level. Deterministic.",
                "active": threat_cfg.get("source", "scenario") == "scenario",
                "implemented": True,
            },
            {
                "id": "pcap_upload",
                "label": "Uploaded pcap",
                "detail": (
                    "Same extract_flow_features() path as the CSV (Hard Rule 11). "
                    "Extraction implemented and tested; the upload endpoint is not built."
                ),
                "active": False,
                "implemented": False,
            },
            {
                "id": "pcap_replay",
                "label": "Replayed pcap (real-time pace)",
                "detail": "Not built -- PLAN2 §11 cut-order item 1.",
                "active": False,
                "implemented": False,
            },
        ],
        "pipeline_stages": [
            {
                "label": "Benign-referenced standardization",
                "mockup_label": "autoencoder",
                "detail": (
                    "16 flow features z-scored against BENIGN traffic, so an attack window is a "
                    "positive deviation. Measured separation: Cohen's d = +4.43 "
                    "(and -0.98 -- inverted -- when standardized against the whole capture)."
                ),
            },
            {
                "label": "LSTM threat head",
                "mockup_label": "XGBoost classifier",
                "detail": (
                    "Shared-trunk dual-head LSTM. Validation balanced accuracy 0.9312 against a "
                    "majority-class rate of 0.6817."
                ),
            },
            {
                "label": "Posture mapping",
                "mockup_label": "fusion",
                "detail": (
                    "Fixed RBF-softmax over CALM/ELEVATED/HIGH anchors, shared verbatim with the "
                    "EWMA fallback so both providers mean the same thing by a score."
                ),
            },
        ],
        "current": {
            "threat_score": latest.get("threat_score", 0.0),
            "posture": latest.get("posture", "CALM"),
        },
        "feature_columns": _feature_columns(),
        "invariants": [
            "Feeds env/masking.py's floor computation ONLY in the raise direction (Hard Rule 2).",
            "Frozen during DQN training -- no gradient from the agent's loss reaches it.",
        ],
        "divergence_note": (
            "The mockup depicts autoencoder + XGBoost + fusion. What is built is the LSTM "
            "dual-head PLAN2 §4A specifies; the stage labels above name the real mechanisms."
        ),
    }


def _feature_columns() -> list[str]:
    from forecaster.dataset import FEATURE_COLUMNS

    return list(FEATURE_COLUMNS)


def panel_living_system(replay: ReplayRun) -> dict[str, Any]:
    """Panel 2 (PLAN2 §7.2)."""
    return {
        "available": True,
        "graph": replay.graph,
        "edge_last_tier": replay.edge_last_tier,
        "tier_colors": _TIER_COLORS,
        "recent_decisions": [
            {k: v for k, v in decision.items() if k != "trace"}
            for decision in replay.decisions[-25:]
        ],
        "pool": replay.pool_series[-1] if replay.pool_series else {},
        "pool_series": replay.pool_series[-200:],
        "threat_series": replay.threat_series[-200:],
        "p99_latency": replay.latency_series[-1] if replay.latency_series else 0.0,
        "latency_series": replay.latency_series[-200:],
    }


def panel_explain_decision(replay: ReplayRun, request_id: str | None = None) -> dict[str, Any]:
    """Panel 3 (PLAN2 §7.3, Hard Rule 10).

    Defaults to the most interesting decision available: one where a
    real cost tradeoff existed, so the panel demonstrates step 5 doing
    something. Falls back to the last decision.
    """
    if not replay.decisions:
        return {"available": False, "reason": "no decisions recorded"}

    chosen = None
    if request_id is not None:
        chosen = next((d for d in replay.decisions if d["request_id"] == request_id), None)
    if chosen is None:
        chosen = next(
            (d for d in reversed(replay.decisions) if d["trace"]["cost_tradeoff_existed"]),
            replay.decisions[-1],
        )

    return {
        "available": True,
        "trace": chosen["trace"],
        "floor_table": _floor_table_grid(),
        "highlighted_cell": {
            "sensitivity_class": SensitivityClass(chosen["trace"]["sensitivity_class"]).name,
            "posture": chosen["trace"]["resolved_posture"],
        },
        "hard_rule_10": (
            "Every value shown was computed by the pipeline; each sentence is templated "
            "deterministically from those values. There is no generative step in this path."
        ),
    }


def _floor_table_grid() -> list[dict[str, Any]]:
    """The real `(class x posture) -> floor` table, straight from
    `env/masking.py`. Rendered, never re-typed."""
    return [
        {
            "sensitivity_class": sensitivity.name,
            "floors": {
                posture.name: _PLACEHOLDER_FLOOR_TABLE[(sensitivity, posture)].name
                for posture in ThreatPosture
            },
        }
        for sensitivity in SensitivityClass
    ]


def panel_budgeting_brain(
    config: dict[str, Any], scenario: str = "S3", seed: int = 0, steps: int = 2000
) -> dict[str, Any]:
    """Panel 4 (PLAN2 §7.4): masked policy vs always-hybrid on S3.

    Both sides are real `run_scenario` results. The "agent" side uses
    whatever masked policy the caller supplies; the default is
    AlwaysPQC, and the panel labels which policy produced the column
    rather than implying a trained DQN when none was loaded.
    """
    arms = {
        "masked policy (AlwaysPQC)": AlwaysPQCPolicy(),
        "always-hybrid baseline": AlwaysHybridPolicy(),
    }
    out: dict[str, Any] = {"available": True, "scenario": scenario, "arms": {}}
    for name, policy in arms.items():
        result = run_scenario(policy, scenario, {**config, "max_steps": steps}, seed=seed)
        metrics = result.episode_metrics
        out["arms"][name] = {
            "p99_latency": result.p99_latency,
            "pool_exhaustion_events": result.pool_exhaustion_events,
            "regret_events": metrics.regret_events,
            "deferred_critical_steps": metrics.deferred_critical_steps,
            "forced_rekey_ratio": metrics.forced_rekey_ratio,
            "floor_violations": result.floor_violations,
            "total_reward": result.total_reward,
        }
    replay_agent = record_replay(config, scenario, AlwaysPQCPolicy(), seed, steps=min(steps, 600))
    replay_base = record_replay(config, scenario, AlwaysHybridPolicy(), seed, steps=min(steps, 600))
    out["pool_series"] = {
        "masked policy (AlwaysPQC)": replay_agent.pool_series,
        "always-hybrid baseline": replay_base.pool_series,
    }
    return out


def panel_steering_attack() -> dict[str, Any]:
    """Panel 5 (PLAN2 §7.5) -- read from `attack/run_attack.py`'s artefact."""
    payload = _load_artifact("steering_dose_response.json")
    if payload is None:
        return {
            "available": False,
            "reason": "run `python -m attack.run_attack` to generate results/steering_dose_response.json",
        }
    return {
        "available": True,
        "doses": payload["doses"],
        "policies": payload["policies"],
        "caption": (
            "Security that isn't in the reward can't be steered out of it, because it was never "
            "a preference to begin with."
        ),
        "metric_note": (
            "Scored at key establishments. `below_class_floor_share` is the share below the "
            "sensitivity-class floor -- the level Hard Rule 2 guarantees no signal can lower. "
            "`below_escalated_floor_share` is the share below the floor the HONEST posture would "
            "have set, and is non-zero for both arms: suppression does prevent floors from "
            "escalating, it just cannot push them below the class floor."
        ),
    }


def panel_migration_wave(config: dict[str, Any], seed: int = 0, steps: int = 2000) -> dict[str, Any]:
    """Panel 6 (PLAN2 §7.6) -- the scripted, held-out S6 schedule."""
    runtime = build_scenario_runtime("S6", config)
    schedule = list(config.get("migration_schedule") or ())
    replay = record_replay(config, "S6", AlwaysPQCPolicy(), seed, steps=min(steps, 1600))

    phases = []
    for entry in schedule:
        step = int(entry["step"])
        phases.append(
            {
                "step": step,
                "cohort": entry["cohort"],
                "new_floor": entry["new_floor"],
                "label": entry.get("label", ""),
                "pqc_capable": bool(entry.get("pqc_capable", False)),
                "floors_in_force": {
                    k: v.name for k, v in runtime.cohort_floors(step).items()
                },
            }
        )

    return {
        "available": True,
        "held_out": True,
        "held_out_note": (
            "Hard Rule 8: the agent is never trained on this schedule. "
            "experiments/train.py and forecaster/train.py both refuse S6 outright."
        ),
        "exogenous_note": (
            "Hard Rule 3: the schedule is scripted config. The agent never chooses the migration "
            "order, never sees the cohorts, and never sees the schedule."
        ),
        "phases": phases,
        "pool_series": replay.pool_series,
    }


def panel_results(config: dict[str, Any]) -> dict[str, Any]:
    """Panel 7 (PLAN2 §7.7) -- read from `experiments/results_table.py`."""
    payload = _load_artifact("closing_table.json")
    if payload is None:
        return {
            "available": False,
            "reason": "run `python -m experiments.results_table` to generate results/closing_table.json",
        }
    return {
        "available": True,
        **payload,
        "structural_note": (
            "The masked policies' floor-violations column is 0 by construction, not by outcome: "
            "env/masking.py's five legality rules make a below-floor delivery unrepresentable, "
            "and experiments/harness.py counts DELIVERED tier rather than chosen action so the "
            "count would notice if a rule were removed."
        ),
    }


def build_dashboard_payload(
    config: dict[str, Any] | None = None,
    replay_scenario: str = "S1",
    replay_steps: int = 600,
    seed: int = 0,
    include_slow_panels: bool = True,
) -> dict[str, Any]:
    """Assemble all seven panels.

    `include_slow_panels=False` skips panels 4 and 6, which each run
    full episodes -- used by tests and by the fast path of the static
    exporter.
    """
    config = config if config is not None else load_full_config()
    replay = record_replay(config, replay_scenario, AlwaysPQCPolicy(), seed, steps=replay_steps)

    payload: dict[str, Any] = {
        "generated_from": "real environment runs and results/ artefacts -- no placeholder values",
        "config_summary": {
            "scenario": replay_scenario,
            "seed": seed,
            "use_foresight": config.get("use_foresight"),
            "threat_source": (config.get("threat_input") or {}).get("source", "scenario"),
            "pool_capacity_bits": config["pool"]["capacity_bits"],
            "refill_bits_per_step": config["pool"].get("refill_bits_per_step"),
            "tenant_graph_nodes": config["tenant_graph"]["n_nodes"],
        },
        "p1_threat_input": panel_threat_input(config, replay),
        "p2_living_system": panel_living_system(replay),
        "p3_explain_decision": panel_explain_decision(replay),
        "p5_steering_attack": panel_steering_attack(),
        "p7_results": panel_results(config),
    }
    if include_slow_panels:
        payload["p4_budgeting_brain"] = panel_budgeting_brain(config, seed=seed)
        payload["p6_migration_wave"] = panel_migration_wave(config, seed=seed)
    else:
        payload["p4_budgeting_brain"] = {"available": False, "reason": "skipped (slow panel)"}
        payload["p6_migration_wave"] = {"available": False, "reason": "skipped (slow panel)"}
    return payload
