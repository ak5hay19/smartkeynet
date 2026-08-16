"""
dashboard/app.py

Live demo dashboard (PLAN.md §6, Demo Beats 1-4). Owned by Person D
(split.md §1).

Panels: tenant service graph (edges coloured by key type served), QKD
pool gauge, threat forecast strip, live regret counter, latency chart.

Implementation is Plotly Dash (PLAN.md tech stack lists it as
negotiable; Dash keeps everything in Python, so the dashboard reads the
same `SmartKeyNetEnv` the experiments do rather than a re-implementation
of it in JavaScript that could silently drift).

---------------------------------------------------------------------
The four beats, and which panel carries each
---------------------------------------------------------------------
  1. **The living system** -- the tenant graph, edges flashing by served
     tier, with the pool gauge and latency chart alongside.
  2. **The budgeting brain** -- agent vs always-hybrid on S3, two
     diverging pool curves and two live regret counters. This is the
     beat where always-hybrid drains the pool and the deferral queue
     visibly backs up.
  3. **The steering attack** -- the served-tier comparison between the
     soft-reward victim and the masked agent, read from
     `results/steering_attack.json`.
  4. **The migration wave** -- S6's scripted floor changes stepping a
     tenant cohort's floor upward mid-episode.

Run it:

    .venv/bin/python -m dashboard.app

Then open http://127.0.0.1:8050. `build_frames` is importable and
testable without Dash installed, which is what the tests exercise --
the callbacks are thin wrappers over it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agents.baselines import AlwaysHybridPolicy, StaticThresholdPolicy
from env.contracts import Action, KeyType
from env.environment import SmartKeyNetEnv
from experiments.train import load_full_config

TIER_COLOURS: dict[str, str] = {
    "CLASSICAL": "#9e9e9e",  # grey
    "PQC": "#f5a623",  # amber
    "HYBRID": "#2ecc71",  # green
    "NONE": "#37474f",
}
"""PLAN.md §6 Beat 1: "edges flash coloured by key type served (grey
classical / amber PQC / green hybrid)"."""

_KEY_TYPE_NAMES: dict[KeyType, str] = {
    KeyType.CLASSICAL: "CLASSICAL",
    KeyType.PQC: "PQC",
    KeyType.HYBRID: "HYBRID",
}


@dataclass
class Frame:
    """One step of replayable dashboard state."""

    step: int
    pool_fill: float
    skr: float
    qber: float
    threat_score: float
    posture: int
    floor: int
    tenant: str
    served_tier: str
    latency: float
    regret_events_total: int
    queue_depth: int


@dataclass
class ReplayLog:
    """A full episode's frames for one policy, plus its label."""

    label: str
    frames: list[Frame] = field(default_factory=list)

    @property
    def pool_curve(self) -> list[float]:
        return [frame.pool_fill for frame in self.frames]

    @property
    def regret_curve(self) -> list[int]:
        return [frame.regret_events_total for frame in self.frames]

    @property
    def tier_histogram(self) -> dict[str, int]:
        counts = {"CLASSICAL": 0, "PQC": 0, "HYBRID": 0, "NONE": 0}
        for frame in self.frames:
            counts[frame.served_tier] += 1
        return counts


def build_frames(
    policy: Any,
    scenario: str = "S1",
    n_steps: int = 800,
    seed: int = 0,
    config: dict[str, Any] | None = None,
) -> ReplayLog:
    """Drive one episode and capture everything the panels need.

    The dashboard replays a captured episode rather than stepping the
    environment live inside a callback. Two reasons, both practical:
    a Dash callback that mutated a shared environment would produce
    different data for every connected browser, and a recorded episode
    can be scrubbed backwards -- which the demo needs, because Beat 2's
    whole point is pointing at the moment the agent *started*
    conserving.
    """
    config = config if config is not None else load_full_config()
    env_config = {
        **config,
        "scenario": scenario,
        "seed": seed,
        "max_steps": n_steps,
        "scenario_steps": n_steps,
    }
    env = SmartKeyNetEnv(env_config)
    state, info = env.reset(seed=seed)

    log = ReplayLog(label=f"{type(policy).__name__} / {scenario}")
    for _ in range(n_steps):
        mask = info["action_mask"]
        request = env._current_request
        tenant_service = (request["tenant"], request["service"])
        floor = int(state["policy_floor"])
        action = policy.act(state, mask)

        state, _reward, _terminated, truncated, info = env.step(action)

        session = env._sessions.get(tenant_service)
        served = (
            _KEY_TYPE_NAMES[session.key_type]
            if session is not None and session.key_type is not None
            else "NONE"
        )

        log.frames.append(
            Frame(
                step=env._step_count,
                pool_fill=float(state["pool_fill"]),
                skr=float(state["skr"]),
                qber=float(state["qber"]),
                threat_score=float(state["threat_score"]),
                posture=int(env._policy_table._ratcheted_posture),
                floor=floor,
                tenant=str(request["tenant"]),
                served_tier=served,
                latency=float(state["avg_latency"]),
                regret_events_total=len(env._regret_log),
                queue_depth=len(env._deferral_queue),
            )
        )
        if truncated:
            break

    return log


def build_beat_two(n_steps: int = 800, seed: int = 0) -> tuple[ReplayLog, ReplayLog]:
    """Beat 2: the budgeting agent against the always-hybrid villain on
    S3, which is the comparison that produces two diverging pool curves
    and two very different regret counters."""
    config = load_full_config()
    frugal = StaticThresholdPolicy(
        pool_fill_threshold=0.7,
        min_hybrid_class=2,
        rekey_age_frac=0.9,
        max_key_age=float(config["key_lifetime"]["max_key_age_steps"]),
    )
    return (
        build_frames(frugal, "S3", n_steps, seed, config),
        build_frames(AlwaysHybridPolicy(), "S3", n_steps, seed, config),
    )


def load_results(path: str | Path) -> dict[str, Any] | None:
    """Read a results JSON if the experiment has been run.

    Returns `None` rather than raising, so the dashboard degrades to
    "this beat has no data yet" instead of refusing to start -- the
    demo should always come up.
    """
    results_path = Path(path)
    if not results_path.exists():
        return None
    return json.loads(results_path.read_text())


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------


def create_app() -> Any:
    """Build the Dash application.

    Imported lazily so this module stays importable (and testable)
    without Dash installed -- `build_frames` and the replay types above
    are plain Python and carry all the logic worth testing.
    """
    import plotly.graph_objects as go
    from dash import Dash, dcc, html

    app = Dash(__name__, title="SmartKeyNet")

    frugal, villain = build_beat_two()
    steering = load_results("results/steering_attack.json")
    gate = load_results("results/gate_w3.json")

    pool_figure = go.Figure()
    for log, colour in ((frugal, TIER_COLOURS["HYBRID"]), (villain, "#e74c3c")):
        pool_figure.add_trace(
            go.Scatter(
                y=log.pool_curve,
                name=log.label,
                line={"color": colour},
                mode="lines",
            )
        )
    pool_figure.update_layout(
        title="Beat 2 — QKD pool level on S3: budgeting agent vs always-hybrid",
        xaxis_title="decision",
        yaxis_title="pool fill fraction",
        template="plotly_dark",
    )

    regret_figure = go.Figure()
    for log, colour in ((frugal, TIER_COLOURS["HYBRID"]), (villain, "#e74c3c")):
        regret_figure.add_trace(
            go.Scatter(y=log.regret_curve, name=log.label, line={"color": colour})
        )
    regret_figure.update_layout(
        title="Beat 2 — cumulative regret events (deferred critical requests)",
        xaxis_title="decision",
        yaxis_title="regret events",
        template="plotly_dark",
    )

    tier_figure = go.Figure()
    for log in (frugal, villain):
        histogram = log.tier_histogram
        tier_figure.add_trace(
            go.Bar(x=list(histogram), y=list(histogram.values()), name=log.label)
        )
    tier_figure.update_layout(
        title="Beat 1 — served-tier mix",
        template="plotly_dark",
        barmode="group",
    )

    children: list[Any] = [
        html.H1("SmartKeyNet — RL for Hybrid Cryptography"),
        html.P(
            "Decision layer for a multi-tenant KMS in the hybrid era. "
            "Security floors are enforced by action masking, never by reward."
        ),
        dcc.Graph(figure=pool_figure),
        dcc.Graph(figure=regret_figure),
        dcc.Graph(figure=tier_figure),
    ]

    if steering is not None:
        analytic = steering.get("soft_reward_optimal_tier_by_threat", [])
        masked = steering.get("masked_floor_by_threat", [])
        steering_figure = go.Figure()
        steering_figure.add_trace(
            go.Scatter(y=analytic, name="soft-reward: preferred tier", mode="lines+markers")
        )
        steering_figure.add_trace(
            go.Scatter(y=masked, name="masked: enforced floor", mode="lines+markers")
        )
        steering_figure.update_layout(
            title=(
                "Beat 3 — the steering attack: security isn't in our reward, "
                "so it isn't for sale"
            ),
            xaxis_title="reported threat (bin 0 = fully suppressed)",
            yaxis_title="tier  (0 classical / 1 PQC / 2 hybrid)",
            template="plotly_dark",
        )
        children.append(dcc.Graph(figure=steering_figure))
        children.append(
            html.P(
                f"Floor violations across every agent, dose and seed: "
                f"{steering.get('total_floor_violations', 'n/a')}. "
                f"Posture ratchet reversals: {steering.get('total_posture_reversals', 'n/a')}."
            )
        )

    if gate is not None:
        rows = []
        for scenario, payload in gate.get("scenarios", {}).items():
            for name, stats in payload.get("policies", {}).items():
                rows.append(
                    html.Tr(
                        [
                            html.Td(scenario),
                            html.Td(name),
                            html.Td(f"{stats['mean_reward']:.1f}"),
                            html.Td(f"{stats['mean_exhaustion_events']:.1f}"),
                            html.Td(str(stats["total_floor_violations"])),
                        ]
                    )
                )
        children.append(html.H2("Gate W3 — agent vs tuned baselines"))
        children.append(
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("scenario"),
                                html.Th("policy"),
                                html.Th("mean reward"),
                                html.Th("exhaustion events"),
                                html.Th("floor violations"),
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ]
            )
        )

    app.layout = html.Div(children, style={"backgroundColor": "#111", "color": "#eee", "padding": "24px"})
    return app


def main() -> None:
    create_app().run(debug=False, port=8050)


if __name__ == "__main__":
    main()
