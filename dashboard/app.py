"""
dashboard/app.py

Live demo dashboard (PLAN.md §6, Demo Beats 1-4). Owned by Person D
(split.md §1).

Panels: tenant service graph (edges coloured by key type served), QKD
pool gauge, threat forecast strip, live regret counter, latency chart.

Implementation is Plotly Dash (PLAN.md tech stack lists it as
negotiable; Dash keeps everything in Python, so the dashboard reads the
same event log the experiments write rather than a re-implementation of the
simulator in JavaScript that could silently drift).

**This module imports nothing from `env/` or `agents/`, by design.**
SMARTKEYNET_BUILD_SPEC.md §S13 requires the dashboard to read
`events.jsonl.gz` and "never reach into env internals" -- so that it can
replay any recorded run, works offline in the viva, and cannot slow training
down. Until 2026-08-19 this file constructed a live `SmartKeyNetEnv` and read
five private attributes off it, which broke all three of those properties.
Recording lives in `experiments/record_demo.py`; this file only renders.
`tests/test_api_and_dashboard.py` AST-scans it to keep the boundary honest.

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
from pathlib import Path
from typing import Any

from dashboard.replay import ReplayEpisode, load_episode

TIER_COLOURS: dict[str, str] = {
    "CLASSICAL": "#9e9e9e",  # grey
    "PQC": "#f5a623",  # amber
    "HYBRID": "#2ecc71",  # green
    "NONE": "#37474f",
}
"""PLAN.md §6 Beat 1: "edges flash coloured by key type served (grey
classical / amber PQC / green hybrid)"."""

_TIER_DISPLAY_ORDER: tuple[str, ...] = ("CLASSICAL", "PQC", "HYBRID", "REUSE")
"""Display order for the tier histogram. Names come from the event log's
`tier_served` field via `dashboard.replay`, not from `env.contracts.KeyType`
-- importing the env's enums here is the first step back to importing the
env, which §S13 forbids."""


def beat_two_logs(
    log_dir: str | Path = "results/demo_logs",
) -> tuple[ReplayEpisode, ReplayEpisode] | None:
    """Load Beat 2's two recorded episodes, or None if they were never
    recorded.

    Returning None rather than recording on demand is deliberate: rendering
    must not be able to start a simulation, or the decoupling §S13 asks for is
    only a convention. Record with:

        .venv/bin/python -m experiments.record_demo
    """
    directory = Path(log_dir)
    frugal_path = directory / "beat2_frugal.jsonl.gz"
    villain_path = directory / "beat2_villain.jsonl.gz"
    if not frugal_path.exists() or not villain_path.exists():
        return None
    return (
        load_episode(frugal_path, label="tuned threshold / S3"),
        load_episode(villain_path, label="always-hybrid / S3"),
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

    beat_two = beat_two_logs()
    if beat_two is None:
        raise SystemExit(
            "no demo logs found -- record them first:\n"
            "    .venv/bin/python -m experiments.record_demo"
        )
    frugal, villain = beat_two
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
        tier_figure.add_trace(go.Bar(x=list(histogram), y=list(histogram.values()), name=log.label))
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
                "Beat 3 — the steering attack: security isn't in our reward, so it isn't for sale"
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

    app.layout = html.Div(
        children, style={"backgroundColor": "#111", "color": "#eee", "padding": "24px"}
    )
    return app


def main() -> None:
    create_app().run(debug=False, port=8050)


if __name__ == "__main__":
    main()
