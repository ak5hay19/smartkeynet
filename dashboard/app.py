"""
dashboard/app.py

Live demo dashboard (PLAN2 §7, Dashboard v2's seven panels). Owned by
Person D (split.md §1).

Two renderings of the same payload, which is assembled entirely from
real runs by `dashboard/data.py`:

  * `build_app()` -- a Plotly Dash app for the live demo (PLAN.md tech
    stack: "Plotly Dash or a small React dashboard"). Returns the app
    object rather than calling `.run()`, so importing this module never
    starts a server.
  * `render_html()` -- a self-contained HTML export mirroring
    `mock.html`'s seven-panel layout with the real numbers in place.
    This is what goes in the report and the PR, because a static file
    can be opened and checked by anyone without running the stack.

`mock.html` is **layout truth and nothing else**. PLAN2's header states
that every number, chart, threat score and "why this decision" sentence
in it was hand-authored for layout demonstration. Nothing from it is
copied here; where a real artefact has not been generated, the panel
renders an explicit "not yet run" note rather than a plausible-looking
placeholder.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from dashboard.data import build_dashboard_payload

_PANEL_TITLES = {
    "p1_threat_input": ("01", "Threat signal input", "Upstream - feeds every panel"),
    "p2_living_system": ("02", "The living system", "Beat 1"),
    "p3_explain_decision": ("03", "Why this key, not another", "Explainability"),
    "p4_budgeting_brain": ("04", "The budgeting brain", "Beat 2"),
    "p5_steering_attack": ("05", "The steering attack", "Beat 3 - never cut"),
    "p6_migration_wave": ("06", "The migration wave", "Beat 4"),
    "p7_results": ("07", "Results across S1-S4, S6", "Closing table"),
}


def load_payload(path: str | Path | None = None, **kwargs: Any) -> dict[str, Any]:
    """Load a previously exported payload, or build a fresh one."""
    if path is not None and Path(path).exists():
        return json.loads(Path(path).read_text())
    return build_dashboard_payload(**kwargs)


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------


def build_app(logs: dict[str, Any] | None = None) -> Any:
    """Construct the Dash app object.

    Returns the app object (rather than calling `.run()`) so tests can
    import this module without starting a server -- the contract this
    function has had since it was a stub.

    `logs` is the dashboard payload (`dashboard/data.build_dashboard_payload`).
    Passing one in is how the demo replays a recorded run instead of
    generating a fresh one on startup.
    """
    from dash import Dash, dcc, html as dhtml

    payload = logs if logs is not None else build_dashboard_payload()
    app = Dash(__name__, title="SmartKeyNet - Dashboard")

    sections = []
    for key, (number, title, kicker) in _PANEL_TITLES.items():
        panel = payload.get(key, {})
        sections.append(
            dhtml.Section(
                [
                    dhtml.Div(kicker, className="kicker"),
                    dhtml.H2(f"{number} - {title}"),
                    _dash_panel_body(dhtml, dcc, key, panel),
                ],
                id=key,
                className="panel",
            )
        )

    app.layout = dhtml.Div(
        [
            dhtml.H1("SmartKeyNet"),
            dhtml.P(payload.get("generated_from", ""), className="provenance"),
            dhtml.Pre(json.dumps(payload.get("config_summary", {}), indent=2)),
            *sections,
        ],
        className="dashboard",
    )
    return app


def _dash_panel_body(dhtml: Any, dcc: Any, key: str, panel: dict[str, Any]) -> Any:
    if not panel.get("available", False):
        return dhtml.Div(
            f"Not yet run - {panel.get('reason', 'no data')}", className="unavailable"
        )
    if key == "p5_steering_attack":
        return dcc.Graph(figure=_dose_response_figure(panel))
    if key == "p2_living_system":
        return dcc.Graph(figure=_pool_figure(panel))
    return dhtml.Pre(json.dumps(panel, indent=2, default=str))


def _dose_response_figure(panel: dict[str, Any]) -> Any:
    import plotly.graph_objects as go

    figure = go.Figure()
    for name, runs in panel["policies"].items():
        figure.add_trace(
            go.Scatter(
                x=[run["dose"] for run in runs],
                y=[run["below_class_floor_share"] for run in runs],
                mode="lines+markers",
                name=name,
            )
        )
    figure.update_layout(
        title="Dose-response: share of key establishments below the sensitivity-class floor",
        xaxis_title="attack strength (dose)",
        yaxis_title="share below class floor",
        yaxis_tickformat=".0%",
    )
    return figure


def _pool_figure(panel: dict[str, Any]) -> Any:
    import plotly.graph_objects as go

    series = panel.get("pool_series", [])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[point["step"] for point in series],
            y=[point["fill_fraction"] for point in series],
            mode="lines",
            name="QKD pool fill",
        )
    )
    figure.update_layout(
        title="QKD pool level", xaxis_title="step", yaxis_title="fill fraction"
    )
    return figure


# ---------------------------------------------------------------------------
# Static HTML export
# ---------------------------------------------------------------------------

_CSS = """
:root{--bg:#0e1116;--panel:#161b22;--line:#2b323c;--fg:#e6edf3;--dim:#8b98a5;
--t0:#e0575b;--t1:#4a9eda;--t2:#37c98b;--warn:#e3b341}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:28px 32px;border-bottom:1px solid var(--line)}
h1{margin:0 0 6px;font-size:22px;letter-spacing:.02em}
.provenance{color:var(--t2);margin:0;font-size:13px}
main{padding:0 32px 64px;max-width:1180px}
section{border:1px solid var(--line);background:var(--panel);border-radius:10px;
padding:22px 24px;margin:26px 0}
.kicker{color:var(--dim);text-transform:uppercase;letter-spacing:.09em;font-size:11px}
h2{margin:6px 0 4px;font-size:18px}
.lede{color:var(--dim);margin:0 0 18px;max-width:80ch}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left}
th{color:var(--dim);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.tier{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600}
.SERVE_CLASSICAL{background:rgba(224,87,91,.16);color:var(--t0)}
.SERVE_PQC{background:rgba(74,158,218,.16);color:var(--t1)}
.SERVE_HYBRID{background:rgba(55,201,139,.16);color:var(--t2)}
.step{border-left:2px solid var(--line);padding:8px 0 8px 16px;margin:0 0 4px 6px}
.step b{display:block;font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:.05em}
.unavailable{color:var(--warn);border:1px dashed var(--warn);padding:12px;border-radius:6px}
.note{color:var(--dim);font-size:12px;border-left:2px solid var(--t2);padding-left:12px;margin:12px 0}
.rule{color:var(--warn);font-size:12px;border-left:2px solid var(--warn);padding-left:12px;margin:12px 0}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;min-width:120px}
.bar>i{display:block;height:100%;background:var(--t1)}
code{background:#0b0e13;padding:1px 5px;border-radius:4px;font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.stat{border:1px solid var(--line);border-radius:8px;padding:12px}
.stat .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.stat .v{font-size:20px;font-variant-numeric:tabular-nums;margin-top:2px}
.struct{color:var(--t2)}
"""


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _stat(key: str, value: Any, cls: str = "") -> str:
    return f'<div class="stat"><div class="k">{_esc(key)}</div><div class="v {cls}">{_esc(value)}</div></div>'


def render_html(payload: dict[str, Any]) -> str:
    """Self-contained dashboard HTML, real numbers only."""
    body = [
        "<header>",
        "<h1>SmartKeyNet - live dashboard</h1>",
        f'<p class="provenance">{_esc(payload.get("generated_from", ""))}</p>',
        "</header><main>",
        _render_config(payload.get("config_summary", {})),
    ]
    renderers = {
        "p1_threat_input": _render_threat_input,
        "p2_living_system": _render_living_system,
        "p3_explain_decision": _render_explain,
        "p4_budgeting_brain": _render_budgeting,
        "p5_steering_attack": _render_steering,
        "p6_migration_wave": _render_migration,
        "p7_results": _render_results,
    }
    for key, (number, title, kicker) in _PANEL_TITLES.items():
        panel = payload.get(key, {})
        body.append(f'<section id="{key}"><div class="kicker">{_esc(kicker)}</div>')
        body.append(f"<h2>{number} &mdash; {_esc(title)}</h2>")
        if not panel.get("available", False):
            body.append(
                f'<div class="unavailable">Not yet run &mdash; {_esc(panel.get("reason", "no data"))}</div>'
            )
        else:
            body.append(renderers[key](panel))
        body.append("</section>")
    body.append("</main>")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>SmartKeyNet - Dashboard</title>"
        f"<style>{_CSS}</style></head><body>{''.join(body)}</body></html>"
    )


def _render_config(summary: dict[str, Any]) -> str:
    cells = "".join(_stat(k, v) for k, v in summary.items())
    return f'<section><div class="kicker">Run configuration</div><div class="grid">{cells}</div></section>'


def _render_threat_input(panel: dict[str, Any]) -> str:
    modes = "".join(
        f"<tr><td>{_esc(m['label'])}</td><td>{_esc(m['detail'])}</td>"
        f"<td>{'active' if m['active'] else ('built' if m['implemented'] else 'not built')}</td></tr>"
        for m in panel["source_modes"]
    )
    stages = "".join(
        f"<tr><td>{_esc(s['label'])}</td><td><code>{_esc(s['mockup_label'])}</code></td>"
        f"<td>{_esc(s['detail'])}</td></tr>"
        for s in panel["pipeline_stages"]
    )
    invariants = "".join(f'<div class="rule">{_esc(i)}</div>' for i in panel["invariants"])
    return (
        '<p class="lede">Wherever the threat score comes from, it goes through the same '
        "feature-extraction path (Hard Rule 11).</p>"
        f'<div class="grid">{_stat("threat_score", round(panel["current"]["threat_score"], 4))}'
        f'{_stat("posture", panel["current"]["posture"])}</div>'
        "<table><tr><th>source</th><th>detail</th><th>status</th></tr>" + modes + "</table>"
        "<table><tr><th>stage (as built)</th><th>mockup label</th><th>what it is</th></tr>"
        + stages
        + "</table>"
        f'<div class="note">{_esc(panel["divergence_note"])}</div>'
        + invariants
    )


def _render_living_system(panel: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{_esc(d['tenant'])}/{_esc(d['service'])}</td><td>{_esc(d['sensitivity_class'])}</td>"
        f"<td><span class='tier {_esc(d['policy_floor'])}'>{_esc(d['policy_floor'])}</span></td>"
        f"<td><span class='tier {_esc(d['delivered_tier'])}'>{_esc(d['delivered_tier'])}</span></td>"
        f"<td>{_esc(d['reason'])}</td></tr>"
        for d in panel["recent_decisions"][-12:]
    )
    pool = panel.get("pool", {})
    fill = pool.get("fill_fraction", 0.0)
    qber = pool.get("qber", 0.0)
    stats = "".join(
        [
            _stat("graph nodes", len(panel["graph"]["nodes"])),
            _stat("graph edges", len(panel["graph"]["edges"])),
            _stat("pool fill", f"{fill:.1%}"),
            _stat("QBER", f"{qber:.4f}"),
            _stat("rolling p99 latency", round(panel["p99_latency"], 3)),
            _stat("deferral queue", int(pool.get("deferral_depth", 0))),
        ]
    )
    return (
        '<p class="lede">Every tenant request flows through the graph. Edge colour is the key '
        "tier actually served.</p>"
        f'<div class="grid">{stats}</div>'
        "<table><tr><th>tenant/service</th><th>class</th><th>floor</th><th>served</th><th>why</th></tr>"
        + rows
        + "</table>"
    )


def _render_explain(panel: dict[str, Any]) -> str:
    trace = panel["trace"]
    steps = "".join(
        f'<div class="step"><b>{s["index"]}. {_esc(s["title"])}</b>{_esc(s["summary"])}</div>'
        for s in trace["steps"]
    )
    grid_rows = "".join(
        "<tr><td>"
        + _esc(row["sensitivity_class"])
        + "</td>"
        + "".join(
            f'<td><span class="tier {_esc(v)}">{_esc(v)}</span></td>' for v in row["floors"].values()
        )
        + "</tr>"
        for row in panel["floor_table"]
    )
    illegal = "".join(
        f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in trace["illegal_reasons"].items()
    ) or "<tr><td colspan=2>every action was legal</td></tr>"
    return (
        '<p class="lede">There is no narrated reasoning here &mdash; the explanation <em>is</em> '
        "the pipeline.</p>"
        + steps
        + "<table><tr><th>class</th><th>CALM</th><th>ELEVATED</th><th>HIGH</th></tr>"
        + grid_rows
        + "</table>"
        + "<table><tr><th>masked action</th><th>reason</th></tr>"
        + illegal
        + "</table>"
        + f'<div class="rule">{_esc(panel["hard_rule_10"])}</div>'
    )


def _render_budgeting(panel: dict[str, Any]) -> str:
    metrics = ["p99_latency", "pool_exhaustion_events", "regret_events",
               "deferred_critical_steps", "forced_rekey_ratio", "floor_violations"]
    header = "".join(f"<th>{_esc(m)}</th>" for m in metrics)
    rows = ""
    for name, values in panel["arms"].items():
        cells = "".join(f'<td class="num">{values[m]:.3f}</td>' for m in metrics)
        rows += f"<tr><td>{_esc(name)}</td>{cells}</tr>"
    return (
        f'<p class="lede">Scenario {_esc(panel["scenario"])} &mdash; QKD degradation, two policies, '
        "same seed.</p>"
        f"<table><tr><th>policy</th>{header}</tr>{rows}</table>"
    )


def _render_steering(panel: dict[str, Any]) -> str:
    blocks = ""
    for name, runs in panel["policies"].items():
        rows = "".join(
            f'<tr><td class="num">{r["dose"]:.2f}</td>'
            f'<td class="num">{r["suppression"]:.1%}</td>'
            f'<td class="num">{r["decisions"]}</td>'
            f'<td class="num">{r["below_class_floor_share"]:.1%}</td>'
            f'<td class="num">{r["below_escalated_floor_share"]:.1%}</td>'
            + "".join(
                f'<td class="num">{v}</td>' for v in r["tier_counts"].values()
            )
            + "</tr>"
            for r in runs
        )
        blocks += (
            f"<h3>{_esc(name)}</h3><table><tr><th>dose</th><th>suppressed</th>"
            "<th>establishments</th><th>below class floor</th><th>below escalated floor</th>"
            "<th>CLASSICAL</th><th>PQC</th><th>HYBRID</th></tr>" + rows + "</table>"
        )
    return (
        f'<p class="lede">{_esc(panel["caption"])}</p>'
        + blocks
        + f'<div class="note">{_esc(panel["metric_note"])}</div>'
    )


def _render_migration(panel: dict[str, Any]) -> str:
    rows = "".join(
        f'<tr><td class="num">{p["step"]}</td><td>{_esc(p["cohort"])}</td>'
        f'<td><span class="tier {_esc(p["new_floor"])}">{_esc(p["new_floor"])}</span></td>'
        f'<td>{"yes" if p["pqc_capable"] else "-"}</td><td>{_esc(p["label"])}</td></tr>'
        for p in panel["phases"]
    )
    return (
        '<p class="lede">A scripted, exogenous compliance timeline &mdash; never chosen by the '
        "agent.</p>"
        "<table><tr><th>step</th><th>cohort</th><th>new floor</th><th>PQC upgrade</th>"
        "<th>phase</th></tr>" + rows + "</table>"
        f'<div class="rule">{_esc(panel["held_out_note"])}</div>'
        f'<div class="rule">{_esc(panel["exogenous_note"])}</div>'
    )


def _render_results(panel: dict[str, Any]) -> str:
    metrics = ["p99_latency", "pool_exhaustion_events", "regret_events",
               "forced_rekey_ratio", "floor_violations"]
    out = ""
    for scenario in panel["scenarios"]:
        header = "".join(f"<th>{_esc(m)}</th>" for m in metrics)
        rows = ""
        for policy in panel["policies"]:
            cell = panel["cells"].get(f"{scenario}|{policy}")
            if cell is None:
                continue
            cells = ""
            for metric in metrics:
                mean = cell[metric]["mean"]
                stdev = cell[metric]["stdev"]
                if metric == "floor_violations" and mean == 0.0:
                    cells += '<td class="num struct">0 &mdash; structural</td>'
                else:
                    cells += f'<td class="num">{mean:.2f} &plusmn; {stdev:.2f}</td>'
            rows += f"<tr><td>{_esc(policy)}</td>{cells}</tr>"
        out += f"<h3>{_esc(scenario)}</h3><table><tr><th>policy</th>{header}</tr>{rows}</table>"
    return (
        '<p class="lede">One number per policy, same scenarios, same seeds.</p>'
        + out
        + f'<div class="rule">{_esc(panel["structural_note"])}</div>'
    )


def export_html(
    out_path: str | Path = "dashboard/index.html",
    payload: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Path:
    """Build the payload and write the static dashboard."""
    payload = payload if payload is not None else build_dashboard_payload(**kwargs)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(payload), encoding="utf-8")
    return path


def main() -> None:
    payload = build_dashboard_payload()
    data_path = Path("results/dashboard_payload.json")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(payload, indent=2, default=str))
    html_path = export_html(payload=payload)
    print(f"wrote {data_path}")
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
