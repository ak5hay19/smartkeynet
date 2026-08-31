"""
dashboard/render_explain.py

Renders dashboard/explain.py's real `DecisionTrace` objects as a
self-contained static HTML page -- the Explain Decision panel's view
layer (PLAN2.md Section 7.3). Visual target: dashboard/mockups/
smartkeynet_dashboard_mockup_v2.html's "Explain Decision" tab (styling
only -- that file's example numbers are 100% fabricated per its own
header warning and are never read by this module).

Hard Rule 10 (central to this module): the renderer is a pure view
over `DecisionTrace`'s own fields. Every dynamic piece of text in the
output is either a `DecisionTrace` field verbatim (or a direct
str()/format of one), or a static structural label (the six step
titles, column/row headers) that names PLAN2.md Section 7.3's six-step
spec itself, not anything about this specific decision. This module
never computes a new value, never invents a reason, and never
generates a narrative -- see `tests/test_render_explain.py`'s Hard
Rule 10 check.

Design decision (recorded in SESSION_LOG.md): plain Python string
templating into one inline-CSS static HTML file. No server, no JS
framework, no build step, zero new dependencies (this deliberately
does not reach for `dash`/`plotly`, both already in requirements.txt
for the *eventual* full live dashboard (`dashboard/app.py`, still a
stub) -- disproportionate for rendering one already-computed trace
object into one static page). A `DecisionTrace` renders to one
self-contained `.html` file, openable directly in a browser, no
Python process needs to keep running.
"""

from __future__ import annotations

import html
from pathlib import Path

from dashboard.explain import DecisionTrace
from env.contracts import Action, SensitivityClass, ThreatPosture

_FLOOR_LABEL: dict[Action, tuple[str, str]] = {
    Action.SERVE_CLASSICAL: ("CL", "cl"),
    Action.SERVE_PQC: ("PQC", "pqc"),
    Action.SERVE_HYBRID: ("HY", "hy"),
}

_POSTURES: list[ThreatPosture] = [ThreatPosture.CALM, ThreatPosture.ELEVATED, ThreatPosture.HIGH]
_SENS_CLASSES: list[SensitivityClass] = [
    SensitivityClass.S0,
    SensitivityClass.S1,
    SensitivityClass.S2,
    SensitivityClass.S3,
]


def _e(value: object) -> str:
    """HTML-escape any dynamic trace value before inlining it."""
    return html.escape(str(value))


def _render_step1_signal(trace: DecisionTrace) -> str:
    return f"""
      <div class="trace-step">
        <div class="step-num">1</div>
        <div class="step-title">Threat signal ingested</div>
        <div class="step-card">
          <div class="signal-row">
            <div class="signal-score">{trace.threat_score:.4f}</div>
            <div class="signal-meta">threat_score (0-1)
              <span class="src">{_e(trace.threat_source)}</span>
            </div>
          </div>
        </div>
      </div>"""


def _render_step2_posture(trace: DecisionTrace) -> str:
    if trace.posture_probs is None:
        body = (
            '<div class="posture-off">No forecaster configured for this decision '
            f'-- posture pinned to <b>{_e(trace.resolved_posture.name)}</b>.</div>'
        )
    else:
        rows: list[str] = []
        for posture in _POSTURES:
            prob = trace.posture_probs.get(posture.name, 0.0)
            pct = prob * 100
            resolved_cls = " resolved" if posture == trace.resolved_posture else ""
            rows.append(
                f"""
            <div class="prob-row{resolved_cls}">
              <div class="prob-label">{_e(posture.name)}</div>
              <div class="prob-track"><div class="prob-fill" style="width:{pct:.2f}%"></div></div>
              <div class="prob-pct">{pct:.1f}%</div>
            </div>"""
            )
        body = f'<div class="prob-bars">{"".join(rows)}</div>'
    return f"""
      <div class="trace-step">
        <div class="step-num">2</div>
        <div class="step-title">Posture classified</div>
        <div class="step-card">{body}</div>
      </div>"""


def _render_step3_floor(trace: DecisionTrace) -> str:
    header_cells = "".join(f'<div class="fg-head">{_e(p.name.title())}</div>' for p in _POSTURES)
    header = f'<div class="fg-head"></div>{header_cells}'

    body_rows = [header]
    for sens_class in _SENS_CLASSES:
        row_cells = [f'<div class="fg-rowlabel">{_e(sens_class.name)}</div>']
        for posture in _POSTURES:
            action = trace.floor_table[(sens_class, posture)]
            label, css_cls = _FLOOR_LABEL[action]
            hit_cls = " hit" if (sens_class, posture) == trace.floor_cell else ""
            data_cell = f"{sens_class.name}-{posture.name}"
            row_cells.append(f'<div class="floor-cell {css_cls}{hit_cls}" data-cell="{data_cell}">{label}</div>')
        body_rows.append("".join(row_cells))
    grid = f'<div class="floor-grid">{"".join(body_rows)}</div>'

    cell_sens, cell_posture = trace.floor_cell
    result = f"""
          <div class="floor-result">
            <span class="k">{_e(cell_sens.name)} + {_e(cell_posture.name)}</span>
            <span class="v">floor = {_e(trace.floor.name)}</span>
          </div>"""

    return f"""
      <div class="trace-step">
        <div class="step-num">3</div>
        <div class="step-title">Policy floor lookup -- (sensitivity class &times; posture)</div>
        <div class="step-card">{grid}{result}</div>
      </div>"""


def _render_step4_mask(trace: DecisionTrace) -> str:
    items: list[str] = []
    for entry in trace.mask:
        legal_cls = " legal" if entry.legal else ""
        items.append(
            f"""
          <div class="mask-item{legal_cls}">
            <div class="a-name">{_e(entry.action.name)}</div>
            <div class="a-reason">{_e(entry.reason)}</div>
          </div>"""
        )
    return f"""
      <div class="trace-step">
        <div class="step-num">4</div>
        <div class="step-title">Action mask computed</div>
        <div class="step-card"><div class="mask-row">{"".join(items)}</div></div>
      </div>"""


def _render_step5_costs(trace: DecisionTrace) -> str:
    max_cost = max((c.latency + c.energy for c in trace.costs), default=0.0) or 1.0
    rows: list[str] = []
    for cost_entry in trace.costs:
        width = (cost_entry.latency + cost_entry.energy) / max_cost * 100
        chosen_cls = " chosen" if cost_entry.chosen else ""
        mark = " ✓" if cost_entry.chosen else ""
        rows.append(
            f"""
          <div class="cost-row{chosen_cls}">
            <div class="cost-name">{_e(cost_entry.action.name)}{mark}</div>
            <div class="cost-track"><div class="cost-fill" style="width:{width:.1f}%"></div></div>
            <div class="cost-nums">lat {cost_entry.latency:g} &middot; en {cost_entry.energy:g}</div>
          </div>"""
        )
    note = f'<div class="cost-note">{_e(trace.cost_note)}</div>' if trace.cost_note else ""
    return f"""
      <div class="trace-step">
        <div class="step-num">5</div>
        <div class="step-title">Cost comparison among legal actions</div>
        <div class="step-card"><div class="cost-rows">{"".join(rows)}</div>{note}</div>
      </div>"""


def _render_step6_final(trace: DecisionTrace) -> str:
    return f"""
      <div class="trace-step">
        <div class="step-num">6</div>
        <div class="step-title">Final decision</div>
        <div class="step-card">
          <div class="final-pick">
            <span class="final-chip">{_e(trace.chosen_action.name)}</span>
            <span class="final-text">{_e(trace.final_text)}</span>
          </div>
        </div>
      </div>"""


_CSS = """
:root{
  --bg:#0A0E13; --panel:#111820; --panel-2:#151D27; --line:#212C39; --line-soft:#171F29;
  --text:#E9EEF4; --text-dim:#8FA0B3; --text-faint:#4C5A6B;
  --classical:#8B95A5; --pqc:#E8A33D; --hybrid:#33D687; --quantum:#6E7EFF;
  --radius:12px; --radius-sm:7px;
  --mono:ui-monospace,SFMono-Regular,Consolas,'Courier New',monospace;
  --disp:-apple-system,'Segoe UI',Roboto,sans-serif;
}
*{box-sizing:border-box;}
html,body{margin:0;padding:0;}
body{
  background:
    radial-gradient(circle at 1px 1px, #16202C 1px, transparent 0) 0 0/28px 28px,
    var(--bg);
  color:var(--text);font-family:var(--disp);padding:28px 20px 60px;
}
.wrap{max-width:760px;margin:0 auto;}
.beat-head{margin-bottom:22px;}
.beat-eyebrow{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--quantum);}
.beat-title{font-family:var(--disp);font-weight:700;font-size:22px;margin-top:4px;}
.beat-desc{color:var(--text-dim);font-size:13px;margin-top:6px;line-height:1.5;}

.trace{position:relative;padding-left:42px;}
.trace::before{content:"";position:absolute;left:14px;top:6px;bottom:6px;width:2px;background:var(--line);}
.trace-step{position:relative;margin-bottom:26px;}
.trace-step:last-child{margin-bottom:0;}
.step-num{
  position:absolute;left:-42px;top:0;width:29px;height:29px;border-radius:50%;
  background:var(--panel);border:2px solid var(--quantum);display:flex;align-items:center;justify-content:center;
  font-family:var(--mono);font-size:12px;font-weight:700;color:var(--quantum);
}
.step-title{font-family:var(--disp);font-weight:600;font-size:14.5px;margin-bottom:10px;}
.step-card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px 17px;}

.signal-row{display:flex;align-items:center;gap:18px;flex-wrap:wrap;}
.signal-score{font-family:var(--mono);font-weight:700;font-size:30px;color:var(--quantum);}
.signal-meta{font-size:12px;color:var(--text-dim);}
.signal-meta .src{font-family:var(--mono);font-size:10.5px;color:var(--text-faint);display:block;margin-top:3px;}

.posture-off{font-size:12.5px;color:var(--text-dim);}

.prob-bars{display:flex;flex-direction:column;gap:8px;}
.prob-row{display:grid;grid-template-columns:82px 1fr 48px;align-items:center;gap:10px;}
.prob-label{font-family:var(--mono);font-size:10.5px;color:var(--text-dim);text-transform:uppercase;letter-spacing:.05em;}
.prob-track{height:9px;border-radius:999px;background:var(--panel-2);border:1px solid var(--line);overflow:hidden;}
.prob-fill{height:100%;border-radius:999px;background:var(--text-faint);}
.prob-row.resolved .prob-label{color:var(--text);font-weight:600;}
.prob-row.resolved .prob-fill{background:var(--quantum);}
.prob-pct{font-family:var(--mono);font-size:11px;color:var(--text-dim);text-align:right;}

.floor-grid{display:grid;grid-template-columns:56px repeat(3,1fr);gap:6px;margin-top:4px;}
.fg-head{font-family:var(--mono);font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--text-faint);text-align:center;padding:6px 2px;}
.fg-rowlabel{font-family:var(--mono);font-size:10.5px;color:var(--text-dim);display:flex;align-items:center;padding-left:2px;}
.floor-cell{
  font-family:var(--mono);font-size:11px;font-weight:600;text-align:center;padding:9px 4px;border-radius:6px;
  border:1px solid var(--line-soft);background:var(--panel-2);color:var(--text-faint);
}
.floor-cell.cl{color:var(--classical);}
.floor-cell.pqc{color:var(--pqc);}
.floor-cell.hy{color:var(--hybrid);}
.floor-cell.hit{border-color:var(--quantum);background:rgba(110,126,255,.14);box-shadow:0 0 0 1px var(--quantum) inset;}

.floor-result{margin-top:12px;display:flex;align-items:center;justify-content:space-between;background:var(--panel-2);border-radius:var(--radius-sm);padding:10px 13px;}
.floor-result .k{font-family:var(--mono);font-size:10.5px;color:var(--text-faint);text-transform:uppercase;letter-spacing:.06em;}
.floor-result .v{font-family:var(--mono);font-weight:700;font-size:14px;}

.mask-row{display:flex;gap:8px;flex-wrap:wrap;}
.mask-item{flex:1;min-width:118px;border-radius:var(--radius-sm);padding:11px 10px;text-align:center;border:1px solid var(--line-soft);background:var(--panel-2);}
.mask-item.legal{border-color:rgba(51,214,135,.32);background:rgba(51,214,135,.06);}
.mask-item .a-name{font-family:var(--mono);font-size:10.5px;font-weight:600;}
.mask-item.legal .a-name{color:var(--hybrid);}
.mask-item:not(.legal) .a-name{color:var(--text-faint);text-decoration:line-through;}
.mask-item .a-reason{font-size:10px;color:var(--text-faint);margin-top:5px;line-height:1.4;}
.mask-item.legal .a-reason{color:var(--hybrid);opacity:.8;}

.cost-rows{display:flex;flex-direction:column;gap:9px;}
.cost-row{display:grid;grid-template-columns:132px 1fr 100px;align-items:center;gap:10px;}
.cost-name{font-family:var(--mono);font-size:11px;color:var(--text-dim);}
.cost-track{height:10px;border-radius:999px;background:var(--panel-2);border:1px solid var(--line);overflow:hidden;}
.cost-fill{height:100%;border-radius:999px;background:var(--text-faint);}
.cost-row.chosen .cost-name{color:var(--text);font-weight:600;}
.cost-row.chosen .cost-fill{background:var(--hybrid);}
.cost-nums{font-family:var(--mono);font-size:10.5px;color:var(--text-faint);text-align:right;}
.cost-note{margin-top:10px;font-size:11.5px;color:var(--text-dim);}

.final-pick{display:flex;align-items:center;gap:14px;flex-wrap:wrap;}
.final-chip{font-family:var(--mono);font-weight:700;font-size:15px;letter-spacing:.02em;padding:9px 16px;border-radius:8px;background:rgba(110,126,255,.14);border:1px solid var(--quantum);color:var(--quantum);}
.final-text{flex:1;min-width:220px;font-size:12.5px;color:var(--text-dim);line-height:1.55;}
"""


def render_trace_html(trace: DecisionTrace, *, title: str = "SmartKeyNet -- Explain Decision") -> str:
    """Render one real `DecisionTrace` (from `dashboard/explain.py`) as
    a self-contained HTML page. Pure view: reads `trace`'s fields only,
    writes nothing back, computes no new value (Hard Rule 10)."""
    steps = "".join(
        [
            _render_step1_signal(trace),
            _render_step2_posture(trace),
            _render_step3_floor(trace),
            _render_step4_mask(trace),
            _render_step5_costs(trace),
            _render_step6_final(trace),
        ]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_e(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<main class="wrap">
  <div class="beat-head">
    <div class="beat-eyebrow">Explainability</div>
    <div class="beat-title">Why this key, not another</div>
    <div class="beat-desc">There's no narrated reasoning here -- the explanation <i>is</i> the pipeline. Signal &rarr; posture &rarr; floor lookup &rarr; action mask &rarr; cost comparison &rarr; pick. Every step below is a real, inspectable computation from this one decision.</div>
  </div>
  <div class="trace">{steps}</div>
</main>
</body>
</html>
"""


def write_trace_html(trace: DecisionTrace, path: str | Path, *, title: str = "SmartKeyNet -- Explain Decision") -> Path:
    """Render `trace` and write it to `path` (parent dirs created as
    needed). Returns the written `Path`."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_trace_html(trace, title=title), encoding="utf-8")
    return out_path
