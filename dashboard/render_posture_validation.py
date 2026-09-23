"""
dashboard/render_posture_validation.py

Renders the 2026-09-23 posture-variance/floor-monotonicity validation
session's real result (SESSION_LOG.md/PROGRESS.md's matching entry,
commit `83f23ba`) as a self-contained static HTML panel, in the same
visual language as the other `dashboard/render_*.py` panels (dark
theme, tier-color legend conventions, honesty callouts).

This module renders an ALREADY-REAL, already-committed result -- it
does not recompute or reinterpret anything itself. `dashboard/
render_posture_validation_demo.py` is the driver that reproduces the
real validation run (same three real sources, same seeds/offsets as
the original validation session) and saves its output to
`dashboard/samples/posture_validation_data.json`; this module turns
that data into HTML.

**Honesty scope, carried over verbatim from this session's own
instruction**: this panel shows the real posture-bucket counts, the
real zero-floor-violation result, the real floor-by-class-per-posture
breakdown, and the real OVERALL action distribution -- all of which
the validation session actually computed and reported. A per-posture
breakdown of WHICH action was chosen WITHIN each posture bucket was
NOT computed or reported by that session (only the aggregate/
per-source action counts were) -- this renderer does not invent or
approximate that breakdown; it states plainly that it isn't available
rather than filling the gap.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

_CSS = """
:root{
  --bg:#0A0E13; --panel:#111820; --panel-2:#151D27; --line:#212C39; --line-soft:#171F29;
  --text:#E9EEF4; --text-dim:#8FA0B3; --text-faint:#4C5A6B;
  --classical:#8B95A5; --pqc:#E8A33D; --hybrid:#33D687; --quantum:#6E7EFF; --danger:#FF5C6C;
  --calm:#33D687; --elevated:#E8A33D; --high:#FF5C6C;
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
.wrap{max-width:820px;margin:0 auto;}
.beat-head{margin-bottom:22px;}
.beat-eyebrow{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--quantum);}
.beat-title{font-family:var(--disp);font-weight:700;font-size:22px;margin-top:4px;}
.beat-desc{color:var(--text-dim);font-size:13px;margin-top:6px;line-height:1.5;}

.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:18px 18px 16px;margin-bottom:16px;}
.card-label{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--text-faint);display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;}

.bar-row{display:flex;align-items:center;gap:10px;margin-bottom:10px;}
.bar-label{width:78px;flex-shrink:0;font-family:var(--mono);font-size:11.5px;color:var(--text-dim);}
.bar-track{flex:1;height:20px;background:var(--panel-2);border-radius:5px;overflow:hidden;}
.bar-fill{height:100%;border-radius:5px 0 0 5px;}
.bar-count{width:46px;flex-shrink:0;text-align:right;font-family:var(--mono);font-size:12px;color:var(--text);}

.headline{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:14px;}
.stat{background:var(--panel-2);border:1px solid var(--line);border-radius:var(--radius-sm);padding:10px 14px;flex:1;min-width:140px;}
.stat-value{font-family:var(--mono);font-size:20px;font-weight:700;}
.stat-value.ok{color:var(--hybrid);}
.stat-label{font-size:11px;color:var(--text-dim);margin-top:2px;}

.callout{margin-top:0;font-size:11.5px;color:var(--text-dim);border-left:2px solid var(--quantum);padding:8px 11px;background:var(--panel-2);border-radius:var(--radius-sm);line-height:1.55;}
.callout.ok{border-left-color:var(--hybrid);}
.callout.note{border-left-color:var(--elevated);}
.callout b{color:var(--text);}

table.floor-table{width:100%;border-collapse:collapse;font-size:12px;}
table.floor-table th,table.floor-table td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line-soft);}
table.floor-table th{color:var(--text-faint);font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.08em;font-weight:600;}
table.floor-table td{color:var(--text-dim);font-family:var(--mono);}
table.floor-table td.posture{color:var(--text);}

.legend-row{display:flex;gap:16px;margin-top:12px;flex-wrap:wrap;}
.legend-item{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--text-dim);}
.legend-swatch{width:10px;height:10px;border-radius:3px;display:inline-block;}

.source-list{font-size:12px;color:var(--text-dim);line-height:1.7;}
.source-list b{color:var(--text);}

.provenance{margin-top:16px;font-family:var(--mono);font-size:10px;color:var(--text-faint);line-height:1.6;}
"""

_POSTURE_COLOR = {"CALM": "var(--calm)", "ELEVATED": "var(--elevated)", "HIGH": "var(--high)"}
_POSTURE_ORDER = ("CALM", "ELEVATED", "HIGH")


def _e(value: object) -> str:
    return html.escape(str(value))


@dataclass(frozen=True)
class ThreatScoreExample:
    """One real (threat_score, sensitivity_class, floor, posture)
    example -- the anecdote this whole validation session set out to
    check, carried through to the panel so the motivating question is
    visible next to its answer."""

    threat_score: float
    sensitivity_class: str
    floor: str
    posture: str


@dataclass(frozen=True)
class FloorsByClass:
    """The real floor(s) observed for one (posture, sensitivity_class)
    pair across the validation sample -- usually one value; a set with
    more than one member means the ratchet caught this class at more
    than one floor level within the same posture bucket across the
    sample (still valid, not a monotonicity violation by itself)."""

    posture: str
    sensitivity_class: str
    floors: tuple[str, ...]


@dataclass(frozen=True)
class SourceInfo:
    """One of the three real data sources this validation combined."""

    name: str
    n_decisions: int
    description: str


@dataclass(frozen=True)
class PostureValidationData:
    total_decisions: int
    posture_counts: dict[str, int]  # CALM/ELEVATED/HIGH -> count
    floor_violations: int
    floor_by_class: tuple[FloorsByClass, ...]
    action_counts: dict[str, int]  # overall action distribution only
    sources: tuple[SourceInfo, ...]
    elevated_transient_note: str
    examples: tuple[ThreatScoreExample, ...]


def _bar_chart(posture_counts: dict[str, int]) -> str:
    total = sum(posture_counts.values()) or 1
    rows = []
    for posture in _POSTURE_ORDER:
        count = posture_counts.get(posture, 0)
        pct = 100.0 * count / total
        rows.append(
            f'<div class="bar-row">'
            f'<span class="bar-label">{_e(posture)}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{_POSTURE_COLOR[posture]}"></div></div>'
            f'<span class="bar-count">{count}</span>'
            f"</div>"
        )
    return "".join(rows)


def render_posture_validation_html(
    data: PostureValidationData,
    *,
    title: str = "SmartKeyNet -- Posture-Variance & Floor-Monotonicity Validation",
) -> str:
    """Render `data` (real, already-computed) as a self-contained HTML
    page. Pure view -- no file I/O, no computation of new numbers."""
    bar_chart = _bar_chart(data.posture_counts)

    violations_ok = data.floor_violations == 0
    violations_class = "ok" if violations_ok else ""
    violations_text = (
        "0 -- no violations found" if violations_ok else f"{data.floor_violations} VIOLATION(S) FOUND"
    )

    example_rows = "\n".join(
        f'<tr><td>{ex.threat_score:.4f}</td><td class="posture">{_e(ex.posture)}</td>'
        f"<td>{_e(ex.sensitivity_class)}</td><td>{_e(ex.floor)}</td></tr>"
        for ex in data.examples
    )

    floor_rows = "\n".join(
        f'<tr><td class="posture">{_e(fbc.posture)}</td><td>{_e(fbc.sensitivity_class)}</td>'
        f'<td>{_e(", ".join(fbc.floors))}</td></tr>'
        for fbc in data.floor_by_class
    )

    action_items = "".join(
        f'<div class="legend-item"><span class="legend-swatch" style="background:var(--quantum)"></span>{_e(action)}: {count}</div>'
        for action, count in data.action_counts.items()
    )

    source_items = "".join(
        f"<div><b>{_e(s.name)}</b> ({s.n_decisions} real decisions) -- {_e(s.description)}</div>"
        for s in data.sources
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
    <div class="beat-eyebrow">Validation &middot; QA session, 2026-09-23</div>
    <div class="beat-title">Posture variance &amp; floor monotonicity, across real data</div>
    <div class="beat-desc">The upload demo showed a HIGHER threat_score producing a LOWER floor than a lower one -- expected if both land in the same posture bucket, since posture (CALM/ELEVATED/HIGH) is discrete, not the floor table's continuous input; sensitivity class does the differentiating work within a bucket. This panel shows the real, broad check of that claim: {data.total_decisions} real decisions across three real data sources, run through the same live forecaster &rarr; posture &rarr; floor &rarr; masking &rarr; DQN pipeline.</div>
  </div>

  <div class="card">
    <div class="card-label"><span>The motivating anecdote</span><span>real values, not illustrative</span></div>
    <table class="floor-table">
      <tr><th>threat_score</th><th>posture</th><th>sensitivity_class</th><th>floor</th></tr>
      {example_rows}
    </table>
    <div class="callout" style="margin-top:10px;">Both scores are comfortably inside CALM (<code>threat_score &lt; 0.25</code>, the RBF-softmax anchor midpoint) -- 0.0689 sits 0.18 below that boundary, 0.1376 sits 0.11 below it. Posture is identical; the floor difference is sensitivity class doing its job, not a bug.</div>
  </div>

  <div class="card">
    <div class="card-label"><span>Posture bucket distribution</span><span>{data.total_decisions} real decisions</span></div>
    {bar_chart}
    <div class="legend-row">
      <div class="legend-item"><span class="legend-swatch" style="background:var(--calm)"></span>CALM</div>
      <div class="legend-item"><span class="legend-swatch" style="background:var(--elevated)"></span>ELEVATED</div>
      <div class="legend-item"><span class="legend-swatch" style="background:var(--high)"></span>HIGH</div>
    </div>
  </div>

  <div class="headline">
    <div class="stat"><div class="stat-value {violations_class}">{violations_text}</div><div class="stat-label">floor-monotonicity violations (higher sensitivity class &rArr; floor never lower, holding posture constant)</div></div>
  </div>

  <div class="callout note">
    <b>Real, honest finding:</b> {_e(data.elevated_transient_note)}
  </div>

  <div class="card">
    <div class="card-label"><span>Floor by (posture, sensitivity class)</span><span>real observed values</span></div>
    <table class="floor-table">
      <tr><th>posture</th><th>sensitivity_class</th><th>floor(s) observed</th></tr>
      {floor_rows}
    </table>
  </div>

  <div class="card">
    <div class="card-label"><span>Action distribution</span><span>overall, {data.total_decisions} decisions</span></div>
    <div class="legend-row">{action_items}</div>
    <div class="callout" style="margin-top:10px;">This is the OVERALL distribution across all three sources combined. A per-posture-bucket breakdown of which action was chosen within each bucket was not computed by the validation session this panel reports -- it is not shown here rather than being approximated.</div>
  </div>

  <div class="card">
    <div class="card-label"><span>Real data sources</span><span>no fabrication</span></div>
    <div class="source-list">{source_items}</div>
  </div>

  <div class="provenance">Source: SESSION_LOG.md/PROGRESS.md's 2026-09-23 posture-variance validation entry (commit 83f23ba); regenerated by dashboard/render_posture_validation_demo.py, data in dashboard/samples/posture_validation_data.json.</div>
</main>
</body>
</html>
"""


def write_posture_validation_html(
    data: PostureValidationData,
    path: str | Path,
    *,
    title: str = "SmartKeyNet -- Posture-Variance & Floor-Monotonicity Validation",
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_posture_validation_html(data, title=title), encoding="utf-8")
    return out_path
