"""
dashboard/render_dose_response.py

Renders the S5 steering-attack dose-response result -- V(pi) (the
below-floor service rate measured against TRUE posture, paper eq. 4)
vs. attack strength alpha, for both the masked DQN and the soft-reward
baseline -- as a self-contained static HTML page with an inline SVG
chart. Visual target: dashboard/mockups/smartkeynet_dashboard_mockup_v2
.html's Steering Attack tab (styling only -- that file's dose-response
curve is 100% fabricated per its own header and is never read here).

Same rendering philosophy as dashboard/render_explain.py: a pure
function over an explicit, real input object, self-contained HTML
(inline CSS + inline SVG, no JS framework, no charting library, no
server), zero new dependencies.

Hard Rule 7 (central to this module, by analogy with Hard Rule 10):
this renderer must show the real measured shape of both curves
honestly, including the masked agent's genuine alpha>=0.9 boundary --
it must never round a nonzero value to zero, never omit an alpha step,
and never present a flat-zero-everywhere version of the masked curve.
The one annotation this module adds (the "first nonzero alpha"
callout) is itself computed directly from the input series, never
hardcoded to a specific alpha -- so it can never quietly go stale or
misdescribe a different dataset than the one actually rendered.
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
.wrap{max-width:800px;margin:0 auto;}
.beat-head{margin-bottom:22px;}
.beat-eyebrow{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--quantum);}
.beat-title{font-family:var(--disp);font-weight:700;font-size:22px;margin-top:4px;}
.beat-desc{color:var(--text-dim);font-size:13px;margin-top:6px;line-height:1.5;}

.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:18px 18px 16px;}
.card-label{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--text-faint);display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;}

.dose-chart{width:100%;height:260px;display:block;}
.dose-axis{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px;color:var(--text-faint);margin-top:6px;}

.legend-row{display:flex;gap:18px;margin-top:12px;flex-wrap:wrap;}
.legend-item{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--text-dim);}
.legend-swatch{width:10px;height:10px;border-radius:3px;display:inline-block;}

.callout{margin-top:14px;font-size:11.5px;color:var(--text-dim);border-left:2px solid var(--quantum);padding:8px 11px;background:var(--panel-2);border-radius:var(--radius-sm);line-height:1.55;}
.callout.boundary{border-left-color:var(--danger);}
.callout b{color:var(--text);}

.provenance{margin-top:16px;font-family:var(--mono);font-size:10px;color:var(--text-faint);line-height:1.6;}
"""

_SERIES_STYLE = {
    "masked": {"color": "#33D687", "css_var": "--hybrid"},
    "soft_reward": {"color": "#FF5C6C", "css_var": "--danger"},
}


@dataclass(frozen=True)
class DoseResponsePoint:
    """One real (alpha, V(pi), spread) measurement -- `mean`/`std` are
    `MultiSeedAttackEvalResult.below_floor_rate_true_mean`/`_std` for
    one alpha, projected verbatim (never recomputed) by the caller."""

    alpha: float
    mean: float
    std: float


@dataclass(frozen=True)
class DoseResponseSeries:
    """One agent's full real dose-response curve. `series_key` selects
    this module's own color mapping (`masked` or `soft_reward`) --
    purely a styling choice, not a data value."""

    label: str
    series_key: str  # "masked" | "soft_reward"
    points: list[DoseResponsePoint]


def _e(value: object) -> str:
    return html.escape(str(value))


def _first_nonzero(points: list[DoseResponsePoint]) -> DoseResponsePoint | None:
    """The first point (in the given, assumed-alpha-ascending order)
    whose real mean is strictly greater than zero -- used only to
    generate the boundary annotation's alpha/value, never to decide
    whether to *show* a point (every point is always rendered)."""
    for point in points:
        if point.mean > 0.0:
            return point
    return None


def _svg_chart(series: list[DoseResponseSeries], *, width: int = 720, height: int = 260) -> str:
    margin_left, margin_right, margin_top, margin_bottom = 44, 16, 14, 28
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_upper = [p.mean + p.std for s in series for p in s.points]
    y_max = max(all_upper) if all_upper else 0.0
    y_max = max(y_max * 1.15, 0.05)  # headroom + a sane minimum so a flat-zero curve isn't a zero-height axis

    def x_of(alpha: float) -> float:
        return margin_left + alpha * plot_w

    def y_of(value: float) -> float:
        return margin_top + (1 - min(value, y_max) / y_max) * plot_h

    parts: list[str] = [
        f'<svg class="dose-chart" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet">'
    ]

    # y-axis gridlines/labels at 0, 0.5*y_max, y_max
    for frac in (0.0, 0.5, 1.0):
        y_val = frac * y_max
        y_px = y_of(y_val)
        parts.append(
            f'<line x1="{margin_left}" y1="{y_px:.1f}" x2="{width - margin_right}" y2="{y_px:.1f}" '
            f'stroke="#212C39" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 6}" y="{y_px + 3:.1f}" text-anchor="end" '
            f'font-family="monospace" font-size="9" fill="#4C5A6B">{y_val:.2f}</text>'
        )

    for s in series:
        style = _SERIES_STYLE[s.series_key]
        color = style["color"]
        pts_sorted = sorted(s.points, key=lambda p: p.alpha)

        # error bars (drawn first, under the line/points)
        for p in pts_sorted:
            x = x_of(p.alpha)
            y_lo = y_of(max(0.0, p.mean - p.std))
            y_hi = y_of(p.mean + p.std)
            parts.append(f'<line x1="{x:.1f}" y1="{y_lo:.1f}" x2="{x:.1f}" y2="{y_hi:.1f}" stroke="{color}" stroke-width="1.4" opacity="0.45"/>')

        line_points = " ".join(f"{x_of(p.alpha):.1f},{y_of(p.mean):.1f}" for p in pts_sorted)
        parts.append(f'<polyline points="{line_points}" fill="none" stroke="{color}" stroke-width="2.2"/>')

        for p in pts_sorted:
            x = x_of(p.alpha)
            y = y_of(p.mean)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{color}" data-alpha="{p.alpha:g}" data-mean="{p.mean:g}" data-std="{p.std:g}"/>')

    parts.append("</svg>")
    return "".join(parts)


def render_dose_response_html(
    series: list[DoseResponseSeries],
    *,
    scenario: str = "S2",
    title: str = "SmartKeyNet -- Steering Attack Dose-Response",
) -> str:
    """Render the real dose-response curves in `series` as a
    self-contained HTML page. Pure view: reads each series' own
    `DoseResponsePoint`s only, invents no alpha step and no value."""
    chart_svg = _svg_chart(series)

    alphas_sorted = sorted({p.alpha for s in series for p in s.points})
    axis_labels = "".join(f"<span>{a:g}</span>" for a in alphas_sorted)

    legend_items = "".join(
        f'<div class="legend-item"><span class="legend-swatch" style="background:{_SERIES_STYLE[s.series_key]["color"]}"></span>{_e(s.label)}</div>'
        for s in series
    )

    boundary_callouts: list[str] = []
    for s in series:
        pts_sorted = sorted(s.points, key=lambda p: p.alpha)
        first = _first_nonzero(pts_sorted)
        if first is None:
            boundary_callouts.append(
                f'<div class="callout"><b>{_e(s.label)}</b>: V(&pi;) measured exactly 0.0000 at every '
                f'alpha in this sweep ({pts_sorted[0].alpha:g}–{pts_sorted[-1].alpha:g}) -- flat zero across the full range shown, honestly, not assumed.</div>'
            )
        else:
            plateau = max(pts_sorted, key=lambda p: p.mean)
            boundary_callouts.append(
                f'<div class="callout boundary"><b>{_e(s.label)}</b>: V(&pi;) first becomes measurably '
                f'nonzero at &alpha;={first.alpha:g} (mean {first.mean:.4f} &plusmn; {first.std:.4f}), '
                f'climbing to its plateau of {plateau.mean:.4f} &plusmn; {plateau.std:.4f} by &alpha;={plateau.alpha:g}.</div>'
            )

    provenance_rows = []
    for s in series:
        n = len(s.points)
        provenance_rows.append(f"{_e(s.label)}: {n} real alpha points, each mean&plusmn;std over its own real eval seeds.")
    provenance = "<br>".join(provenance_rows)

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
    <div class="beat-eyebrow">Steering Attack &middot; Scenario {_e(scenario)}</div>
    <div class="beat-title">Dose-response: V(&pi;) vs. attack strength</div>
    <div class="beat-desc">Below-floor service rate measured against the TRUE (unshaped) posture -- paper eq. 4 -- as the equation-7 input-shaping attack strength &alpha; increases from 0.0 to 1.0. Every point below is a real measured value; error bars show the real spread across eval seeds.</div>
  </div>
  <div class="card">
    <div class="card-label"><span>V(&pi;) &middot; below-floor service rate (true posture)</span><span>&alpha; 0.0 &rarr; 1.0</span></div>
    {chart_svg}
    <div class="dose-axis">{axis_labels}</div>
    <div class="legend-row">{legend_items}</div>
  </div>
  {"".join(boundary_callouts)}
  <div class="provenance">{provenance}</div>
</main>
</body>
</html>
"""


def write_dose_response_html(
    series: list[DoseResponseSeries],
    path: str | Path,
    *,
    scenario: str = "S2",
    title: str = "SmartKeyNet -- Steering Attack Dose-Response",
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_dose_response_html(series, scenario=scenario, title=title), encoding="utf-8")
    return out_path
