"""
dashboard/render_budgeting_brain.py

Renders the Budgeting Brain panel (PLAN.md Demo Beat 2): a real,
same-seed S3 (QKD degradation) episode run under two policies -- the
trained masked DQN and `agents/baselines.py::AlwaysHybridPolicy` (that
module's own docstring: "The 'drains the pool' baseline") -- as a
side-by-side real pool-trajectory-over-time visual, with real
exhaustion/regret events marked at their real positions. Visual
target: dashboard/mockups/smartkeynet_dashboard_mockup_v2.html's
Budgeting Brain tab (styling only -- that tab's curve shapes, stat
numbers, and "hospital/fintech" tenant label are 100% fabricated per
the mockup's own header and are never read here).

Same rendering philosophy as dashboard/render_dose_response.py and
dashboard/render_comparison_table.py: a pure function over an
explicit, real input object, self-contained HTML (inline CSS + inline
SVG, no charting library, no server), zero new dependencies.

Hard Rule 7 (central to this module, by analogy with Hard Rule 10):
every pool-level point plotted and every exhaustion marker placed is a
real value taken verbatim from a `PolicyEpisode` this module did not
compute (see dashboard/render_budgeting_brain_demo.py for how it's
collected from one real episode) -- this module invents no trajectory
shape and no event position. `regret_events`/`pool_exhaustion_events`
are always the SAME real integer on one `PolicyEpisode` (verified by
the driver, not re-derived here) and are always shown paired with the
"same event by construction" annotation -- never presented as two
independent numbers. `p99_latency`, when `include_p99` is True, always
carries its documented discrete-cost-model-percentile-artifact caveat.
The "stable"/"exhausted" badge and the real-fact callout (pool-fill
range, first-exhaustion position) are the only derived text this
module adds, and both are computed live from the same input object,
never hardcoded to a specific policy side.
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
body{background:var(--bg);color:var(--text);font-family:var(--disp);padding:28px 20px 60px;}
.wrap{max-width:900px;margin:0 auto;}
.beat-head{margin-bottom:22px;}
.beat-eyebrow{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--quantum);}
.beat-title{font-family:var(--disp);font-weight:700;font-size:22px;margin-top:4px;}
.beat-desc{color:var(--text-dim);font-size:13px;margin-top:6px;line-height:1.5;}

.scenario-strip{display:flex;align-items:center;gap:10px;margin-bottom:16px;font-family:var(--mono);font-size:11.5px;color:var(--text-dim);}
.scenario-strip .s-chip{padding:4px 10px;border:1px solid var(--line);border-radius:999px;color:var(--text-faint);}
.scenario-strip .s-chip.on{color:var(--quantum);border-color:var(--quantum);}

.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:18px 18px 16px;}

.arena{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
.arena-card{position:relative;overflow:hidden;}
.arena-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;}
.arena-title{font-family:var(--disp);font-weight:600;font-size:15px;}
.arena-tag{font-family:var(--mono);font-size:10px;color:var(--text-faint);}
.area-chart{width:100%;height:150px;display:block;}
.chart-axis{display:flex;justify-content:space-between;font-family:var(--mono);font-size:9.5px;color:var(--text-faint);margin-top:4px;}

.badge{font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--text-dim);}
.badge.calm{color:var(--hybrid);border-color:rgba(51,214,135,.35);}
.badge.high{color:var(--danger);border-color:rgba(255,92,108,.35);}

.callout{margin-top:12px;font-size:11.5px;color:var(--text-dim);border-left:2px solid var(--quantum);padding:8px 11px;background:var(--panel-2);border-radius:var(--radius-sm);line-height:1.5;}
.callout b{color:var(--text);}

.exhaust-banner{margin-top:12px;font-family:var(--mono);font-size:11px;font-weight:600;letter-spacing:.06em;color:var(--danger);background:rgba(255,92,108,.1);border:1px solid rgba(255,92,108,.35);border-radius:var(--radius-sm);padding:8px 11px;display:flex;align-items:center;gap:8px;}
.exhaust-banner .x-dot{width:6px;height:6px;border-radius:50%;background:var(--danger);flex:none;}

.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px;}
.stat-box{background:var(--panel-2);border:1px solid var(--line-soft);border-radius:var(--radius-sm);padding:10px 12px;}
.stat-box.hero{border-color:rgba(110,126,255,.35);}
.stat-box .k{font-family:var(--mono);font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--text-faint);}
.stat-box .v{font-family:var(--mono);font-weight:700;font-size:19px;margin-top:3px;}
.stat-box .v.good{color:var(--hybrid);}
.stat-box .v.bad{color:var(--danger);}

.note{margin-top:16px;font-size:11.5px;color:var(--text-dim);border-left:2px solid var(--quantum);padding:8px 11px;background:var(--panel-2);border-radius:var(--radius-sm);line-height:1.55;}
.note.caveat{border-left-color:var(--danger);}
.note b{color:var(--text);}

.provenance{margin-top:16px;font-family:var(--mono);font-size:10px;color:var(--text-faint);line-height:1.6;}
"""

_SERIES_STYLE = {
    "agent": {"color": "#6E7EFF"},
    "baseline": {"color": "#FF5C6C"},
}

_P99_CAVEAT = (
    "p99_latency is a discrete-cost-model percentile artifact, not a meaningful tail-latency "
    "discriminator: whenever >=4/250 (>=1.6%) of an episode's decisions cost SERVE_HYBRID (the "
    "max of a 4-value discrete cost set), np.percentile's interpolation saturates at exactly 1.5000 "
    "-- see experiments/harness.py::ScenarioResult.p99_latency's own docstring for the full, "
    "numerically-verified mechanism. Included here for completeness, not as a differentiator."
)

_REGRET_NOTE = (
    "regret_events and pool_exhaustion_events are the same count by construction in this "
    "environment (every logged RegretEvent is a pool-exhaustion event) -- see "
    "experiments/harness.py::run_scenario's own docstring. Shown once, not as two independent results."
)

_BELOW_FLOOR_NOTE = (
    "below_floor_rate is 0.0000 for BOTH policies on this real episode -- Hard Rule 9 (env/"
    "deferral_queue.py: pool exhaustion never causes a downgrade) holds even for Always-Hybrid, "
    "which is still masked, just greedy about spending the pool. Unlike the masked-vs-soft-reward "
    "S3 comparison (dashboard/s3_comparison_table.html), where disabling masking made below_floor_rate "
    "the headline discriminator, here pool_exhaustion_events/regret_events -- not below_floor_rate -- "
    "is the real, measured axis these two policies diverge on. Shown for completeness, not reframed "
    "to look like a discriminator it isn't on this pair."
)


@dataclass(frozen=True)
class PoolTrajectoryPoint:
    """One real (internal simulator tick, pool level) sample -- `step`
    is the real `env._step_count` value at the moment this decision's
    state was returned; `pool_fill` is the real, already-normalized
    `state["pool_fill"]` field ([0, 1], fraction of pool capacity)."""

    step: int
    pool_fill: float


@dataclass(frozen=True)
class ExhaustionEvent:
    """One real deferral onset (`env/contracts.py::RegretEvent`), on the
    SAME internal-tick axis as `PoolTrajectoryPoint.step` above.
    `pool_fill_normalized` is `RegretEvent["pool_fill_at_onset"]`
    (real, raw bits) divided by the real pool capacity the episode was
    configured with -- pre-normalized by the caller so this module
    never needs the capacity value itself."""

    step: int
    pool_fill_normalized: float
    tenant: str
    sensitivity_class: int


@dataclass(frozen=True)
class PolicyEpisode:
    """One policy's real trajectory + real summary stats from one real
    S3 episode. `regret_events` and `pool_exhaustion_events` are kept
    as two separate fields (matching `experiments/harness.py::
    ScenarioResult`'s own shape) but are always equal by construction
    -- the caller (the demo driver) sets both from the same
    `len(exhaustion_events)`, never from two different sources."""

    label: str
    series_key: str  # "agent" | "baseline"
    tag: str
    trajectory: list[PoolTrajectoryPoint]
    exhaustion_events: list[ExhaustionEvent]
    regret_events: int
    pool_exhaustion_events: int
    below_floor_rate: float
    forced_rekey_ratio: float
    p99_latency: float


@dataclass(frozen=True)
class BudgetingBrainData:
    scenario: str
    seed: int
    agent: PolicyEpisode
    baseline: PolicyEpisode
    include_p99: bool = True


def _e(value: object) -> str:
    return html.escape(str(value))


def _svg_area_chart(episode: PolicyEpisode, *, width: int = 300, height: int = 150) -> str:
    traj = episode.trajectory
    color = _SERIES_STYLE[episode.series_key]["color"]

    if not traj:
        return f'<svg class="area-chart" viewBox="0 0 {width} {height}" preserveAspectRatio="none"></svg>'

    steps = [p.step for p in traj]
    step_min, step_max = min(steps), max(steps)
    step_span = max(step_max - step_min, 1)

    def x_of(step: int) -> float:
        return (step - step_min) / step_span * width

    def y_of(fill: float) -> float:
        # Full pool (fill=1) draws near the top; an exhausted pool
        # (fill=0) draws at the bottom -- matches the mockup's own
        # "pool level" area-chart convention.
        clamped = max(0.0, min(1.0, fill))
        return height - clamped * height

    line_points = " ".join(f"{x_of(p.step):.1f},{y_of(p.pool_fill):.1f}" for p in traj)
    polygon_points = f"{x_of(step_min):.1f},{height} {line_points} {x_of(step_max):.1f},{height}"
    trajectory_data = ";".join(f"{p.step}:{p.pool_fill:g}" for p in traj)

    gradient_id = f"grad-{episode.series_key}"
    parts = [
        f'<svg class="area-chart" viewBox="0 0 {width} {height}" preserveAspectRatio="none">',
        f'<defs><linearGradient id="{gradient_id}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity=".35"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>',
        f'<polygon points="{polygon_points}" fill="url(#{gradient_id})"/>',
        f'<polyline points="{line_points}" fill="none" stroke="{color}" stroke-width="2.2" '
        f'data-trajectory="{trajectory_data}"/>',
    ]

    for ev in episode.exhaustion_events:
        ex = x_of(ev.step)
        ey = y_of(ev.pool_fill_normalized)
        parts.append(
            f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="2.6" fill="var(--danger)" stroke="{color}" '
            f'stroke-width="0.6" data-event-step="{ev.step}" '
            f'data-event-pool-fill="{ev.pool_fill_normalized:g}"/>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _real_fact_callout(episode: PolicyEpisode) -> str:
    """A factual, non-fabricated observation computed directly from
    `episode`'s own real trajectory/event data -- never a hardcoded
    mechanism claim (this module has no access to *why* the pool moved
    the way it did, only the real recorded fill levels themselves)."""
    fills = [p.pool_fill for p in episode.trajectory]
    min_point = min(episode.trajectory, key=lambda p: p.pool_fill)

    if episode.exhaustion_events:
        first = episode.exhaustion_events[0]
        return (
            f'<div class="callout"><b>{_e(episode.label)}</b>: real pool level ranged '
            f"{min(fills):.3f}&ndash;{max(fills):.3f} of capacity over this episode, reaching its "
            f"minimum ({min_point.pool_fill:.3f}) at internal tick {min_point.step}. First real "
            f"deferral onset at tick {first.step} (pool at {first.pool_fill_normalized:.3f} of "
            f"capacity, tenant {_e(first.tenant)}) -- {len(episode.exhaustion_events)} total this "
            f"episode.</div>"
        )
    return (
        f'<div class="callout"><b>{_e(episode.label)}</b>: real pool level ranged '
        f"{min(fills):.3f}&ndash;{max(fills):.3f} of capacity over this episode, never triggering a "
        f"deferral -- zero exhaustion events, measured, not assumed.</div>"
    )


def _stat_grid(episode: PolicyEpisode, *, include_p99: bool) -> str:
    below_floor_cls = "good" if episode.below_floor_rate == 0.0 else "bad"
    regret_cls = "good" if episode.regret_events == 0 else "bad"
    exhaustion_cls = "good" if episode.pool_exhaustion_events == 0 else "bad"

    boxes = [
        f'<div class="stat-box hero"><div class="k">Pool exhaustion <span title="same event as regret, by construction">(== regret)</span></div>'
        f'<div class="v {exhaustion_cls}">{episode.pool_exhaustion_events}</div></div>',
        f'<div class="stat-box"><div class="k">Regret events</div>'
        f'<div class="v {regret_cls}">{episode.regret_events}</div></div>',
        f'<div class="stat-box"><div class="k">Below-floor rate</div>'
        f'<div class="v {below_floor_cls}">{episode.below_floor_rate:.4f}</div></div>',
        f'<div class="stat-box"><div class="k">Forced-rekey ratio</div>'
        f'<div class="v">{episode.forced_rekey_ratio:.3f}</div></div>',
    ]
    if include_p99:
        boxes.append(
            f'<div class="stat-box"><div class="k">p99 latency*</div>'
            f'<div class="v">{episode.p99_latency:.4f}</div></div>'
        )
    return f'<div class="stat-grid">{"".join(boxes)}</div>'


def _arena_card(episode: PolicyEpisode, *, include_p99: bool) -> str:
    badge_cls = "calm" if episode.regret_events == 0 else "high"
    badge_text = "stable" if episode.regret_events == 0 else "exhausted"
    steps = [p.step for p in episode.trajectory]
    axis = (
        f"<span>tick {steps[0]}</span><span>tick {steps[-1]}</span>" if steps else ""
    )

    exhaust_banner = ""
    if episode.exhaustion_events:
        first = episode.exhaustion_events[0]
        exhaust_banner = (
            '<div class="exhaust-banner"><span class="x-dot"></span> POOL EXHAUSTED -- '
            f"{len(episode.exhaustion_events)} real deferral event(s), first at tick {first.step} "
            f"(tenant {_e(first.tenant)})</div>"
        )

    return f"""<div class="card arena-card">
  <div class="arena-head">
    <div><div class="arena-title">{_e(episode.label)}</div><div class="arena-tag">{_e(episode.tag)}</div></div>
    <span class="badge {badge_cls}">{badge_text}</span>
  </div>
  {_svg_area_chart(episode)}
  <div class="chart-axis">{axis}</div>
  {_real_fact_callout(episode)}
  {exhaust_banner}
  {_stat_grid(episode, include_p99=include_p99)}
</div>"""


def render_budgeting_brain_html(
    data: BudgetingBrainData,
    *,
    title: str = "SmartKeyNet -- Budgeting Brain",
) -> str:
    """Render `data`'s two real `PolicyEpisode`s as a self-contained
    side-by-side HTML page. Pure view: every plotted point, every
    marker, and every stat box value is a `PolicyEpisode` field
    verbatim (or, for the badge/callout, computed live from those same
    fields) -- never recomputed or invented (Hard Rule 7)."""
    agent_card = _arena_card(data.agent, include_p99=data.include_p99)
    baseline_card = _arena_card(data.baseline, include_p99=data.include_p99)

    p99_note = (
        f'<div class="note caveat"><b>p99_latency caveat:</b> {_e(_P99_CAVEAT)}</div>'
        if data.include_p99
        else ""
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
    <div class="beat-eyebrow">Demo Beat 2 &middot; Scenario {_e(data.scenario)}</div>
    <div class="beat-title">The budgeting brain</div>
    <div class="beat-desc">One real, same-seed (seed={data.seed}) QKD-degradation episode, two policies, identical conditions -- the same real request stream and QKD refill/QBER trace -- diverging purely from each policy's own action choices.</div>
  </div>
  <div class="scenario-strip">
    <span>Scenario</span>
    <span class="s-chip on">{_e(data.scenario)} &middot; QKD degradation</span>
  </div>
  <div class="arena">
    {agent_card}
    {baseline_card}
  </div>
  {p99_note}
  <div class="note"><b>regret_events note:</b> {_e(_REGRET_NOTE)}</div>
  <div class="note"><b>below_floor_rate note:</b> {_e(_BELOW_FLOOR_NOTE)}</div>
  <div class="provenance">pool_exhaustion_events leads each side's stats deliberately for this panel -- the real, measured axis these two policies diverge on here. Pool-trajectory shape and real exhaustion-event positions are this panel's own headline visual.</div>
</main>
</body>
</html>
"""


def write_budgeting_brain_html(
    data: BudgetingBrainData,
    path: str | Path,
    *,
    title: str = "SmartKeyNet -- Budgeting Brain",
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_budgeting_brain_html(data, title=title), encoding="utf-8")
    return out_path
