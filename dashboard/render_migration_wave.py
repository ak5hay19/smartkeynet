"""
dashboard/render_migration_wave.py

Migration Wave panel (PLAN.md §5 S6; PLAN.md §6 Demo Beat 4). Renders
S6's scripted, exogenous floor-ratchet schedule (`configs/scenarios/
s6_migration.yaml::migration_schedule`) against real per-decision floor
data from one real held-out S6 episode. Same rendering philosophy as
every prior panel: self-contained static HTML, inline SVG, zero heavy
JS deps, no server, real measured data only. Visual target:
`dashboard/mockups/smartkeynet_dashboard_mockup_v2.html`'s Migration
Wave tab (styling only -- that tab's phase-selector JS, hospital/
fintech tenant names, and its hand-drawn "pool dips and recovers" curve
are 100% fabricated per the mockup's own header and are never read
here; this module renders all three real scripted events statically,
stacked, not as a JS tab switcher, matching every prior panel's "zero
heavy JS deps" convention).

**Hard Rule 7 -- central to this module, read before extending it**:
two prior sessions (S3 Gate W3, the Living System panel) independently
found that `env/forecast_provider.py`'s placeholder threat-feature
formula's `load` term ratchets posture up within the first 1-2
decisions of a real episode, well before any scripted schedule fires.
This module's whole narrative is "the floor ratchets at scheduled
migration steps" -- exactly the story that finding could falsify if a
tenant's floor were already at its post-migration level from
load-driven posture alone. `attribute_floor_change` below is the
honesty gate: it never labels a floor change "scripted" unless the
real, observed posture was IDENTICAL immediately before and after the
event window (see its docstring for the other three honestly-reported
outcomes: no observation to compare against, posture itself moved too
so the two causes can't be cleanly isolated, or the class change had no
visible effect on the floor at all). See `dashboard/
render_migration_wave_demo.py` for what this real S6 episode actually
showed under each of these four outcomes -- not assumed here, verified.

Every value plotted or shown in a `MigrationEventView`/
`PoolTrajectoryPoint` is real, supplied by the driver from one real
episode (or `None`, shown honestly as "not observed," never defaulted).
This module invents no tenant identity, no floor value, and no
attribution it can't derive from the real `before`/`after`
observations it's handed.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from env.contracts import Action, KeyType, SensitivityClass, ThreatPosture
from env.masking import _PLACEHOLDER_FLOOR_TABLE

# ---------------------------------------------------------------------------
# Data model -- real fields only, populated by the demo driver from one real
# S6 episode, never hand-authored (see render_migration_wave_demo.py).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloorObservation:
    """One real decision for one tenant: the real request's
    `sensitivity_class`, the real resolved posture, the real
    `PolicyTable.floor(...)` value that decision's mask was built from
    (`state["policy_floor"]`), and the real tier that decision actually
    resolved to serving (`SmartKeyNetEnv._resulting_key_type` -- the
    same ground-truth function the environment itself uses)."""

    step: int
    tenant: str
    service: str
    sensitivity_class: SensitivityClass
    posture: ThreatPosture
    floor: Action
    served_tier: KeyType


@dataclass(frozen=True)
class MigrationEventView:
    """One real scripted event from `s6_migration.yaml::migration_schedule`,
    plus the real before/after `FloorObservation`s the driver found
    bracketing it (each may be `None` -- see `attribute_floor_change`)
    and the honest attribution this module computed from them."""

    step: int
    tenant_id: str
    old_sensitivity_class: SensitivityClass
    new_sensitivity_class: SensitivityClass
    before: FloorObservation | None
    after: FloorObservation | None
    attribution: str
    attribution_note: str


@dataclass(frozen=True)
class PoolTrajectoryPoint:
    """One real (internal simulator tick, pool level) sample -- same
    shape/convention as `render_budgeting_brain.py::PoolTrajectoryPoint`."""

    step: int
    pool_fill: float


@dataclass(frozen=True)
class MigrationWaveData:
    scenario: str
    seed: int
    policy_label: str
    checkpoint_note: str
    n_decisions: int
    trajectory: tuple[PoolTrajectoryPoint, ...]
    events: tuple[MigrationEventView, ...]


# ---------------------------------------------------------------------------
# Attribution -- the honesty gate (Hard Rule 7). Pure function of the real
# before/after observations; never reaches into an episode itself.
# ---------------------------------------------------------------------------


def _calm_floor(sensitivity_class: SensitivityClass) -> Action:
    """The real floor this class would get at CALM posture -- looked up
    directly off the same real `_PLACEHOLDER_FLOOR_TABLE` `env/masking.py`
    itself uses, never a second, hand-copied table."""
    return _PLACEHOLDER_FLOOR_TABLE[(sensitivity_class, ThreatPosture.CALM)]


def attribute_floor_change(
    *,
    event_step: int,
    old_sensitivity_class: SensitivityClass,
    new_sensitivity_class: SensitivityClass,
    before: FloorObservation | None,
    after: FloorObservation | None,
) -> tuple[str, str]:
    """Decide -- honestly, from real observations only -- what actually
    caused (or didn't cause) an observed floor change around one
    scripted migration event. Returns `(attribution, note)`.

    Four possible outcomes, in the order checked (Hard Rule 7 -- never
    forces the "scripted migration raised the floor" story unless the
    real data actually isolates it):

    1. `"no_after_observation"` -- no real decision for this tenant, at
       any point in the rest of the episode, ever showed the new
       `sensitivity_class`. Nothing to report about this event's effect.
    2. `"no_before_observation"` -- no real decision for this tenant
       exists before the event step. There is nothing to compare the
       post-event floor against, so no floor-*change* claim is made;
       only the real post-event floor itself is reportable.
    3. `"posture_confound"` -- the real resolved posture itself differs
       between the last before-observation and the first after-
       observation. Any floor change in this window could be the
       scripted class change, the posture shift, or both -- can't be
       cleanly isolated from these two observations alone, reported as
       such rather than credited to the schedule.
    4. `"no_visible_change"` -- posture held constant, but the floor
       value itself didn't change: the old and new classes already
       resolve to the same floor at this posture (a genuine, real case
       where the scripted schedule's effect isn't observable at the
       floor level).
    5. `"scripted"` -- posture held constant AND the floor genuinely
       rose. The only case this module credits to the schedule, and
       only because the one variable that could confound it (posture)
       is pinned to a real, identical, observed value on both sides.
    """
    if after is None:
        return (
            "no_after_observation",
            f"No real decision for tenant {before.tenant if before else '(unknown)'} anywhere in the rest of "
            f"this episode ever showed sensitivity_class={new_sensitivity_class.name} -- the scripted event at "
            f"step {event_step} fired (upstream, in the request-arrival process), but this tenant never "
            "generated a request afterward that this episode's real traffic actually observed reflecting it.",
        )

    if before is None:
        return (
            "no_before_observation",
            f"No real decision for tenant {after.tenant} was observed before step {event_step} in this "
            "episode -- there is nothing to compare the post-event floor against, so no floor-change claim is "
            f"made for this event. The only real fact available: the first decision observed after the event "
            f"that actually reflects the new class {new_sensitivity_class.name} (t={after.step}) showed "
            f"floor={after.floor.name} at posture {after.posture.name}.",
        )

    if before.posture != after.posture:
        return (
            "posture_confound",
            f"Posture itself changed between the last before-observation ({before.posture.name} at "
            f"t={before.step}) and the first after-observation reflecting the new class "
            f"({after.posture.name} at t={after.step}) -- any floor change in this window cannot be cleanly "
            "attributed to the scripted migration alone; the posture shift may have contributed too.",
        )

    if before.floor == after.floor:
        return (
            "no_visible_change",
            f"The scripted class change ({old_sensitivity_class.name} -> {new_sensitivity_class.name}) did "
            f"NOT visibly raise this tenant's floor: both the last observation before (t={before.step}, "
            f"floor={before.floor.name}) and the first observation reflecting the new class after "
            f"(t={after.step}, floor={after.floor.name}) resolve to the same floor at posture "
            f"{before.posture.name} -- the {old_sensitivity_class.name} and {new_sensitivity_class.name} rows "
            "of the real floor table already agreed at this posture level before the migration fired.",
        )

    calm_before = _calm_floor(old_sensitivity_class)
    already_elevated_note = ""
    if int(before.floor) > int(calm_before):
        already_elevated_note = (
            f" Note: the pre-event floor ({before.floor.name}) was already above the "
            f"{old_sensitivity_class.name}/CALM baseline ({calm_before.name}) -- this episode's own "
            f"load-driven posture had already ratcheted to {before.posture.name} before this event fired "
            "(a standing, separately-flagged property of the placeholder threat-feature formula, not this "
            f"migration). The migration's own real contribution here is the further step from "
            f"{before.floor.name} to {after.floor.name}, not a climb starting from the CALM baseline."
        )
    return (
        "scripted",
        f"Real, attributable: the last observed decision before the event (t={before.step}) showed "
        f"floor={before.floor.name} at {old_sensitivity_class.name}+{before.posture.name}; the first decision "
        f"observed after the event that actually reflects the new class {new_sensitivity_class.name} "
        f"(t={after.step}) showed floor={after.floor.name} at {new_sensitivity_class.name}+{after.posture.name} "
        f"-- the SAME posture ({before.posture.name}) throughout, so this floor increase is attributable to "
        "the scripted migration, not a posture shift." + already_elevated_note,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_TIER_COLOR: dict[Action, str] = {
    Action.SERVE_CLASSICAL: "#8B95A5",
    Action.SERVE_PQC: "#E8A33D",
    Action.SERVE_HYBRID: "#33D687",
}

_ATTRIBUTION_BADGE: dict[str, tuple[str, str]] = {
    "scripted": ("scripted", "good"),
    "no_visible_change": ("no visible change", "neutral"),
    "posture_confound": ("posture confound", "bad"),
    "no_before_observation": ("no before data", "neutral"),
    "no_after_observation": ("no after data", "neutral"),
}

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
.wrap{max-width:960px;margin:0 auto;}
.beat-head{margin-bottom:22px;}
.beat-eyebrow{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--quantum);}
.beat-title{font-family:var(--disp);font-weight:700;font-size:22px;margin-top:4px;}
.beat-desc{color:var(--text-dim);font-size:13px;margin-top:6px;line-height:1.5;}

.scenario-strip{display:flex;align-items:center;gap:10px;margin-bottom:16px;font-family:var(--mono);font-size:11.5px;color:var(--text-dim);}
.scenario-strip .s-chip{padding:4px 10px;border:1px solid var(--line);border-radius:999px;color:var(--text-faint);}
.scenario-strip .s-chip.on{color:var(--quantum);border-color:var(--quantum);}

.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px 18px;}

.phase-list{display:flex;flex-direction:column;gap:14px;}
.phase-card{}
.phase-head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;}
.phase-title{font-family:var(--disp);font-weight:600;font-size:15px;}
.phase-tag{font-family:var(--mono);font-size:10.5px;color:var(--text-faint);}

.badge{font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--text-dim);}
.badge.good{color:var(--hybrid);border-color:rgba(51,214,135,.35);}
.badge.bad{color:var(--danger);border-color:rgba(255,92,108,.35);}
.badge.neutral{color:var(--pqc);border-color:rgba(232,163,61,.35);}

.mig-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px;}
@media (max-width:760px){.mig-grid{grid-template-columns:1fr;}}

.subgraph-tags{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;}
.sg-tag{font-family:var(--mono);font-size:11px;color:var(--text-dim);background:var(--panel-2);border:1px solid var(--line-soft);border-radius:999px;padding:5px 10px;}
.sg-tag .arrow{color:var(--text-faint);margin:0 4px;}
.sg-tag .to{color:var(--text);font-weight:600;}

.floor-stat-row{display:flex;align-items:center;gap:10px;margin-top:10px;}
.floor-stat{flex:1;background:var(--panel-2);border:1px solid var(--line-soft);border-radius:var(--radius-sm);padding:9px 11px;}
.floor-stat .k{font-family:var(--mono);font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--text-faint);}
.floor-stat .v{font-family:var(--mono);font-weight:700;font-size:15px;margin-top:3px;}
.floor-stat .meta{font-family:var(--mono);font-size:9.5px;color:var(--text-faint);margin-top:2px;}
.floor-arrow{font-family:var(--mono);color:var(--text-faint);font-size:16px;}

.note{margin-top:10px;font-size:11.5px;color:var(--text-dim);border-left:2px solid var(--quantum);padding:8px 11px;background:var(--panel-2);border-radius:var(--radius-sm);line-height:1.55;}
.note.good{border-left-color:var(--hybrid);}
.note.bad{border-left-color:var(--danger);}
.note.neutral{border-left-color:var(--pqc);}
.note b{color:var(--text);}

.area-chart{width:100%;height:120px;display:block;margin-top:10px;}
.chart-axis{display:flex;justify-content:space-between;font-family:var(--mono);font-size:9.5px;color:var(--text-faint);margin-top:4px;}

.provenance{margin-top:16px;font-family:var(--mono);font-size:10px;color:var(--text-faint);line-height:1.6;}
"""


def _e(value: object) -> str:
    return html.escape(str(value))


def _obs_stat_html(label: str, obs: FloorObservation | None) -> str:
    if obs is None:
        return (
            f'<div class="floor-stat"><div class="k">{_e(label)}</div>'
            f'<div class="v" style="color:var(--text-faint)">not observed</div>'
            f'<div class="meta">no real decision found</div></div>'
        )
    color = _TIER_COLOR[obs.floor]
    return (
        f'<div class="floor-stat"><div class="k">{_e(label)}</div>'
        f'<div class="v" style="color:{color}">{_e(obs.floor.name)}</div>'
        f'<div class="meta">t={obs.step} &middot; {_e(obs.sensitivity_class.name)}+{_e(obs.posture.name)} '
        f"&middot; served {_e(obs.served_tier.name)}</div></div>"
    )


def _phase_card(index: int, ev: MigrationEventView) -> str:
    badge_text, badge_cls = _ATTRIBUTION_BADGE[ev.attribution]

    tags: list[str] = []
    tag_target = ev.after if ev.after is not None else ev.before
    if tag_target is not None:
        tags.append(
            f'<span class="sg-tag">{_e(ev.tenant_id)} &middot; {_e(tag_target.service)} '
            f'<span class="arrow">&rarr;</span> <span class="to">{_e(tag_target.floor.name)}</span></span>'
        )

    return f"""<div class="card phase-card">
  <div class="phase-head">
    <div><div class="phase-title">Phase {index} &middot; step {ev.step}</div>
    <div class="phase-tag">{_e(ev.tenant_id)}: {_e(ev.old_sensitivity_class.name)} &rarr; {_e(ev.new_sensitivity_class.name)}</div></div>
    <span class="badge {badge_cls}">{_e(badge_text)}</span>
  </div>
  <div class="subgraph-tags">{"".join(tags)}</div>
  <div class="mig-grid">
    {_obs_stat_html("Floor before", ev.before)}
    {_obs_stat_html("Floor after", ev.after)}
  </div>
  <div class="note {badge_cls}"><b>attribution:</b> {_e(ev.attribution_note)}</div>
</div>"""


def _svg_pool_trajectory(trajectory: tuple[PoolTrajectoryPoint, ...], events: tuple[MigrationEventView, ...], *, width: int = 900, height: int = 120) -> str:
    if not trajectory:
        return f'<svg class="area-chart" viewBox="0 0 {width} {height}" preserveAspectRatio="none"></svg>'

    steps = [p.step for p in trajectory]
    step_min, step_max = min(steps), max(steps)
    step_span = max(step_max - step_min, 1)

    def x_of(step: int) -> float:
        return (step - step_min) / step_span * width

    def y_of(fill: float) -> float:
        clamped = max(0.0, min(1.0, fill))
        return height - clamped * height

    line_points = " ".join(f"{x_of(p.step):.1f},{y_of(p.pool_fill):.1f}" for p in trajectory)
    trajectory_data = ";".join(f"{p.step}:{p.pool_fill:g}" for p in trajectory)

    parts = [
        f'<svg class="area-chart" viewBox="0 0 {width} {height}" preserveAspectRatio="none">',
        f'<polyline points="{line_points}" fill="none" stroke="#6E7EFF" stroke-width="2" '
        f'data-trajectory="{trajectory_data}"/>',
    ]
    for ev in events:
        ex = x_of(ev.step)
        parts.append(
            f'<line x1="{ex:.1f}" y1="0" x2="{ex:.1f}" y2="{height}" stroke="#212C39" '
            f'stroke-dasharray="3 4" data-event-step="{ev.step}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _pool_trajectory_note(trajectory: tuple[PoolTrajectoryPoint, ...]) -> str:
    if not trajectory:
        return "No real pool trajectory data was supplied."
    fills = [p.pool_fill for p in trajectory]
    span = max(fills) - min(fills)
    if span < 0.01:
        return (
            f"Real pool_fill stayed effectively flat ({min(fills):.4f}&ndash;{max(fills):.4f}) across this "
            "entire episode -- S6 has no QKD-degradation mechanism (unlike S3), so this migration wave "
            "produces no observable pool-scarcity response on its own; the interesting real signal here is "
            "the per-tenant floor change, not pool pressure."
        )
    return (
        f"Real pool_fill ranged {min(fills):.4f}&ndash;{max(fills):.4f} of capacity across this episode "
        "(dashed lines mark the three real scripted event steps)."
    )


def render_migration_wave_html(
    data: MigrationWaveData, *, title: str = "SmartKeyNet -- Migration Wave"
) -> str:
    """Render one real `MigrationWaveData` as a self-contained HTML
    page. Pure view: every phase card, floor stat, and trajectory point
    is a real field off `data` (or an honestly-labeled `None`); the
    only derived text is the attribution note already computed by
    `attribute_floor_change` and the pool-trajectory note computed live
    from the real trajectory's own span -- never a fabricated claim."""
    phases = "".join(_phase_card(i + 1, ev) for i, ev in enumerate(data.events))
    chart = _svg_pool_trajectory(data.trajectory, data.events)
    pool_note = _pool_trajectory_note(data.trajectory)
    steps = [p.step for p in data.trajectory]
    axis = f"<span>tick {steps[0]}</span><span>tick {steps[-1]}</span>" if steps else ""

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
    <div class="beat-eyebrow">Demo Beat 4 &middot; Scenario {_e(data.scenario)}</div>
    <div class="beat-title">The migration wave</div>
    <div class="beat-desc">A scripted, exogenous compliance timeline -- never chosen by the agent -- ratchets three real tenants' sensitivity classes at three fixed steps in one real, held-out episode. {_e(data.policy_label)} ({_e(data.checkpoint_note)}) was never trained on this schedule (Hard Rule 8).</div>
  </div>
  <div class="scenario-strip">
    <span>Scenario</span>
    <span class="s-chip on">{_e(data.scenario)} &middot; migration wave &middot; seed={data.seed}</span>
  </div>
  <div class="phase-list">
    {phases}
  </div>
  <div class="card" style="margin-top:16px;">
    <div class="phase-head"><div class="phase-title">Pool response</div><div class="phase-tag">real pool_fill trajectory, {data.n_decisions} decisions</div></div>
    {chart}
    <div class="chart-axis">{axis}</div>
    <div class="note neutral">{pool_note}</div>
  </div>
  <div class="provenance">Every floor/posture/tier value shown is a real field from one real held-out S6 episode; attribution labels are computed by attribute_floor_change() directly from the real before/after observations bracketing each scripted event -- never assumed to be "scripted" by default.</div>
</main>
</body>
</html>
"""


def write_migration_wave_html(
    data: MigrationWaveData, path: str | Path, *, title: str = "SmartKeyNet -- Migration Wave"
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_migration_wave_html(data, title=title), encoding="utf-8")
    return out_path
