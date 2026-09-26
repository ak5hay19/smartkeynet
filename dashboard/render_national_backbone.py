"""
dashboard/render_national_backbone.py

"National Backbone -- Illustrative Scenario" panel: one real held-out
S6-mechanism episode (`configs/scenarios/s_narrative_national_backbone.
yaml`) in which four tenants are shown under illustrative South-Korean
government-agency display names. Same rendering philosophy as every
prior panel: self-contained static HTML, inline SVG, no JS deps, real
measured data only.

**What is and isn't labeled (read before extending)**: display names are
the ONLY thing this module adds. Every tenant identity (`tenant_N`),
sensitivity class, floor, served tier and posture shown is a real field
off a real episode. The sensitivity classes were not assigned to fit the
story -- the driver's seed search found a real `build_tenant_graph` seed
whose untouched random assignment already has the story's shape. Every
agency row therefore shows the underlying `tenant_N` next to the name,
and the page carries a permanent banner saying the names are
illustrative and not a real-deployment claim.

The Passport Office's mid-run reclassification is S6's existing
`migration_schedule` mechanism, unchanged; its before/after attribution
comes from `render_migration_wave.attribute_floor_change` (reused, not
reimplemented), so it is never credited to the schedule unless posture
was held constant.

Nothing here is computed beyond counting: the floor-guarantee card is a
count of real decisions whose real served tier was below the real floor.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from dashboard.render_migration_wave import (
    _ATTRIBUTION_BADGE,
    _CSS as _BASE_CSS,
    _TIER_COLOR,
    FloorObservation,
    MigrationEventView,
    PoolTrajectoryPoint,
)
from env.contracts import Action, SensitivityClass

NAMES_DISCLAIMER = (
    "Illustrative agency names for this scenario; underlying tenant identities and sensitivity "
    "assignment are unchanged from the rest of the system. Not a claim about any real deployment."
)


@dataclass(frozen=True)
class AgencyView:
    """One labeled tenant: the display name, the real tenant id it maps
    onto, the real starting class, and every real decision that tenant
    received in the episode (chronological)."""

    display_name: str
    tenant_id: str
    initial_class: SensitivityClass
    observations: tuple[FloorObservation, ...]


@dataclass(frozen=True)
class NationalBackboneData:
    scenario: str
    seed: int
    graph_seed: int
    policy_label: str
    checkpoint_note: str
    n_decisions: int  # all real decisions in the episode, labeled tenants or not
    n_below_floor: int  # over ALL n_decisions: real served tier < real floor
    agencies: tuple[AgencyView, ...]
    reclass_agency: str  # display_name of the reclassified agency
    reclass_event: MigrationEventView
    trajectory: tuple[PoolTrajectoryPoint, ...]


def _e(value: object) -> str:
    return html.escape(str(value))


def validate_identity_fidelity(data: NationalBackboneData) -> None:
    """Raise if any observation isn't really that agency's tenant, or if
    the reclassified agency/event don't line up -- a mislabeled row must
    fail loudly, not render."""
    seen: set[str] = set()
    for a in data.agencies:
        if a.tenant_id in seen:
            raise ValueError(f"tenant {a.tenant_id} labeled twice")
        seen.add(a.tenant_id)
        for o in a.observations:
            if o.tenant != a.tenant_id:
                raise ValueError(f"{a.display_name} ({a.tenant_id}) holds an observation for {o.tenant}")
    matches = [a for a in data.agencies if a.display_name == data.reclass_agency]
    if len(matches) != 1 or matches[0].tenant_id != data.reclass_event.tenant_id:
        raise ValueError("reclassified agency does not match the reclassification event's tenant")


def _floor_range(obs: tuple[FloorObservation, ...]) -> str:
    if not obs:
        return "no decisions"
    names = []
    for o in obs:
        if o.floor.name not in names:
            names.append(o.floor.name)
    return " &rarr; ".join(_e(n.replace("SERVE_", "")) for n in names)


def _class_cell(a: AgencyView, reclass: MigrationEventView) -> str:
    if reclass.tenant_id == a.tenant_id:
        return (
            f"{_e(reclass.old_sensitivity_class.name)} &rarr; "
            f'<b style="color:var(--hybrid)">{_e(reclass.new_sensitivity_class.name)}</b> at step {reclass.step}'
        )
    return f"<b>{_e(a.initial_class.name)}</b> (constant)"


def _agency_table(data: NationalBackboneData) -> str:
    rows = []
    for a in data.agencies:
        rows.append(
            f'<tr data-tenant="{_e(a.tenant_id)}"><td><b>{_e(a.display_name)}</b></td>'
            f'<td class="mono">{_e(a.tenant_id)}</td><td class="mono">{_class_cell(a, data.reclass_event)}</td>'
            f'<td class="mono">{len(a.observations)}</td><td class="mono">{_floor_range(a.observations)}</td></tr>'
        )
    return (
        '<table class="agency-table"><thead><tr><th>Agency</th><th>Tenant</th><th>Sensitivity class</th>'
        "<th>Real decisions</th><th>Floor over run</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _svg_timeline(data: NationalBackboneData, *, width: int = 900, row_h: int = 44) -> str:
    steps = [p.step for p in data.trajectory] or [o.step for a in data.agencies for o in a.observations]
    step_min, step_max = (min(steps), max(steps)) if steps else (0, 1)
    span = max(step_max - step_min, 1)
    label_w = 150
    plot_w = width - label_w - 10
    height = row_h * len(data.agencies) + 22

    def x_of(step: int) -> float:
        return label_w + (step - step_min) / span * plot_w

    parts = [f'<svg class="timeline" viewBox="0 0 {width} {height}" role="img" aria-label="Per-agency decisions">']
    for i, a in enumerate(data.agencies):
        y = 10 + i * row_h
        parts.append(f'<text x="0" y="{y + 16}" fill="#E9EEF4" font-size="12">{_e(a.display_name)}</text>')
        parts.append(f'<text x="0" y="{y + 30}" fill="#4C5A6B" font-size="10">{_e(a.tenant_id)}</text>')
        parts.append(f'<line x1="{label_w}" y1="{y + 20}" x2="{width - 10}" y2="{y + 20}" stroke="#171F29"/>')
        for o in a.observations:
            x = x_of(o.step)
            floor_c = _TIER_COLOR[o.floor]
            served_c = _TIER_COLOR[Action(int(o.served_tier))]
            below = int(o.served_tier) < int(o.floor)
            stroke = "#FF5C6C" if below else "none"
            parts.append(
                f'<rect x="{x - 3:.1f}" y="{y + 25}" width="6" height="5" fill="{floor_c}" '
                f'data-kind="floor" data-tenant="{_e(o.tenant)}" data-step="{o.step}" data-tier="{_e(o.floor.name)}"/>'
            )
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y + 14}" r="4.5" fill="{served_c}" stroke="{stroke}" stroke-width="2" '
                f'data-kind="served" data-tenant="{_e(o.tenant)}" data-step="{o.step}" '
                f'data-class="{_e(o.sensitivity_class.name)}" data-tier="{_e(o.served_tier.name)}"/>'
            )
    ex = x_of(data.reclass_event.step)
    parts.append(
        f'<line x1="{ex:.1f}" y1="0" x2="{ex:.1f}" y2="{height - 14}" stroke="#6E7EFF" stroke-dasharray="4 4" '
        f'data-event-step="{data.reclass_event.step}"/>'
    )
    parts.append(
        f'<text x="{ex + 4:.1f}" y="{height - 3}" fill="#6E7EFF" font-size="10">'
        f"step {data.reclass_event.step}: {_e(data.reclass_agency)} reclassified</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _legend() -> str:
    chips = "".join(
        f'<span class="sg-tag"><span style="color:{_TIER_COLOR[a]}">&#9679;</span> {_e(a.name.replace("SERVE_", ""))}</span>'
        for a in (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID)
    )
    return (
        f'<div class="subgraph-tags">{chips}<span class="sg-tag">circle = tier served &middot; small bar = required floor '
        "&middot; red ring = served below floor</span></div>"
    )


def _obs_stat(label: str, obs: FloorObservation | None) -> str:
    if obs is None:
        return (
            f'<div class="floor-stat"><div class="k">{_e(label)}</div>'
            '<div class="v" style="color:var(--text-faint)">not observed</div></div>'
        )
    return (
        f'<div class="floor-stat"><div class="k">{_e(label)}</div>'
        f'<div class="v" style="color:{_TIER_COLOR[obs.floor]}">{_e(obs.floor.name)}</div>'
        f'<div class="meta">t={obs.step} &middot; {_e(obs.sensitivity_class.name)}+{_e(obs.posture.name)} '
        f"&middot; served {_e(obs.served_tier.name)}</div></div>"
    )


def _reclass_card(data: NationalBackboneData) -> str:
    ev = data.reclass_event
    badge_text, badge_cls = _ATTRIBUTION_BADGE[ev.attribution]
    return f"""<div class="card" style="margin-top:16px;">
  <div class="phase-head"><div><div class="phase-title">{_e(data.reclass_agency)} reclassified &middot; step {ev.step}</div>
  <div class="phase-tag">{_e(ev.tenant_id)}: {_e(ev.old_sensitivity_class.name)} &rarr; {_e(ev.new_sensitivity_class.name)} (S6's existing scripted-reclassification mechanism)</div></div>
  <span class="badge {badge_cls}">{_e(badge_text)}</span></div>
  <div class="mig-grid">{_obs_stat("Floor before", ev.before)}{_obs_stat("Floor after", ev.after)}</div>
  <div class="note {badge_cls}"><b>attribution:</b> {_e(ev.attribution_note)}</div>
</div>"""


def _floor_guarantee_card(data: NationalBackboneData) -> str:
    fills = [p.pool_fill for p in data.trajectory]
    labeled = sum(len(a.observations) for a in data.agencies)
    verdict_cls = "good" if data.n_below_floor == 0 else "bad"
    if fills and max(fills) - min(fills) < 0.01:
        pool = (
            f"<b>Honest limit:</b> the shared pool did NOT come under pressure in this run -- real pool_fill was "
            f"{min(fills):.2f}&ndash;{max(fills):.2f} of capacity for every decision (S6 has no QKD-degradation "
            "mechanism, and changing pool capacity/initial fill in the config had no effect). So this run shows "
            "the floor holding across a reclassification with a full shared pool; it does NOT demonstrate the "
            "floor holding under pool scarcity. The S3 Comparison Table and Budgeting Brain panels cover that case."
        )
    elif fills:
        pool = f"Real pool_fill ranged {min(fills):.2f}&ndash;{max(fills):.2f} of capacity across the run."
    else:
        pool = "No real pool trajectory was supplied."
    return f"""<div class="card" style="margin-top:16px;">
  <div class="phase-head"><div class="phase-title">Floor guarantee</div><div class="phase-tag">counted over all {data.n_decisions} real decisions</div></div>
  <div class="note {verdict_cls}"><b>{data.n_below_floor} of {data.n_decisions}</b> real decisions were served below their real floor
  ({labeled} of them belong to the four labeled agencies; the rest are the six unlabeled tenants sharing the same pool).</div>
  <div class="note neutral">{pool}</div>
</div>"""


_EXTRA_CSS = """
.banner{background:var(--panel-2);border:1px solid var(--pqc);border-radius:var(--radius-sm);padding:9px 12px;font-size:12px;color:var(--pqc);margin-bottom:14px;line-height:1.5;}
.agency-table{width:100%;border-collapse:collapse;font-size:12.5px;}
.agency-table th{text-align:left;font-family:var(--mono);font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--text-faint);padding:6px 8px;border-bottom:1px solid var(--line);}
.agency-table td{padding:8px;border-bottom:1px solid var(--line-soft);}
.mono{font-family:var(--mono);}
.timeline{width:100%;height:auto;display:block;margin-top:10px;font-family:var(--mono);}
"""


def render_national_backbone_html(
    data: NationalBackboneData, *, title: str = "SmartKeyNet -- National Backbone (Illustrative)"
) -> str:
    """Render one real `NationalBackboneData`. Raises `ValueError` if the
    agency/tenant mapping is inconsistent (see
    `validate_identity_fidelity`)."""
    validate_identity_fidelity(data)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_e(title)}</title>
<style>{_BASE_CSS}{_EXTRA_CSS}</style>
</head>
<body>
<main class="wrap">
  <div class="beat-head">
    <div class="beat-eyebrow">Illustrative scenario &middot; {_e(data.scenario)} mechanism &middot; seed={data.seed}</div>
    <div class="beat-title">National Backbone &mdash; one shared key pool, four agencies</div>
    <div class="beat-desc">A national quantum backbone serves government agencies from a single shared key pool. One real, held-out episode: each agency's identity and tier, and one agency reclassified mid-operation. {_e(data.policy_label)} ({_e(data.checkpoint_note)}) was never trained on this schedule (Hard Rule 8).</div>
  </div>
  <div class="banner" data-role="names-disclaimer">{_e(NAMES_DISCLAIMER)} The tiers were not assigned to fit the story: real tenant-graph seed {data.graph_seed} was found by search because its untouched random assignment already had this shape.</div>
  <div class="card">{_agency_table(data)}</div>
  <div class="card" style="margin-top:16px;">
    <div class="phase-head"><div class="phase-title">Every real decision, per agency</div><div class="phase-tag">tick {min((p.step for p in data.trajectory), default=0)}&ndash;{max((p.step for p in data.trajectory), default=0)}</div></div>
    {_legend()}
    {_svg_timeline(data)}
  </div>
  {_reclass_card(data)}
  {_floor_guarantee_card(data)}
  <div class="provenance">Every tenant id, class, floor, tier and posture shown is a real field from one real episode; only the agency display names are illustrative. Attribution is computed by attribute_floor_change() from real before/after observations.</div>
</main>
</body>
</html>
"""


def write_national_backbone_html(
    data: NationalBackboneData, path: str | Path, *, title: str = "SmartKeyNet -- National Backbone (Illustrative)"
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_national_backbone_html(data, title=title), encoding="utf-8")
    return out_path
