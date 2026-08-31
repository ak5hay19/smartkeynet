"""
dashboard/render_living_system.py

Living System panel backend + view (PLAN2.md's dashboard panel set;
Hard Rule 10 by analogy with `dashboard/render_explain.py`,
`render_dose_response.py`, `render_comparison_table.py`). Visual
target: `dashboard/mockups/smartkeynet_dashboard_mockup_v2.html`'s
"Living System" tab (styling only -- that file's tenant names
hospital/fintech/logging/iot-telemetry and its 6 hand-placed node
coordinates are 100% fabricated illustrative examples and are never
read by this module; the real tenant graph's nodes are generic
`tenant_0..tenant_N` identities from `env/request_generator.py::
build_tenant_graph`, rendered faithfully as such).

Hard Rule 10 (central to this module, same discipline as every prior
rendered panel): every tenant identity, sensitivity_class, pqc_capable
flag, action, and served tier drawn here is a real field off a real
`LivingSystemSnapshot` -- never invented. Exactly two things this
module computes rather than reads verbatim, both pure functions of
real input, documented at their definition:
  1. `_tenant_positions` -- deterministic hub-and-spoke ring layout, a
     graphical placement derived only from *how many* real tenant
     nodes this snapshot's graph has (never a fixed/hardcoded count,
     never the mockup's 6 hand-placed coordinates).
  2. `build_snapshot`'s "most recently served tier per tenant" -- a
     real, traceable fold over the real decision history up to a real
     cutoff index (never a fabricated or default tier -- a tenant with
     no decision yet in the window renders as `last_served_tier=None`,
     shown honestly as "no traffic yet", not defaulted to some tier).

Design decision (recorded in SESSION_LOG.md): static snapshots, not a
live-animating view -- see `dashboard/render_living_system_demo.py`'s
module docstring for the full argument. A `LivingSystemSnapshot` is
one real, frozen moment in a real episode; `render_living_system_html`
renders exactly one snapshot into one self-contained `.html` file
(inline CSS, inline SVG, zero JS, zero server, zero new dependencies --
same philosophy as every prior renderer in this package).
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from env.contracts import Action, KeyType, SensitivityClass, ThreatPosture

# ---------------------------------------------------------------------------
# Data model -- real fields only, populated by the demo driver from a real
# episode, never hand-authored (see render_living_system_demo.py).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TenantNodeView:
    """One real tenant node's persistent graph attributes
    (`env/request_generator.py::build_tenant_graph`) plus the real tier
    it was most recently served at, as of one snapshot's cutoff.
    `last_served_tier=None` means this tenant has no decision yet in
    the window this snapshot covers -- a real, honest "no traffic yet"
    state, never defaulted to a tier."""

    tenant_id: str
    sensitivity_class: SensitivityClass
    pqc_capable: bool
    traffic_rate: float
    services: tuple[str, ...]
    last_served_tier: KeyType | None


@dataclass(frozen=True)
class RecentDecisionView:
    """One real decision from a real episode: which real tenant/service
    request it was, the real action chosen, and the real tier that
    action actually resolved to serving (`SmartKeyNetEnv.
    _resulting_key_type` -- the same ground-truth resolution the
    environment itself uses, never re-derived by this module)."""

    step: int
    tenant: str
    service: str
    sensitivity_class: SensitivityClass
    action: Action
    served_tier: KeyType


@dataclass(frozen=True)
class LivingSystemSnapshot:
    """One real, static moment in a real episode -- everything a
    rendered page needs, already resolved to real values."""

    label: str
    step: int
    pool_fill: float
    posture: ThreatPosture
    hub_id: str
    tenants: tuple[TenantNodeView, ...]
    recent_decisions: tuple[RecentDecisionView, ...]


def build_snapshot(
    *,
    label: str,
    hub_id: str,
    tenant_attrs: dict[str, dict],
    all_decisions: Sequence[RecentDecisionView],
    snapshot_index: int,
    pool_fill: float,
    posture: ThreatPosture,
    recent_window: int = 8,
) -> LivingSystemSnapshot:
    """Build one real, static snapshot as of `all_decisions[snapshot_index]`
    (inclusive) -- an ordinal cutoff into the real decision list, not a
    step-number threshold, since two decisions can legitimately share
    the same internal tick (one arrival queue can hold more than one
    pending request per tick); ordinal cutoff is unambiguous about
    "decisions so far" where a step-number cutoff would not be.

    `tenant_attrs`: `{tenant_id: node_attrs}` read directly off the
    real `nx.Graph` built by `build_tenant_graph` (never re-sampled or
    relabeled here). Every tenant node in the graph appears in the
    output, including ones this window never served (`last_served_tier
    =None`) -- the rendered node count is therefore always the real
    graph's real tenant-node count, never a fixed number.

    "Most recently served tier per tenant" is folded directly from
    `all_decisions[:snapshot_index + 1]`, in step order -- a later
    decision for a tenant always overwrites an earlier one, exactly
    matching "most recent" -- never a fabricated or default tier.
    """
    decisions_so_far = list(all_decisions[: snapshot_index + 1])
    if not decisions_so_far:
        raise ValueError("snapshot_index must select at least one real decision")

    last_tier_by_tenant: dict[str, KeyType] = {}
    for decision in decisions_so_far:
        last_tier_by_tenant[decision.tenant] = decision.served_tier

    tenants = tuple(
        TenantNodeView(
            tenant_id=tenant_id,
            sensitivity_class=SensitivityClass(int(attrs["sensitivity_class"])),
            pqc_capable=bool(attrs["pqc_capable"]),
            traffic_rate=float(attrs["traffic_rate"]),
            services=tuple(attrs["services"]),
            last_served_tier=last_tier_by_tenant.get(tenant_id),
        )
        for tenant_id, attrs in tenant_attrs.items()
    )
    recent = tuple(sorted(decisions_so_far, key=lambda d: d.step, reverse=True)[:recent_window])

    return LivingSystemSnapshot(
        label=label,
        step=decisions_so_far[-1].step,
        pool_fill=pool_fill,
        posture=posture,
        hub_id=hub_id,
        tenants=tenants,
        recent_decisions=recent,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_TIER_COLOR: dict[KeyType, str] = {
    KeyType.CLASSICAL: "#8B95A5",
    KeyType.PQC: "#E8A33D",
    KeyType.HYBRID: "#33D687",
}
_TIER_LEGEND_LABEL: dict[KeyType, str] = {
    KeyType.CLASSICAL: "Classical",
    KeyType.PQC: "PQC (ML-KEM-768)",
    KeyType.HYBRID: "Hybrid (+ QKD)",
}
_NO_TRAFFIC_COLOR = "#4C5A6B"  # matches render_explain.py's --text-faint token

_POSTURE_BADGE_CLASS: dict[ThreatPosture, str] = {
    ThreatPosture.CALM: "calm",
    ThreatPosture.ELEVATED: "elevated",
    ThreatPosture.HIGH: "high",
}

_HUB_CX, _HUB_CY = 320.0, 190.0
_HUB_RADIUS = 15.0
_TENANT_RING_RADIUS = 148.0
_TENANT_NODE_RADIUS = 11.0


def _e(value: object) -> str:
    """HTML-escape any dynamic value before inlining it."""
    return html.escape(str(value))


def _tenant_positions(n: int) -> list[tuple[float, float]]:
    """Deterministic hub-and-spoke ring layout for `n` real tenant
    nodes, evenly spaced around the hub starting at 12 o'clock. A real
    layout of however many real tenant nodes this graph actually has --
    never the mockup's fixed 6 hand-placed coordinates."""
    positions = []
    for i in range(n):
        angle = (2 * math.pi * i / n) - (math.pi / 2)
        cx = _HUB_CX + _TENANT_RING_RADIUS * math.cos(angle)
        cy = _HUB_CY + _TENANT_RING_RADIUS * math.sin(angle)
        positions.append((cx, cy))
    return positions


def _tier_color(tier: KeyType | None) -> str:
    return _TIER_COLOR[tier] if tier is not None else _NO_TRAFFIC_COLOR


def _render_graph_svg(snapshot: LivingSystemSnapshot) -> str:
    positions = _tenant_positions(len(snapshot.tenants))
    edges: list[str] = []
    nodes: list[str] = []
    for tenant, (cx, cy) in zip(snapshot.tenants, positions):
        color = _tier_color(tenant.last_served_tier)
        edges.append(
            f'<line x1="{_HUB_CX:.1f}" y1="{_HUB_CY:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" '
            f'stroke="{color}" stroke-width="2" data-tenant="{_e(tenant.tenant_id)}"></line>'
        )
        nodes.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{_TENANT_NODE_RADIUS}" fill="#151D27" '
            f'stroke="{color}" stroke-width="2" data-tenant="{_e(tenant.tenant_id)}" '
            f'data-tier="{_e(tenant.last_served_tier.name if tenant.last_served_tier is not None else "NONE")}"></circle>'
            f'<text x="{cx:.1f}" y="{cy - 18:.1f}" text-anchor="middle" class="node-label">{_e(tenant.tenant_id)}</text>'
        )
    hub = (
        f'<circle cx="{_HUB_CX}" cy="{_HUB_CY}" r="{_HUB_RADIUS}" fill="#151D27" '
        f'stroke="#6E7EFF" stroke-width="2"></circle>'
        f'<text x="{_HUB_CX}" y="{_HUB_CY + 4}" text-anchor="middle" class="node-label hub">{_e(snapshot.hub_id)}</text>'
    )
    return (
        '<svg viewBox="0 0 640 380" xmlns="http://www.w3.org/2000/svg">'
        + "".join(edges)
        + hub
        + "".join(nodes)
        + "</svg>"
    )


def _render_legend() -> str:
    swatches = "".join(
        f'<div class="legend-item"><span class="legend-swatch" style="background:{color}"></span>{_e(label)}</div>'
        for tier, color in _TIER_COLOR.items()
        for label in [_TIER_LEGEND_LABEL[tier]]
    )
    swatches += (
        f'<div class="legend-item"><span class="legend-swatch" style="background:{_NO_TRAFFIC_COLOR}"></span>'
        "No traffic yet</div>"
    )
    return f'<div class="graph-legend">{swatches}</div>'


def _render_recent_decisions(snapshot: LivingSystemSnapshot) -> str:
    rows: list[str] = []
    for decision in snapshot.recent_decisions:
        color = _TIER_COLOR[decision.served_tier]
        rows.append(
            f"""
          <div class="req-row">
            <span class="req-dot" style="background:{color}"></span>
            <span class="req-tenant">{_e(decision.tenant)}</span>
            <span class="req-service">{_e(decision.service)}</span>
            <span class="req-sens">{_e(decision.sensitivity_class.name)}</span>
            <span class="req-action">{_e(decision.action.name)}</span>
            <span class="req-tier">{_e(decision.served_tier.name)}</span>
            <span class="req-step">t={decision.step}</span>
          </div>"""
        )
    return f'<div class="req-list">{"".join(rows)}</div>'


def _render_side_stack(snapshot: LivingSystemSnapshot) -> str:
    pool_pct = snapshot.pool_fill * 100.0
    posture_cls = _POSTURE_BADGE_CLASS[snapshot.posture]
    return f"""
      <div class="side-stack">
        <div class="card">
          <div class="card-label"><span>QKD pool</span><span>bits available</span></div>
          <div class="gauge-track"><div class="gauge-fill" style="width:{pool_pct:.1f}%"></div></div>
          <div class="gauge-row"><span class="gauge-big">{pool_pct:.0f}%</span><span class="gauge-meta">pool_fill (real, normalized)</span></div>
        </div>
        <div class="card">
          <div class="card-label"><span>Threat posture</span><span class="badge {posture_cls}">{_e(snapshot.posture.name.lower())}</span></div>
          <div class="gauge-meta">resolved posture at t={snapshot.step} (real, from the forecaster's argmax)</div>
        </div>
      </div>"""


_CSS = """
:root{
  --bg:#0A0E13; --panel:#111820; --panel-2:#151D27; --line:#212C39; --line-soft:#171F29;
  --text:#E9EEF4; --text-dim:#8FA0B3; --text-faint:#4C5A6B;
  --classical:#8B95A5; --pqc:#E8A33D; --hybrid:#33D687; --quantum:#6E7EFF;
  --calm:#4C5A6B; --elevated:#E8A33D; --high:#FF5C5C;
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
.wrap{max-width:960px;margin:0 auto;}
.beat-head{margin-bottom:22px;}
.beat-eyebrow{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--quantum);}
.beat-title{font-family:var(--disp);font-weight:700;font-size:22px;margin-top:4px;}
.beat-desc{color:var(--text-dim);font-size:13px;margin-top:6px;line-height:1.5;}

.grid-1{display:grid;grid-template-columns:1.6fr 1fr;gap:18px;align-items:start;}
@media (max-width:820px){.grid-1{grid-template-columns:1fr;}}

.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px 17px;}
.card-label{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--text-faint);margin-bottom:10px;}

.graph-wrap svg{width:100%;height:auto;display:block;}
.node-label{font-family:var(--mono);font-size:10px;fill:var(--text-dim);}
.node-label.hub{fill:var(--quantum);font-weight:700;}

.graph-legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;font-size:11px;color:var(--text-dim);}
.legend-item{display:flex;align-items:center;gap:6px;}
.legend-swatch{width:10px;height:10px;border-radius:3px;display:inline-block;}

.req-list{display:flex;flex-direction:column;gap:6px;max-height:320px;overflow-y:auto;}
.req-row{
  display:grid;grid-template-columns:8px 78px 1fr 34px 90px 66px 56px;align-items:center;gap:8px;
  font-family:var(--mono);font-size:10.5px;color:var(--text-dim);
  background:var(--panel-2);border:1px solid var(--line-soft);border-radius:6px;padding:7px 9px;
}
.req-dot{width:8px;height:8px;border-radius:50%;}
.req-tenant{color:var(--text);font-weight:600;}
.req-service{color:var(--text-faint);}
.req-sens{color:var(--text-faint);}
.req-tier{text-align:right;}
.req-step{color:var(--text-faint);text-align:right;}

.side-stack{display:flex;flex-direction:column;gap:14px;}
.gauge-track{height:10px;border-radius:999px;background:var(--panel-2);border:1px solid var(--line);overflow:hidden;}
.gauge-fill{height:100%;border-radius:999px;background:var(--quantum);}
.gauge-row{display:flex;justify-content:space-between;align-items:baseline;margin-top:8px;}
.gauge-big{font-family:var(--mono);font-weight:700;font-size:20px;}
.gauge-meta{font-size:11px;color:var(--text-faint);}

.badge{font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.05em;padding:3px 8px;border-radius:999px;}
.badge.calm{background:rgba(76,90,107,.25);color:var(--text-dim);}
.badge.elevated{background:rgba(232,163,61,.18);color:var(--pqc);}
.badge.high{background:rgba(255,92,92,.18);color:var(--high);}
"""


def render_living_system_html(
    snapshot: LivingSystemSnapshot, *, title: str = "SmartKeyNet -- Living System"
) -> str:
    """Render one real `LivingSystemSnapshot` as a self-contained HTML
    page. Pure view: reads `snapshot`'s fields only (plus the two
    documented derived-layout/color functions above), writes nothing
    back, invents no tenant identity or tier (Hard Rule 10)."""
    graph_svg = _render_graph_svg(snapshot)
    legend = _render_legend()
    recent = _render_recent_decisions(snapshot)
    side_stack = _render_side_stack(snapshot)

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
    <div class="beat-eyebrow">{_e(snapshot.label)} &middot; t={snapshot.step}</div>
    <div class="beat-title">The living system</div>
    <div class="beat-desc">A real, static snapshot of the real tenant graph -- edge/node color is the key tier each tenant was most recently served at, as of this exact moment in one real episode. Not animated; not live.</div>
  </div>
  <div class="grid-1">
    <div class="card graph-wrap">
      <div class="card-label"><span>Tenant network ({len(snapshot.tenants)} tenants)</span><span>t={snapshot.step}</span></div>
      {graph_svg}
      {legend}
      <div class="card-label" style="margin-top:22px;"><span>Recent decisions</span><span>most recent {len(snapshot.recent_decisions)}</span></div>
      {recent}
    </div>
    {side_stack}
  </div>
</main>
</body>
</html>
"""


def write_living_system_html(
    snapshot: LivingSystemSnapshot, path: str | Path, *, title: str = "SmartKeyNet -- Living System"
) -> Path:
    """Render `snapshot` and write it to `path` (parent dirs created as
    needed). Returns the written `Path`."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_living_system_html(snapshot, title=title), encoding="utf-8")
    return out_path
