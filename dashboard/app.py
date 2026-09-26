"""
dashboard/app.py

Minimal local web server for the already-rendered dashboard panels in
`dashboard/samples/` (Living System -- S1 and S2, Explain Decision,
dose-response, S3 comparison table, Budgeting Brain, Migration Wave --
see README.md's "How to see the results" table, the source this
module's `PANELS` registry mirrors). Turns "open a dozen separate HTML
files by hand" into "run one command, open one URL." Living System is
four panel cards, not one -- S1 (steady/calm traffic) and S2 (elevated
HNDL posture) under the demo's original `StaticThresholdPolicy`, plus a
trained-masked-DQN-driven variant of each (`checkpoints/dqn_s1.pt` /
`dqn_s2.pt`) -- four distinct real episodes through the same real
pipeline, added alongside each other so the demo shows both the
StaticThresholdPolicy saturation finding and the genuine tier variety a
trained agent produces (see `dashboard/render_living_system_demo.py`'s
module docstring for the full honest finding on what each one actually
shows).

This is a presentation-layer convenience over already-rendered, real
static artifacts -- it computes nothing and runs no episode. It does
not stream a live `SmartKeyNetEnv` (PLAN2.md Hard Rule 11 scopes live
capture/streaming out) and does not regenerate panels on request; it
only serves the committed files as they exist on disk right now,
byte-for-byte.

Deliberately standard-library only (`http.server`), for the same
zero-new-dependency reason `dashboard/render_explain.py` gives for its
own no-`dash`/no-`plotly` choice: serving ten already-rendered static
files is disproportionate to either dependency.

Route surface is a small explicit whitelist (`PANELS` below), not a
general static file server over `dashboard/samples/` -- so there is no
path-traversal surface to reason about, and a typo'd or removed sample
file fails loudly (404) rather than silently serving something outside
the registry.

Two entry points, both served over the same route whitelist: `/`
(default, aliased with `/v2`/`/v2.html` -- the tabbed shell styled
after `dashboard/mockups/smartkeynet_dashboard_mockup_v2.html`'s
visual language -- dark theme, tab nav, tier-color legend -- that
embeds each real sample file via a same-origin iframe rather than
re-rendering it, so the honesty-guard-tested markup each panel's own
render_*.py driver produced is shown byte-for-byte) and `/classic`
(aliased with `/classic.html` -- the original card-grid index, kept
as an explicit fallback, e.g. for a presentation where the tabbed
shell needs a known-good backstop). Both read from the same `PANELS`
registry, so neither can drift from the other or from the mockup's
own fabricated placeholder data. Flipped to `/v2` as the default
2026-09-14 after a parity audit confirmed no panel, description,
honesty framing, legend, or navigation path present on the old grid
was missing from the tabbed shell -- see PROGRESS.md's newest entry.

Run directly: `python -m dashboard.app` (prints the URL to open).
"""

from __future__ import annotations

import html
import http.server
import json
import urllib.parse
from pathlib import Path
from typing import NamedTuple

from dashboard.render_upload import render_upload_error_html, render_upload_form_html, render_upload_result_html
from dashboard.upload_demo import InvalidPcapError, run_pcap_upload_demo

_SAMPLES_DIR = Path(__file__).resolve().parent / "samples"


class PanelFile(NamedTuple):
    filename: str
    label: str


class Panel(NamedTuple):
    title: str
    description: str
    files: tuple[PanelFile, ...]


# Mirrors README.md's "How to see the results" table exactly -- real
# filenames confirmed against `dashboard/samples/` on disk, not assumed.
PANELS: tuple[Panel, ...] = (
    Panel(
        "Living System -- S1 (steady/calm)",
        "The same real tenant graph and policy under S1 (steady, unscripted traffic), driven by the real, trained forecaster (RTIoT2022ReplayProvider) -- shown honestly: this run still saturates to hybrid on every real decision, but for a single, now-isolated cause -- StaticThresholdPolicy itself serves hybrid whenever pool_fill > 0.5, and pool_fill never drops that low here; posture stays CALM throughout under the real forecaster (no longer an artifact-driven ratchet).",
        (
            PanelFile("living_system_s1_01_first_decision.html", "1. First decision"),
            PanelFile("living_system_s1_02_graph_fully_populated.html", "2. Graph fully populated"),
            PanelFile("living_system_s1_03_final_decision.html", "3. Final decision"),
        ),
    ),
    Panel(
        "Living System -- S2 (elevated HNDL)",
        "A tenant service graph under S2 (HNDL posture); nodes/edges colored by the tier each tenant was most recently served.",
        (
            PanelFile("living_system_01_first_decision.html", "1. First decision"),
            PanelFile("living_system_02_graph_fully_populated.html", "2. Graph fully populated"),
            PanelFile("living_system_03_final_decision.html", "3. Final decision"),
        ),
    ),
    Panel(
        "Living System -- S1, trained DQN",
        "Same real tenant graph/episode as the S1 card above (real RTIoT2022ReplayProvider forecaster), but driven by a trained masked-DQN checkpoint (checkpoints/dqn_s1.pt, retrained under the real forecaster) instead of StaticThresholdPolicy -- genuine tier variety (250 real decisions: 68 classical, 83 hybrid, 99 PQC), unlike that card's 100% hybrid saturation.",
        (
            PanelFile("living_system_s1_dqn_01_first_decision.html", "1. First decision"),
            PanelFile("living_system_s1_dqn_02_graph_fully_populated.html", "2. Graph fully populated"),
            PanelFile("living_system_s1_dqn_03_final_decision.html", "3. Final decision"),
        ),
    ),
    Panel(
        "Living System -- S2, trained DQN",
        "Same real tenant graph/episode as the S2 card above (real LSTMForecastProvider), but driven by a trained masked-DQN checkpoint (checkpoints/dqn_s2.pt, retrained under the real forecaster) instead of StaticThresholdPolicy -- genuine tier variety (250 real decisions: 6 classical, 163 hybrid, 81 PQC), unlike that card's 100% hybrid saturation.",
        (
            PanelFile("living_system_s2_dqn_01_first_decision.html", "1. First decision"),
            PanelFile("living_system_s2_dqn_02_graph_fully_populated.html", "2. Graph fully populated"),
            PanelFile("living_system_s2_dqn_03_final_decision.html", "3. Final decision"),
        ),
    ),
    Panel(
        "Explain Decision",
        "The floor lookup, action mask legality, cost comparison, and final decision for three real, structurally-different S2 decisions, under the real, trained LSTMForecastProvider and the recalibrated floor table.",
        (
            PanelFile("01_first_decision.html", "1. First decision"),
            PanelFile("02_floor_driven_only_hybrid_clears.html", "2. Floor-driven (only HYBRID clears)"),
            PanelFile("03_real_cost_tradeoff.html", "3. Real cost tradeoff"),
        ),
    ),
    Panel(
        "Dose-Response",
        "The steering-attack headline result: V(pi) vs. attack strength alpha, masked agent vs. soft-reward baseline -- real, trained LSTMForecastProvider (cold-start fixed), a two-plateau step curve, not the earlier placeholder-forecaster sweep's smooth rise.",
        (PanelFile("dose_response_chart.html", "Open panel"),),
    ),
    Panel(
        "S3 Comparison Table",
        "Masked DQN vs. soft-reward baseline on S3 (QKD degradation), real RTIoT2022ReplayProvider forecaster + recalibrated floor table: p99 latency, total reward, rekey ratio, regret/exhaustion, below-floor rate.",
        (PanelFile("s3_comparison_table.html", "Open panel"),),
    ),
    Panel(
        "Budgeting Brain",
        "Real S3 pool-trajectory comparison, masked DQN vs. AlwaysHybridPolicy -- same seed, same conditions, real exhaustion-event markers, real RTIoT2022ReplayProvider forecaster.",
        (PanelFile("budgeting_brain.html", "Open panel"),),
    ),
    Panel(
        "Migration Wave",
        "S6's three scripted floor-ratchet events against a real held-out episode (real RTIoT2022ReplayProvider forecaster) from an agent never trained on S6, with an honesty-gated attribution.",
        (PanelFile("migration_wave.html", "Open panel"),),
    ),
    Panel(
        "Posture Validation",
        "A real 100-decision validation (pcap-calm + real held-out RT-IoT2022 benign->attack transition + steady-state attack) confirming the upload demo's higher-threat_score/lower-floor anecdote is expected: CALM 66 / ELEVATED 4 / HIGH 30, zero floor-monotonicity violations, and the honest finding that ELEVATED posture is real but only transiently reachable.",
        (PanelFile("posture_validation.html", "Open panel"),),
    ),
    Panel(
        "National Backbone -- Illustrative Scenario",
        "Illustrative: one shared key pool serving four labeled government agencies (Defense, Health, Tax, Passport Office), with the Passport Office reclassified S1 -> S3 mid-run via S6's existing mechanism. Real held-out episode; agency names are display labels only, tiers are the real seed-30 assignment. Honest limit stated on the panel: the pool never came under pressure in this run.",
        (PanelFile("national_backbone.html", "Open panel"),),
    ),
)

# Flat filename -> on-disk path, the server's actual route whitelist.
_ROUTES: dict[str, Path] = {
    panel_file.filename: _SAMPLES_DIR / panel_file.filename for panel in PANELS for panel_file in panel.files
}


_TIER_LEGEND: tuple[tuple[str, str], ...] = (
    ("Classical", "#8B95A5"),
    ("PQC (ML-KEM-768)", "#E8A33D"),
    ("Hybrid (+ QKD)", "#33D687"),
)


def render_index_html() -> str:
    """Pure function: build the index page linking all panels.
    No file I/O, no server -- safe to call directly in tests."""
    cards = []
    for i, panel in enumerate(PANELS, start=1):
        links = "\n".join(
            f'<a class="panel-link" href="/samples/{html.escape(pf.filename)}">{html.escape(pf.label)}'
            '<span class="arrow">&rarr;</span></a>'
            for pf in panel.files
        )
        cards.append(
            f"""<div class="card">
<div class="card-label"><span>{i:02d}</span><span>{len(panel.files)} file{"s" if len(panel.files) != 1 else ""}</span></div>
<div class="card-title">{html.escape(panel.title)}</div>
<div class="card-desc">{html.escape(panel.description)}</div>
<div class="card-links">{links}</div>
</div>"""
        )
    cards_html = "\n".join(cards)

    legend_html = "\n".join(
        f'<div class="legend-item"><span class="legend-swatch" style="background:{color}"></span>{html.escape(label)}</div>'
        for label, color in _TIER_LEGEND
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartKeyNet -- Dashboard</title>
<style>
:root{{
  --bg:#0A0E13; --panel:#111820; --panel-2:#151D27; --line:#212C39; --line-soft:#171F29;
  --text:#E9EEF4; --text-dim:#8FA0B3; --text-faint:#4C5A6B; --quantum:#6E7EFF;
  --classical:#8B95A5; --pqc:#E8A33D; --hybrid:#33D687;
  --radius:12px; --radius-sm:7px;
  --mono:ui-monospace,SFMono-Regular,Consolas,'Courier New',monospace;
  --disp:-apple-system,'Segoe UI',Roboto,sans-serif;
}}
*{{box-sizing:border-box;}}
html,body{{margin:0;padding:0;}}
body{{
  background:
    radial-gradient(circle at 1px 1px, #16202C 1px, transparent 0) 0 0/28px 28px,
    var(--bg);
  color:var(--text);font-family:var(--disp);min-height:100vh;
}}
a{{color:inherit;}}

.topbar{{
  display:flex;align-items:center;gap:12px;padding:18px 24px;
  border-bottom:1px solid var(--line);
  background:linear-gradient(180deg, rgba(17,24,32,.9), rgba(17,24,32,.55));
}}
.brand-name{{font-weight:700;font-size:16px;letter-spacing:.05em;}}
.brand-sub{{font-family:var(--mono);font-size:10px;color:var(--text-faint);letter-spacing:.12em;text-transform:uppercase;margin-top:1px;}}

main{{max-width:1080px;margin:0 auto;padding:36px 22px 60px;}}
.thesis{{
  font-size:16px;line-height:1.55;max-width:680px;margin:0 0 6px;color:var(--text);
}}
.thesis b{{color:var(--quantum);}}
.subtitle{{color:var(--text-dim);font-size:13px;margin-bottom:22px;}}

.legend-row{{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:28px;}}
.legend-item{{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--text-dim);}}
.legend-swatch{{width:10px;height:10px;border-radius:3px;display:inline-block;}}

.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px 17px;transition:border-color .15s ease;}}
.card:hover{{border-color:#2C3B4C;}}
.card-label{{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--text-faint);margin-bottom:10px;}}
.card-title{{font-weight:700;font-size:15px;margin-bottom:6px;}}
.card-desc{{color:var(--text-dim);font-size:12.5px;line-height:1.5;margin-bottom:14px;}}
.card-links{{display:flex;flex-direction:column;gap:2px;border-top:1px solid var(--line-soft);padding-top:10px;}}
.panel-link{{
  display:flex;align-items:center;justify-content:space-between;gap:8px;
  color:var(--quantum);font-family:var(--mono);font-size:11.5px;text-decoration:none;
  padding:6px 2px;border-radius:var(--radius-sm);
}}
.panel-link:hover{{background:var(--panel-2);}}
.panel-link .arrow{{color:var(--text-faint);}}

footer{{max-width:1080px;margin:8px auto 0;padding:16px 22px 0;font-family:var(--mono);font-size:10.5px;color:var(--text-faint);border-top:1px solid var(--line);}}
</style>
</head>
<body>
<header class="topbar">
  <svg width="26" height="26" viewBox="0 0 30 30" fill="none">
    <polygon points="15,1.5 27,8.25 27,21.75 15,28.5 3,21.75 3,8.25" stroke="#6E7EFF" stroke-width="1.6"/>
    <path d="M11 15a4 4 0 1 1 4 4v4.2M11 15a4 4 0 0 1 4-4M15 19v-1.4" stroke="#6E7EFF" stroke-width="1.6" stroke-linecap="round"/>
    <circle cx="19.3" cy="10.7" r="1.4" fill="#6E7EFF"/>
  </svg>
  <div>
    <div class="brand-name">SMARTKEYNET</div>
    <div class="brand-sub">Hybrid KMS &middot; Decision Layer</div>
  </div>
</header>
<main>
<p class="thesis">Security is enforced as a <b>hard, structural constraint</b> &mdash; action masking, never a term in the reward.</p>
<div class="subtitle">Real, pre-rendered panels from dashboard/samples/, served as-is &mdash; nothing computed live, nothing in these pages is a placeholder. This is the classic fallback view; <a href="/" style="color:var(--quantum);">the default tabbed view is here &rarr;</a></div>
<div class="legend-row">{legend_html}</div>
<div class="grid">
{cards_html}
</div>
</main>
<footer>SmartKeyNet &mdash; decision layer for a multi-tenant hybrid-crypto KMS</footer>
</body>
</html>"""


def _v2_tab_groups() -> tuple[tuple[str, tuple[Panel, ...]], ...]:
    """Groups PANELS into v2-shell tabs: every "Living System -- ..."
    panel collapses into one "Living System" tab (switched client-side
    via a variant selector); every other panel gets its own tab.
    Derived directly from PANELS' own titles -- there is no separate,
    hardcoded tab/file list that could silently drift out of sync with
    the real registry `render_index_html()` already uses."""
    living = tuple(p for p in PANELS if p.title.startswith("Living System"))
    rest = tuple(p for p in PANELS if not p.title.startswith("Living System"))
    return (("Living System", living),) + tuple((p.title, (p,)) for p in rest)


def _v2_panel_data() -> list[dict]:
    """PANELS, reshaped into the tab/variant/file JSON the v2 shell's
    client-side JS switches between -- filenames and descriptions are
    copied verbatim from PANELS, nothing is re-derived or fabricated."""
    tabs = []
    for tab_label, group_panels in _v2_tab_groups():
        multi = len(group_panels) > 1
        variants = []
        for p in group_panels:
            variant_label = p.title.split(" -- ", 1)[1] if multi and " -- " in p.title else None
            variants.append(
                {
                    "variantLabel": variant_label,
                    "description": p.description,
                    "files": [{"filename": pf.filename, "label": pf.label} for pf in p.files],
                }
            )
        tabs.append({"tabLabel": tab_label, "variants": variants})
    return tabs


def render_index_v2_html() -> str:
    """Pure function: build the tabbed-shell entry point, styled after
    dashboard/mockups/smartkeynet_dashboard_mockup_v2.html's visual
    language (dark theme, tab nav, tier-color legend, card/typography
    system) -- but every tab/sub-tab/file comes from `_v2_panel_data()`,
    i.e. straight from the real PANELS registry `render_index_html()`
    already uses, never from the mockup itself. The mockup remains a
    style reference only: no number, finding, or fabricated data point
    from it is copied here. Each real sample file is embedded via a
    same-origin iframe (auto-resized client-side) rather than
    re-rendered -- so the honesty-guard-tested markup each panel's own
    render_*.py driver already produced is shown byte-for-byte, not
    reimplemented in parallel. No file I/O, no server -- safe to call
    directly in tests, matching `render_index_html()`'s own contract."""
    tabs = _v2_panel_data()
    tabs_json = json.dumps(tabs).replace("</", "<\\/")

    legend_html = "\n".join(
        f'<div class="legend-item"><span class="legend-swatch" style="background:{color}"></span>{html.escape(label)}</div>'
        for label, color in _TIER_LEGEND
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartKeyNet -- Dashboard</title>
<style>
:root{{
  --bg:#0A0E13; --panel:#111820; --panel-2:#151D27; --line:#212C39; --line-soft:#171F29;
  --text:#E9EEF4; --text-dim:#8FA0B3; --text-faint:#4C5A6B; --quantum:#6E7EFF;
  --classical:#8B95A5; --pqc:#E8A33D; --hybrid:#33D687;
  --radius:12px; --radius-sm:7px;
  --mono:ui-monospace,SFMono-Regular,Consolas,'Courier New',monospace;
  --disp:-apple-system,'Segoe UI',Roboto,sans-serif;
}}
*{{box-sizing:border-box;}}
html,body{{margin:0;padding:0;}}
body{{
  background:
    radial-gradient(circle at 1px 1px, #16202C 1px, transparent 0) 0 0/28px 28px,
    var(--bg);
  color:var(--text);font-family:var(--disp);min-height:100vh;
}}
a{{color:inherit;}}
button{{font-family:inherit;}}

.topbar{{
  display:flex;align-items:center;gap:14px;padding:14px 22px;
  border-bottom:1px solid var(--line);
  background:linear-gradient(180deg, rgba(17,24,32,.9), rgba(17,24,32,.55));
  position:sticky;top:0;z-index:50;flex-wrap:wrap;
}}
.brand{{display:flex;align-items:center;gap:10px;margin-right:auto;}}
.brand-name{{font-weight:700;font-size:16px;letter-spacing:.05em;}}
.brand-sub{{font-family:var(--mono);font-size:10px;color:var(--text-faint);letter-spacing:.12em;text-transform:uppercase;margin-top:1px;}}
.pill{{
  font-family:var(--mono);font-size:11px;color:var(--text-dim);
  background:var(--panel-2);border:1px solid var(--line);
  padding:5px 10px;border-radius:999px;white-space:nowrap;
}}
.classic-link{{
  font-family:var(--mono);font-size:11px;color:var(--quantum);text-decoration:none;
  border:1px solid rgba(110,126,255,.35);border-radius:999px;padding:5px 10px;white-space:nowrap;
}}
.classic-link:hover{{background:rgba(110,126,255,.1);}}

.tab-nav{{display:flex;gap:2px;padding:0 22px;border-bottom:1px solid var(--line);overflow-x:auto;}}
.tab-btn{{
  font-family:var(--disp);font-weight:600;font-size:13px;
  color:var(--text-faint);background:transparent;border:none;
  padding:12px 14px 10px;cursor:pointer;border-bottom:2px solid transparent;
  display:flex;align-items:center;gap:7px;white-space:nowrap;
}}
.tab-btn .n{{font-family:var(--mono);font-size:10px;color:var(--text-faint);border:1px solid var(--line);border-radius:4px;padding:1px 5px;}}
.tab-btn:hover{{color:var(--text-dim);}}
.tab-btn.active{{color:var(--text);border-bottom-color:var(--quantum);}}
.tab-btn.active .n{{border-color:var(--quantum);color:var(--quantum);}}

main{{max-width:1200px;margin:0 auto;padding:24px 22px 60px;}}
.tab-head{{margin-bottom:16px;}}
.tab-title-row{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}}
.tab-title{{font-weight:700;font-size:22px;letter-spacing:-.01em;}}
.file-count-badge{{font-family:var(--mono);font-size:10.5px;color:var(--text-faint);border:1px solid var(--line);border-radius:4px;padding:2px 7px;text-transform:uppercase;letter-spacing:.08em;}}
.tab-desc{{color:var(--text-dim);font-size:13px;max-width:760px;margin-top:6px;line-height:1.55;}}

.chip-row{{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;}}
.chip-btn{{
  font-family:var(--mono);font-size:11px;color:var(--text-dim);background:var(--panel-2);
  border:1px solid var(--line);border-radius:999px;padding:6px 13px;cursor:pointer;
}}
.chip-btn.active{{color:#0A0E13;background:var(--quantum);border-color:var(--quantum);font-weight:600;}}

.iframe-card{{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);overflow:hidden;}}
.iframe-card iframe{{display:block;width:100%;height:900px;border:none;background:var(--bg);}}

.legend-row{{display:flex;gap:16px;flex-wrap:wrap;}}
.legend-item{{display:flex;align-items:center;gap:7px;font-size:11.5px;color:var(--text-dim);}}
.legend-swatch{{width:10px;height:10px;border-radius:3px;display:inline-block;}}

footer{{max-width:1200px;margin:14px auto 0;padding:16px 22px 0;font-family:var(--mono);font-size:10.5px;color:var(--text-faint);border-top:1px solid var(--line);display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;}}
</style>
</head>
<body>
<header class="topbar">
  <div class="brand">
    <div>
      <div class="brand-name">SMARTKEYNET</div>
      <div class="brand-sub">Hybrid KMS &middot; Decision Layer</div>
    </div>
  </div>
  <span class="pill">Static, pre-rendered panels</span>
  <span class="pill">Nothing computed live &middot; nothing is a placeholder</span>
  <a class="classic-link" href="/upload">Upload a pcap (live) &rarr;</a>
  <a class="classic-link" href="/classic">Classic card view</a>
</header>

<nav class="tab-nav" id="tabNav"></nav>

<main>
  <div class="tab-head">
    <div class="tab-title-row">
      <div class="tab-title" id="tabTitle"></div>
      <span class="file-count-badge" id="fileCountBadge"></span>
    </div>
    <div class="tab-desc" id="tabDesc"></div>
  </div>
  <div class="chip-row" id="variantNav"></div>
  <div class="chip-row" id="momentNav"></div>
  <div class="iframe-card"><iframe id="panelFrame" title="panel"></iframe></div>
  <div class="legend-row" style="margin-top:18px;">{legend_html}</div>
</main>

<footer>
  <span>SmartKeyNet &mdash; decision layer for a multi-tenant hybrid-crypto KMS</span>
  <span>Visual shell only &mdash; every panel embedded here is the exact file linked from the classic view</span>
</footer>

<script>
  const TABS = {tabs_json};
  let activeTab = 0, activeVariant = 0, activeMoment = 0;

  const tabNav = document.getElementById('tabNav');
  const variantNav = document.getElementById('variantNav');
  const momentNav = document.getElementById('momentNav');
  const tabTitle = document.getElementById('tabTitle');
  const tabDesc = document.getElementById('tabDesc');
  const fileCountBadge = document.getElementById('fileCountBadge');
  const frame = document.getElementById('panelFrame');

  function renderTabNav(){{
    tabNav.innerHTML = '';
    TABS.forEach((t, i) => {{
      const btn = document.createElement('button');
      btn.className = 'tab-btn' + (i === activeTab ? ' active' : '');
      btn.innerHTML = '<span class="n">' + String(i + 1).padStart(2, '0') + '</span> ' + t.tabLabel;
      btn.addEventListener('click', () => selectTab(i));
      tabNav.appendChild(btn);
    }});
  }}

  function renderVariantNav(){{
    const variants = TABS[activeTab].variants;
    variantNav.innerHTML = '';
    variantNav.hidden = variants.length < 2;
    variants.forEach((v, i) => {{
      const btn = document.createElement('button');
      btn.className = 'chip-btn' + (i === activeVariant ? ' active' : '');
      btn.textContent = v.variantLabel || TABS[activeTab].tabLabel;
      btn.addEventListener('click', () => {{ activeVariant = i; activeMoment = 0; renderVariantNav(); renderMomentNav(); renderFrame(); }});
      variantNav.appendChild(btn);
    }});
  }}

  function renderMomentNav(){{
    const files = TABS[activeTab].variants[activeVariant].files;
    momentNav.innerHTML = '';
    momentNav.hidden = files.length < 2;
    files.forEach((f, i) => {{
      const btn = document.createElement('button');
      btn.className = 'chip-btn' + (i === activeMoment ? ' active' : '');
      btn.textContent = f.label;
      btn.addEventListener('click', () => {{ activeMoment = i; renderMomentNav(); renderFrame(); }});
      momentNav.appendChild(btn);
    }});
  }}

  function renderFrame(){{
    const variant = TABS[activeTab].variants[activeVariant];
    const file = variant.files[activeMoment];
    tabDesc.textContent = variant.description;
    fileCountBadge.textContent = variant.files.length + (variant.files.length === 1 ? ' FILE' : ' FILES');
    frame.src = '/samples/' + file.filename;
  }}

  function selectTab(i){{
    activeTab = i; activeVariant = 0; activeMoment = 0;
    renderTabNav();
    tabTitle.textContent = TABS[activeTab].tabLabel;
    renderVariantNav();
    renderMomentNav();
    renderFrame();
  }}

  frame.addEventListener('load', () => {{
    try {{
      const doc = frame.contentWindow.document;
      const h = Math.max(doc.body.scrollHeight, doc.documentElement.scrollHeight);
      frame.style.height = (h + 20) + 'px';
    }} catch (e) {{
      frame.style.height = '900px';
    }}
  }});

  selectTab(0);
</script>
</body>
</html>"""


def resolve_route(path: str) -> tuple[int, str, bytes]:
    """Pure routing logic: request path -> (status, content_type, body).
    No socket I/O -- the handler below is a thin adapter over this.

    `/` is aliased to the tabbed `/v2` shell (the default, since the
    2026-09-14 parity audit) -- `/index.html` follows `/` rather than
    the classic grid, so "index" always means "whatever's current
    default." The classic card-grid index remains reachable, unchanged,
    at the explicit `/classic` fallback path -- not deleted, since it's
    a presentation-critical known-good backstop."""
    if path in ("/", "/index.html", "/v2", "/v2.html"):
        return 200, "text/html; charset=utf-8", render_index_v2_html().encode("utf-8")

    if path in ("/classic", "/classic.html"):
        return 200, "text/html; charset=utf-8", render_index_html().encode("utf-8")

    if path in ("/upload", "/upload.html"):
        return 200, "text/html; charset=utf-8", render_upload_form_html().encode("utf-8")

    if path.startswith("/samples/"):
        filename = path[len("/samples/") :]
        file_path = _ROUTES.get(filename)
        if file_path is not None and file_path.is_file():
            return 200, "text/html; charset=utf-8", file_path.read_bytes()

    return 404, "text/plain; charset=utf-8", b"Not Found"


def resolve_upload_post(path: str, body: bytes, filename_header: str | None) -> tuple[int, str, bytes]:
    """Pure POST handler for the v2 Phase 2 upload-and-run demo
    (`dashboard/upload_demo.py`/`dashboard/render_upload.py`), mirroring
    `resolve_route`'s "pure function, no socket" testability contract.

    The uploaded pcap's raw bytes are sent as the request body directly
    (not multipart/form-data) -- this server stays standard-library
    only (no `cgi`, deprecated/slated for removal; no new dependency),
    matching `dashboard/app.py`'s own module docstring's
    zero-new-dependency convention. `filename_header` (the request's
    `X-Filename` header, if any) is display-only.

    Any real, honest failure (`InvalidPcapError` -- not a valid pcap,
    zero extractable flows, oversized upload) renders a plain error
    page (400) instead of a fabricated result, per this session's Step
    5 instruction."""
    if path not in ("/upload", "/upload.html"):
        return 404, "text/plain; charset=utf-8", b"Not Found"

    filename = urllib.parse.unquote(filename_header) if filename_header else "upload.pcap"

    try:
        result = run_pcap_upload_demo(body, filename)
    except InvalidPcapError as exc:
        return 400, "text/html; charset=utf-8", render_upload_error_html(filename, str(exc)).encode("utf-8")

    return 200, "text/html; charset=utf-8", render_upload_result_html(result).encode("utf-8")


class DashboardRequestHandler(http.server.BaseHTTPRequestHandler):
    """Thin adapter: delegates all routing/content decisions to
    `resolve_route` so that logic is testable without binding a socket."""

    def do_GET(self) -> None:  # noqa: N802 (stdlib-mandated name)
        status, content_type, body = resolve_route(self.path)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 (stdlib-mandated name)
        content_length = int(self.headers.get("Content-Length", 0))
        request_body = self.rfile.read(content_length) if content_length > 0 else b""
        status, content_type, body = resolve_upload_post(self.path, request_body, self.headers.get("X-Filename"))
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # keep the terminal quiet; this is a local convenience server


def build_server(host: str = "127.0.0.1", port: int = 8000) -> http.server.HTTPServer:
    """Construct the server (does not start serving). Port 0 lets the
    OS pick a free ephemeral port, e.g. for tests."""
    return http.server.HTTPServer((host, port), DashboardRequestHandler)


def main() -> None:
    server = build_server()
    host, port = server.server_address[:2]
    print(f"SmartKeyNet dashboard running at http://{host}:{port}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
