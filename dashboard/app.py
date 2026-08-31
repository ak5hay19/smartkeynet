"""
dashboard/app.py

Minimal local web server for the already-rendered dashboard panels in
`dashboard/samples/` (Living System -- S1 and S2, Explain Decision,
dose-response, S3 comparison table, Budgeting Brain, Migration Wave --
see README.md's "How to see the results" table, the source this
module's `PANELS` registry mirrors). Turns "open a dozen separate HTML
files by hand" into "run one command, open one URL." Living System is
two panel cards, not one -- S1 (steady/calm traffic) and S2 (elevated
HNDL posture) are two distinct real episodes through the same real
pipeline, added alongside each other so the demo shows both (see
`dashboard/render_living_system_demo.py`'s module docstring for why,
and its honest finding on what each one actually shows).

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

Run directly: `python -m dashboard.app` (prints the URL to open).
"""

from __future__ import annotations

import html
import http.server
from pathlib import Path
from typing import NamedTuple

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
        "The same real tenant graph and policy under S1 (steady, unscripted traffic) -- shown honestly: this run also saturates to hybrid on every real decision (see the panel for why -- pool_fill stays above the demo policy's threshold throughout, compounding the same placeholder-forecaster posture ratchet documented on S2).",
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
        "Explain Decision",
        "The floor lookup, action mask legality, cost comparison, and final decision for three real, structurally-different S2 decisions.",
        (
            PanelFile("01_first_decision.html", "1. First decision"),
            PanelFile("02_floor_driven_only_hybrid_clears.html", "2. Floor-driven (only HYBRID clears)"),
            PanelFile("03_real_cost_tradeoff.html", "3. Real cost tradeoff"),
        ),
    ),
    Panel(
        "Dose-Response",
        "The steering-attack headline result: V(pi) vs. attack strength alpha, masked agent vs. soft-reward baseline.",
        (PanelFile("dose_response_chart.html", "Open panel"),),
    ),
    Panel(
        "S3 Comparison Table",
        "Masked DQN vs. soft-reward baseline on S3 (QKD degradation): p99 latency, total reward, rekey ratio, regret/exhaustion, below-floor rate.",
        (PanelFile("s3_comparison_table.html", "Open panel"),),
    ),
    Panel(
        "Budgeting Brain",
        "Real S3 pool-trajectory comparison, masked DQN vs. AlwaysHybridPolicy -- same seed, same conditions, real exhaustion-event markers.",
        (PanelFile("budgeting_brain.html", "Open panel"),),
    ),
    Panel(
        "Migration Wave",
        "S6's three scripted floor-ratchet events against a real held-out episode from an agent never trained on S6, with an honesty-gated attribution.",
        (PanelFile("migration_wave.html", "Open panel"),),
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
<div class="subtitle">Real, pre-rendered panels from dashboard/samples/, served as-is &mdash; nothing computed live, nothing in these pages is a placeholder.</div>
<div class="legend-row">{legend_html}</div>
<div class="grid">
{cards_html}
</div>
</main>
<footer>SmartKeyNet &mdash; decision layer for a multi-tenant hybrid-crypto KMS</footer>
</body>
</html>"""


def resolve_route(path: str) -> tuple[int, str, bytes]:
    """Pure routing logic: request path -> (status, content_type, body).
    No socket I/O -- the handler below is a thin adapter over this."""
    if path in ("/", "/index.html"):
        return 200, "text/html; charset=utf-8", render_index_html().encode("utf-8")

    if path.startswith("/samples/"):
        filename = path[len("/samples/") :]
        file_path = _ROUTES.get(filename)
        if file_path is not None and file_path.is_file():
            return 200, "text/html; charset=utf-8", file_path.read_bytes()

    return 404, "text/plain; charset=utf-8", b"Not Found"


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
