"""
dashboard/app.py

Minimal local web server for the six already-rendered dashboard panels
in `dashboard/samples/` (Living System, Explain Decision, dose-response,
S3 comparison table, Budgeting Brain, Migration Wave -- see README.md's
"How to see the results" table, the source this module's `PANELS`
registry mirrors). Turns "open six separate HTML files by hand" into
"run one command, open one URL."

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
        "Living System",
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


def render_index_html() -> str:
    """Pure function: build the index page linking all six panels.
    No file I/O, no server -- safe to call directly in tests."""
    cards = []
    for panel in PANELS:
        links = "\n".join(
            f'<a class="panel-link" href="/samples/{html.escape(pf.filename)}">{html.escape(pf.label)}</a>'
            for pf in panel.files
        )
        cards.append(
            f"""<div class="card">
<div class="card-title">{html.escape(panel.title)}</div>
<div class="card-desc">{html.escape(panel.description)}</div>
<div class="card-links">{links}</div>
</div>"""
        )
    cards_html = "\n".join(cards)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartKeyNet -- Dashboard</title>
<style>
:root{{
  --bg:#0A0E13; --panel:#111820; --panel-2:#151D27; --line:#212C39;
  --text:#E9EEF4; --text-dim:#8FA0B3; --text-faint:#4C5A6B; --quantum:#6E7EFF;
  --radius:12px;
  --mono:ui-monospace,SFMono-Regular,Consolas,'Courier New',monospace;
  --disp:-apple-system,'Segoe UI',Roboto,sans-serif;
}}
*{{box-sizing:border-box;}}
html,body{{margin:0;padding:0;}}
body{{background:var(--bg);color:var(--text);font-family:var(--disp);padding:32px 20px 60px;}}
.wrap{{max-width:900px;margin:0 auto;}}
h1{{font-size:22px;margin:0 0 6px;}}
.subtitle{{color:var(--text-dim);font-size:13px;margin-bottom:28px;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px;}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:18px;}}
.card-title{{font-weight:700;font-size:15px;margin-bottom:8px;}}
.card-desc{{color:var(--text-dim);font-size:12.5px;line-height:1.5;margin-bottom:14px;}}
.card-links{{display:flex;flex-direction:column;gap:6px;}}
.panel-link{{color:var(--quantum);font-family:var(--mono);font-size:11.5px;text-decoration:none;}}
.panel-link:hover{{text-decoration:underline;}}
</style>
</head>
<body>
<div class="wrap">
<h1>SmartKeyNet Dashboard</h1>
<div class="subtitle">Six real, pre-rendered panels from dashboard/samples/ -- served as-is, nothing computed live.</div>
<div class="grid">
{cards_html}
</div>
</div>
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
