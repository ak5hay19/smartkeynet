"""
dashboard/render_upload.py

Pure HTML renderers for the v2 Phase 2 upload-and-run demo
(`dashboard/upload_demo.py` is the real driver; this module only turns
its already-computed, real output into HTML -- no I/O, no server, no
computation, matching `dashboard/render_explain.py`'s own
pure-renderer convention). Styled to match `dashboard/app.py`'s `/v2`
shell (same dark theme, same tier-color legend) for visual consistency,
per this session's scope instruction -- not a separate visual system.
"""

from __future__ import annotations

import html

from dashboard.live_capture import CAPTURE_DURATION_SECONDS, CaptureInterface
from dashboard.upload_demo import DecisionRow, UPLOAD_CAVEAT_TEXT, UploadDemoResult

_BASE_STYLE = """
:root{
  --bg:#0A0E13; --panel:#111820; --panel-2:#151D27; --line:#212C39; --line-soft:#171F29;
  --text:#E9EEF4; --text-dim:#8FA0B3; --text-faint:#4C5A6B; --quantum:#6E7EFF;
  --classical:#8B95A5; --pqc:#E8A33D; --hybrid:#33D687; --warn:#E85D5D;
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
  color:var(--text);font-family:var(--disp);min-height:100vh;
}
a{color:var(--quantum);}
.topbar{
  display:flex;align-items:center;gap:14px;padding:14px 22px;
  border-bottom:1px solid var(--line);
  background:linear-gradient(180deg, rgba(17,24,32,.9), rgba(17,24,32,.55));
}
.brand-name{font-weight:700;font-size:16px;letter-spacing:.05em;}
.brand-sub{font-family:var(--mono);font-size:10px;color:var(--text-faint);letter-spacing:.12em;text-transform:uppercase;margin-top:1px;}
.pill{
  font-family:var(--mono);font-size:11px;color:var(--text-dim);
  background:var(--panel-2);border:1px solid var(--line);
  padding:5px 10px;border-radius:999px;white-space:nowrap;
}
.classic-link{
  font-family:var(--mono);font-size:11px;color:var(--quantum);text-decoration:none;
  border:1px solid rgba(110,126,255,.35);border-radius:999px;padding:5px 10px;white-space:nowrap;margin-left:auto;
}
main{max-width:1200px;margin:0 auto;padding:24px 22px 60px;}
.caveat{
  background:rgba(232,93,93,.08);border:1px solid rgba(232,93,93,.35);border-radius:var(--radius);
  padding:14px 16px;font-size:13px;line-height:1.55;color:#F2C7C7;margin-bottom:20px;
}
.caveat b{color:var(--warn);}
.summary-row{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:22px;}
.summary-card{
  background:var(--panel);border:1px solid var(--line);border-radius:var(--radius-sm);
  padding:10px 14px;font-family:var(--mono);font-size:12px;color:var(--text-dim);
}
.summary-card b{display:block;font-family:var(--disp);font-size:18px;color:var(--text);margin-top:2px;}
table{width:100%;border-collapse:collapse;font-size:12.5px;background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);overflow:hidden;}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line-soft);vertical-align:top;}
th{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-faint);background:var(--panel-2);}
tr:last-child td{border-bottom:none;}
.tier-CLASSICAL{color:var(--classical);}
.tier-PQC{color:var(--pqc);}
.tier-HYBRID{color:var(--hybrid);}
.mono{font-family:var(--mono);}
.legal-chip{display:inline-block;font-family:var(--mono);font-size:10px;color:var(--text-dim);background:var(--panel-2);border:1px solid var(--line);border-radius:4px;padding:1px 5px;margin:1px 2px 1px 0;}
.chosen{font-weight:700;}
h1{font-size:20px;margin:0 0 4px;}
h2{font-size:14px;color:var(--text-dim);margin:28px 0 10px;text-transform:uppercase;letter-spacing:.06em;font-family:var(--mono);}
.upload-card{
  max-width:560px;margin:60px auto;background:var(--panel);border:1px solid var(--line);
  border-radius:var(--radius);padding:28px;text-align:center;
}
.upload-card input[type=file]{margin:18px 0;color:var(--text-dim);}
.upload-card button{
  font-family:var(--disp);font-weight:600;font-size:13px;color:#0A0E13;background:var(--quantum);
  border:none;border-radius:8px;padding:10px 20px;cursor:pointer;
}
.upload-card button:disabled{opacity:.5;cursor:default;}
#status{margin-top:14px;font-family:var(--mono);font-size:12px;color:var(--text-dim);}
.error-box{
  max-width:640px;margin:60px auto;background:rgba(232,93,93,.08);border:1px solid rgba(232,93,93,.35);
  border-radius:var(--radius);padding:24px;color:#F2C7C7;
}
"""


def _topbar(extra_pill: str = "") -> str:
    pill = f'<span class="pill">{html.escape(extra_pill)}</span>' if extra_pill else ""
    return f"""<header class="topbar">
  <div>
    <div class="brand-name">SMARTKEYNET</div>
    <div class="brand-sub">Hybrid KMS &middot; Decision Layer</div>
  </div>
  {pill}
  <a class="classic-link" href="/">&larr; Dashboard</a>
</header>"""


def render_upload_form_html() -> str:
    """The upload page itself: a file picker that POSTs the raw pcap
    bytes to `/upload` (no multipart parsing needed -- this server
    stays standard-library only, per `dashboard/app.py`'s own
    zero-new-dependency convention) and swaps the whole page for the
    server's real, live-computed result page."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartKeyNet -- Upload pcap</title>
<style>{_BASE_STYLE}</style>
</head>
<body>
{_topbar()}
<main>
<div class="upload-card">
  <h1>Upload a .pcap</h1>
  <p style="color:var(--text-dim);font-size:13px;line-height:1.5;">
    Real tshark-backed flow extraction, the real trained LSTM threat forecaster,
    and a real trained masked-DQN checkpoint -- computed live, right now, on
    whatever you upload. See the honesty note on the results page before
    reading anything into the output.
  </p>
  <input type="file" id="pcapFile" accept=".pcap,.pcapng">
  <br>
  <button id="uploadBtn">Run the pipeline</button>
  <div id="status"></div>
</div>
</main>
<script>
  const btn = document.getElementById('uploadBtn');
  const status = document.getElementById('status');
  const fileInput = document.getElementById('pcapFile');

  btn.addEventListener('click', async () => {{
    const file = fileInput.files[0];
    if (!file) {{ status.textContent = 'Choose a .pcap file first.'; return; }}
    btn.disabled = true;
    status.textContent = 'Uploading and running the real pipeline (tshark extraction, LSTM forecast, masked DQN) -- this can take a few seconds...';
    try {{
      const buf = await file.arrayBuffer();
      const resp = await fetch('/upload', {{
        method: 'POST',
        headers: {{ 'X-Filename': encodeURIComponent(file.name), 'Content-Type': 'application/octet-stream' }},
        body: buf,
      }});
      const text = await resp.text();
      document.open();
      document.write(text);
      document.close();
    }} catch (e) {{
      status.textContent = 'Upload failed: ' + e;
      btn.disabled = false;
    }}
  }});
</script>
</body>
</html>"""


def render_capture_form_html(interfaces: tuple[CaptureInterface, ...], default_index: str) -> str:
    """The v2 Phase 3 live-capture form: an interface picker instead of
    a file input, POSTing to `/capture?interface=<index>` (no request
    body needed -- unlike `/upload`, there's no file to send). Reuses
    this module's own style system and results/error pages verbatim
    (Step 4: same results table, same honesty caveat, same error
    philosophy as the upload flow -- this is an alternate TRIGGER for
    the identical pipeline, not a new visual system)."""
    options_html = "\n".join(
        f'<option value="{html.escape(iface.index)}"'
        f'{" selected" if iface.index == default_index else ""}>'
        f"{html.escape(iface.friendly_name)} (interface #{html.escape(iface.index)})</option>"
        for iface in interfaces
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartKeyNet -- Live capture</title>
<style>{_BASE_STYLE}</style>
</head>
<body>
{_topbar()}
<main>
<div class="upload-card">
  <h1>Start a live capture</h1>
  <p style="color:var(--text-dim);font-size:13px;line-height:1.5;">
    Captures real traffic from a chosen interface on this machine for a
    fixed {CAPTURE_DURATION_SECONDS}-second window, then runs it through
    the exact same real pipeline as uploading a .pcap (tshark extraction,
    the real trained LSTM threat forecaster, a real trained masked-DQN
    checkpoint). See the honesty note on the results page before reading
    anything into the output.
  </p>
  <select id="ifaceSelect" style="width:100%;padding:9px;background:var(--panel-2);color:var(--text);border:1px solid var(--line);border-radius:8px;margin:18px 0;font-family:var(--mono);font-size:12px;">
{options_html}
  </select>
  <br>
  <button id="captureBtn">Capture for {CAPTURE_DURATION_SECONDS}s and run the pipeline</button>
  <div id="status"></div>
</div>
</main>
<script>
  const btn = document.getElementById('captureBtn');
  const status = document.getElementById('status');
  const ifaceSelect = document.getElementById('ifaceSelect');

  btn.addEventListener('click', async () => {{
    const iface = ifaceSelect.value;
    btn.disabled = true;
    status.textContent = 'Capturing live traffic for {CAPTURE_DURATION_SECONDS}s, then running the real pipeline -- this will take about {CAPTURE_DURATION_SECONDS} seconds...';
    try {{
      const resp = await fetch('/capture?interface=' + encodeURIComponent(iface), {{ method: 'POST' }});
      const text = await resp.text();
      document.open();
      document.write(text);
      document.close();
    }} catch (e) {{
      status.textContent = 'Capture failed: ' + e;
      btn.disabled = false;
    }}
  }});
</script>
</body>
</html>"""


def render_capture_unavailable_html(message: str) -> str:
    """An honest, plain page for when live capture can't even be
    offered (e.g. no interfaces enumerable, tshark not on PATH) --
    reuses the same error-box styling as `render_upload_error_html`,
    never a fabricated interface list or a silent blank page."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartKeyNet -- Live capture unavailable</title>
<style>{_BASE_STYLE}</style>
</head>
<body>
{_topbar()}
<main>
<div class="error-box">
  <h1 style="margin-top:0;">Live capture is not available here</h1>
  <p class="mono" style="font-size:12.5px;line-height:1.6;">{html.escape(message)}</p>
  <p><a href="/upload">Use the upload flow instead &rarr;</a></p>
</div>
</main>
</body>
</html>"""


def _decision_row_html(row: DecisionRow) -> str:
    legal_html = "".join(f'<span class="legal-chip">{html.escape(a)}</span>' for a in row.legal_actions)
    return f"""<tr>
<td class="mono">{row.decision_num}</td>
<td>{html.escape(row.demo_tenant)}<div style="color:var(--text-faint);font-size:11px;">{html.escape(row.demo_service)} &middot; {html.escape(row.demo_sensitivity_class)}</div></td>
<td class="mono">#{row.pcap_flow_index}<div style="color:var(--text-faint);font-size:11px;">{html.escape(row.pcap_flow_proto)} / {html.escape(row.pcap_flow_service)}</div></td>
<td class="mono">{row.threat_score:.4f}</td>
<td>{html.escape(row.posture)}</td>
<td>{html.escape(row.floor)}</td>
<td>{legal_html}</td>
<td class="tier-{html.escape(row.chosen_action.replace('SERVE_', '').split('_')[0])} chosen">{html.escape(row.chosen_action)}</td>
<td style="max-width:320px;color:var(--text-dim);font-size:11.5px;">{html.escape(row.final_text)}</td>
</tr>"""


def render_upload_result_html(result: UploadDemoResult) -> str:
    """The real, live-computed result page. Every number here came
    from `dashboard/upload_demo.py::run_pcap_upload_demo` actually
    running the real pipeline against the uploaded file -- nothing on
    this page is pre-rendered or fabricated."""
    services_html = ", ".join(
        f"{html.escape(str(svc))}: {count}" for svc, count in sorted(result.service_counts.items(), key=lambda kv: -kv[1])
    )
    rows_html = "\n".join(_decision_row_html(r) for r in result.decisions)

    n_decisions = len(result.decisions)
    wraparound_note_html = ""
    if result.n_flows < n_decisions:
        wraparound_note_html = (
            '<div class="caveat"><b>Replay note:</b> '
            f"{result.n_flows} real flows extracted; this {n_decisions}-decision episode revisits "
            "some flows more than once (a real replay of the same real flow, not new traffic) "
            "since the episode length exceeds the flow count.</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartKeyNet -- Upload results</title>
<style>{_BASE_STYLE}</style>
</head>
<body>
{_topbar("Live, computed at request time -- not pre-rendered")}
<main>
<h1>{html.escape(result.pcap_filename)}</h1>
<p style="color:var(--text-dim);font-size:13px;">
  {html.escape(result.checkpoint_name)} &middot; {html.escape(result.config_name)} &middot;
  {len(result.decisions)} real decisions
</p>

<div class="caveat">
  <b>Honesty note:</b> {html.escape(UPLOAD_CAVEAT_TEXT)}
</div>
{wraparound_note_html}
<div class="summary-row">
  <div class="summary-card">Packets (total)<b>{result.n_packets_total}</b></div>
  <div class="summary-card">Packets used (IPv4 tcp/udp/icmp)<b>{result.n_packets_used}</b></div>
  <div class="summary-card">Packets skipped (non-IPv4)<b>{result.n_packets_skipped_non_ipv4}</b></div>
  <div class="summary-card">Real flows extracted<b>{result.n_flows}</b></div>
  <div class="summary-card" style="flex:2;">Extracted service mix<b style="font-size:13px;">{services_html or "(none)"}</b></div>
</div>

<h2>Decisions</h2>
<table>
<thead><tr>
  <th>#</th>
  <th>Demo tenant / service</th>
  <th>Pcap flow (real)</th>
  <th>threat_score</th>
  <th>Posture</th>
  <th>Floor</th>
  <th>Legal actions</th>
  <th>Chosen</th>
  <th>Why (cheapest-legal)</th>
</tr></thead>
<tbody>
{rows_html}
</tbody>
</table>

<p style="margin-top:22px;"><a href="/upload">&larr; Upload another pcap</a></p>
</main>
</body>
</html>"""


def render_upload_error_html(filename: str, message: str) -> str:
    """An honest, plain error page -- never a fabricated/placeholder
    result. Step 5 of this session's brief."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartKeyNet -- Upload failed</title>
<style>{_BASE_STYLE}</style>
</head>
<body>
{_topbar()}
<main>
<div class="error-box">
  <h1 style="margin-top:0;">Could not process {html.escape(filename)}</h1>
  <p class="mono" style="font-size:12.5px;line-height:1.6;">{html.escape(message)}</p>
  <p><a href="/upload">&larr; Try a different file</a></p>
</div>
</main>
</body>
</html>"""
