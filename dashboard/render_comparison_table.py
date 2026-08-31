"""
dashboard/render_comparison_table.py

Renders the masked-DQN-vs-soft-reward-baseline S3 comparison (PLAN.md
Table V shape) as a self-contained static HTML table. Visual target:
dashboard/mockups/smartkeynet_dashboard_mockup_v2.html's Results tab
(styling only -- that file's table numbers are 100% fabricated per its
own header and are never read here).

Same rendering philosophy as dashboard/render_explain.py and
dashboard/render_dose_response.py: a pure function over an explicit,
real input object, self-contained HTML, zero new dependencies.

Hard Rule 7 (central to this module): the table must never present
`p99_latency` as a clean discriminating metric -- its documented
status (`experiments/harness.py::ScenarioResult.p99_latency`'s own
docstring) is a discrete-cost-model percentile artifact, not a
meaningful tail-latency signal. If `include_p99` is True, the renderer
ALWAYS attaches the caveat text next to the p99 row (a test asserts
this pairing can't be split apart). `below_floor_rate` -- the cleanest,
most thesis-relevant metric -- always leads the table. `regret_events`
is always annotated as identical to `pool_exhaustion_events` by
construction (the same real design note `experiments/harness.py`'s own
docstring carries), so a reader never mistakes them for two
independent results.
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
.wrap{max-width:820px;margin:0 auto;}
.beat-head{margin-bottom:22px;}
.beat-eyebrow{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--quantum);}
.beat-title{font-family:var(--disp);font-weight:700;font-size:22px;margin-top:4px;}
.beat-desc{color:var(--text-dim);font-size:13px;margin-top:6px;line-height:1.5;}

.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:var(--radius);}
table{width:100%;border-collapse:collapse;font-size:12.5px;}
thead th{text-align:left;font-family:var(--mono);font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--text-faint);background:var(--panel-2);padding:12px 16px;border-bottom:1px solid var(--line);white-space:nowrap;}
tbody td{padding:13px 16px;border-bottom:1px solid var(--line-soft);font-family:var(--mono);}
tbody tr:last-child td{border-bottom:none;}
tbody tr.hero{background:rgba(110,126,255,.06);}
tbody tr.hero td:first-child{position:relative;}
tbody tr.hero td:first-child::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--quantum);}
tbody tr.caveat td{background:rgba(255,92,108,.04);}
.metric-name{color:var(--text);}
.num-good{color:var(--hybrid);}
.num-bad{color:var(--danger);}
.spread{color:var(--text-faint);font-size:10.5px;}

.note{margin-top:14px;font-size:11.5px;color:var(--text-dim);border-left:2px solid var(--quantum);padding:8px 11px;background:var(--panel-2);border-radius:var(--radius-sm);line-height:1.55;}
.note.caveat{border-left-color:var(--danger);}
.note b{color:var(--text);}

.provenance{margin-top:16px;font-family:var(--mono);font-size:10px;color:var(--text-faint);line-height:1.6;}
"""

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


@dataclass(frozen=True)
class AgentMetrics:
    """One agent's real, checkpoint-averaged S3 metrics -- every field
    is `mean`/`std` of that many real per-training-seed
    `MultiSeedEvalResult.<field>_mean` values (the same "mean of
    already-averaged eval-seed means, spread = training-seed std"
    methodology Gate W3 and the masked-vs-soft-reward S3 comparison
    session both used), computed by the caller from real harness
    output -- never invented here."""

    label: str
    below_floor_rate_mean: float
    below_floor_rate_std: float
    total_reward_mean: float
    total_reward_std: float
    forced_rekey_ratio_mean: float
    forced_rekey_ratio_std: float
    regret_events_mean: float
    regret_events_std: float
    p99_latency_mean: float
    p99_latency_std: float
    floor_violations_total: int
    n_training_seeds: int
    n_eval_seeds_per_checkpoint: int


@dataclass(frozen=True)
class ComparisonTableData:
    scenario: str
    masked: AgentMetrics
    soft_reward: AgentMetrics
    include_p99: bool = True


def _e(value: object) -> str:
    return html.escape(str(value))


def _fmt(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} <span class=\"spread\">&plusmn; {std:.{digits}f}</span>"


def render_comparison_table_html(
    data: ComparisonTableData,
    *,
    title: str = "SmartKeyNet -- Masked DQN vs Soft-Reward Baseline",
) -> str:
    """Render `data`'s real, checkpoint-averaged metrics as a
    self-contained HTML table. Pure view: every number shown is a
    `AgentMetrics` field verbatim, formatted -- never recomputed or
    invented (Hard Rule 7)."""
    m, s = data.masked, data.soft_reward

    def below_floor_cls(agent: AgentMetrics) -> str:
        return "num-good" if agent.below_floor_rate_mean == 0.0 else "num-bad"

    rows: list[str] = []

    rows.append(
        f'<tr class="hero">'
        f'<td class="metric-name">below_floor_rate <span class="spread">(V(&pi;), eq. 4)</span></td>'
        f'<td class="{below_floor_cls(m)}">{_fmt(m.below_floor_rate_mean, m.below_floor_rate_std)}</td>'
        f'<td class="{below_floor_cls(s)}">{_fmt(s.below_floor_rate_mean, s.below_floor_rate_std)}</td>'
        f"</tr>"
    )

    rows.append(
        f'<tr><td class="metric-name">total_reward</td>'
        f"<td>{_fmt(m.total_reward_mean, m.total_reward_std, digits=2)}</td>"
        f"<td>{_fmt(s.total_reward_mean, s.total_reward_std, digits=2)}</td>"
        f"</tr>"
    )

    rows.append(
        f'<tr><td class="metric-name">forced_rekey_ratio</td>'
        f"<td>{_fmt(m.forced_rekey_ratio_mean, m.forced_rekey_ratio_std, digits=3)}</td>"
        f"<td>{_fmt(s.forced_rekey_ratio_mean, s.forced_rekey_ratio_std, digits=3)}</td>"
        f"</tr>"
    )

    rows.append(
        f'<tr class="caveat"><td class="metric-name">regret_events <span class="spread">(== pool_exhaustion_events)</span></td>'
        f"<td>{_fmt(m.regret_events_mean, m.regret_events_std, digits=2)}</td>"
        f"<td>{_fmt(s.regret_events_mean, s.regret_events_std, digits=2)}</td>"
        f"</tr>"
    )

    rows.append(
        f'<tr><td class="metric-name">floor_violations_total <span class="spread">(summed, all episodes)</span></td>'
        f"<td>{m.floor_violations_total}</td>"
        f"<td>{s.floor_violations_total}</td>"
        f"</tr>"
    )

    p99_row = ""
    p99_note = ""
    if data.include_p99:
        rows.append(
            f'<tr class="caveat"><td class="metric-name">p99_latency <span class="spread">(see caveat below)</span></td>'
            f"<td>{_fmt(m.p99_latency_mean, m.p99_latency_std)}</td>"
            f"<td>{_fmt(s.p99_latency_mean, s.p99_latency_std)}</td>"
            f"</tr>"
        )
        p99_note = f'<div class="note caveat"><b>p99_latency caveat:</b> {_e(_P99_CAVEAT)}</div>'

    table_body = "".join(rows)

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
    <div class="beat-eyebrow">Results &middot; Scenario {_e(data.scenario)}</div>
    <div class="beat-title">Masked DQN vs. soft-reward baseline</div>
    <div class="beat-desc">Checkpoint-averaged real measurements ({m.n_training_seeds} training seeds &times; {m.n_eval_seeds_per_checkpoint} eval seeds per checkpoint, masked agent; {s.n_training_seeds} &times; {s.n_eval_seeds_per_checkpoint}, soft-reward agent). Spread is the standard deviation across training-seed checkpoints, never a bare point estimate.</div>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Metric</th><th>{_e(m.label)}</th><th>{_e(s.label)}</th></tr></thead>
      <tbody>{table_body}</tbody>
    </table>
  </div>
  {p99_note}
  <div class="note"><b>regret_events note:</b> {_e(_REGRET_NOTE)}</div>
  <div class="provenance">below_floor_rate leads this table deliberately -- the cleanest, most thesis-relevant metric (Hard Rule 2's guarantee for the masked agent, measured, not assumed).</div>
</main>
</body>
</html>
"""


def write_comparison_table_html(
    data: ComparisonTableData,
    path: str | Path,
    *,
    title: str = "SmartKeyNet -- Masked DQN vs Soft-Reward Baseline",
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_comparison_table_html(data, title=title), encoding="utf-8")
    return out_path
