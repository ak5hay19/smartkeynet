"""Behavioral tests for `dashboard.render_comparison_table` (Hard Rule
7 -- the rendered table must lead with `below_floor_rate`, never
present `p99_latency` without its documented-artifact caveat, and
always annotate `regret_events`/`pool_exhaustion_events` as the same
event by construction).

`AgentMetrics`/`ComparisonTableData` are constructed directly with
known, real-shaped values (the actual figures from SESSION_LOG.md's
2026-08-25 masked-vs-soft-reward S3 comparison entry, reproduced fresh
by this session's own driver -- see `dashboard/render_results_demo.py`)
-- the renderer's contract is "render exactly these values," so
testing it as a pure view over an explicit input object is the correct
level (mirrors `tests/test_render_explain.py`'s directly-constructed
`DecisionTrace`s).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from dashboard.render_comparison_table import (
    AgentMetrics,
    ComparisonTableData,
    render_comparison_table_html,
    write_comparison_table_html,
)

# The real, measured S3 comparison values (SESSION_LOG.md 2026-08-25
# entry, reproduced byte-for-byte by this session's fresh re-run).
_MASKED = AgentMetrics(
    label="Masked DQN",
    below_floor_rate_mean=0.0000,
    below_floor_rate_std=0.0000,
    total_reward_mean=-10214.82,
    total_reward_std=1303.25,
    forced_rekey_ratio_mean=0.156,
    forced_rekey_ratio_std=0.037,
    regret_events_mean=0.00,
    regret_events_std=0.00,
    p99_latency_mean=1.5000,
    p99_latency_std=0.0000,
    floor_violations_total=0,
    n_training_seeds=3,
    n_eval_seeds_per_checkpoint=8,
)

_SOFT_REWARD = AgentMetrics(
    label="Soft-reward baseline",
    below_floor_rate_mean=0.1687,
    below_floor_rate_std=0.0759,
    total_reward_mean=-11010.19,
    total_reward_std=12380.91,
    forced_rekey_ratio_mean=0.703,
    forced_rekey_ratio_std=0.408,
    regret_events_mean=3.54,
    regret_events_std=5.01,
    p99_latency_mean=1.4064,
    p99_latency_std=0.1324,
    floor_violations_total=1012,
    n_training_seeds=3,
    n_eval_seeds_per_checkpoint=8,
)


def _real_data(include_p99: bool = True) -> ComparisonTableData:
    return ComparisonTableData(scenario="S3", masked=_MASKED, soft_reward=_SOFT_REWARD, include_p99=include_p99)


class _BalancedTagChecker(HTMLParser):
    _VOID_ELEMENTS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in self._VOID_ELEMENTS:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        assert self.stack, f"</{tag}> with no matching open tag"
        assert self.stack[-1] == tag, f"expected </{self.stack[-1]}>, got </{tag}>"
        self.stack.pop()


# ---------------------------------------------------------------------------
# Real values appear exactly
# ---------------------------------------------------------------------------


def test_below_floor_rate_leads_the_table_with_real_values():
    html_out = render_comparison_table_html(_real_data())

    below_floor_idx = html_out.index("below_floor_rate")
    total_reward_idx = html_out.index("total_reward")
    forced_rekey_idx = html_out.index("forced_rekey_ratio")
    assert below_floor_idx < total_reward_idx < forced_rekey_idx

    assert "0.0000" in html_out and "&plusmn; 0.0000" in html_out  # masked below_floor_rate
    assert "0.1687" in html_out and "&plusmn; 0.0759" in html_out  # soft-reward below_floor_rate


def test_real_total_reward_and_forced_rekey_ratio_values_appear():
    html_out = render_comparison_table_html(_real_data())

    assert "-10214.82" in html_out
    assert "-11010.19" in html_out
    assert "0.156" in html_out
    assert "0.703" in html_out


def test_real_regret_and_floor_violation_values_appear():
    html_out = render_comparison_table_html(_real_data())

    assert "3.54" in html_out
    assert ">0<" in html_out or "<td>0</td>" in html_out  # masked floor_violations_total
    assert "<td>1012</td>" in html_out


# ---------------------------------------------------------------------------
# Hard Rule 7: p99_latency honesty
# ---------------------------------------------------------------------------


def test_p99_latency_always_carries_its_caveat_when_shown():
    html_out = render_comparison_table_html(_real_data(include_p99=True))

    assert "p99_latency" in html_out
    assert "1.5000" in html_out
    assert "1.4064" in html_out
    assert "discrete-cost-model percentile artifact" in html_out
    assert "not a meaningful tail-latency discriminator" in html_out
    # the caveat must actually be reachable near the p99 row, not just
    # floating anywhere in the page unrelated to it
    assert "p99_latency caveat" in html_out


def test_p99_latency_omitted_entirely_means_no_orphaned_caveat():
    html_out = render_comparison_table_html(_real_data(include_p99=False))

    assert "p99_latency" not in html_out
    assert "discrete-cost-model percentile artifact" not in html_out
    assert "1.4064" not in html_out


def test_regret_events_always_annotated_as_same_event_as_pool_exhaustion():
    html_out = render_comparison_table_html(_real_data())

    assert "regret_events" in html_out
    assert "pool_exhaustion_events" in html_out
    assert "same count by construction" in html_out


# ---------------------------------------------------------------------------
# No-fabrication check
# ---------------------------------------------------------------------------


def test_every_numeric_metric_value_traces_back_to_agent_metrics_fields():
    """Extract every `<tbody>` row's two data-column numeric leads
    (the value before the `&plusmn;` spread, or the bare int for
    floor_violations_total) and confirm each is exactly one of the
    real `AgentMetrics` fields, formatted at this renderer's own
    documented precision -- never an unaccounted-for number."""
    data = _real_data()
    html_out = render_comparison_table_html(data)

    expected_masked = {
        f"{data.masked.below_floor_rate_mean:.4f}",
        f"{data.masked.total_reward_mean:.2f}",
        f"{data.masked.forced_rekey_ratio_mean:.3f}",
        f"{data.masked.regret_events_mean:.2f}",
        f"{data.masked.p99_latency_mean:.4f}",
        str(data.masked.floor_violations_total),
    }
    expected_soft = {
        f"{data.soft_reward.below_floor_rate_mean:.4f}",
        f"{data.soft_reward.total_reward_mean:.2f}",
        f"{data.soft_reward.forced_rekey_ratio_mean:.3f}",
        f"{data.soft_reward.regret_events_mean:.2f}",
        f"{data.soft_reward.p99_latency_mean:.4f}",
        str(data.soft_reward.floor_violations_total),
    }

    body_match = re.search(r"<tbody>(.*)</tbody>", html_out, re.DOTALL)
    assert body_match is not None
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", body_match.group(1), re.DOTALL)
    assert len(rows) == 6  # below_floor_rate, total_reward, forced_rekey_ratio, regret_events, floor_violations_total, p99_latency

    for row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        assert len(cells) == 3
        masked_lead = re.search(r"-?\d[\d.]*", cells[1])
        soft_lead = re.search(r"-?\d[\d.]*", cells[2])
        assert masked_lead is not None and soft_lead is not None
        assert masked_lead.group(0) in expected_masked, f"unaccounted-for masked value: {masked_lead.group(0)!r}"
        assert soft_lead.group(0) in expected_soft, f"unaccounted-for soft-reward value: {soft_lead.group(0)!r}"


# ---------------------------------------------------------------------------
# Well-formedness
# ---------------------------------------------------------------------------


def test_rendered_html_has_balanced_tags():
    html_out = render_comparison_table_html(_real_data())
    checker = _BalancedTagChecker()
    checker.feed(html_out)
    checker.close()
    assert checker.stack == []


def test_rendered_html_without_p99_has_balanced_tags():
    html_out = render_comparison_table_html(_real_data(include_p99=False))
    checker = _BalancedTagChecker()
    checker.feed(html_out)
    checker.close()
    assert checker.stack == []


def test_write_comparison_table_html_writes_matching_content(tmp_path):
    data = _real_data()
    out_path = tmp_path / "table.html"
    returned = write_comparison_table_html(data, out_path)

    assert returned == out_path
    assert out_path.read_text(encoding="utf-8") == render_comparison_table_html(data)
