"""Behavioral tests for `dashboard.render_budgeting_brain` (Hard Rule 7 --
by analogy with Hard Rule 10 -- the rendered pool trajectory and
exhaustion markers must show real per-step episode data honestly:
never a dramatized exhaustion that didn't really happen, never a
hidden one that did).

Every test constructs `PoolTrajectoryPoint`/`ExhaustionEvent`/
`PolicyEpisode` objects directly with real values -- the renderer's own
contract is "render exactly these values," so testing it as a pure
view over an explicit input object is the correct level (mirrors
`tests/test_render_dose_response.py`'s use of directly-constructed
`DoseResponseSeries`). The values below are a representative subset of
the actual real S3 (seed=900) episode this session's own
`dashboard/render_budgeting_brain_demo.py` produced (see
`dashboard/samples/budgeting_data.json` for the full real run): the
masked DQN never drops below 0.7568 of pool capacity and hits zero
real deferral events; `AlwaysHybridPolicy` drains to 0.0016 by tick
131 and stays pinned there through 18 real deferral events until the
degradation window ends at tick 200. All 18 real exhaustion events
below are copied verbatim from that real run, not invented.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from dashboard.render_budgeting_brain import (
    BudgetingBrainData,
    ExhaustionEvent,
    PolicyEpisode,
    PoolTrajectoryPoint,
    render_budgeting_brain_html,
    write_budgeting_brain_html,
)

# Real, representative subset of the agent (masked DQN) trajectory --
# actual (step, pool_fill) pairs from the real seed=900 S3 episode.
_AGENT_TRAJECTORY = [
    PoolTrajectoryPoint(step=2, pool_fill=1.0),
    PoolTrajectoryPoint(step=50, pool_fill=0.9872),
    PoolTrajectoryPoint(step=100, pool_fill=0.872),
    PoolTrajectoryPoint(step=150, pool_fill=0.8336),
    PoolTrajectoryPoint(step=199, pool_fill=0.7568),  # real minimum this episode
    PoolTrajectoryPoint(step=200, pool_fill=1.0),
    PoolTrajectoryPoint(step=262, pool_fill=1.0),
]

# Real, representative subset of the baseline (AlwaysHybridPolicy)
# trajectory -- actual (step, pool_fill) pairs, same real episode/seed.
_BASELINE_TRAJECTORY = [
    PoolTrajectoryPoint(step=2, pool_fill=1.0),
    PoolTrajectoryPoint(step=50, pool_fill=0.9872),
    PoolTrajectoryPoint(step=100, pool_fill=0.3472),
    PoolTrajectoryPoint(step=131, pool_fill=0.0016),  # real minimum this episode
    PoolTrajectoryPoint(step=198, pool_fill=0.0016),
    PoolTrajectoryPoint(step=200, pool_fill=1.0),
    PoolTrajectoryPoint(step=272, pool_fill=1.0),
]

# All 18 real RegretEvents from the baseline's real run, verbatim.
_BASELINE_EXHAUSTION_EVENTS = [
    ExhaustionEvent(step=144, pool_fill_normalized=0.0016, tenant="iot-telemetry", sensitivity_class=3),
    ExhaustionEvent(step=146, pool_fill_normalized=0.0016, tenant="logging", sensitivity_class=1),
    ExhaustionEvent(step=150, pool_fill_normalized=0.0016, tenant="hospital", sensitivity_class=3),
    ExhaustionEvent(step=150, pool_fill_normalized=0.0016, tenant="fintech", sensitivity_class=1),
    ExhaustionEvent(step=154, pool_fill_normalized=0.0016, tenant="hospital", sensitivity_class=0),
    ExhaustionEvent(step=159, pool_fill_normalized=0.0016, tenant="logging", sensitivity_class=3),
    ExhaustionEvent(step=162, pool_fill_normalized=0.0016, tenant="hospital", sensitivity_class=3),
    ExhaustionEvent(step=162, pool_fill_normalized=0.0016, tenant="hospital", sensitivity_class=2),
    ExhaustionEvent(step=168, pool_fill_normalized=0.0016, tenant="iot-telemetry", sensitivity_class=1),
    ExhaustionEvent(step=171, pool_fill_normalized=0.0016, tenant="hospital", sensitivity_class=1),
    ExhaustionEvent(step=174, pool_fill_normalized=0.0016, tenant="fintech", sensitivity_class=3),
    ExhaustionEvent(step=174, pool_fill_normalized=0.0016, tenant="logging", sensitivity_class=3),
    ExhaustionEvent(step=183, pool_fill_normalized=0.0016, tenant="iot-telemetry", sensitivity_class=2),
    ExhaustionEvent(step=187, pool_fill_normalized=0.0016, tenant="iot-telemetry", sensitivity_class=3),
    ExhaustionEvent(step=190, pool_fill_normalized=0.0016, tenant="logging", sensitivity_class=3),
    ExhaustionEvent(step=190, pool_fill_normalized=0.0016, tenant="logging", sensitivity_class=3),
    ExhaustionEvent(step=196, pool_fill_normalized=0.0016, tenant="logging", sensitivity_class=2),
    ExhaustionEvent(step=198, pool_fill_normalized=0.0016, tenant="logging", sensitivity_class=0),
]


def _real_agent_episode(*, include_p99: bool = True) -> PolicyEpisode:
    return PolicyEpisode(
        label="Masked DQN (agent)",
        series_key="agent",
        tag="foresight: ewma",
        trajectory=_AGENT_TRAJECTORY,
        exhaustion_events=[],
        regret_events=0,
        pool_exhaustion_events=0,
        below_floor_rate=0.0,
        forced_rekey_ratio=0.20202020202020202,
        p99_latency=1.5,
    )


def _real_baseline_episode() -> PolicyEpisode:
    return PolicyEpisode(
        label="Always-Hybrid (baseline)",
        series_key="baseline",
        tag="no budgeting",
        trajectory=_BASELINE_TRAJECTORY,
        exhaustion_events=_BASELINE_EXHAUSTION_EVENTS,
        regret_events=18,
        pool_exhaustion_events=18,
        below_floor_rate=0.0,
        forced_rekey_ratio=0.08097165991902834,
        p99_latency=1.5,
    )


def _real_data(*, include_p99: bool = True) -> BudgetingBrainData:
    return BudgetingBrainData(
        scenario="S3",
        seed=900,
        agent=_real_agent_episode(),
        baseline=_real_baseline_episode(),
        include_p99=include_p99,
    )


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


def _extract_trajectory(html_out: str, series_key: str) -> list[tuple[int, float]]:
    """Every (step, pool_fill) pair from the polyline's own
    `data-trajectory` attribute, for the given series."""
    gradient_marker = f'grad-{series_key}'
    assert gradient_marker in html_out
    # isolate the arena-card block for this series by its gradient id,
    # then pull that block's own data-trajectory attribute
    start = html_out.index(gradient_marker)
    block = html_out[start : start + 20000]
    match = re.search(r'data-trajectory="([^"]*)"', block)
    assert match, "no data-trajectory attribute found"
    pairs = match.group(1).split(";")
    return [(int(s), float(f)) for s, f in (p.split(":") for p in pairs if p)]


def _extract_event_markers(html_out: str) -> list[tuple[int, float]]:
    matches = re.findall(r'data-event-step="([^"]+)" data-event-pool-fill="([^"]+)"', html_out)
    return [(int(s), float(f)) for s, f in matches]


# ---------------------------------------------------------------------------
# Real trajectory + event values appear verbatim
# ---------------------------------------------------------------------------


def test_every_real_agent_trajectory_point_appears_verbatim():
    data = _real_data()
    html_out = render_budgeting_brain_html(data)

    rendered = _extract_trajectory(html_out, "agent")
    expected = [(p.step, p.pool_fill) for p in _AGENT_TRAJECTORY]
    assert rendered == expected


def test_every_real_baseline_trajectory_point_appears_verbatim():
    data = _real_data()
    html_out = render_budgeting_brain_html(data)

    rendered = _extract_trajectory(html_out, "baseline")
    expected = [(p.step, p.pool_fill) for p in _BASELINE_TRAJECTORY]
    assert rendered == expected


def test_all_18_real_exhaustion_events_appear_at_their_real_positions():
    data = _real_data()
    html_out = render_budgeting_brain_html(data)

    rendered = _extract_event_markers(html_out)
    expected = [(ev.step, ev.pool_fill_normalized) for ev in _BASELINE_EXHAUSTION_EVENTS]
    assert rendered == expected


# ---------------------------------------------------------------------------
# The central Hard Rule 7 check: real contrast, neither dramatized nor hidden
# ---------------------------------------------------------------------------


def test_conserving_agent_shows_zero_exhaustion_markers():
    data = _real_data()
    html_out = render_budgeting_brain_html(data)

    # the agent side's own card block must carry no event markers
    start = html_out.index("Masked DQN (agent)")
    end = html_out.index("Always-Hybrid (baseline)")
    agent_block = html_out[start:end]
    assert "data-event-step" not in agent_block
    assert "POOL EXHAUSTED" not in agent_block
    assert 'class="badge calm">stable' in agent_block


def test_exhausting_baseline_shows_real_exhaustion_markers_and_banner():
    data = _real_data()
    html_out = render_budgeting_brain_html(data)

    start = html_out.index("Always-Hybrid (baseline)")
    baseline_block = html_out[start:]
    assert baseline_block.count("data-event-step") == 18
    assert "POOL EXHAUSTED" in baseline_block
    assert "18 real deferral event(s)" in baseline_block
    assert "first at tick 144" in baseline_block
    assert "tenant iot-telemetry" in baseline_block
    assert 'class="badge high">exhausted' in baseline_block


def test_real_minimum_pool_fill_values_appear_in_callouts():
    data = _real_data()
    html_out = render_budgeting_brain_html(data)
    # real minimums from the constructed trajectories
    assert "0.757" in html_out  # agent's real min, 0.7568, rounded to 3dp
    assert "0.002" in html_out  # baseline's real min, 0.0016, rounded to 3dp


# ---------------------------------------------------------------------------
# regret == pool_exhaustion annotation, always present when both shown
# ---------------------------------------------------------------------------


def test_regret_equals_pool_exhaustion_annotation_present():
    data = _real_data()
    html_out = render_budgeting_brain_html(data)
    assert "same count by construction" in html_out
    assert "== regret" in html_out


# ---------------------------------------------------------------------------
# below_floor_rate honesty note (this panel's own real finding: 0.0000 for
# both, not a discriminator here, unlike the masked-vs-soft-reward table)
# ---------------------------------------------------------------------------


def test_below_floor_rate_note_present_and_explains_why_its_zero_for_both():
    data = _real_data()
    html_out = render_budgeting_brain_html(data)
    assert "below_floor_rate is 0.0000 for BOTH policies" in html_out
    assert "Hard Rule 9" in html_out


# ---------------------------------------------------------------------------
# p99_latency honesty: always paired with its caveat when shown, always
# absent (value AND caveat) when include_p99=False
# ---------------------------------------------------------------------------


def test_p99_latency_always_carries_its_caveat_when_shown():
    data = _real_data(include_p99=True)
    html_out = render_budgeting_brain_html(data)
    assert "p99 latency" in html_out
    assert "discrete-cost-model percentile artifact" in html_out


def test_p99_latency_and_caveat_both_absent_when_excluded():
    data = _real_data(include_p99=False)
    html_out = render_budgeting_brain_html(data)
    assert "p99 latency" not in html_out
    assert "discrete-cost-model percentile artifact" not in html_out


# ---------------------------------------------------------------------------
# No-fabrication check
# ---------------------------------------------------------------------------


def test_no_rendered_trajectory_point_is_absent_from_the_input():
    data = _real_data()
    html_out = render_budgeting_brain_html(data)

    real_agent_points = {(p.step, p.pool_fill) for p in _AGENT_TRAJECTORY}
    real_baseline_points = {(p.step, p.pool_fill) for p in _BASELINE_TRAJECTORY}

    for pair in _extract_trajectory(html_out, "agent"):
        assert pair in real_agent_points
    for pair in _extract_trajectory(html_out, "baseline"):
        assert pair in real_baseline_points


def test_no_rendered_event_marker_is_absent_from_the_input():
    data = _real_data()
    html_out = render_budgeting_brain_html(data)

    real_events = {(ev.step, ev.pool_fill_normalized) for ev in _BASELINE_EXHAUSTION_EVENTS}
    for pair in _extract_event_markers(html_out):
        assert pair in real_events


def test_stat_box_values_match_the_real_policyepisode_fields_exactly():
    data = _real_data()
    html_out = render_budgeting_brain_html(data)

    # agent: zero events, zero below-floor rate, its real forced_rekey_ratio
    assert ">0.0000<" in html_out  # below_floor_rate, both sides
    assert ">0.202<" in html_out  # agent forced_rekey_ratio, 3dp
    assert ">0.081<" in html_out  # baseline forced_rekey_ratio, 3dp
    assert ">18<" in html_out  # baseline regret/pool-exhaustion count


# ---------------------------------------------------------------------------
# Well-formedness
# ---------------------------------------------------------------------------


def test_rendered_html_has_balanced_tags():
    data = _real_data()
    html_out = render_budgeting_brain_html(data)
    checker = _BalancedTagChecker()
    checker.feed(html_out)
    checker.close()
    assert checker.stack == []


def test_rendered_html_has_balanced_tags_without_p99():
    data = _real_data(include_p99=False)
    html_out = render_budgeting_brain_html(data)
    checker = _BalancedTagChecker()
    checker.feed(html_out)
    checker.close()
    assert checker.stack == []


def test_write_budgeting_brain_html_writes_matching_content(tmp_path):
    data = _real_data()
    out_path = tmp_path / "budgeting.html"
    returned = write_budgeting_brain_html(data, out_path)

    assert returned == out_path
    assert out_path.read_text(encoding="utf-8") == render_budgeting_brain_html(data)
