"""Behavioral tests for `dashboard.render_dose_response` (Hard Rule 7 --
by analogy with last session's Hard Rule 10 -- the rendered chart must
show the real measured V(pi) shape honestly, including the masked
agent's genuine alpha>=0.9 boundary, never a fabricated flat-zero
version).

Every test constructs `DoseResponsePoint`/`DoseResponseSeries` objects
directly with known, explicit real-shaped values -- the renderer's own
contract is "render exactly these values," so testing it as a pure
view over an explicit input object is the correct level (mirrors
`tests/test_render_explain.py`'s use of directly-constructed
`DecisionTrace`s). The real numbers used below are the actual figures
from SESSION_LOG.md's 2026-08-26 S5 sweep entry (reproduced fresh by
this session's own driver -- see `dashboard/render_results_demo.py`).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from dashboard.render_dose_response import (
    DoseResponsePoint,
    DoseResponseSeries,
    render_dose_response_html,
    write_dose_response_html,
)

# The real, measured S5 sweep values (SESSION_LOG.md 2026-08-26 entry,
# reproduced byte-for-byte by this session's fresh re-run).
_MASKED_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_MASKED_MEANS = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0016, 0.0024, 0.0048, 0.0096, 0.3000, 0.3000]
_MASKED_STDS = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0020, 0.0020, 0.0039, 0.0032, 0.0076, 0.0076]

_SOFT_MEANS = [0.2896] * 9 + [0.3256, 0.3256]
_SOFT_STDS = [0.0678] * 9 + [0.0799, 0.0799]


def _real_masked_series() -> DoseResponseSeries:
    return DoseResponseSeries(
        label="Masked DQN",
        series_key="masked",
        points=[
            DoseResponsePoint(alpha=a, mean=m, std=s)
            for a, m, s in zip(_MASKED_ALPHAS, _MASKED_MEANS, _MASKED_STDS)
        ],
    )


def _real_soft_series() -> DoseResponseSeries:
    return DoseResponseSeries(
        label="Soft-reward baseline",
        series_key="soft_reward",
        points=[
            DoseResponsePoint(alpha=a, mean=m, std=s)
            for a, m, s in zip(_MASKED_ALPHAS, _SOFT_MEANS, _SOFT_STDS)
        ],
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


def _extract_points(html_out: str) -> list[tuple[float, float, float]]:
    """Every (alpha, mean, std) triple actually rendered as a chart
    point (the `data-*` attributes on each `<circle>`)."""
    matches = re.findall(
        r'data-alpha="([^"]+)" data-mean="([^"]+)" data-std="([^"]+)"', html_out
    )
    return [(float(a), float(m), float(s)) for a, m, s in matches]


# ---------------------------------------------------------------------------
# Real values appear at the right positions
# ---------------------------------------------------------------------------


def test_every_real_point_appears_with_exact_alpha_mean_std():
    masked = _real_masked_series()
    html_out = render_dose_response_html([masked])

    rendered = _extract_points(html_out)
    expected = [(p.alpha, p.mean, p.std) for p in masked.points]
    assert rendered == expected


def test_two_series_render_all_22_real_points():
    series = [_real_masked_series(), _real_soft_series()]
    html_out = render_dose_response_html(series)

    rendered = _extract_points(html_out)
    expected = [(p.alpha, p.mean, p.std) for s in series for p in s.points]
    assert rendered == expected


# ---------------------------------------------------------------------------
# The central Hard Rule 7 check: no flattening the masked curve
# ---------------------------------------------------------------------------


def test_masked_agent_alpha_0_9_nonzero_value_is_faithfully_shown():
    masked = _real_masked_series()
    html_out = render_dose_response_html([masked])

    # The real alpha=0.9 point (mean 0.3000) must appear verbatim --
    # this is the exact value a "clean flat-zero-everywhere" renderer
    # would have to drop or round away.
    assert 'data-alpha="0.9" data-mean="0.3" data-std="0.0076"' in html_out


def test_masked_curve_is_not_flat_zero_everywhere():
    masked = _real_masked_series()
    html_out = render_dose_response_html([masked])

    rendered = _extract_points(html_out)
    means = [m for _a, m, _s in rendered]
    assert any(m > 0.0 for m in means), (
        "renderer must show the masked agent's real nonzero V(pi) values -- "
        "a flat-zero-everywhere render would be the tempting-but-false 'clean' version"
    )
    # the real magnitude at the plateau, not just "some small nonzero noise"
    assert max(means) == 0.3


def test_boundary_callout_names_the_real_first_nonzero_alpha_and_value():
    masked = _real_masked_series()
    html_out = render_dose_response_html([masked])

    assert "&alpha;=0.5" in html_out
    assert "0.0016" in html_out
    assert "0.3000" in html_out  # the real plateau value
    assert "&alpha;=0.9" in html_out


def test_flat_zero_series_gets_the_honest_flat_zero_callout_not_a_fake_boundary():
    flat_zero = DoseResponseSeries(
        label="Hypothetically flat",
        series_key="masked",
        points=[DoseResponsePoint(alpha=a, mean=0.0, std=0.0) for a in [0.0, 0.5, 1.0]],
    )
    html_out = render_dose_response_html([flat_zero])

    assert "flat zero across the full range shown" in html_out
    assert "first becomes measurably nonzero" not in html_out


# ---------------------------------------------------------------------------
# No-fabrication check
# ---------------------------------------------------------------------------


def test_no_rendered_point_has_a_value_absent_from_the_input():
    series = [_real_masked_series(), _real_soft_series()]
    html_out = render_dose_response_html(series)

    real_points = {(p.alpha, p.mean, p.std) for s in series for p in s.points}
    for triple in _extract_points(html_out):
        assert triple in real_points


# ---------------------------------------------------------------------------
# Well-formedness
# ---------------------------------------------------------------------------


def test_rendered_html_has_balanced_tags():
    html_out = render_dose_response_html([_real_masked_series(), _real_soft_series()])
    checker = _BalancedTagChecker()
    checker.feed(html_out)
    checker.close()
    assert checker.stack == []


def test_write_dose_response_html_writes_matching_content(tmp_path):
    series = [_real_masked_series()]
    out_path = tmp_path / "dose.html"
    returned = write_dose_response_html(series, out_path)

    assert returned == out_path
    assert out_path.read_text(encoding="utf-8") == render_dose_response_html(series)
