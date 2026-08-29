"""Behavioral tests for `dashboard.render_explain` (Hard Rule 10 --
the rendered Explain Decision panel may only display values the real
`DecisionTrace` object already carries, never a generated narrative).

Every trace rendered here comes from calling the real
`dashboard.explain.explain_decision`/`explain_decision_from_env`
functions (constructed inputs for the targeted cases, a real stepped
`SmartKeyNetEnv` for the end-to-end/Hard-Rule-10 case) -- never a
hand-built `DecisionTrace` object, so a passing test is evidence about
the real rendering path, not a mock.
"""

from __future__ import annotations

import html as html_lib
import itertools
import re
from html.parser import HTMLParser

import pytest

from dashboard.explain import DecisionTrace, explain_decision, explain_decision_from_env
from dashboard.render_explain import render_trace_html, write_trace_html
from env.contracts import Action, KeyType, Request, SensitivityClass, ThreatPosture
from env.environment import SmartKeyNetEnv
from env.masking import PolicyTable, load_key_lifetime_config

MAX_KEY_AGE = load_key_lifetime_config()["max_key_age_steps"]

_COLD_START_ONEHOT = [0.0, 0.0, 0.0]


def _onehot(key_type: KeyType | None) -> list[float]:
    onehot = [0.0, 0.0, 0.0]
    if key_type is not None:
        onehot[int(key_type)] = 1.0
    return onehot


def make_request(sensitivity_class: int = 0, hybrid_mandatory: bool = False) -> Request:
    return Request(
        request_id="r0",
        step=0,
        tenant="hospital",
        service="export",
        sensitivity_class=sensitivity_class,
        pqc_capable=True,
        hybrid_mandatory=hybrid_mandatory,
    )


def _default_kwargs(**overrides):
    kwargs = dict(
        request=make_request(sensitivity_class=0),
        threat_score=0.1,
        threat_source="test",
        posture_probs=None,
        floor=Action.SERVE_CLASSICAL,
        key_age=0.0,
        max_key_age=MAX_KEY_AGE,
        pool_can_draw=True,
        key_type_onehot=_COLD_START_ONEHOT,
        chosen_action=Action.SERVE_CLASSICAL,
    )
    kwargs.update(overrides)
    return kwargs


def _base_config(**overrides):
    config = {
        "scenario": "S1",
        "seed": 0,
        "use_foresight": "off",
        "pool": {"capacity_bits": 1_000_000, "initial_fill_frac": 0.5, "bits_per_hybrid_draw": 256},
        "key_lifetime": {"max_key_age_steps": MAX_KEY_AGE},
        "reward": {
            "w_lat": 1.0,
            "w_en": 0.1,
            "w_fr": 0.1,
            "w_qkd": 1.0,
            "r_starve": 10.0,
            "c_rekey_base": 1.0,
            "c_rekey_load_beta": 1.0,
        },
        "max_steps": 20,
    }
    config.update(overrides)
    return config


class _BalancedTagChecker(HTMLParser):
    """Tracks open/close tags (ignoring void elements) to confirm the
    rendered page is well-formed enough to actually open in a browser."""

    _VOID_ELEMENTS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in self._VOID_ELEMENTS:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass  # explicit self-closing tag (e.g. <br/>) -- never pushed

    def handle_endtag(self, tag):
        assert self.stack, f"</{tag}> with no matching open tag"
        assert self.stack[-1] == tag, f"expected </{self.stack[-1]}>, got </{tag}>"
        self.stack.pop()


# ---------------------------------------------------------------------------
# Real values from a known, constructed DecisionTrace appear verbatim
# ---------------------------------------------------------------------------


def test_render_shows_the_real_floor_chosen_action_and_final_sentence():
    trace = explain_decision(
        **_default_kwargs(
            request=make_request(sensitivity_class=int(SensitivityClass.S2)),
            floor=Action.SERVE_PQC,
            posture_probs=[0.9, 0.08, 0.02],
            key_type_onehot=_COLD_START_ONEHOT,
            chosen_action=Action.SERVE_PQC,
        )
    )
    html_out = render_trace_html(trace)

    assert f'<span class="final-chip">{trace.chosen_action.name}</span>' in html_out
    assert html_lib.escape(trace.final_text) in html_out
    assert f"floor = {trace.floor.name}" in html_out
    assert f"{trace.sensitivity_class.name} + {trace.resolved_posture.name}" in html_out


def test_render_shows_every_real_per_action_mask_reason():
    trace = explain_decision(
        **_default_kwargs(
            request=make_request(sensitivity_class=int(SensitivityClass.S3)),
            floor=Action.SERVE_HYBRID,
            key_age=MAX_KEY_AGE,  # cold-start sessions are pinned here by environment.py
            pool_can_draw=True,
            key_type_onehot=_COLD_START_ONEHOT,
            chosen_action=Action.SERVE_HYBRID,
        )
    )
    html_out = render_trace_html(trace)

    for entry in trace.mask:
        assert entry.action.name in html_out
        assert html_lib.escape(entry.reason) in html_out


def test_render_shows_real_cost_numbers_for_a_genuine_tradeoff():
    trace = explain_decision(
        **_default_kwargs(
            request=make_request(sensitivity_class=int(SensitivityClass.S0)),
            floor=Action.SERVE_CLASSICAL,
            key_type_onehot=_COLD_START_ONEHOT,
            chosen_action=Action.SERVE_HYBRID,  # cheaper SERVE_CLASSICAL was legal but not chosen
        )
    )
    html_out = render_trace_html(trace)

    assert len(trace.costs) > 1
    assert trace.cost_note is None
    for cost_entry in trace.costs:
        assert f"lat {cost_entry.latency:g}" in html_out
        assert f"en {cost_entry.energy:g}" in html_out
    assert "the policy's own preference among legal options" in trace.final_text
    assert html_lib.escape(trace.final_text) in html_out


def test_render_shows_the_real_cost_note_when_only_one_legal_action_exists():
    # HYBRID-floor-with-empty-pool: SERVE_CLASSICAL/PQC below floor,
    # SERVE_HYBRID blocked by the pool, REUSE blocked by cold start --
    # only REKEY_NOW legal, so dashboard/explain.py's own _cost_entries
    # sets cost_note (the real "no tradeoff" degradation this session's
    # brief asks the renderer to show honestly, not fake).
    trace = explain_decision(
        **_default_kwargs(
            request=make_request(sensitivity_class=int(SensitivityClass.S3)),
            floor=Action.SERVE_HYBRID,
            key_age=MAX_KEY_AGE,  # cold-start sessions are pinned here by environment.py
            pool_can_draw=False,
            key_type_onehot=_COLD_START_ONEHOT,
            chosen_action=Action.REKEY_NOW,
        )
    )
    assert trace.cost_note is not None
    assert len(trace.costs) == 1

    html_out = render_trace_html(trace)
    assert f'<div class="cost-note">{html_lib.escape(trace.cost_note)}</div>' in html_out
    assert trace.chosen_action.name in html_out


# ---------------------------------------------------------------------------
# Step 3: the floor grid highlights the REAL fired cell, not a hardcoded one
# ---------------------------------------------------------------------------


def test_floor_grid_highlights_exactly_the_traces_real_floor_cell():
    for sensitivity_class, posture in itertools.product(SensitivityClass, ThreatPosture):
        table = PolicyTable()
        real_floor = table.floor(sensitivity_class, posture)
        probs = [0.01, 0.01, 0.01]
        probs[int(posture)] = 1.0

        trace = explain_decision(
            **_default_kwargs(
                request=make_request(sensitivity_class=int(sensitivity_class)),
                posture_probs=probs,
                floor=real_floor,
                chosen_action=real_floor,
            )
        )
        html_out = render_trace_html(trace)

        expected_cell = f"{sensitivity_class.name}-{posture.name}"
        # render_explain.py emits `class="floor-cell <tier> [hit]" data-cell="<cell>"`
        # in that fixed attribute order -- match it directly.
        all_cells = re.findall(r'<div class="floor-cell ([^"]*)" data-cell="([^"]+)">', html_out)
        hit_cells = [cell for classes, cell in all_cells if "hit" in classes.split()]
        assert hit_cells == [expected_cell], (
            f"expected exactly one 'hit' cell at {expected_cell}, found {hit_cells}"
        )
        assert f'floor = {real_floor.name}' in html_out


# ---------------------------------------------------------------------------
# Hard Rule 10: the renderer adds no narrative beyond the trace's own fields
# ---------------------------------------------------------------------------


def test_hard_rule_10_every_rendered_reason_and_final_text_comes_from_the_trace():
    """End-to-end over a real stepped env: render every real decision's
    trace and verify (a) every mask reason shown is exactly one of that
    trace's own `MaskEntry.reason` strings (never invented), and (b) the
    final sentence shown is exactly `trace.final_text` (never a
    different, renderer-generated sentence)."""
    env = SmartKeyNetEnv(_base_config())
    state, info = env.reset(seed=0)

    checked = 0
    for _ in range(15):
        mask = info["action_mask"]
        chosen = next(a for a in Action if bool(mask[int(a)]))
        trace = explain_decision_from_env(env, state, chosen)
        html_out = render_trace_html(trace)

        # (a) per-action reasons: exactly the trace's own reasons, in order
        rendered_reasons = re.findall(r'<div class="a-reason">(.*?)</div>', html_out, re.DOTALL)
        real_reasons = [html_lib.escape(entry.reason) for entry in trace.mask]
        assert rendered_reasons == real_reasons

        # (b) final sentence: exactly trace.final_text, nothing else
        final_text_match = re.search(r'<span class="final-text">(.*?)</span>', html_out, re.DOTALL)
        assert final_text_match is not None
        assert final_text_match.group(1) == html_lib.escape(trace.final_text)

        # (c) chosen action name only ever appears alongside real trace values
        assert trace.chosen_action.name in html_out

        checked += 1
        state, _reward, _terminated, truncated, info = env.step(chosen)
        if truncated:
            break

    assert checked > 0


def test_hard_rule_10_cost_rows_are_exactly_the_traces_own_legal_actions():
    env = SmartKeyNetEnv(_base_config(use_foresight="ewma"))
    state, info = env.reset(seed=1)

    mask = info["action_mask"]
    chosen = next(a for a in Action if bool(mask[int(a)]))
    trace = explain_decision_from_env(env, state, chosen)
    html_out = render_trace_html(trace)

    rendered_cost_names = re.findall(r'<div class="cost-name">([A-Z_]+)', html_out)
    real_cost_names = [c.action.name for c in trace.costs]
    assert rendered_cost_names == real_cost_names


# ---------------------------------------------------------------------------
# Basic HTML well-formedness
# ---------------------------------------------------------------------------


def test_rendered_html_has_balanced_tags():
    trace = explain_decision(**_default_kwargs())
    html_out = render_trace_html(trace)

    checker = _BalancedTagChecker()
    checker.feed(html_out)
    checker.close()
    assert checker.stack == []


def test_rendered_html_is_well_formed_for_a_real_end_to_end_trace():
    env = SmartKeyNetEnv(_base_config(use_foresight="ewma"))
    state, info = env.reset(seed=2)
    mask = info["action_mask"]
    chosen = next(a for a in Action if bool(mask[int(a)]))
    trace = explain_decision_from_env(env, state, chosen)

    checker = _BalancedTagChecker()
    checker.feed(render_trace_html(trace))
    checker.close()
    assert checker.stack == []


# ---------------------------------------------------------------------------
# write_trace_html
# ---------------------------------------------------------------------------


def test_write_trace_html_writes_the_same_content_to_disk(tmp_path):
    trace = explain_decision(**_default_kwargs())
    out_path = tmp_path / "nested" / "trace.html"

    returned = write_trace_html(trace, out_path)

    assert returned == out_path
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == render_trace_html(trace)
