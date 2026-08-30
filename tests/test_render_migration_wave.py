"""Behavioral tests for `dashboard.render_migration_wave` (Hard Rule 7,
central this session): the rendered timeline must show S6's real
scripted schedule and each tenant's real before/after floor honestly --
never crediting the scripted migration for a floor change the real
data shows was already there (load-driven posture), and never hiding a
genuine floor change that really happened.

The event/observation values below are copied verbatim from the real,
held-out S6 episode (seed=900, `checkpoints/dqn_s1.pt`) this session's
own `dashboard/render_migration_wave_demo.py` produced -- see
`dashboard/samples/migration_data.json` for the full real run. The
real scripted schedule itself (steps 60/130/190, tenant_index 0/3/4,
new_sensitivity_class 3/3/2) is `configs/scenarios/s6_migration.yaml::
migration_schedule`, read directly, not retyped by hand from memory.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from dashboard.render_migration_wave import (
    FloorObservation,
    MigrationEventView,
    MigrationWaveData,
    PoolTrajectoryPoint,
    attribute_floor_change,
    render_migration_wave_html,
    write_migration_wave_html,
)
from env.contracts import Action, KeyType, SensitivityClass, ThreatPosture
from env.masking import _PLACEHOLDER_FLOOR_TABLE
from experiments.train import load_full_config

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REAL_SCHEDULE = load_full_config(
    _REPO_ROOT / "configs" / "scenarios" / "s6_migration.yaml"
)["migration_schedule"]


# ---------------------------------------------------------------------------
# attribute_floor_change -- the honesty gate (Hard Rule 7)
# ---------------------------------------------------------------------------


def _obs(step, tenant, sens, posture, floor, served=KeyType.PQC):
    return FloorObservation(
        step=step, tenant=tenant, service="ingest", sensitivity_class=sens, posture=posture,
        floor=floor, served_tier=served,
    )


def test_no_before_observation_never_claims_scripted():
    after = _obs(79, "tenant_0", SensitivityClass.S3, ThreatPosture.ELEVATED, Action.SERVE_HYBRID)
    attribution, note = attribute_floor_change(
        event_step=60,
        old_sensitivity_class=SensitivityClass.S1,
        new_sensitivity_class=SensitivityClass.S3,
        before=None,
        after=after,
    )
    assert attribution == "no_before_observation"
    assert attribution != "scripted"
    assert "no floor-change claim is made" in note


def test_no_after_observation_never_claims_scripted():
    before = _obs(39, "tenant_0", SensitivityClass.S1, ThreatPosture.ELEVATED, Action.SERVE_PQC)
    attribution, note = attribute_floor_change(
        event_step=60,
        old_sensitivity_class=SensitivityClass.S1,
        new_sensitivity_class=SensitivityClass.S3,
        before=before,
        after=None,
    )
    assert attribution == "no_after_observation"
    assert attribution != "scripted"


def test_posture_confound_never_claims_scripted():
    """A floor increase that coincides with a posture shift (not just
    the scripted class change) must not be credited to the schedule."""
    before = _obs(121, "tenant_3", SensitivityClass.S2, ThreatPosture.CALM, Action.SERVE_CLASSICAL)
    after = _obs(157, "tenant_3", SensitivityClass.S3, ThreatPosture.HIGH, Action.SERVE_HYBRID)
    attribution, note = attribute_floor_change(
        event_step=130,
        old_sensitivity_class=SensitivityClass.S2,
        new_sensitivity_class=SensitivityClass.S3,
        before=before,
        after=after,
    )
    assert attribution == "posture_confound"
    assert attribution != "scripted"
    assert "cannot be cleanly attributed" in note


def test_no_visible_change_when_floor_already_at_post_migration_level():
    """The central Hard Rule 7 guard: if the real floor was ALREADY at
    the post-migration tier before the event fired (because posture
    alone already forced it there), the panel must NOT claim the
    scripted migration raised the floor -- this is exactly the failure
    mode the session's framing warned about."""
    before = _obs(118, "tenant_3", SensitivityClass.S2, ThreatPosture.HIGH, Action.SERVE_HYBRID)
    after = _obs(157, "tenant_3", SensitivityClass.S3, ThreatPosture.HIGH, Action.SERVE_HYBRID)
    attribution, note = attribute_floor_change(
        event_step=130,
        old_sensitivity_class=SensitivityClass.S2,
        new_sensitivity_class=SensitivityClass.S3,
        before=before,
        after=after,
    )
    assert attribution == "no_visible_change"
    assert attribution != "scripted"
    assert "did NOT visibly raise" in note


def test_scripted_when_posture_held_constant_and_floor_genuinely_rises():
    """The real seed=900 tenant_3 case: posture identical (ELEVATED)
    before and after, floor genuinely rises PQC -> HYBRID."""
    before = _obs(121, "tenant_3", SensitivityClass.S2, ThreatPosture.ELEVATED, Action.SERVE_PQC)
    after = _obs(243, "tenant_3", SensitivityClass.S3, ThreatPosture.ELEVATED, Action.SERVE_HYBRID)
    attribution, note = attribute_floor_change(
        event_step=130,
        old_sensitivity_class=SensitivityClass.S2,
        new_sensitivity_class=SensitivityClass.S3,
        before=before,
        after=after,
    )
    assert attribution == "scripted"
    assert "SAME posture (ELEVATED) throughout" in note


def test_scripted_note_flags_when_pre_event_floor_already_above_calm_baseline():
    """Even in a genuine "scripted" case, the note must surface that
    the pre-event floor was already elevated above the CALM baseline
    for the OLD class by this episode's own load-driven posture ratchet
    -- not silently implying the climb started from CLASSICAL."""
    before = _obs(179, "tenant_4", SensitivityClass.S0, ThreatPosture.ELEVATED, Action.SERVE_CLASSICAL)
    after = _obs(199, "tenant_4", SensitivityClass.S2, ThreatPosture.ELEVATED, Action.SERVE_PQC)
    attribution, note = attribute_floor_change(
        event_step=190,
        old_sensitivity_class=SensitivityClass.S0,
        new_sensitivity_class=SensitivityClass.S2,
        before=before,
        after=after,
    )
    assert attribution == "scripted"
    # S0/CALM's real floor is SERVE_CLASSICAL too, so before.floor == calm baseline here --
    # this case should NOT trigger the "already elevated" caveat (real, honest: check both sides).
    assert _PLACEHOLDER_FLOOR_TABLE[(SensitivityClass.S0, ThreatPosture.CALM)] == Action.SERVE_CLASSICAL
    assert "already above" not in note

    # Now a genuinely already-elevated case (S1/ELEVATED gives PQC, above S1/CALM's CLASSICAL):
    before2 = _obs(39, "tenant_0", SensitivityClass.S1, ThreatPosture.ELEVATED, Action.SERVE_PQC)
    after2 = _obs(84, "tenant_0", SensitivityClass.S3, ThreatPosture.ELEVATED, Action.SERVE_HYBRID)
    attribution2, note2 = attribute_floor_change(
        event_step=60,
        old_sensitivity_class=SensitivityClass.S1,
        new_sensitivity_class=SensitivityClass.S3,
        before=before2,
        after=after2,
    )
    assert attribution2 == "scripted"
    assert "already above the S1/CALM baseline" in note2


def test_before_after_floors_match_real_masking_table_lookup():
    """Cross-check against the REAL floor table (imported, not
    reimplemented) -- the real seed=900 tenant_3/tenant_4 observations'
    floors must equal what `env/masking.py`'s own table gives for their
    real (class, posture) pairs."""
    assert _PLACEHOLDER_FLOOR_TABLE[(SensitivityClass.S2, ThreatPosture.ELEVATED)] == Action.SERVE_PQC
    assert _PLACEHOLDER_FLOOR_TABLE[(SensitivityClass.S3, ThreatPosture.ELEVATED)] == Action.SERVE_HYBRID
    assert _PLACEHOLDER_FLOOR_TABLE[(SensitivityClass.S0, ThreatPosture.ELEVATED)] == Action.SERVE_CLASSICAL
    assert _PLACEHOLDER_FLOOR_TABLE[(SensitivityClass.S2, ThreatPosture.ELEVATED)] == Action.SERVE_PQC


# ---------------------------------------------------------------------------
# Real schedule / rendering
# ---------------------------------------------------------------------------


def test_real_schedule_values_loaded_from_config_are_the_expected_three_events():
    """Sanity: the real schedule this test module reads directly off
    `configs/scenarios/s6_migration.yaml` is the one this panel and its
    demo driver were built against."""
    assert len(_REAL_SCHEDULE) == 3
    assert _REAL_SCHEDULE[0] == {"step": 60, "tenant_index": 0, "new_sensitivity_class": 3}
    assert _REAL_SCHEDULE[1] == {"step": 130, "tenant_index": 3, "new_sensitivity_class": 3}
    assert _REAL_SCHEDULE[2] == {"step": 190, "tenant_index": 4, "new_sensitivity_class": 2}


def _real_events() -> tuple[MigrationEventView, ...]:
    """The three real `MigrationEventView`s from the actual seed=900
    `dashboard/render_migration_wave_demo.py` run, values copied
    verbatim from `dashboard/samples/migration_data.json`."""
    ev1_after = _obs(79, "tenant_0", SensitivityClass.S3, ThreatPosture.ELEVATED, Action.SERVE_HYBRID, KeyType.HYBRID)
    ev1_attr, ev1_note = attribute_floor_change(
        event_step=60, old_sensitivity_class=SensitivityClass.S1, new_sensitivity_class=SensitivityClass.S3,
        before=None, after=ev1_after,
    )
    ev1 = MigrationEventView(
        step=60, tenant_id="tenant_0", old_sensitivity_class=SensitivityClass.S1,
        new_sensitivity_class=SensitivityClass.S3, before=None, after=ev1_after,
        attribution=ev1_attr, attribution_note=ev1_note,
    )

    ev2_before = _obs(121, "tenant_3", SensitivityClass.S2, ThreatPosture.ELEVATED, Action.SERVE_PQC, KeyType.PQC)
    ev2_after = _obs(243, "tenant_3", SensitivityClass.S3, ThreatPosture.ELEVATED, Action.SERVE_HYBRID, KeyType.HYBRID)
    ev2_attr, ev2_note = attribute_floor_change(
        event_step=130, old_sensitivity_class=SensitivityClass.S2, new_sensitivity_class=SensitivityClass.S3,
        before=ev2_before, after=ev2_after,
    )
    ev2 = MigrationEventView(
        step=130, tenant_id="tenant_3", old_sensitivity_class=SensitivityClass.S2,
        new_sensitivity_class=SensitivityClass.S3, before=ev2_before, after=ev2_after,
        attribution=ev2_attr, attribution_note=ev2_note,
    )

    ev3_before = _obs(179, "tenant_4", SensitivityClass.S0, ThreatPosture.ELEVATED, Action.SERVE_CLASSICAL, KeyType.PQC)
    ev3_after = _obs(199, "tenant_4", SensitivityClass.S2, ThreatPosture.ELEVATED, Action.SERVE_PQC, KeyType.PQC)
    ev3_attr, ev3_note = attribute_floor_change(
        event_step=190, old_sensitivity_class=SensitivityClass.S0, new_sensitivity_class=SensitivityClass.S2,
        before=ev3_before, after=ev3_after,
    )
    ev3 = MigrationEventView(
        step=190, tenant_id="tenant_4", old_sensitivity_class=SensitivityClass.S0,
        new_sensitivity_class=SensitivityClass.S2, before=ev3_before, after=ev3_after,
        attribution=ev3_attr, attribution_note=ev3_note,
    )
    return (ev1, ev2, ev3)


def _real_data() -> MigrationWaveData:
    trajectory = tuple(PoolTrajectoryPoint(step=s, pool_fill=1.0) for s in (2, 60, 130, 190, 250))
    return MigrationWaveData(
        scenario="S6",
        seed=900,
        policy_label="Masked DQN",
        checkpoint_note="checkpoints/dqn_s1.pt, trained on S1 (steady state), held-out eval on S6",
        n_decisions=250,
        trajectory=trajectory,
        events=_real_events(),
    )


def test_rendered_html_shows_all_three_real_schedule_events():
    html_out = render_migration_wave_html(_real_data())
    assert "step 60" in html_out
    assert "tenant_0: S1" in html_out and "S3" in html_out
    assert "step 130" in html_out
    assert "tenant_3: S2" in html_out
    assert "step 190" in html_out
    assert "tenant_4: S0" in html_out


def test_rendered_html_shows_no_before_data_badge_for_event_without_observation():
    html_out = render_migration_wave_html(_real_data())
    assert "no before data" in html_out
    assert "not observed" in html_out


def test_rendered_html_shows_scripted_badge_for_attributable_events():
    html_out = render_migration_wave_html(_real_data())
    assert html_out.count('class="badge good">scripted<') == 2


def test_rendered_html_shows_real_floor_values_for_observed_events():
    html_out = render_migration_wave_html(_real_data())
    assert "SERVE_PQC" in html_out
    assert "SERVE_HYBRID" in html_out
    assert "SERVE_CLASSICAL" in html_out
    assert "t=121" in html_out
    assert "t=243" in html_out
    assert "t=179" in html_out
    assert "t=199" in html_out


def test_flat_pool_trajectory_gets_honest_no_scarcity_caption():
    """S6 has no QKD-degradation mechanism -- a flat real trajectory
    must get the honest "no observable pool-scarcity response" caption,
    never a fabricated dip/recovery narrative (the mockup's own, 100%
    fabricated, "pool dips and recovers" story must not leak in)."""
    html_out = render_migration_wave_html(_real_data())
    assert "no observable pool-scarcity response" in html_out
    assert "reallocates from discretionary hybrid serves" not in html_out  # the mockup's fabricated line


def test_nonflat_pool_trajectory_gets_range_caption_not_the_flat_caption():
    trajectory = (
        PoolTrajectoryPoint(step=0, pool_fill=1.0),
        PoolTrajectoryPoint(step=100, pool_fill=0.4),
        PoolTrajectoryPoint(step=200, pool_fill=0.9),
    )
    data = MigrationWaveData(
        scenario="S6", seed=900, policy_label="Masked DQN", checkpoint_note="test",
        n_decisions=3, trajectory=trajectory, events=(),
    )
    html_out = render_migration_wave_html(data)
    assert "no observable pool-scarcity response" not in html_out
    assert "0.4000" in html_out or "0.4" in html_out


def test_no_fabrication_every_tenant_and_step_is_from_real_input():
    data = _real_data()
    html_out = render_migration_wave_html(data)
    real_tenants = {ev.tenant_id for ev in data.events}
    # every "tenant_N" substring appearing must be a member of the real input's tenants
    import re as _re
    for match in _re.findall(r"tenant_\d+", html_out):
        assert match in real_tenants, f"fabricated tenant id: {match}"


def test_mockup_fabricated_tenant_names_never_appear():
    html_out = render_migration_wave_html(_real_data())
    for fake_name in ("hospital", "fintech", "logging", "iot-telemetry"):
        assert fake_name not in html_out


# ---------------------------------------------------------------------------
# HTML well-formedness
# ---------------------------------------------------------------------------


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


def test_rendered_html_has_balanced_tags():
    html_out = render_migration_wave_html(_real_data())
    checker = _BalancedTagChecker()
    checker.feed(html_out)
    checker.close()
    assert checker.stack == []


def test_rendered_html_with_no_events_is_still_well_formed():
    data = MigrationWaveData(
        scenario="S6", seed=1, policy_label="x", checkpoint_note="x",
        n_decisions=0, trajectory=(), events=(),
    )
    html_out = render_migration_wave_html(data)
    checker = _BalancedTagChecker()
    checker.feed(html_out)
    checker.close()
    assert checker.stack == []


def test_write_migration_wave_html_writes_matching_content(tmp_path):
    data = _real_data()
    out_path = tmp_path / "migration.html"
    returned = write_migration_wave_html(data, out_path)

    assert returned == out_path
    assert out_path.read_text(encoding="utf-8") == render_migration_wave_html(data)
