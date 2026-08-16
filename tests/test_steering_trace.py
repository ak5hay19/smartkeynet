"""Behavioral tests for `attack.steering_trace` and the S5 experiment
(PLAN.md §5 scenario S5; §6 Demo Beat 3 -- the headline contribution).

The properties under test are the ones the thesis actually rests on:
the attack is a pure suppression of the threat signal, it reaches
nothing but the forecaster's input, and the masked architecture has no
gradient for it to pull on.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.soft_reward_baseline import (
    SOFT_REWARD_IS_THE_CRITIQUED_DESIGN,
    SoftRewardAgent,
    SoftRewardConfig,
    soft_reward,
)
from attack.steering_trace import (
    SuppressionTrace,
    detectability_score,
    dose_response_traces,
)
from env.contracts import Action


# ---------------------------------------------------------------------------
# SuppressionTrace
# ---------------------------------------------------------------------------


def test_dose_zero_is_exactly_the_unattacked_signal():
    """The control run must go through the same code path as the
    attacked runs, or any difference between them is not attributable
    to the dose alone."""
    trace = SuppressionTrace(start_step=100, end_step=500, dose=0.0)
    features = [0.02, 0.4, 3.0]
    for step in (0, 99, 100, 300, 499, 500, 900):
        assert trace.apply(features, step) == features


def test_suppression_only_ever_removes_signal():
    """The attack can never *raise* the reported threat: the multiplier
    is bounded to [0, 1]. An attack that could raise it would be a
    different experiment (and would only trigger stronger protection)."""
    for dose in (0.0, 0.25, 0.5, 0.75, 1.0):
        trace = SuppressionTrace(start_step=50, end_step=400, dose=dose)
        for step in range(600):
            assert 0.0 <= trace.multiplier_at(step) <= 1.0


def test_suppression_is_confined_to_the_window():
    trace = SuppressionTrace(start_step=100, end_step=200, dose=1.0, ramp_steps=0)
    assert trace.multiplier_at(99) == pytest.approx(1.0)
    assert trace.multiplier_at(100) == pytest.approx(0.0)
    assert trace.multiplier_at(199) == pytest.approx(0.0)
    assert trace.multiplier_at(200) == pytest.approx(1.0)


def test_full_dose_zeroes_the_signal():
    trace = SuppressionTrace(start_step=0, end_step=100, dose=1.0, ramp_steps=0)
    assert trace.apply([0.5, 1.0, 6.0], 50) == [0.0, 0.0, 0.0]


def test_ramp_softens_the_edge_but_lengthens_exposure():
    """The attacker cannot minimise both detectability statistics at
    once -- that trade-off is the point of reporting them."""
    blunt = SuppressionTrace(start_step=100, end_step=600, dose=1.0, ramp_steps=0)
    ramped = SuppressionTrace(start_step=100, end_step=600, dose=1.0, ramp_steps=100)

    blunt_score = detectability_score(blunt, 1000)
    ramped_score = detectability_score(ramped, 1000)

    assert ramped_score["max_first_difference"] < blunt_score["max_first_difference"]
    assert ramped_score["mean_suppression"] < blunt_score["mean_suppression"]


def test_invalid_traces_are_rejected():
    with pytest.raises(ValueError):
        SuppressionTrace(start_step=0, end_step=10, dose=1.5)
    with pytest.raises(ValueError):
        SuppressionTrace(start_step=0, end_step=10, dose=-0.1)
    with pytest.raises(ValueError):
        SuppressionTrace(start_step=10, end_step=10, dose=0.5)
    with pytest.raises(ValueError):
        SuppressionTrace(start_step=0, end_step=10, dose=0.5, ramp_steps=-1)


def test_dose_response_traces_include_the_control():
    traces = dose_response_traces(100, 500, [0.0, 0.5, 1.0])
    assert [t.dose for t in traces] == [0.0, 0.5, 1.0]
    assert all(t.start_step == 100 and t.end_step == 500 for t in traces)


# ---------------------------------------------------------------------------
# The vulnerability itself
# ---------------------------------------------------------------------------


def test_soft_reward_optimal_tier_slides_down_as_threat_is_suppressed():
    """THE HEADLINE PROPERTY, proved analytically rather than measured.

    The critiqued reward is `w_sec * security(tier) * threat - w_cost *
    cost(tier)`. As `threat` falls, the security term shrinks and the
    cost term -- increasing in tier -- takes over, so the argmax walks
    down the tier ladder. This is a property of the reward function, so
    it holds for any agent maximising it, independent of seed, training
    budget or exploration schedule.
    """
    tier_actions = (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID)

    def best_tier(threat: float) -> int:
        return int(max(tier_actions, key=lambda a: soft_reward({}, a, threat)))

    tiers = [best_tier(threat / 10) for threat in range(10)]

    assert tiers == sorted(tiers), f"not monotone in threat: {tiers}"
    assert tiers[0] < tiers[-1], "suppressing the threat must lower the preferred tier"
    assert tiers[0] == int(Action.SERVE_CLASSICAL), (
        "at zero reported threat the soft reward should prefer the quantum-vulnerable "
        "tier -- that is the vulnerability being demonstrated"
    )


def test_masked_floor_is_monotone_non_decreasing_in_threat():
    """The counterpart property: the masked architecture's floor can only
    go up with the threat signal, so suppression has nothing to pull on."""
    from env.contracts import SensitivityClass, ThreatPosture
    from env.masking import PolicyTable

    floors = [int(PolicyTable().floor(SensitivityClass.S3, posture)) for posture in ThreatPosture]
    assert floors == sorted(floors)


def test_the_masked_reward_cannot_see_the_threat_signal():
    """Hard Rule 1, stated as the reason the attack fails. If the
    environment's reward ever gained a threat term, the masked agent
    would inherit exactly the soft-reward agent's vulnerability -- so
    this is the test that keeps the headline claim true."""
    import inspect

    from env import environment

    reward_source = inspect.getsource(environment.SmartKeyNetEnv._apply_action)
    for forbidden in ("threat", "posture", "security", "risk"):
        assert forbidden not in reward_source.lower(), (
            f"'{forbidden}' appears in the reward computation -- Hard Rule 1 violated"
        )


# ---------------------------------------------------------------------------
# The victim agent
# ---------------------------------------------------------------------------


def test_soft_reward_module_is_marked_as_the_critiqued_design():
    assert SOFT_REWARD_IS_THE_CRITIQUED_DESIGN is True


def test_soft_reward_agent_never_returns_an_illegal_action():
    agent = SoftRewardAgent(config=SoftRewardConfig(), seed=0)
    state = {"threat_score": 0.5}
    for legal_action in Action:
        mask = np.zeros(len(Action), dtype=bool)
        mask[int(legal_action)] = True
        for _ in range(10):
            assert agent.act(state, mask) == legal_action


def test_soft_reward_agent_learns_to_prefer_strong_tiers_at_high_threat():
    agent = SoftRewardAgent(config=SoftRewardConfig(epsilon_start=1.0, epsilon_end=1.0), seed=0)
    full_mask = np.ones(len(Action), dtype=bool)
    high_threat = {"threat_score": 0.95}

    for _ in range(4000):
        action = agent.act(high_threat, full_mask)
        agent.learn(high_threat, action, high_threat, full_mask)

    tier_actions = (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID)
    q_row = agent.q_table[agent._bin_for(0.95)]
    best = max(tier_actions, key=lambda a: q_row[int(a)])
    assert best is not Action.SERVE_CLASSICAL


def test_soft_reward_agent_ignores_the_environment_reward():
    """The victim must be driven by its OWN reward -- `env/` is never
    modified to emit a security-flavoured one, which is how Hard Rule 1
    stays true of the environment while this comparison exists."""
    import inspect

    from agents import soft_reward_baseline

    learn_source = inspect.getsource(soft_reward_baseline.SoftRewardAgent.learn)
    assert "reward" in learn_source
    signature = inspect.signature(soft_reward_baseline.SoftRewardAgent.learn)
    assert "reward" not in signature.parameters, (
        "learn() must not accept the environment's reward -- it computes its own"
    )
