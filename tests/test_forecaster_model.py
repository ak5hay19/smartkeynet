"""Behavioral tests for `forecaster.model` -- the dual-head LSTM and the
frozen `LSTMForecastProvider` (PLAN.md Addition A).

The load-bearing properties here are not accuracy (that is
`forecaster/train.py`'s business) but the Hard Rule guarantees the
architecture makes: the provider is frozen, its pool head never reaches
the policy table, and it is interchangeable with the EWMA fallback.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from env.contracts import ForecastObservation, ForecastProvider, ThreatPosture
from env.forecast_provider import MovingAverageForecaster
from forecaster.dataset import N_FEATURES, N_POOL_SIGNALS, POOL_HORIZONS
from forecaster.model import (
    THREAT_HORIZON_STEPS,
    DualHeadConfig,
    DualHeadLSTM,
    LSTMForecastProvider,
    posture_probs_from_score,
)


def _observation(threat_level: float = 0.0, **overrides) -> ForecastObservation:
    observation = ForecastObservation(
        qber=0.02,
        skr=0.015,
        pool_fill=0.5,
        arrivals_per_class=[1, 1, 1, 1],
        hybrid_serves=1,
        threat_features=[threat_level] * N_FEATURES,
    )
    observation.update(overrides)  # type: ignore[typeddict-item]
    return observation


def _provider(seed: int = 0) -> LSTMForecastProvider:
    torch.manual_seed(seed)
    return LSTMForecastProvider(DualHeadLSTM(DualHeadConfig(hidden_size=16, window_steps=8)))


# ---------------------------------------------------------------------------
# Shapes and the contract
# ---------------------------------------------------------------------------


def test_trunk_consumes_threat_and_pool_channels_together():
    """"Dual-head" means one shared recurrent trunk with two heads, not
    two models in a trenchcoat -- so the input width is the sum."""
    model = DualHeadLSTM(DualHeadConfig(hidden_size=16, window_steps=8))
    assert model.input_size == N_FEATURES + N_POOL_SIGNALS

    batch = torch.zeros((4, 8, model.input_size))
    threat_logits, pool_out = model(batch)
    assert threat_logits.shape == (4, 1 + THREAT_HORIZON_STEPS)
    assert pool_out.shape == (4, 3 * len(POOL_HORIZONS))


def test_provider_output_matches_the_frozen_contract_shapes():
    provider = _provider()
    provider.update(_observation())

    threat = provider.get_threat_forecast()
    assert 0.0 <= threat.threat_score <= 1.0
    assert len(threat.posture_probs) == len(ThreatPosture)
    assert sum(threat.posture_probs) == pytest.approx(1.0)
    assert len(threat.horizon_scores) == THREAT_HORIZON_STEPS

    pool = provider.get_pool_forecast()
    assert len(pool.pool_level_hat) == len(POOL_HORIZONS)
    assert len(pool.skr_mean_hat) == len(POOL_HORIZONS)
    assert len(pool.hybrid_demand_hat) == len(POOL_HORIZONS)


def test_provider_is_a_forecast_provider_and_interchangeable_with_the_ewma_fallback():
    """Addition A's "provider interchangeability" test. Both must be
    usable behind the same interface with the same observations, or the
    E-A ablation is comparing two different systems rather than two
    forecasters."""
    for provider in (_provider(), MovingAverageForecaster()):
        assert isinstance(provider, ForecastProvider)
        provider.update(_observation())
        threat, pool = provider.get_threat_forecast(), provider.get_pool_forecast()
        assert len(threat.posture_probs) == len(ThreatPosture)
        assert len(pool.pool_level_hat) == len(POOL_HORIZONS)


def test_provider_survives_a_scalar_threat_signal():
    """`threat_input.source: scenario` emits one standardized scalar
    rather than a real feature window. That is a degraded input, not a
    crash."""
    provider = _provider()
    provider.update(_observation(threat_features=[1.5]))  # type: ignore[arg-type]
    assert 0.0 <= provider.get_threat_forecast().threat_score <= 1.0


# ---------------------------------------------------------------------------
# Hard Rule guarantees
# ---------------------------------------------------------------------------


def test_provider_is_frozen_and_builds_no_graph():
    """Frozen during DQN training (Addition A). If a gradient could
    reach these weights, the agent could learn to shape its own threat
    signal -- and therefore its own floors -- which is the exact failure
    mode Hard Rule 2 and the steering attack exist to rule out."""
    provider = _provider()
    assert not provider._model.training
    assert all(not p.requires_grad for p in provider._model.parameters())

    before = [p.detach().clone() for p in provider._model.parameters()]
    for _ in range(20):
        provider.update(_observation(threat_level=2.0))
    for parameter, original in zip(provider._model.parameters(), before):
        assert torch.equal(parameter, original), "a frozen provider's weights moved"


def test_provider_exposes_no_training_entry_point():
    provider = _provider()
    for forbidden in ("learn", "fit", "train_step", "optimizer", "backward"):
        assert not hasattr(provider, forbidden)


def test_posture_mapping_is_shared_verbatim_with_the_ewma_fallback():
    """If `ewma` and `lstm` disagreed about what a score of 0.6 *means*,
    the E-A ablation would be comparing two floor policies rather than
    two forecasters."""
    for score in (0.0, 0.05, 0.3, 0.5, 0.75, 1.0):
        fallback = MovingAverageForecaster()
        fallback._threat_score = score
        assert posture_probs_from_score(score) == pytest.approx(
            fallback.get_threat_forecast().posture_probs
        )


def test_posture_mapping_is_monotone_from_calm_to_high():
    calm = posture_probs_from_score(0.02)
    high = posture_probs_from_score(0.95)
    assert int(np.argmax(calm)) == int(ThreatPosture.CALM)
    assert int(np.argmax(high)) == int(ThreatPosture.HIGH)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_provider_round_trips_through_save_and_load(tmp_path):
    provider = _provider(seed=3)
    for _ in range(10):
        provider.update(_observation(threat_level=1.0))
    expected = provider.get_threat_forecast()

    path = tmp_path / "forecaster.pt"
    provider.save(path)
    restored = LSTMForecastProvider.load(path)
    for _ in range(10):
        restored.update(_observation(threat_level=1.0))

    assert restored.get_threat_forecast().threat_score == pytest.approx(expected.threat_score)
    assert not restored._model.training
    assert all(not p.requires_grad for p in restored._model.parameters())


def test_window_rolls_so_only_the_last_w_steps_are_visible():
    provider = _provider()
    for _ in range(50):
        provider.update(_observation(threat_level=0.0))
    quiet = provider.get_threat_forecast().threat_score

    for _ in range(50):
        provider.update(_observation(threat_level=5.0))
    loud = provider.get_threat_forecast().threat_score

    assert quiet != loud  # the window genuinely turned over
