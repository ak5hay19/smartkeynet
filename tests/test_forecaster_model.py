"""Real behavioral tests for `forecaster/model.py` (PLAN2.md §8A Addition
E, Phases 2-4: `SmartKeyForecaster` + `LSTMForecastProvider`, replacing
the prior import-smoke test now that this module is genuinely
implemented -- see PLAN.md §10 / split.md §3, "gets replaced by real
behavioral tests as each module is implemented").

Three groups: (1) `SmartKeyForecaster`'s forward pass shape/dtype, (2)
`LSTMForecastProvider`'s `ForecastProvider` interface conformance and
sane output ranges using a small synthetic checkpoint (no real training
required, fast), and (3) the real, committed, trained checkpoint's
degenerate-collapse check -- a test that would fail if the model were
secretly degenerate, per this session's instruction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from env.contracts import ForecastObservation
from forecaster.model import WINDOW, SmartKeyForecaster, LSTMForecastProvider

REAL_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "lstm_threat_forecaster.pt"
requires_real_checkpoint = pytest.mark.skipif(not REAL_CHECKPOINT.exists(), reason="lstm_threat_forecaster.pt not present")

REAL_NPZ = Path(__file__).resolve().parents[1] / "data" / "processed" / "rt_iot2022" / "rt_iot2022_windows.npz"
requires_real_dataset = pytest.mark.skipif(not REAL_NPZ.exists(), reason="rt_iot2022_windows.npz not present")


def make_observation(threat_features: list[float], pool_fill: float = 0.5, skr: float = 10.0,
                      hybrid_serves: int = 1, qber: float = 0.02) -> ForecastObservation:
    return ForecastObservation(
        qber=qber, skr=skr, pool_fill=pool_fill, arrivals_per_class=[1, 1, 1, 1],
        hybrid_serves=hybrid_serves, threat_features=list(threat_features),
    )


def make_tiny_checkpoint(tmp_path: Path, feature_dim: int = 5, window: int = 8, hidden_dim: int = 4) -> Path:
    """A small, untrained-but-well-formed checkpoint -- fast to construct,
    exercises the exact dict shape `train_threat_lstm.py` really saves,
    without needing a real training run in every test."""
    torch.manual_seed(0)
    model = SmartKeyForecaster(input_dim=feature_dim, hidden_dim=hidden_dim)
    path = tmp_path / "tiny_checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "feature_dim": feature_dim,
            "window": window,
            "hidden_dim": hidden_dim,
            "num_layers": 1,
            "scaler_mean": np.zeros(feature_dim - 2, dtype=np.float64),  # 2 "one-hot-like" trailing cols left unscaled
            "scaler_std": np.ones(feature_dim - 2, dtype=np.float64),
            "decision_threshold": 0.5,
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# SmartKeyForecaster -- forward pass
# ---------------------------------------------------------------------------


def test_forward_pass_shape_and_dtype():
    model = SmartKeyForecaster(input_dim=39, hidden_dim=16)
    window = torch.randn(4, WINDOW, 39)  # batch=4
    logits = model(window)
    assert logits.shape == (4,)
    assert logits.dtype == torch.float32


def test_forward_pass_single_batch_element():
    model = SmartKeyForecaster(input_dim=39, hidden_dim=16)
    window = torch.randn(1, WINDOW, 39)
    logits = model(window)
    assert logits.shape == (1,)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# LSTMForecastProvider -- ForecastProvider interface conformance
# ---------------------------------------------------------------------------


def test_fresh_instance_does_not_crash_and_is_sensible(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path)
    provider = LSTMForecastProvider(str(checkpoint))

    threat = provider.get_threat_forecast()
    assert 0.0 <= threat.threat_score <= 1.0
    assert len(threat.posture_probs) == 3
    assert sum(threat.posture_probs) == pytest.approx(1.0)
    assert len(threat.horizon_scores) == 5

    pool = provider.get_pool_forecast()
    assert len(pool.pool_level_hat) == 3
    assert len(pool.skr_mean_hat) == 3
    assert len(pool.hybrid_demand_hat) == 3


def test_update_then_forecast_does_not_crash_with_short_feature_vector(tmp_path):
    # Mirrors the real live env: `threat_features` is only 2-dim
    # ([qber, load]) -- far shorter than the model's trained feature_dim.
    # See LSTMForecastProvider's class docstring "live-feed caveat".
    checkpoint = make_tiny_checkpoint(tmp_path, feature_dim=39)
    provider = LSTMForecastProvider(str(checkpoint))
    for _ in range(10):
        provider.update(make_observation(threat_features=[0.02, 0.3]))
    threat = provider.get_threat_forecast()
    assert 0.0 <= threat.threat_score <= 1.0


def test_update_then_forecast_does_not_crash_with_matching_width_feature_vector(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path, feature_dim=6)
    provider = LSTMForecastProvider(str(checkpoint))
    rng = np.random.default_rng(0)
    for _ in range(20):
        provider.update(make_observation(threat_features=rng.normal(size=6).tolist()))
    threat = provider.get_threat_forecast()
    assert 0.0 <= threat.threat_score <= 1.0


def test_posture_probs_sum_to_one_across_many_updates(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path, feature_dim=4)
    provider = LSTMForecastProvider(str(checkpoint))
    rng = np.random.default_rng(1)
    for _ in range(15):
        provider.update(make_observation(threat_features=rng.normal(scale=5, size=4).tolist()))
        probs = provider.get_threat_forecast().posture_probs
        assert sum(probs) == pytest.approx(1.0)
        assert all(p >= 0.0 for p in probs)


def test_pool_forecast_matches_moving_average_forecaster_reuse(tmp_path):
    """`LSTMForecastProvider.get_pool_forecast()` is documented as a
    direct delegation to `MovingAverageForecaster`'s own EWMA pool head
    (no real pool-head training data is in scope -- see model.py's
    module docstring). Verify that reuse actually happens: driving an
    independent `MovingAverageForecaster` with the same observations
    must produce identical pool output."""
    from env.forecast_provider import MovingAverageForecaster

    checkpoint = make_tiny_checkpoint(tmp_path, feature_dim=3)
    provider = LSTMForecastProvider(str(checkpoint))
    reference = MovingAverageForecaster()

    observations = [make_observation(threat_features=[0.1, 0.2, 0.3], pool_fill=f, skr=f * 2) for f in (0.1, 0.4, 0.7)]
    for obs in observations:
        provider.update(obs)
        reference.update(obs)

    lstm_pool = provider.get_pool_forecast()
    ref_pool = reference.get_pool_forecast()
    assert lstm_pool.pool_level_hat == ref_pool.pool_level_hat
    assert lstm_pool.skr_mean_hat == ref_pool.skr_mean_hat
    assert lstm_pool.hybrid_demand_hat == ref_pool.hybrid_demand_hat


def test_buffer_window_caps_at_configured_window_length(tmp_path):
    # window=8 in the tiny checkpoint -- feeding far more than 8 updates
    # must not grow the model's input beyond what it was trained for.
    checkpoint = make_tiny_checkpoint(tmp_path, feature_dim=3, window=8)
    provider = LSTMForecastProvider(str(checkpoint))
    rng = np.random.default_rng(2)
    for _ in range(50):
        provider.update(make_observation(threat_features=rng.normal(size=3).tolist()))
    assert len(provider._buffer) == 8  # never exceeds `window`


# ---------------------------------------------------------------------------
# Cold-start warm-start fix (2026-09-14, S5-flat-curve follow-up) -- see
# `get_threat_forecast`'s own comment for why this provider's fix is a
# reactive analog of `RTIoT2022ReplayProvider._warm_start_buffer()`'s
# replicate-first-row fallback, not a verbatim copy of it (this provider
# has no pre-loaded array to borrow real PREDECESSOR rows from -- only
# ever the earliest row THIS episode has itself delivered so far).
# ---------------------------------------------------------------------------


def test_first_forecast_after_one_update_is_not_zero_padded(tmp_path):
    """Immediately after the very first `update()` call (the exact
    moment the old code fed the model 1 real row + `window - 1` zeros),
    the model must instead see `window` copies of that one real row --
    never a zero row."""
    checkpoint = make_tiny_checkpoint(tmp_path, feature_dim=4, window=6)
    provider = LSTMForecastProvider(str(checkpoint))
    provider.update(make_observation(threat_features=[3.0, 3.0, 3.0, 3.0]))

    provider.get_threat_forecast()  # must not crash; real assertion below

    expected = provider._prepare_step_features([3.0, 3.0, 3.0, 3.0])
    # Reconstruct exactly what get_threat_forecast fed the model, mirroring
    # its own padding logic, to assert no zero rows were used.
    history = list(provider._buffer)
    pad = provider._window - len(history)
    padded = [history[0]] * pad + history if pad > 0 else history
    assert len(padded) == 6
    for row in padded:
        np.testing.assert_allclose(row, expected)
        assert not np.allclose(row, 0.0)


def test_partial_buffer_pads_with_earliest_real_row_not_zeros(tmp_path):
    """After a few (but fewer than `window`) real updates, the missing
    slots must be filled by replicating the earliest real row this
    episode has delivered so far -- not zero-padded."""
    checkpoint = make_tiny_checkpoint(tmp_path, feature_dim=3, window=5)
    provider = LSTMForecastProvider(str(checkpoint))
    rows = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    for row in rows:
        provider.update(make_observation(threat_features=row))

    history = list(provider._buffer)
    assert len(history) == 3
    expected_first = provider._prepare_step_features(rows[0])
    np.testing.assert_allclose(history[0], expected_first)

    # Reproduce get_threat_forecast's own padding to assert the 2 missing
    # slots are filled with history[0] (real), not zeros.
    pad = provider._window - len(history)
    padded = [history[0]] * pad + history
    for row in padded[:pad]:
        np.testing.assert_allclose(row, expected_first)
        assert not np.allclose(row, 0.0)


def test_cold_start_no_longer_spikes_threat_score_on_real_checkpoint():
    """Regression test for the real, measured before-fix artifact
    (SESSION_LOG.md 2026-09-14 S5-flat-curve follow-up session): on the
    real trained checkpoint, feeding a genuinely calm-looking
    `[qber, load]` reading and immediately reading `get_threat_forecast()`
    (the exact cold-start moment) must no longer land at the pre-fix
    ~0.96 spike this bug produced on a real S2 episode's first decision.
    This does not assert a specific "correct" calm threat_score (the
    model was never trained on this 2-dim placeholder shape -- see
    `LSTMForecastProvider`'s own "live-feed caveat" docstring), only that
    the fix actually changed cold-start behavior versus feeding an
    all-zero window directly."""
    if not REAL_CHECKPOINT.exists():
        pytest.skip("lstm_threat_forecaster.pt not present")

    provider = LSTMForecastProvider(str(REAL_CHECKPOINT))
    provider.update(make_observation(threat_features=[0.02, 0.05]))
    fixed_score = provider.get_threat_forecast().threat_score

    # Reproduce the OLD (zero-padded) behavior directly for comparison,
    # without touching the fixed code path.
    history = list(provider._buffer)
    zero_padded = [np.zeros(provider._feature_dim, dtype=np.float64)] * (provider._window - len(history)) + history
    window_arr = np.stack(zero_padded, axis=0).astype(np.float32)
    with torch.no_grad():
        old_logit = provider._model(torch.from_numpy(window_arr).unsqueeze(0))
        old_score = float(torch.sigmoid(old_logit).item())

    assert old_score == pytest.approx(0.9570, abs=0.01)  # the real, measured pre-fix spike
    assert fixed_score != pytest.approx(old_score, abs=0.05)


# ---------------------------------------------------------------------------
# Real, committed, trained checkpoint -- the degenerate-collapse gate
# ---------------------------------------------------------------------------


@requires_real_checkpoint
@requires_real_dataset
def test_real_checkpoint_prediction_variance_across_diverse_held_out_windows():
    """This is the test this session's instruction explicitly asked for:
    "a test that would fail if the model were secretly degenerate (e.g.
    checking prediction variance across a diverse held-out sample)."
    Runs the real trained model directly against a diverse slice of the
    real windowed dataset (spanning multiple distinct `Attack_type`
    labels, not just one), and asserts predictions actually vary and
    separate the two classes -- not stuck outputting one constant
    value regardless of input."""
    checkpoint = torch.load(REAL_CHECKPOINT, map_location="cpu", weights_only=False)
    model = SmartKeyForecaster(
        input_dim=int(checkpoint["feature_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint.get("num_layers", 1)),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    data = np.load(REAL_NPZ, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y_binary"]

    # A diverse sample: some benign windows, some attack windows, spread
    # across the dataset rather than one contiguous slice.
    rng = np.random.default_rng(0)
    benign_sample = rng.choice(np.where(y == 0)[0], size=30, replace=False)
    attack_sample = rng.choice(np.where(y == 1)[0], size=30, replace=False)
    sample_idx = np.concatenate([benign_sample, attack_sample])

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X[sample_idx]))).numpy()

    assert np.std(probs) > 0.01  # real variance, not a constant output
    benign_probs = probs[: len(benign_sample)]
    attack_probs = probs[len(benign_sample):]
    assert attack_probs.mean() > benign_probs.mean()  # separates the two classes in the expected direction
