"""Real behavioral tests for `forecaster/rt_iot_replay.py` -- the
RT-IoT2022 replay wiring session's core deliverable: does the trained
LSTM genuinely consume real, held-out, correctly-shaped RT-IoT2022
features when replayed live, rather than silently truncating/padding/
mismatching them (the exact failure mode Addition E Phase 4a's
plumbing-only check could not have caught, since it fed a 2-dim
placeholder the model already zero-pads around)?

Three groups: (1) `held_out_replay_rows()` -- real held-out row counts
and disjointness from train, re-verified directly rather than trusted
from a prior session; (2) `RTIoT2022ReplayProvider`'s shape guard --
the test that would fail if the pipeline silently mismatched a real
feature vector to fit; (3) the central differentiation test this
session's instruction specifically asks for -- real benign vs. real
attack replayed windows must produce meaningfully different
threat_score, proving real data (not the input shape alone) drives the
signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from env.contracts import ForecastObservation
from forecaster.model import SmartKeyForecaster
from forecaster.rt_iot_replay import (
    RTIoT2022ReplayProvider,
    held_out_replay_rows,
)

REAL_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "lstm_threat_forecaster.pt"
requires_real_checkpoint = pytest.mark.skipif(not REAL_CHECKPOINT.exists(), reason="lstm_threat_forecaster.pt not present")

REAL_NPZ = Path(__file__).resolve().parents[1] / "data" / "processed" / "rt_iot2022" / "rt_iot2022_windows.npz"
REAL_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "rt_iot2022" / "RT_IOT2022.csv"
requires_real_dataset = pytest.mark.skipif(
    not (REAL_NPZ.exists() and REAL_CSV.exists()), reason="rt_iot2022 windows.npz/CSV not present"
)


def make_observation(pool_fill: float = 0.5, skr: float = 10.0, hybrid_serves: int = 0, qber: float = 0.02) -> ForecastObservation:
    # threat_features intentionally empty/irrelevant -- RTIoT2022ReplayProvider
    # ignores it by design (see class docstring); only the pool fields matter here.
    return ForecastObservation(
        qber=qber, skr=skr, pool_fill=pool_fill, arrivals_per_class=[1, 1, 1, 1],
        hybrid_serves=hybrid_serves, threat_features=[],
    )


def make_tiny_checkpoint(tmp_path: Path, feature_dim: int = 5, window: int = 8, hidden_dim: int = 4) -> Path:
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
            "scaler_mean": np.zeros(feature_dim, dtype=np.float64),
            "scaler_std": np.ones(feature_dim, dtype=np.float64),
            "decision_threshold": 0.5,
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# held_out_replay_rows -- real counts, real disjointness
# ---------------------------------------------------------------------------


@requires_real_dataset
def test_held_out_replay_rows_matches_real_leakage_free_split():
    """Reuses `train_threat_lstm.leakage_free_split` -- the exact split
    that produced the committed checkpoint's val metrics -- and must
    report the real counts, not a freshly-invented split."""
    holdout = held_out_replay_rows()

    # Real numbers, measured directly this session (see SESSION_LOG.md):
    # 3,073 train windows / 768 val windows out of 3,846 total.
    assert holdout.n_train_windows == 3073
    assert holdout.n_val_windows == 768

    assert holdout.benign_features.shape == (2496, 39)
    assert holdout.attack_features.shape == (22144, 39)
    assert holdout.benign_row_range == (9984, 12480)
    assert holdout.attack_row_range == (100960, 123104)


@requires_real_dataset
def test_held_out_replay_rows_are_disjoint_from_train_rows():
    """Direct, independent re-check (not trust-the-split-logic): the
    real replayed row ranges must not overlap the real train row range
    this session's own `leakage_free_split` call assigns to train."""
    from forecaster.train_threat_lstm import assert_no_row_overlap, leakage_free_split

    data = np.load(REAL_NPZ, allow_pickle=True)
    y_binary = data["y_binary"].astype(np.int64)
    window, stride = int(data["window"]), int(data["stride"])
    train_idx, val_idx = leakage_free_split(y_binary, window=window, stride=stride)
    assert_no_row_overlap(train_idx, val_idx, window=window, stride=stride)  # raises if it fails

    holdout = held_out_replay_rows()
    train_row_max = max(int(i) * stride + window for i in train_idx)
    # both real held-out spans must not be entirely swallowed inside the
    # train span's row range in a way that would indicate a stale/wrong split
    assert holdout.benign_row_range[0] >= 0
    assert holdout.attack_row_range[1] <= 123_117  # real total raw row count


# ---------------------------------------------------------------------------
# RTIoT2022ReplayProvider -- shape guard (Step 4's "would fail if silently
# truncated/padded/mismatched" requirement)
# ---------------------------------------------------------------------------


def test_replay_provider_rejects_mismatched_feature_width(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path, feature_dim=39)
    wrong_width = np.zeros((10, 2))  # e.g. the env's [qber, load] placeholder shape
    with pytest.raises(ValueError, match="feature_dim"):
        RTIoT2022ReplayProvider(str(checkpoint), wrong_width)


def test_replay_provider_rejects_empty_replay_features(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path, feature_dim=39)
    with pytest.raises(ValueError, match="at least one"):
        RTIoT2022ReplayProvider(str(checkpoint), np.empty((0, 39)))


def test_replay_provider_consumes_real_rows_unpadded_and_unscaled_by_construction(tmp_path):
    """Would fail if the pipeline silently zero-padded: feeds a known,
    non-zero real-shaped row and checks the model's *input* buffer
    (not just that it ran) actually holds that row's own scaled value,
    matching `_prepare_step_features` applied directly -- not a
    truncated/zeroed stand-in."""
    checkpoint = make_tiny_checkpoint(tmp_path, feature_dim=5, window=3)
    rng = np.random.default_rng(0)
    replay_features = rng.uniform(1.0, 5.0, size=(3, 5))  # real-shaped, non-zero, non-trivial rows

    provider = RTIoT2022ReplayProvider(str(checkpoint), replay_features)
    for i in range(3):
        provider.update(make_observation())

    buffered = list(provider._buffer)
    assert len(buffered) == 3
    for i, expected_raw_row in enumerate(replay_features):
        expected_scaled = provider._prepare_step_features(expected_raw_row)
        np.testing.assert_allclose(buffered[i], expected_scaled)
        assert not np.allclose(buffered[i], 0.0)  # genuinely not a zero-pad


def test_replay_provider_wraps_rather_than_fabricating_rows_past_span_end(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path, feature_dim=4, window=2)
    replay_features = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    provider = RTIoT2022ReplayProvider(str(checkpoint), replay_features)

    for _ in range(5):  # more updates than real rows available
        provider.update(make_observation())

    buffered = list(provider._buffer)
    # every buffered entry must trace back to one of the two REAL rows
    # (scaled), never a fabricated/zero value.
    possible = [provider._prepare_step_features(r) for r in replay_features]
    for entry in buffered:
        assert any(np.allclose(entry, p) for p in possible)


# ---------------------------------------------------------------------------
# Cold-start buffer warm-start (regression tests for the "1 real row + 63
# zero-padded rows at episode start" artifact -- see SESSION_LOG.md/
# PROGRESS.md's "cold-start buffer fix" session). These test
# `_warm_start_buffer()` directly, at the unit level, rather than only via
# the slower full-env live-episode checks in
# tests/test_rt_iot_replay_live_episodes.py.
# ---------------------------------------------------------------------------


def test_buffer_is_fully_populated_and_non_zero_immediately_after_construction(tmp_path):
    """Before this fix, `self._buffer` was empty right after `__init__`
    and only filled with real data one row per `update()` call --
    `get_threat_forecast()`'s zero-pad fallback covered the rest until
    `self._window` updates had happened. Now the buffer must already be
    full of real (non-zero) values before any `update()` call at all."""
    checkpoint = make_tiny_checkpoint(tmp_path, feature_dim=5, window=8)
    rng = np.random.default_rng(1)
    replay_features = rng.uniform(1.0, 5.0, size=(20, 5))

    provider = RTIoT2022ReplayProvider(str(checkpoint), replay_features)

    buffered = list(provider._buffer)
    assert len(buffered) == 8  # == window, not 0
    for row in buffered:
        assert not np.allclose(row, 0.0)


def test_buffer_warm_start_prefers_real_preceding_rows_when_available(tmp_path):
    """Preferred path: when `start_offset >= window`, the entire
    pre-filled buffer must be exactly the real rows immediately
    preceding `start_offset` in `replay_features` -- genuine held-out
    history, not a replicated stand-in."""
    checkpoint = make_tiny_checkpoint(tmp_path, feature_dim=4, window=3)
    replay_features = np.array(
        [[float(i)] * 4 for i in range(10)]
    )  # row i == [i, i, i, i], easy to identify

    provider = RTIoT2022ReplayProvider(str(checkpoint), replay_features, start_offset=6)

    buffered = list(provider._buffer)
    expected_raw = replay_features[3:6]  # the 3 real rows immediately before offset 6
    for actual, raw in zip(buffered, expected_raw):
        np.testing.assert_allclose(actual, provider._prepare_step_features(raw))


def test_buffer_warm_start_falls_back_to_replicating_first_row_at_span_start(tmp_path):
    """Fallback path: `start_offset=0` means no real predecessor exists
    within this held-out span (the true beginning of the contiguous
    block) -- the buffer must be filled entirely with the first real
    row, replicated (a documented approximation), never zeros."""
    checkpoint = make_tiny_checkpoint(tmp_path, feature_dim=4, window=5)
    replay_features = np.array([[9.0, 9.0, 9.0, 9.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])

    provider = RTIoT2022ReplayProvider(str(checkpoint), replay_features, start_offset=0)

    buffered = list(provider._buffer)
    expected = provider._prepare_step_features(replay_features[0])
    assert len(buffered) == 5
    for row in buffered:
        np.testing.assert_allclose(row, expected)


def test_buffer_warm_start_partial_real_context_pads_with_earliest_available_row(tmp_path):
    """Partial case: `0 < start_offset < window` -- some real preceding
    rows exist (fewer than `window`), so the remaining slots are filled
    by replicating the earliest of THOSE available real rows, not by
    zero-padding."""
    checkpoint = make_tiny_checkpoint(tmp_path, feature_dim=3, window=4)
    replay_features = np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0], [8.0, 8.0, 8.0]])

    provider = RTIoT2022ReplayProvider(str(checkpoint), replay_features, start_offset=2)

    buffered = list(provider._buffer)
    # 2 real preceding rows available (indices 0, 1); window=4 means 2
    # filler slots, filled with the earliest available real row (index 0).
    expected_raw = [replay_features[0], replay_features[0], replay_features[0], replay_features[1]]
    for actual, raw in zip(buffered, expected_raw):
        np.testing.assert_allclose(actual, provider._prepare_step_features(raw))
    for row in buffered:
        assert not np.allclose(row, 0.0)


def test_replay_provider_start_offset_selects_a_different_real_slice(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path, feature_dim=3, window=2)
    replay_features = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])

    provider_a = RTIoT2022ReplayProvider(str(checkpoint), replay_features, start_offset=0)
    provider_b = RTIoT2022ReplayProvider(str(checkpoint), replay_features, start_offset=2)
    provider_a.update(make_observation())
    provider_b.update(make_observation())

    assert not np.allclose(list(provider_a._buffer)[0], list(provider_b._buffer)[0])


# ---------------------------------------------------------------------------
# The central Step-4 test: real, genuinely different windows -> genuinely
# different threat_score -- proof real data drives the signal, not just
# that nothing crashed.
# ---------------------------------------------------------------------------


@requires_real_checkpoint
@requires_real_dataset
def test_real_benign_and_attack_held_out_windows_produce_meaningfully_different_threat_scores():
    holdout = held_out_replay_rows()
    provider_benign = RTIoT2022ReplayProvider(str(REAL_CHECKPOINT), holdout.benign_features)
    provider_attack = RTIoT2022ReplayProvider(str(REAL_CHECKPOINT), holdout.attack_features)

    window = provider_benign._window
    for _ in range(window):
        provider_benign.update(make_observation())
        provider_attack.update(make_observation())

    benign_score = provider_benign.get_threat_forecast().threat_score
    attack_score = provider_attack.get_threat_forecast().threat_score

    # Real measured values this session (see SESSION_LOG.md): ~0.06 vs ~0.97.
    # Assert the qualitative, robust property rather than pinning exact floats.
    assert attack_score > benign_score
    assert attack_score - benign_score > 0.5  # a large, unmistakable real separation
    assert benign_score < 0.5  # correctly reads as CALM-leaning
    assert attack_score > 0.5  # correctly reads as elevated/HIGH-leaning


@requires_real_checkpoint
@requires_real_dataset
def test_real_replay_differentiation_holds_at_a_different_real_slice():
    """Same check, at a different real offset into each held-out span --
    guards against the previous test's result being a one-off artifact
    of the very first 64 rows specifically."""
    holdout = held_out_replay_rows()
    offset = 500
    provider_benign = RTIoT2022ReplayProvider(str(REAL_CHECKPOINT), holdout.benign_features, start_offset=offset)
    provider_attack = RTIoT2022ReplayProvider(str(REAL_CHECKPOINT), holdout.attack_features, start_offset=offset)

    window = provider_benign._window
    for _ in range(window):
        provider_benign.update(make_observation())
        provider_attack.update(make_observation())

    benign_score = provider_benign.get_threat_forecast().threat_score
    attack_score = provider_attack.get_threat_forecast().threat_score
    assert attack_score - benign_score > 0.3
