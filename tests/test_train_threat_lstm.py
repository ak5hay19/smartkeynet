"""Real behavioral tests for `forecaster/train_threat_lstm.py` (PLAN2.md
§8A Addition E, Phases 2-4: training the real LSTM threat forecaster on
RT-IoT2022).

Covers the two real risks this session's instruction flagged before any
training was run: (1) a leakage-free train/val split despite RT-IoT2022's
raw file being one contiguous block per label and `rt_iot_features.py`'s
50%-overlapping sliding windows, and (2) detecting (not silently
tolerating) a degenerate always-predict-majority collapse on this
dataset's real 390/3,456 (10.1%/89.9%) benign/attack class imbalance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forecaster import train_threat_lstm as ttl

REAL_NPZ = Path(__file__).resolve().parents[1] / "data" / "processed" / "rt_iot2022" / "rt_iot2022_windows.npz"
requires_real_dataset = pytest.mark.skipif(not REAL_NPZ.exists(), reason="rt_iot2022_windows.npz not present")

REAL_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "lstm_threat_forecaster.pt"
requires_real_checkpoint = pytest.mark.skipif(not REAL_CHECKPOINT.exists(), reason="lstm_threat_forecaster.pt not present")


# ---------------------------------------------------------------------------
# leakage_free_split -- the core no-overlap guarantee
# ---------------------------------------------------------------------------


def test_split_has_zero_raw_row_overlap_on_synthetic_two_block_data():
    # Mirrors the real data's shape: one contiguous benign block, one
    # contiguous attack block, with 50%-overlapping windows (window=64,
    # stride=32) -- exactly the structure that caused a real bug during
    # development (adjacent runs' touching train/val windows overlapping
    # at the run boundary) before the global cleanup pass was added.
    y_binary = np.array([0] * 200 + [1] * 800, dtype=np.int64)
    train_idx, val_idx = ttl.leakage_free_split(y_binary, window=64, stride=32, val_frac=0.2)
    ttl.assert_no_row_overlap(train_idx, val_idx, window=64, stride=32)  # raises on any overlap
    assert len(train_idx) > 0
    assert len(val_idx) > 0


def test_split_catches_run_boundary_overlap_regression():
    """Directly reproduces the exact bug this module's development hit:
    without the global post-split cleanup pass, the last val window of
    one run and the first train window of the next run are index-adjacent
    (distance 1 < the required gap of 2), so their raw row ranges
    overlap. This test would fail if that cleanup pass were removed."""
    y_binary = np.array([0] * 390 + [1] * 3456, dtype=np.int64)
    train_idx, val_idx = ttl.leakage_free_split(y_binary, window=64, stride=32, val_frac=0.2)
    ttl.assert_no_row_overlap(train_idx, val_idx, window=64, stride=32)


def test_split_train_and_val_never_share_an_index():
    y_binary = np.array([0] * 50 + [1] * 50 + [0] * 60 + [1] * 200, dtype=np.int64)
    train_idx, val_idx = ttl.leakage_free_split(y_binary, window=64, stride=32, val_frac=0.25)
    assert set(train_idx.tolist()).isdisjoint(set(val_idx.tolist()))


def test_split_both_classes_represented_in_val_when_runs_are_large_enough():
    y_binary = np.array([0] * 200 + [1] * 800, dtype=np.int64)
    train_idx, val_idx = ttl.leakage_free_split(y_binary, window=64, stride=32, val_frac=0.2)
    val_labels = y_binary[val_idx]
    assert (val_labels == 0).sum() > 0
    assert (val_labels == 1).sum() > 0


def test_split_tiny_run_below_min_falls_entirely_to_train_not_dropped():
    # A run shorter than min_run_for_val can't afford both a gap and a
    # meaningful val slice -- it must go entirely to train, not vanish.
    y_binary = np.array([0] * 3 + [1] * 500, dtype=np.int64)
    train_idx, val_idx = ttl.leakage_free_split(y_binary, window=64, stride=32, val_frac=0.2, min_run_for_val=6)
    assert set(range(3)).issubset(set(train_idx.tolist()))
    assert not set(range(3)).intersection(set(val_idx.tolist()))


@requires_real_dataset
def test_split_on_real_rt_iot2022_windows_has_zero_overlap_and_both_classes_in_val():
    data = np.load(REAL_NPZ, allow_pickle=True)
    y_binary = data["y_binary"]
    window, stride = int(data["window"]), int(data["stride"])
    train_idx, val_idx = ttl.leakage_free_split(y_binary, window=window, stride=stride, val_frac=0.2)
    ttl.assert_no_row_overlap(train_idx, val_idx, window=window, stride=stride)
    val_labels = y_binary[val_idx]
    assert (val_labels == 0).sum() > 0  # benign represented in val
    assert (val_labels == 1).sum() > 0  # attack represented in val
    # real, checked numbers (Phase 1's 390/3,456 split) -- most windows
    # land in train/val, only a small boundary-gap handful dropped
    assert len(train_idx) + len(val_idx) >= len(y_binary) - 20


# ---------------------------------------------------------------------------
# compute_class_weights -- upweights the real minority class (benign)
# ---------------------------------------------------------------------------


def test_class_weights_upweight_the_minority_class():
    y = np.array([0] * 10 + [1] * 90, dtype=np.int64)  # benign is rare, like the real data
    weights = ttl.compute_class_weights(y)
    assert weights[0] > weights[1]  # minority (0=benign) gets the larger weight


def test_class_weights_balanced_when_classes_equal():
    y = np.array([0] * 50 + [1] * 50, dtype=np.int64)
    weights = ttl.compute_class_weights(y)
    assert weights[0] == pytest.approx(weights[1])


# ---------------------------------------------------------------------------
# evaluate / precision_recall_f1 -- correctness on hand-computable cases
# ---------------------------------------------------------------------------


def test_precision_recall_f1_hand_computed():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])  # 1 FP (idx1), 1 FN (idx4), 2 TP, 1 TN
    precision, recall, f1 = ttl.precision_recall_f1(y_true, y_pred, positive=1)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3)


def test_evaluate_perfect_predictions_score_one():
    y_true = np.array([0, 1, 0, 1, 1])
    probs = np.array([0.1, 0.9, 0.2, 0.8, 0.95])
    metrics = ttl.evaluate(y_true, probs, threshold=0.5)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["benign"]["recall"] == pytest.approx(1.0)
    assert metrics["attack"]["recall"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# detect_degenerate_collapse -- the core Step-3 gate this session's
# instruction requires before any success can be declared
# ---------------------------------------------------------------------------


def test_detects_always_predict_majority_collapse():
    # A model that always outputs a high attack-probability, never
    # recovering any benign example -- the exact degenerate shortcut a
    # 90/10 split makes trivially loss-minimizing.
    y_true = np.array([0] * 39 + [1] * 351, dtype=np.int64)  # 39/351, ~10% benign like the real split
    probs = np.full(390, 0.95)  # constant "always attack" prediction, zero variance
    metrics = ttl.evaluate(y_true, probs, threshold=0.5)
    assert ttl.detect_degenerate_collapse(metrics) is True
    assert metrics["accuracy"] > 0.85  # the whole point: accuracy alone looks fine here


def test_does_not_flag_genuine_variance_and_recall_as_collapsed():
    y_true = np.array([0] * 39 + [1] * 351, dtype=np.int64)
    rng = np.random.default_rng(0)
    probs = np.where(y_true == 1, rng.uniform(0.6, 0.99, size=390), rng.uniform(0.01, 0.4, size=390))
    metrics = ttl.evaluate(y_true, probs, threshold=0.5)
    assert ttl.detect_degenerate_collapse(metrics) is False
    assert metrics["benign"]["recall"] > 0.9


# ---------------------------------------------------------------------------
# Real end-to-end training -- confirms the real, trained checkpoint this
# session produced is genuinely non-degenerate, not just the synthetic
# unit tests above
# ---------------------------------------------------------------------------


@requires_real_dataset
def test_real_training_run_a_few_epochs_does_not_crash(tmp_path):
    report = ttl.train_threat_lstm(
        npz_path=REAL_NPZ,
        checkpoint_out=tmp_path / "smoke_checkpoint.pt",
        epochs=3,
        hidden_dim=16,
        seed=0,
    )
    assert report["n_train"] > 0
    assert report["n_val"] > 0
    assert (tmp_path / "smoke_checkpoint.pt").exists()


@requires_real_checkpoint
def test_committed_checkpoint_report_shows_genuine_non_degenerate_learning():
    """This session's real, committed training report
    (`checkpoints/lstm_threat_forecaster.train_report.json`) must show
    the model actually passed the degenerate-collapse check -- this test
    would fail if a future retrain regressed to a majority-class
    shortcut without anyone noticing."""
    import json

    report_path = REAL_CHECKPOINT.with_suffix(".train_report.json")
    report = json.loads(report_path.read_text())

    assert report["degenerate_collapse"] is False
    assert report["final_val_metrics"]["benign"]["recall"] > 0.5
    assert report["final_val_metrics"]["pred_std"] > 0.01
