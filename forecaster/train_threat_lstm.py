"""
forecaster/train_threat_lstm.py

Offline supervised training for `forecaster.model.SmartKeyForecaster`'s
threat head, on the real windowed RT-IoT2022 dataset
`forecaster/rt_iot_features.py::run_pipeline` produced (PLAN2.md §8A
Addition E, Phases 2-4).

Deliberately separate from `forecaster/train.py` (which remains the
future rollout-log-based dual-head trainer, still a real
`NotImplementedError` stub -- see `forecaster/model.py`'s module
docstring for why the two are not merged this session).

**Class imbalance, measured directly off `rt_iot2022_windows.npz`
(3,846 total windows): 390 benign (10.14%) vs. 3,456 attack (89.86%).**
A model that always predicts "attack" scores ~90% raw accuracy while
learning nothing -- see `evaluate()`'s per-class precision/recall/F1
and `detect_degenerate_collapse()` below, which is what this module
reports and gates success on instead.

**Leakage-free split, the other real risk this module is built
around:** `rt_iot_features.py::build_windows` slides window=64/
stride=32 over RAW ROW ORDER with NO shuffling, so window i and window
i+1 share 32 of their 64 raw rows (50% overlap) -- a random per-window
train/val split would put overlapping windows on both sides, inflating
validation metrics. Worse, RT-IoT2022's raw file is not row-shuffled at
all: every one of its 12 `Attack_type` labels occupies exactly ONE
contiguous run of raw rows (verified directly: 11 label transitions
total across 123,117 rows, i.e. 12 uninterrupted blocks -- confirmed
independently in window-index space too: `y_multiclass` has exactly 11
contiguous runs across the 3,846 windows, since window ordering
preserves raw row ordering). All three benign `Attack_type` labels
happen to sit first in the file, so the 390 benign windows and 3,456
attack windows are themselves each ONE contiguous block in window-index
space (indices 0-389 / 390-3845). `leakage_free_split()` below exploits
this directly: it finds each contiguous same-binary-label run, and
splits each run temporally (earliest windows -> train, latest -> val,
in original order, never shuffled across the boundary), dropping a
small gap of windows at each split point so that no val window's raw
row range can overlap any train window's raw row range (gap size
derived from `window`/`stride` below, not hardcoded).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from forecaster.model import SmartKeyForecaster

DEFAULT_NPZ = Path("data/processed/rt_iot2022/rt_iot2022_windows.npz")
DEFAULT_CHECKPOINT_OUT = Path("checkpoints/lstm_threat_forecaster.pt")


# ---------------------------------------------------------------------------
# Leakage-free split
# ---------------------------------------------------------------------------


def _contiguous_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Returns `(start, end_inclusive, label)` for each maximal run of
    equal adjacent values in `labels` (index order)."""
    runs = []
    start = 0
    current = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != current:
            runs.append((start, i - 1, int(current)))
            start = i
            current = labels[i]
    runs.append((start, len(labels) - 1, int(current)))
    return runs


def leakage_free_split(
    y_binary: np.ndarray,
    window: int,
    stride: int,
    val_frac: float = 0.2,
    min_run_for_val: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Splits window indices into (train_idx, val_idx) with zero raw-row
    overlap between the two sides and both classes represented in each
    (given the real data's contiguous-run structure -- see module
    docstring).

    Two windows at index distance `d` apart have raw row starts
    `d * stride` apart; they cannot overlap once `d * stride >= window`,
    i.e. `d >= ceil(window / stride)`. Each contiguous same-label run is
    split at `val_frac` (latest windows -> val, in original order),
    with exactly that many indices dropped as a gap immediately before
    the val slice -- so the true minimum distance between any train
    index and any val index is >= `ceil(window / stride)`, guaranteeing
    no raw-row overlap.

    A run shorter than `min_run_for_val` windows is assigned entirely
    to train (too little data to carve out a meaningful, leakage-free
    val slice from) -- reported by the caller, not silently hidden.
    """
    gap = math.ceil(window / stride)
    runs = _contiguous_runs(y_binary)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for start, end, _label in runs:
        run_len = end - start + 1
        if run_len < min_run_for_val:
            train_idx.extend(range(start, end + 1))
            continue

        n_val = max(1, round(run_len * val_frac))
        n_train = run_len - n_val - gap
        if n_train < 1:
            # Run too short to afford both the gap and a val slice --
            # fall back to whole-run-to-train rather than a leakage risk.
            train_idx.extend(range(start, end + 1))
            continue

        train_end = start + n_train  # exclusive
        val_start = train_end + gap
        train_idx.extend(range(start, train_end))
        val_idx.extend(range(val_start, end + 1))

    train_arr = np.array(sorted(train_idx), dtype=np.int64)
    val_arr = np.array(sorted(val_idx), dtype=np.int64)

    # The per-run split above only guarantees the gap WITHIN each run.
    # Adjacent runs (e.g. the benign run's val tail sitting immediately
    # before the next run's train head) can still touch with distance 1
    # at the RUN BOUNDARY -- a real bug caught by `assert_no_row_overlap`
    # during development, not a hypothetical. Global cleanup: drop any
    # val index within `gap` of the NEAREST train index (train_arr is
    # left untouched -- trimming only val is sufficient and terminates
    # in one pass, no cascading).
    if len(val_arr) > 0 and len(train_arr) > 0:
        pos = np.searchsorted(train_arr, val_arr)
        left = train_arr[np.clip(pos - 1, 0, len(train_arr) - 1)]
        right = train_arr[np.clip(pos, 0, len(train_arr) - 1)]
        nearest_dist = np.minimum(np.abs(val_arr - left), np.abs(val_arr - right))
        val_arr = val_arr[nearest_dist >= gap]

    return train_arr, val_arr


def assert_no_row_overlap(train_idx: np.ndarray, val_idx: np.ndarray, window: int, stride: int) -> None:
    """Directly verifies, in raw-row space, that no train window and no
    val window share a single raw row -- the property `leakage_free_split`
    is designed to guarantee, checked rather than trusted."""
    def row_range(idx: int) -> tuple[int, int]:
        s = idx * stride
        return s, s + window  # [s, e)

    train_rows = set()
    for idx in train_idx:
        s, e = row_range(int(idx))
        train_rows.update(range(s, e))
    for idx in val_idx:
        s, e = row_range(int(idx))
        if train_rows.intersection(range(s, e)):
            raise AssertionError(f"val window {idx} overlaps a train window's raw rows")


# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Balanced inverse-frequency weights (`sklearn`'s `class_weight="balanced"`
    formula: `n_samples / (n_classes * n_samples_for_class)`), computed
    from the TRAIN split only so val's distribution never leaks into
    training. Upweights the minority class (benign, label 0) rather
    than the majority (attack, label 1) -- the direction that matters
    given attack is the majority here, unlike the textbook rare-positive
    framing."""
    counts = np.bincount(y, minlength=2)
    n = len(y)
    n_classes = 2
    return {c: n / (n_classes * counts[c]) if counts[c] > 0 else 0.0 for c in range(n_classes)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, positive: int) -> tuple[float, float, float]:
    tp = int(np.sum((y_pred == positive) & (y_true == positive)))
    fp = int(np.sum((y_pred == positive) & (y_true != positive)))
    fn = int(np.sum((y_pred != positive) & (y_true == positive)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    y_pred = (probs >= threshold).astype(np.int64)
    benign_p, benign_r, benign_f1 = precision_recall_f1(y_true, y_pred, positive=0)
    attack_p, attack_r, attack_f1 = precision_recall_f1(y_true, y_pred, positive=1)
    accuracy = float(np.mean(y_pred == y_true))
    macro_f1 = (benign_f1 + attack_f1) / 2.0
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "benign": {"precision": benign_p, "recall": benign_r, "f1": benign_f1, "support": int(np.sum(y_true == 0))},
        "attack": {"precision": attack_p, "recall": attack_r, "f1": attack_f1, "support": int(np.sum(y_true == 1))},
        "pred_positive_rate": float(np.mean(y_pred)),
        "pred_std": float(np.std(probs)),
    }


def best_threshold_by_macro_f1(y_true: np.ndarray, probs: np.ndarray, candidates: Sequence[float] | None = None) -> tuple[float, dict]:
    """Scans candidate decision thresholds and returns the one
    maximizing macro-F1 on the given (val) set -- the "precision/
    recall-aware... criterion instead of raw accuracy" this session's
    instruction asked for, since 0.5 is not a defensible default under
    a 90/10 class split with a class-weighted loss shifting the
    natural decision boundary."""
    if candidates is None:
        candidates = [i / 100.0 for i in range(5, 100, 5)]
    best_t, best_metrics = 0.5, evaluate(y_true, probs, 0.5)
    for t in candidates:
        metrics = evaluate(y_true, probs, t)
        if metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_t, best_metrics = t, metrics
    return best_t, best_metrics


def detect_degenerate_collapse(metrics: dict, min_benign_recall: float = 0.05) -> bool:
    """True iff the model has collapsed to an always-predict-majority
    (or near enough) shortcut: essentially zero recall on the minority
    (benign) class, or near-zero prediction variance. This is the check
    this session's instruction requires BEFORE declaring success --
    raw accuracy on this 90/10 split cannot detect it (a constant
    "attack" predictor scores ~90% accuracy)."""
    return metrics["benign"]["recall"] < min_benign_recall or metrics["pred_std"] < 1e-4


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_threat_lstm(
    npz_path: Path = DEFAULT_NPZ,
    checkpoint_out: Path = DEFAULT_CHECKPOINT_OUT,
    epochs: int = 60,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    batch_size: int = 64,
    val_frac: float = 0.2,
    seed: int = 0,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y_binary = data["y_binary"].astype(np.int64)
    window = int(data["window"])
    stride = int(data["stride"])

    train_idx, val_idx = leakage_free_split(y_binary, window=window, stride=stride, val_frac=val_frac)
    assert_no_row_overlap(train_idx, val_idx, window=window, stride=stride)

    X_train, y_train = X[train_idx], y_binary[train_idx]
    X_val, y_val = X[val_idx], y_binary[val_idx]

    class_weights = compute_class_weights(y_train)
    sample_weights_train = torch.tensor([class_weights[int(c)] for c in y_train], dtype=torch.float32)

    model = SmartKeyForecaster(input_dim=X.shape[2], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val.astype(np.float32))

    n_train = X_train_t.shape[0]
    history = []
    best_val_macro_f1 = -1.0
    best_state = None
    best_threshold = 0.5

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            batch_idx = perm[start : start + batch_size]
            xb, yb, wb = X_train_t[batch_idx], y_train_t[batch_idx], sample_weights_train[batch_idx]

            optimizer.zero_grad()
            logits = model(xb)
            if use_focal_loss:
                p = torch.sigmoid(logits)
                pt = torch.where(yb == 1, p, 1 - p)
                bce = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                loss = (wb * ((1 - pt) ** focal_gamma) * bce).mean()
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, weight=wb, reduction="mean")
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_idx)
        epoch_loss /= n_train

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val_t)).numpy()
        threshold, val_metrics = best_threshold_by_macro_f1(y_val, val_probs)
        history.append({"epoch": epoch, "train_loss": epoch_loss, **val_metrics})

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_threshold = threshold

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_val_probs = torch.sigmoid(model(X_val_t)).numpy()
        final_train_probs = torch.sigmoid(model(X_train_t)).numpy()
    final_val_metrics = evaluate(y_val, final_val_probs, best_threshold)
    final_train_metrics = evaluate(y_train, final_train_probs, best_threshold)
    collapsed = detect_degenerate_collapse(final_val_metrics)

    checkpoint_out = Path(checkpoint_out)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "feature_dim": int(X.shape[2]),
            "window": window,
            "hidden_dim": hidden_dim,
            "num_layers": 1,
            "scaler_mean": data["scaler_mean"],
            "scaler_std": data["scaler_std"],
            "decision_threshold": best_threshold,
        },
        checkpoint_out,
    )

    report = {
        "npz_path": str(npz_path),
        "checkpoint_out": str(checkpoint_out),
        "n_windows_total": int(len(y_binary)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_dropped_as_gap_or_unsplittable_tiny_run": int(len(y_binary) - len(train_idx) - len(val_idx)),
        "class_weights_train": class_weights,
        "best_threshold": best_threshold,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "degenerate_collapse": collapsed,
        "history": history,
    }
    report_path = checkpoint_out.with_suffix(".train_report.json")
    report_path.write_text(json.dumps(report, indent=2))

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--checkpoint-out", type=Path, default=DEFAULT_CHECKPOINT_OUT)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--focal-loss", action="store_true")
    args = parser.parse_args()

    result = train_threat_lstm(
        npz_path=args.npz,
        checkpoint_out=args.checkpoint_out,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        seed=args.seed,
        use_focal_loss=args.focal_loss,
    )
    # history is verbose (one entry per epoch) -- omit from the printed summary
    summary = {k: v for k, v in result.items() if k != "history"}
    print(json.dumps(summary, indent=2))
