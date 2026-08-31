"""
forecaster/rt_iot_features.py

Real feature-extraction pipeline for RT-IoT2022 (PLAN.md "Datasets &
Provenance" Slot 1; PLAN2.md §6, Addition E Phase 1). Standalone from
`forecaster/dataset.py` (which builds LSTM training windows from
*environment rollout logs* -- inputs already include a `threat_features`
field -- not raw network data). This module is the missing upstream
piece: it turns RT-IoT2022's raw per-flow CSV rows into the feature
representation an LSTM threat head would consume, via ONE shared
extraction function meant for two callers (PLAN2.md Addition D: "one
implementation, two callers"):

  1. Offline training path (this session): `run_pipeline()` windows
     the FULL RT-IoT2022 CSV into fixed-length sequences and saves
     them to disk as this repo's first real LSTM training input.
  2. Future single-window inference path: `extract_flow_features()` is
     called identically on a single live flow's raw fields (a 1-row
     DataFrame) once real-time ingestion exists (PLAN2.md §7.1/§8:
     "same feature-extraction path... offline RT-IoT2022 training
     data" / "uploaded pcap" / "replayed pcap"). It is deterministic
     and stateless (no fitted statistics), so it is guaranteed to
     produce identical output for the same row regardless of caller --
     see `tests/test_rt_iot_features.py::
     test_single_row_and_batch_extraction_agree`.

SCOPE BOUNDARY (this session's instruction, not a project Hard Rule):
this module's output is NOT wired into `env/forecast_provider.py` or
`env/environment.py`. `env/environment.py::_threat_features_placeholder`
(`[qber, load]`) remains the system's active threat-feature source
until a future, separate, deliberate integration session -- see
PLAN2.md's "Addition E" section.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from forecaster.model import WINDOW as DEFAULT_WINDOW

# ---------------------------------------------------------------------------
# Feature design -- real network-flow intrusion-detection engineering
# practice, reasoning documented per group. RT-IoT2022's columns are
# CICFlowMeter/Zeek-style per-flow aggregates (verified against the
# real CSV header + `.describe()` on the full 123,117-row file: all
# columns used below are non-negative; several span 5+ orders of
# magnitude, e.g. `flow_iat.std` ranges 0..1.34e8). log1p is the
# standard transform for that shape (compresses the tail, fixes zero
# at zero, monotonic, cheap to invert).
# ---------------------------------------------------------------------------

NUMERIC_LOG1P_COLUMNS: tuple[str, ...] = (
    # packet/byte RATES -- volumetric attack fingerprints (SYN floods,
    # DDoS): a flood flow's packets/bytes-per-second sits orders of
    # magnitude above ordinary telemetry traffic.
    "fwd_pkts_per_sec", "bwd_pkts_per_sec", "flow_pkts_per_sec",
    "payload_bytes_per_second",
    # packet/byte VOLUME -- separates bulk transfer from near-empty
    # scan probes (this dataset's NMAP_* labels are, almost by
    # definition, low-payload flows).
    "fwd_pkts_tot", "bwd_pkts_tot", "fwd_data_pkts_tot", "bwd_data_pkts_tot",
    "flow_pkts_payload.avg", "flow_pkts_payload.std",
    # TIMING -- flow duration, inter-arrival stats, active/idle
    # periods: captures beaconing/periodicity vs. single-burst floods
    # vs. long idle legitimate sessions.
    "flow_duration", "flow_iat.avg", "flow_iat.std", "active.avg", "idle.avg",
    # HEADER OVERHEAD -- header bytes relative to (often near-zero)
    # payload is itself a scan/malformed-packet signal.
    "fwd_header_size_tot", "bwd_header_size_tot",
    # TCP WINDOW SIZE -- classic OS/stack-fingerprinting feature,
    # directly relevant here: NMAP_OS_DETECTION is one of this
    # dataset's real labels.
    "fwd_init_window_size", "bwd_init_window_size",
    # TCP CONTROL-FLAG COUNTS -- SYN/RST/FIN/ACK/PSH/URG counts are
    # the textbook scan/flood fingerprint (SYN flood: SYN >> ACK;
    # NMAP FIN/XMAS scans: unusual FIN/URG/PSH combos with no
    # completed handshake). Kept as (log1p'd) counts rather than
    # ratios of total packets: `down_up_ratio` already gives a
    # normalized asymmetry signal, and a same-ratio normalization here
    # would make a 3-packet SYN probe and an 8-packet SYN flood look
    # identical, discarding real magnitude information.
    "flow_SYN_flag_count", "flow_RST_flag_count", "flow_FIN_flag_count",
    "flow_ACK_flag_count", "fwd_PSH_flag_count", "fwd_URG_flag_count",
)

# Already a bounded ratio (real observed range on the full dataset:
# [0.0, 6.09]) -- used as-is, no log1p.
RAW_RATIO_COLUMNS: tuple[str, ...] = ("down_up_ratio",)

# Categorical columns, one-hot encoded against a FIXED category list
# (not `pd.get_dummies` on whatever categories happen to appear in a
# given slice), so a single flow's inference-time feature vector has
# the same shape as a full-dataset training window (PLAN2.md Addition
# D unit test: "feature-extraction output is identical in shape
# whether sourced from the training CSV or a pcap window"). Both
# tuples are the real, complete category sets observed on the full
# RT-IoT2022 CSV (`df["proto"].unique()` / `df["service"].unique()`).
PROTO_CATEGORIES: tuple[str, ...] = ("tcp", "udp", "icmp")
SERVICE_CATEGORIES: tuple[str, ...] = (
    "-", "dns", "mqtt", "http", "ssl", "ntp", "dhcp", "irc", "ssh", "radius",
)

CONTINUOUS_COLUMNS: tuple[str, ...] = NUMERIC_LOG1P_COLUMNS + RAW_RATIO_COLUMNS
FEATURE_NAMES: tuple[str, ...] = (
    CONTINUOUS_COLUMNS
    + tuple(f"proto={p}" for p in PROTO_CATEGORIES)
    + tuple(f"service={s}" for s in SERVICE_CATEGORIES)
)
FEATURE_DIM = len(FEATURE_NAMES)

REQUIRED_RAW_COLUMNS: tuple[str, ...] = NUMERIC_LOG1P_COLUMNS + RAW_RATIO_COLUMNS + ("proto", "service")
LABEL_COLUMN = "Attack_type"

# Documented, REASONED (not officially cited) benign/attack split --
# flagged explicitly as an inference from label semantics, not a
# verified citation against RT-IoT2022's own paper. These three
# `Attack_type` values name ordinary IoT device communication activity
# (an MQTT publish message, a ThingSpeak cloud-telemetry push, a Wipro
# smart-bulb device's traffic) rather than a named attack technique,
# unlike the other 9 values, which all name a specific attack
# tool/technique (DOS/DDOS floods, ARP poisoning, NMAP reconnaissance
# scan variants, a Metasploit brute-force module). A real calibration
# pass against RT-IoT2022's own documentation is future work before
# this binary label feeds anything beyond pipeline validation -- see
# SESSION_LOG.md.
BENIGN_ATTACK_TYPES: frozenset[str] = frozenset({"MQTT_Publish", "Thing_Speak", "Wipro_bulb"})


# ---------------------------------------------------------------------------
# Degeneracy screening (PLAN.md/PLAN2.md "Datasets & Provenance": "a
# file where one label covers nearly the whole set gets discarded, not
# sampled around" -- the DO-NOT-USE `context_dataset_basic.csv`/
# `context_dataset_advanced.csv` precedent is a single label at 100%).
# Generic over any label column so the same function also correctly
# rejects a constructed synthetic degenerate file in tests.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DegeneracyResult:
    is_degenerate: bool
    dominant_label: object
    dominant_fraction: float
    n_unique_labels: int
    n_rows: int


def screen_label_degeneracy(labels: pd.Series, dominant_frac_threshold: float = 0.98) -> DegeneracyResult:
    """Flags a label column as degenerate if one label covers
    `dominant_frac_threshold` or more of all rows.

    `dominant_frac_threshold=0.98` is a practical data-hygiene default
    (NOT a Hard Rule 4 security constant -- that rule governs tier
    mappings, not data-screening thresholds), chosen to sit comfortably
    below the project's own 100%-one-label DO-NOT-USE precedent while
    still catching a file that is *almost* entirely one label.
    """
    n_rows = len(labels)
    if n_rows == 0:
        return DegeneracyResult(is_degenerate=True, dominant_label=None, dominant_fraction=0.0, n_unique_labels=0, n_rows=0)
    counts = labels.value_counts()
    dominant_label = counts.index[0]
    dominant_fraction = float(counts.iloc[0]) / n_rows
    return DegeneracyResult(
        is_degenerate=dominant_fraction >= dominant_frac_threshold,
        dominant_label=dominant_label,
        dominant_fraction=dominant_fraction,
        n_unique_labels=int(counts.shape[0]),
        n_rows=n_rows,
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_rt_iot2022(csv_path: Path | str) -> pd.DataFrame:
    """Loads the full RT-IoT2022 CSV, validating the columns this
    pipeline needs are present and the file is non-empty. Does not
    subsample -- callers get every row in the file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found -- run `python data/get_data.py --dataset rt_iot2022` first."
        )
    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_RAW_COLUMNS) | {LABEL_COLUMN}
    missing -= set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{csv_path} loaded but has zero rows")
    return df


# ---------------------------------------------------------------------------
# Extraction (the shared function -- see module docstring)
# ---------------------------------------------------------------------------


def extract_flow_features(df: pd.DataFrame) -> np.ndarray:
    """Deterministic, stateless per-flow feature extraction. Returns an
    array of shape `(len(df), FEATURE_DIM)` in `FEATURE_NAMES` order.

    Stateless by design: no fitted statistics (mean/std) are used here,
    so calling this on a single-row `df` (a future live-flow caller)
    and calling it on that same row embedded in a larger `df` (the
    offline training caller) produce identical output. Standardization
    -- which DOES need fitted statistics -- is a deliberately separate
    step (`fit_scaler` / `ContinuousScaler.apply`) so it can be fit
    once offline and the same fitted scaler reused verbatim by a
    future inference caller.
    """
    missing = set(REQUIRED_RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing required raw columns: {sorted(missing)}")

    blocks = []
    for col in NUMERIC_LOG1P_COLUMNS:
        values = df[col].to_numpy(dtype=np.float64)
        values = np.clip(values, a_min=0.0, a_max=None)  # defensive: column is documented non-negative
        blocks.append(np.log1p(values).reshape(-1, 1))
    for col in RAW_RATIO_COLUMNS:
        blocks.append(df[col].to_numpy(dtype=np.float64).reshape(-1, 1))

    proto_onehot = pd.get_dummies(df["proto"]).reindex(columns=list(PROTO_CATEGORIES), fill_value=0)
    blocks.append(proto_onehot.to_numpy(dtype=np.float64))

    service_onehot = pd.get_dummies(df["service"]).reindex(columns=list(SERVICE_CATEGORIES), fill_value=0)
    blocks.append(service_onehot.to_numpy(dtype=np.float64))

    return np.concatenate(blocks, axis=1)


@dataclass(frozen=True)
class ContinuousScaler:
    """z-score scaler fit on the continuous (log1p + ratio) columns
    ONLY -- the one-hot columns are already in {0, 1} and left
    untouched. Fit once, offline, on the training pipeline; meant to
    be persisted and reused verbatim by a future single-window
    inference caller (fitting a fresh scaler per inference call would
    make a live flow's features incomparable to what the model was
    trained on)."""

    mean: np.ndarray
    std: np.ndarray

    def apply(self, features: np.ndarray) -> np.ndarray:
        n_continuous = len(CONTINUOUS_COLUMNS)
        out = features.copy()
        out[:, :n_continuous] = (features[:, :n_continuous] - self.mean) / self.std
        return out

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def fit_scaler(features: np.ndarray, eps: float = 1e-8) -> ContinuousScaler:
    n_continuous = len(CONTINUOUS_COLUMNS)
    continuous = features[:, :n_continuous]
    mean = continuous.mean(axis=0)
    std = continuous.std(axis=0)
    std = np.where(std < eps, 1.0, std)  # guard a constant column against divide-by-zero
    return ContinuousScaler(mean=mean, std=std)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowedDataset:
    X: np.ndarray  # (n_windows, window, FEATURE_DIM), standardized
    y_binary: np.ndarray  # (n_windows,) int64 -- 0 benign, 1 attack, majority vote
    y_multiclass: np.ndarray  # (n_windows,) int64 -- index into `class_names`, majority vote
    class_names: tuple[str, ...]
    window: int
    stride: int


def build_windows(
    features: np.ndarray,
    labels: pd.Series,
    window: int = DEFAULT_WINDOW,
    stride: int = 32,
) -> WindowedDataset:
    """Slides a length-`window` window over `features`/`labels` (row
    order as given -- RT-IoT2022 ships no explicit timestamp column,
    so file row order is the only available ordering signal) at the
    given `stride`, producing one training example per window.

    Each window's label is a MAJORITY VOTE over its `window` rows'
    `Attack_type` (ties broken by `pandas.Series.mode()`'s documented
    behavior: alphabetically-first tied label) -- a fixed-length LSTM
    input window needs exactly one label per window, and majority vote
    is the standard way to turn per-row labels into a window-level one.

    `stride=32` (50% overlap for the default `window=64`) is a
    standard sliding-window choice: enough overlap for good label
    coverage near label-transition boundaries without the redundancy
    (and dataset blow-up) of a stride-1 window.
    """
    n_rows = features.shape[0]
    class_names = tuple(sorted(labels.unique()))

    if n_rows < window:
        return WindowedDataset(
            X=np.empty((0, window, features.shape[1])),
            y_binary=np.empty((0,), dtype=np.int64),
            y_multiclass=np.empty((0,), dtype=np.int64),
            class_names=class_names,
            window=window,
            stride=stride,
        )

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    labels_arr = labels.to_numpy()

    starts = list(range(0, n_rows - window + 1, stride))
    X = np.empty((len(starts), window, features.shape[1]), dtype=np.float64)
    y_multiclass = np.empty((len(starts),), dtype=np.int64)
    y_binary = np.empty((len(starts),), dtype=np.int64)

    for i, start in enumerate(starts):
        end = start + window
        X[i] = features[start:end]
        window_labels = pd.Series(labels_arr[start:end])
        majority_label = window_labels.mode().iloc[0]
        y_multiclass[i] = class_to_idx[majority_label]
        y_binary[i] = 0 if majority_label in BENIGN_ATTACK_TYPES else 1

    return WindowedDataset(X=X, y_binary=y_binary, y_multiclass=y_multiclass, class_names=class_names, window=window, stride=stride)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    csv_path: Path | str,
    out_dir: Path | str,
    window: int = DEFAULT_WINDOW,
    stride: int = 32,
    degeneracy_threshold: float = 0.98,
) -> dict:
    """Loads the FULL RT-IoT2022 CSV, screens it for label degeneracy,
    extracts + standardizes + windows it, and saves the result to
    `out_dir` as `rt_iot2022_windows.npz` (+ a human-readable JSON
    report alongside it). Returns the same report as a dict.

    Raises `ValueError` if the file fails degeneracy screening --
    per PLAN.md's established practice, a degenerate file is
    discarded, not sampled around, so this pipeline refuses to produce
    a windowed dataset from it rather than silently proceeding.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rt_iot2022(csv_path)
    degeneracy = screen_label_degeneracy(df[LABEL_COLUMN], dominant_frac_threshold=degeneracy_threshold)
    if degeneracy.is_degenerate:
        raise ValueError(
            f"{csv_path} failed degeneracy screening: label "
            f"'{degeneracy.dominant_label}' covers "
            f"{degeneracy.dominant_fraction:.4%} of rows "
            f"(threshold {degeneracy_threshold:.0%}) -- discarded per "
            f"PLAN.md's established degeneracy-screening practice, not "
            f"sampled around."
        )

    raw_features = extract_flow_features(df)
    scaler = fit_scaler(raw_features)
    features = scaler.apply(raw_features)

    windows = build_windows(features, df[LABEL_COLUMN], window=window, stride=stride)

    out_path = out_dir / "rt_iot2022_windows.npz"
    np.savez_compressed(
        out_path,
        X=windows.X,
        y_binary=windows.y_binary,
        y_multiclass=windows.y_multiclass,
        class_names=np.array(windows.class_names),
        feature_names=np.array(FEATURE_NAMES),
        scaler_mean=scaler.mean,
        scaler_std=scaler.std,
        window=np.array(windows.window),
        stride=np.array(windows.stride),
    )

    binary_counts = np.bincount(windows.y_binary, minlength=2)
    multiclass_counts = np.bincount(windows.y_multiclass, minlength=len(windows.class_names))

    report = {
        "csv_path": str(csv_path),
        "out_path": str(out_path),
        "n_rows_raw": int(len(df)),
        "n_unique_raw_labels": int(df[LABEL_COLUMN].nunique()),
        "raw_label_counts": {str(k): int(v) for k, v in df[LABEL_COLUMN].value_counts().items()},
        "degeneracy": {
            "is_degenerate": degeneracy.is_degenerate,
            "dominant_label": str(degeneracy.dominant_label),
            "dominant_fraction": degeneracy.dominant_fraction,
            "threshold": degeneracy_threshold,
        },
        "n_windows": int(windows.X.shape[0]),
        "window": window,
        "stride": stride,
        "feature_dim": FEATURE_DIM,
        "binary_class_balance": {"benign": int(binary_counts[0]), "attack": int(binary_counts[1])},
        "multiclass_class_names": list(windows.class_names),
        "multiclass_class_counts": {c: int(n) for c, n in zip(windows.class_names, multiclass_counts)},
    }

    report_path = out_dir / "rt_iot2022_windows_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/rt_iot2022/RT_IOT2022.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/rt_iot2022"))
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--stride", type=int, default=32)
    args = parser.parse_args()
    result = run_pipeline(args.csv, args.out_dir, window=args.window, stride=args.stride)
    print(json.dumps(result, indent=2))
