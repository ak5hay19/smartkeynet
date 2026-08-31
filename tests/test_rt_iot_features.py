"""Real behavioral tests for `forecaster/rt_iot_features.py` (this
session's RT-IoT2022 feature-extraction pipeline -- PLAN2.md Addition
E Phase 1). Not smoke tests: hand-verified numeric expectations,
synthetic degenerate/non-degenerate fixtures, and a real end-to-end
run against the full vendored RT-IoT2022 CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecaster import rt_iot_features as rif

REAL_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "rt_iot2022" / "RT_IOT2022.csv"
requires_real_dataset = pytest.mark.skipif(
    not REAL_CSV.exists(), reason="full RT-IoT2022 CSV not present at data/raw/rt_iot2022/RT_IOT2022.csv"
)


def _minimal_row(**overrides) -> dict:
    """One fully-specified flow row with every required raw column
    zeroed/defaulted, so tests only need to set the fields they care
    about."""
    row = {col: 0.0 for col in rif.NUMERIC_LOG1P_COLUMNS}
    row.update({col: 0.0 for col in rif.RAW_RATIO_COLUMNS})
    row["proto"] = "tcp"
    row["service"] = "-"
    row[rif.LABEL_COLUMN] = "MQTT_Publish"
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Degeneracy screening
# ---------------------------------------------------------------------------


def test_degeneracy_screening_rejects_synthetic_degenerate_file():
    labels = pd.Series(["critical"] * 995 + ["other"] * 5)  # 99.5% one label
    result = rif.screen_label_degeneracy(labels, dominant_frac_threshold=0.98)
    assert result.is_degenerate is True
    assert result.dominant_label == "critical"
    assert result.dominant_fraction == pytest.approx(0.995)
    assert result.n_unique_labels == 2
    assert result.n_rows == 1000


def test_degeneracy_screening_rejects_fully_uniform_label():
    labels = pd.Series(["critical"] * 422)  # the DO-NOT-USE precedent: 100% one value
    result = rif.screen_label_degeneracy(labels)
    assert result.is_degenerate is True
    assert result.dominant_fraction == 1.0


def test_degeneracy_screening_accepts_balanced_synthetic_data():
    labels = pd.Series(["a", "b", "c", "d"] * 250)  # perfectly balanced, 4 classes
    result = rif.screen_label_degeneracy(labels, dominant_frac_threshold=0.98)
    assert result.is_degenerate is False
    assert result.dominant_fraction == pytest.approx(0.25)
    assert result.n_unique_labels == 4


def test_degeneracy_screening_empty_series_is_degenerate():
    result = rif.screen_label_degeneracy(pd.Series([], dtype=object))
    assert result.is_degenerate is True
    assert result.n_rows == 0


@requires_real_dataset
def test_degeneracy_screening_on_real_rt_iot2022_labels():
    df = rif.load_rt_iot2022(REAL_CSV)
    result = rif.screen_label_degeneracy(df[rif.LABEL_COLUMN])
    # Real, checked result: DOS_SYN_Hping is the dominant label at
    # ~76.9% of 123,117 rows -- well below the 98% degeneracy
    # threshold, so the real file passes screening (not discarded).
    assert result.is_degenerate is False
    assert result.dominant_label == "DOS_SYN_Hping"
    assert 0.75 < result.dominant_fraction < 0.78
    assert result.n_unique_labels == 12
    assert result.n_rows == 123_117


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_rt_iot2022_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        rif.load_rt_iot2022("does/not/exist.csv")


def test_load_rt_iot2022_missing_columns_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"proto": ["tcp"], "service": ["-"]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        rif.load_rt_iot2022(bad_csv)


def test_load_rt_iot2022_empty_file_raises(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    columns = list(rif.REQUIRED_RAW_COLUMNS) + [rif.LABEL_COLUMN]
    pd.DataFrame(columns=columns).to_csv(empty_csv, index=False)
    with pytest.raises(ValueError, match="zero rows"):
        rif.load_rt_iot2022(empty_csv)


@requires_real_dataset
def test_load_rt_iot2022_full_dataset_shape_and_dtypes():
    df = rif.load_rt_iot2022(REAL_CSV)
    assert df.shape == (123_117, 85)
    assert df["proto"].dtype == object
    assert df["service"].dtype == object
    assert df[rif.LABEL_COLUMN].dtype == object
    for col in rif.NUMERIC_LOG1P_COLUMNS + rif.RAW_RATIO_COLUMNS:
        assert pd.api.types.is_numeric_dtype(df[col]), col
    assert df.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Extraction -- hand-verified exact values
# ---------------------------------------------------------------------------


def test_extract_flow_features_hand_calculated_single_row():
    row = _minimal_row(
        fwd_pkts_per_sec=9.0,  # log1p(9) = ln(10)
        down_up_ratio=2.5,
        proto="udp",
        service="dns",
    )
    df = pd.DataFrame([row])
    features = rif.extract_flow_features(df)

    assert features.shape == (1, rif.FEATURE_DIM)

    expected = np.zeros(rif.FEATURE_DIM)
    idx_fwd_pkts = rif.FEATURE_NAMES.index("fwd_pkts_per_sec")
    expected[idx_fwd_pkts] = np.log1p(9.0)
    idx_ratio = rif.FEATURE_NAMES.index("down_up_ratio")
    expected[idx_ratio] = 2.5
    idx_proto_udp = rif.FEATURE_NAMES.index("proto=udp")
    expected[idx_proto_udp] = 1.0
    idx_service_dns = rif.FEATURE_NAMES.index("service=dns")
    expected[idx_service_dns] = 1.0

    np.testing.assert_allclose(features[0], expected, atol=1e-12)
    assert np.log1p(9.0) == pytest.approx(np.log(10.0))


def test_extract_flow_features_clips_negative_values_defensively():
    row = _minimal_row(flow_duration=-5.0)
    df = pd.DataFrame([row])
    features = rif.extract_flow_features(df)
    idx = rif.FEATURE_NAMES.index("flow_duration")
    assert features[0, idx] == 0.0  # clipped to 0 before log1p(0) = 0


def test_extract_flow_features_missing_column_raises():
    df = pd.DataFrame([{"proto": "tcp", "service": "-"}])
    with pytest.raises(ValueError, match="missing required raw columns"):
        rif.extract_flow_features(df)


def test_extract_flow_features_unknown_category_maps_to_all_zero_onehot():
    row = _minimal_row(proto="sctp", service="unknown-service")  # not in fixed category lists
    df = pd.DataFrame([row])
    features = rif.extract_flow_features(df)
    proto_start = rif.FEATURE_NAMES.index("proto=tcp")
    proto_block = features[0, proto_start : proto_start + len(rif.PROTO_CATEGORIES)]
    assert proto_block.sum() == 0.0
    service_start = rif.FEATURE_NAMES.index(f"service={rif.SERVICE_CATEGORIES[0]}")
    service_block = features[0, service_start : service_start + len(rif.SERVICE_CATEGORIES)]
    assert service_block.sum() == 0.0


def test_single_row_and_batch_extraction_agree():
    """The 'one implementation, two callers' guarantee (PLAN2.md
    Addition D): calling `extract_flow_features` on a lone row must
    give the identical result whether that row is passed alone (a
    future single-flow inference caller) or embedded in a larger
    batch (the offline training caller)."""
    rows = [
        _minimal_row(fwd_pkts_per_sec=3.0, proto="tcp", service="http"),
        _minimal_row(fwd_pkts_per_sec=500.0, down_up_ratio=1.2, proto="udp", service="dns"),
        _minimal_row(flow_SYN_flag_count=4.0, proto="icmp", service="-"),
    ]
    batch_df = pd.DataFrame(rows)
    batch_features = rif.extract_flow_features(batch_df)

    for i, row in enumerate(rows):
        single_df = pd.DataFrame([row])
        single_features = rif.extract_flow_features(single_df)
        np.testing.assert_allclose(single_features[0], batch_features[i], atol=1e-12)


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------


def test_fit_scaler_and_apply_zero_means_continuous_columns():
    rows = [_minimal_row(fwd_pkts_per_sec=v) for v in (1.0, 9.0, 99.0)]
    df = pd.DataFrame(rows)
    raw = rif.extract_flow_features(df)
    scaler = rif.fit_scaler(raw)
    scaled = scaler.apply(raw)
    n_continuous = len(rif.CONTINUOUS_COLUMNS)
    np.testing.assert_allclose(scaled[:, :n_continuous].mean(axis=0), 0.0, atol=1e-10)
    # one-hot block must be untouched by scaling
    np.testing.assert_allclose(scaled[:, n_continuous:], raw[:, n_continuous:])


def test_fit_scaler_handles_constant_column_without_dividing_by_zero():
    rows = [_minimal_row() for _ in range(5)]  # every continuous column constant at 0
    df = pd.DataFrame(rows)
    raw = rif.extract_flow_features(df)
    scaler = rif.fit_scaler(raw)
    assert np.all(scaler.std > 0)
    scaled = scaler.apply(raw)
    assert not np.any(np.isnan(scaled))
    assert not np.any(np.isinf(scaled))


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def _synthetic_features_and_labels(n_rows: int, feature_dim: int = 4):
    rng = np.random.default_rng(0)
    features = rng.normal(size=(n_rows, feature_dim))
    return features


def test_build_windows_shape_and_count():
    n_rows, window, stride, feature_dim = 100, 10, 5, 4
    features = _synthetic_features_and_labels(n_rows, feature_dim)
    labels = pd.Series(["benign"] * n_rows)
    windows = rif.build_windows(features, labels, window=window, stride=stride)
    expected_n_windows = (n_rows - window) // stride + 1
    assert windows.X.shape == (expected_n_windows, window, feature_dim)
    assert windows.y_binary.shape == (expected_n_windows,)
    assert windows.y_multiclass.shape == (expected_n_windows,)


def test_build_windows_majority_vote_labeling_hand_verified():
    # 10 rows: first 6 "attack_a", last 4 "attack_b" -> one window of
    # size 10, stride irrelevant (only one window fits) -- majority is
    # "attack_a" (6 > 4).
    features = _synthetic_features_and_labels(10, feature_dim=2)
    labels = pd.Series(["attack_a"] * 6 + ["attack_b"] * 4)
    windows = rif.build_windows(features, labels, window=10, stride=10)
    assert windows.X.shape[0] == 1
    class_names = windows.class_names
    majority_idx = class_names.index("attack_a")
    assert windows.y_multiclass[0] == majority_idx
    assert windows.y_binary[0] == 1  # "attack_a" not in BENIGN_ATTACK_TYPES


def test_build_windows_benign_majority_gives_binary_zero():
    features = _synthetic_features_and_labels(10, feature_dim=2)
    labels = pd.Series(["MQTT_Publish"] * 7 + ["DOS_SYN_Hping"] * 3)
    windows = rif.build_windows(features, labels, window=10, stride=10)
    assert windows.y_binary[0] == 0


def test_build_windows_too_few_rows_returns_empty():
    features = _synthetic_features_and_labels(5, feature_dim=3)
    labels = pd.Series(["benign"] * 5)
    windows = rif.build_windows(features, labels, window=10, stride=5)
    assert windows.X.shape == (0, 10, 3)
    assert windows.y_binary.shape == (0,)
    assert windows.y_multiclass.shape == (0,)


def test_build_windows_exact_row_slicing():
    n_rows, window, stride, feature_dim = 20, 5, 5, 2
    features = _synthetic_features_and_labels(n_rows, feature_dim)
    labels = pd.Series(["benign"] * n_rows)
    windows = rif.build_windows(features, labels, window=window, stride=stride)
    # window 0 must be exactly rows [0:5), window 1 exactly rows [5:10)
    np.testing.assert_allclose(windows.X[0], features[0:5])
    np.testing.assert_allclose(windows.X[1], features[5:10])


# ---------------------------------------------------------------------------
# Full pipeline -- real, end-to-end, against the FULL dataset
# ---------------------------------------------------------------------------


def test_run_pipeline_raises_on_degenerate_input(tmp_path):
    columns = list(rif.REQUIRED_RAW_COLUMNS) + [rif.LABEL_COLUMN]
    rows = [_minimal_row() for _ in range(998)] + [
        _minimal_row(**{rif.LABEL_COLUMN: "DOS_SYN_Hping"}) for _ in range(2)
    ]
    df = pd.DataFrame(rows)[columns]
    csv_path = tmp_path / "degenerate.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="degeneracy"):
        rif.run_pipeline(csv_path, tmp_path / "out")


@requires_real_dataset
def test_run_pipeline_end_to_end_real_full_dataset(tmp_path):
    out_dir = tmp_path / "processed"
    report = rif.run_pipeline(REAL_CSV, out_dir, window=64, stride=32)

    # Real dataset stats, sanity-checked, not hand-fixed to exact
    # numbers that would break on any future re-derivation:
    assert report["n_rows_raw"] == 123_117
    assert report["degeneracy"]["is_degenerate"] is False
    assert report["n_windows"] > 0
    assert report["feature_dim"] == rif.FEATURE_DIM
    assert sum(report["binary_class_balance"].values()) == report["n_windows"]
    assert sum(report["multiclass_class_counts"].values()) == report["n_windows"]

    saved_path = out_dir / "rt_iot2022_windows.npz"
    assert saved_path.exists()
    report_path = out_dir / "rt_iot2022_windows_report.json"
    assert report_path.exists()

    loaded = np.load(saved_path, allow_pickle=True)
    X = loaded["X"]
    y_binary = loaded["y_binary"]
    y_multiclass = loaded["y_multiclass"]

    assert X.shape[0] == report["n_windows"] > 0
    assert X.shape[1] == 64
    assert X.shape[2] == rif.FEATURE_DIM
    assert not np.any(np.isnan(X))
    assert not np.any(np.isinf(X))
    # not degenerate: no single feature column constant across every window
    flattened = X.reshape(-1, X.shape[-1])
    assert np.all(flattened.std(axis=0) > 0)

    assert y_binary.shape == (report["n_windows"],)
    assert y_multiclass.shape == (report["n_windows"],)
    assert set(np.unique(y_binary)).issubset({0, 1})

    scaler_mean = loaded["scaler_mean"]
    scaler_std = loaded["scaler_std"]
    assert scaler_mean.shape == (len(rif.CONTINUOUS_COLUMNS),)
    assert scaler_std.shape == (len(rif.CONTINUOUS_COLUMNS),)


# ---------------------------------------------------------------------------
# Shape/interface documentation test (LSTM input-layer contract)
# ---------------------------------------------------------------------------


def test_feature_dim_matches_documented_constant():
    assert rif.FEATURE_DIM == len(rif.CONTINUOUS_COLUMNS) + len(rif.PROTO_CATEGORIES) + len(rif.SERVICE_CATEGORIES)
    assert len(rif.FEATURE_NAMES) == rif.FEATURE_DIM


def test_window_dimension_matches_forecaster_model_window():
    from forecaster.model import WINDOW

    assert rif.DEFAULT_WINDOW == WINDOW
