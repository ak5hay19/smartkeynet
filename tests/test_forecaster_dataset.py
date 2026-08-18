"""Behavioral tests for `forecaster.dataset` (PLAN.md Addition A;
PLAN2 §6 provenance rules, §8 Addition D's "one implementation, two
callers" feature extractor).
"""

from __future__ import annotations

import numpy as np
import pytest

from forecaster.dataset import (
    BENIGN_ATTACK_TYPES,
    FEATURE_COLUMNS,
    N_FEATURES,
    FeatureStandardizer,
    RTIoT2022Dataset,
    ThreatTraceSampler,
    extract_flow_features,
    is_attack,
    load_rt_iot2022,
    pool_signal_vector,
    resolve_dataset_path,
)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _dataset_available() -> bool:
    try:
        resolve_dataset_path()
        return True
    except FileNotFoundError:
        return False


requires_dataset = pytest.mark.skipif(
    not _dataset_available(),
    reason="RT-IoT2022 is gitignored; skipped where the operator has not placed it",
)


# ---------------------------------------------------------------------------
# The shared feature extractor (Hard Rule 11: one implementation, two callers)
# ---------------------------------------------------------------------------


def _csv_style_row(**overrides) -> dict[str, str]:
    row = {column: "1.0" for column in FEATURE_COLUMNS}
    row.update({k: str(v) for k, v in overrides.items()})
    return row


def _pcap_style_record(**overrides) -> dict[str, float]:
    record = {column: 1.0 for column in FEATURE_COLUMNS}
    record.update(overrides)
    return record


def test_extractor_output_shape_is_identical_for_csv_rows_and_pcap_records():
    """Addition D's stated unit test: "feature-extraction output is
    identical in shape whether sourced from the training CSV or a pcap
    window". Same function, two callers -- there is no second pipeline."""
    from_csv = extract_flow_features([_csv_style_row() for _ in range(5)])
    from_pcap = extract_flow_features([_pcap_style_record() for _ in range(5)])

    assert from_csv.shape == from_pcap.shape == (5, N_FEATURES)
    assert np.allclose(from_csv, from_pcap)


def test_extractor_tolerates_fields_a_pcap_flow_may_not_carry():
    """A pcap-derived flow legitimately may not have every field a
    Zeek-derived CSV row does. A hole must not take down inference."""
    partial = extract_flow_features([{"fwd_pkts_per_sec": 10.0}])
    assert partial.shape == (1, N_FEATURES)
    assert np.all(np.isfinite(partial))


def test_extractor_replaces_unparseable_and_non_finite_values_with_zero():
    features = extract_flow_features(
        [_pcap_style_record(down_up_ratio="not-a-number", flow_duration=float("inf"))]
    )
    assert np.all(np.isfinite(features))


def test_extractor_returns_an_empty_matrix_of_the_right_width():
    assert extract_flow_features([]).shape == (0, N_FEATURES)


def test_heavy_tailed_columns_are_log_scaled_before_standardization():
    """Without it a single ~10^6 packets/sec flood flow dominates the
    column's statistics and every benign flow standardizes to roughly
    the same value."""
    small = extract_flow_features([_pcap_style_record(fwd_pkts_per_sec=1.0)])[0]
    huge = extract_flow_features([_pcap_style_record(fwd_pkts_per_sec=1_000_000.0)])[0]
    index = FEATURE_COLUMNS.index("fwd_pkts_per_sec")
    assert huge[index] < 20.0  # log1p(1e6) ~ 13.8, not 1e6
    assert huge[index] > small[index]


# ---------------------------------------------------------------------------
# Standardizer
# ---------------------------------------------------------------------------


def test_standardizer_centres_the_data_it_was_fit_on():
    features = np.random.default_rng(0).normal(5.0, 2.0, size=(500, N_FEATURES)).astype(np.float32)
    standardizer = FeatureStandardizer.fit(features)
    transformed = standardizer.transform(features)
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-4)


def test_standardizer_passes_constant_columns_through_instead_of_dividing_by_zero():
    features = np.ones((10, N_FEATURES), dtype=np.float32)
    transformed = FeatureStandardizer.fit(features).transform(features)
    assert np.all(np.isfinite(transformed))


def test_standardizer_round_trips_through_a_dict():
    features = np.random.default_rng(1).normal(size=(50, N_FEATURES)).astype(np.float32)
    original = FeatureStandardizer.fit(features)
    restored = FeatureStandardizer.from_dict(original.to_dict())
    assert np.allclose(original.transform(features), restored.transform(features))


# ---------------------------------------------------------------------------
# Labels and provenance
# ---------------------------------------------------------------------------


def test_benign_classes_are_the_datasets_own_normal_operation_patterns():
    assert BENIGN_ATTACK_TYPES == {"Thing_Speak", "MQTT_Publish", "Wipro_bulb"}
    assert not is_attack("Thing_Speak")
    assert is_attack("DOS_SYN_Hping")
    assert is_attack("NMAP_XMAS_TREE_SCAN")


def test_degenerate_label_variety_is_refused(tmp_path):
    """PLAN2 §6's DO-NOT-USE rule, enforced on load rather than trusted."""
    path = tmp_path / "degenerate.csv"
    header = ",".join([*FEATURE_COLUMNS, "Attack_type"])
    rows = "\n".join(",".join(["1.0"] * len(FEATURE_COLUMNS) + ["DOS_SYN_Hping"]) for _ in range(500))
    path.write_text(f"{header}\n{rows}\n")

    with pytest.raises(ValueError, match="label variety"):
        load_rt_iot2022(path=path)


def test_missing_feature_columns_are_reported_by_name(tmp_path):
    path = tmp_path / "wrong_schema.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing expected feature columns"):
        load_rt_iot2022(path=path)


# ---------------------------------------------------------------------------
# ThreatTraceSampler -- the scenario/capture join
# ---------------------------------------------------------------------------


def _toy_dataset(seed: int = 0) -> RTIoT2022Dataset:
    rng = np.random.default_rng(seed)
    benign = rng.normal(0.0, 1.0, size=(400, N_FEATURES)).astype(np.float32)
    attack = rng.normal(3.0, 1.0, size=(400, N_FEATURES)).astype(np.float32)
    return RTIoT2022Dataset(
        benign=benign,
        attack=attack,
        standardizer=FeatureStandardizer.fit(benign),
        label_counts={"Thing_Speak": 400, "DOS_SYN_Hping": 400},
    )


def test_attack_probability_rises_monotonically_with_threat_level():
    levels = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    probabilities = [ThreatTraceSampler.attack_probability(level) for level in levels]
    assert probabilities == sorted(probabilities)
    assert probabilities[1] < 0.1   # benign (level 0) is mostly benign traffic
    assert probabilities[-1] > 0.9  # a strongly elevated level is mostly attack traffic


def test_sampler_draws_contiguous_windows_of_the_requested_shape():
    sampler = ThreatTraceSampler(_toy_dataset(), window_size=16, seed=0)
    window = sampler.window_for_level(0.0)
    assert window.shape == (16, N_FEATURES)


def test_sampler_mixture_follows_the_threat_level():
    calm = ThreatTraceSampler(_toy_dataset(), window_size=8, seed=0)
    high = ThreatTraceSampler(_toy_dataset(), window_size=8, seed=0)

    calm_attack_share = np.mean([calm.labelled_window_for_level(0.0)[1] for _ in range(400)])
    high_attack_share = np.mean([high.labelled_window_for_level(4.0)[1] for _ in range(400)])

    assert calm_attack_share < 0.15
    assert high_attack_share > 0.85


def test_sampler_is_seed_reproducible():
    a = ThreatTraceSampler(_toy_dataset(), window_size=8, seed=7)
    b = ThreatTraceSampler(_toy_dataset(), window_size=8, seed=7)
    assert np.allclose(a.window_for_level(1.0), b.window_for_level(1.0))


def test_sampler_handles_a_source_shorter_than_the_window():
    tiny = RTIoT2022Dataset(
        benign=np.zeros((3, N_FEATURES), dtype=np.float32),
        attack=np.ones((3, N_FEATURES), dtype=np.float32),
        standardizer=FeatureStandardizer.fit(np.zeros((3, N_FEATURES), dtype=np.float32)),
        label_counts={},
    )
    sampler = ThreatTraceSampler(tiny, window_size=16, seed=0)
    assert sampler.window_for_level(0.0).shape == (16, N_FEATURES)


def test_pool_signal_vector_is_fixed_order_and_excludes_the_tenant_mix():
    """`arrivals_per_class` is deliberately not a pool signal: it encodes
    the tenant mix, and routing it into a model whose threat head feeds
    the policy table would blur Hard Rule 3's separation between "which
    requests arrive" and "what the agent sees"."""
    observation = {"qber": 0.02, "skr": 0.015, "pool_fill": 0.5, "hybrid_serves": 2,
                   "arrivals_per_class": [9, 9, 9, 9], "threat_features": []}
    assert pool_signal_vector(observation) == [0.02, 0.015, 0.5, 2.0]


# ---------------------------------------------------------------------------
# The real capture
# ---------------------------------------------------------------------------


@requires_dataset
def test_real_capture_loads_with_both_classes_present():
    dataset = load_rt_iot2022(max_rows=20_000)
    assert len(dataset.benign) > 0
    assert len(dataset.attack) > 0
    assert dataset.benign.shape[1] == N_FEATURES
    assert 0.0 < dataset.attack_fraction < 1.0


@requires_dataset
def test_real_capture_attack_windows_are_separable_from_benign_ones():
    """The premise the threat head depends on: after benign-referenced
    standardization, attack traffic is a positive deviation. If this
    ever fails, the threat signal is pointing the wrong way and every
    posture it produces is backwards."""
    dataset = load_rt_iot2022(max_rows=40_000)
    benign_standardizer = FeatureStandardizer.fit(dataset.benign)
    benign = np.abs(benign_standardizer.transform(dataset.benign)).mean(axis=1)
    attack = np.abs(benign_standardizer.transform(dataset.attack)).mean(axis=1)
    assert attack.mean() > benign.mean()
