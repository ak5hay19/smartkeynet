"""Tests for `forecaster/pcap_flow_features.py` (v2 Phase 1).

`tests/fixtures/pcap_flow_features_fixture.pcap` is a small, synthetic,
committed pcap built with scapy (dev-only tool, NOT a project
dependency -- see `scratchpad/gen_pcap_fixtures.py`'s docstring,
scratchpad/ is never committed per repo convention) covering, in one
file:
  - ARP + IPv6/ICMPv6 packets (must be SKIPPED, out of this module's
    documented IPv4-only scope, not silently miscounted as a flow).
  - Four TCP "flows" sharing the SAME 4-tuple, exercising both
    flow-instance-split rules: (A) a clean SYN..FIN teardown; (B) a
    fresh SYN ~100s later (< the 300s TCP inactivity timeout) that
    must still split, because A closed via FIN; (C) a fresh SYN 200s
    after B's RST (< 300s) that must ALSO split, because B closed via
    RST; (D) a fresh SYN 400s after C's last packet (> 300s) that must
    split purely on the inactivity backstop, since C itself never
    closed (no FIN/RST) -- the one scenario that specifically isolates
    the timeout rule from the closed+SYN rule.
  - A 5-packet TCP flow with an intentional >5s gap, to check the
    ACTIVE_IDLE_SPLIT_SEC-based active/idle burst split.
  - A real IPv4/UDP DNS query+response (service -> `dns`).
  - A minimal TLS-record-shaped TCP flow (service -> `ssl`).
  - A generic/unidentified UDP flow (service -> `-`).
  - A UDP flow split into two bursts >60s apart (UDP inactivity split).

Real per-flow values below were computed by hand from the fixture's
own construction and cross-checked against a real run of the extractor
(not asserted blind) before being written into these tests.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from forecaster.pcap_flow_features import (
    ACTIVE_IDLE_SPLIT_SEC,
    INACTIVITY_TIMEOUT_TCP_SEC,
    INACTIVITY_TIMEOUT_UDP_SEC,
    SERVICE_MAP,
    extract_pcap_flow_features,
    map_service,
)
from forecaster.rt_iot_features import FEATURE_DIM, PROTO_CATEGORIES, SERVICE_CATEGORIES

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pcap_flow_features_fixture.pcap"
REAL_SAMPLE = Path(__file__).resolve().parents[1] / "scratchpad" / "sample.pcap"

requires_tshark = pytest.mark.skipif(shutil.which("tshark") is None, reason="tshark not found on PATH")
requires_real_sample = pytest.mark.skipif(
    not REAL_SAMPLE.exists(), reason="scratchpad/sample.pcap not present (gitignored, session-local capture)"
)


# ---------------------------------------------------------------------------
# Service-mapping policy -- pure-function tests (no pcap/tshark needed)
# ---------------------------------------------------------------------------


class TestServiceMapping:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("DNS", "dns"),
            ("MQTT", "mqtt"),
            ("HTTP", "http"),
            ("HTTP2", "http"),
            ("SSL", "ssl"),
            ("TLSv1.2", "ssl"),
            ("TLSv1.3", "ssl"),
            ("NTP", "ntp"),
            ("DHCP", "dhcp"),
            ("BOOTP", "dhcp"),
            ("IRC", "irc"),
            ("SSH", "ssh"),
            ("RADIUS", "radius"),
        ],
    )
    def test_named_protocols_map_to_their_rt_iot2022_bucket(self, raw, expected):
        assert map_service(raw) == expected

    @pytest.mark.parametrize("raw", ["QUIC", "MDNS", "TCP", "UDP", "ARP", "PKIX-CRL", "ICMPv6", "SOMETHING_UNKNOWN"])
    def test_unmapped_protocols_fall_back_to_dash_not_a_named_bucket(self, raw):
        """The explicit 'unmapped/other' policy: QUIC and anything else
        with no real RT-IoT2022-era analog maps to `-` (RT-IoT2022's
        OWN 'no identified service' bucket), never silently forced
        into an unrelated named category."""
        assert map_service(raw) == "-"

    def test_mapping_is_case_insensitive(self):
        assert map_service("dns") == map_service("Dns") == map_service("DNS") == "dns"

    def test_every_mapped_value_is_a_real_rt_iot2022_service_category(self):
        assert set(SERVICE_MAP.values()) <= set(SERVICE_CATEGORIES)

    def test_every_non_dash_rt_iot2022_category_has_at_least_one_mapping(self):
        mapped_targets = set(SERVICE_MAP.values())
        for category in SERVICE_CATEGORIES:
            if category == "-":
                continue
            assert category in mapped_targets, f"{category} has no tshark protocol mapped to it"


# ---------------------------------------------------------------------------
# Full extractor, against the committed synthetic fixture
# ---------------------------------------------------------------------------


@requires_tshark
class TestExtractPcapFlowFeatures:
    @pytest.fixture(scope="class")
    def result(self):
        return extract_pcap_flow_features(FIXTURE)

    def test_non_ipv4_packets_are_skipped_and_counted_not_silently_dropped(self, result):
        assert result.n_packets_total == 36
        assert result.n_packets_skipped_non_ipv4 == 2  # 1 ARP + 1 IPv6/ICMPv6
        assert result.n_packets_used == 34

    def test_flow_count_and_output_shape(self, result):
        assert result.n_flows == 10
        assert result.raw_flow_df.shape[0] == 10
        assert result.features.shape == (10, FEATURE_DIM)

    def test_feature_output_is_finite_no_nan_no_inf(self, result):
        assert np.isfinite(result.features).all()

    def test_every_row_proto_is_a_real_proto_category(self, result):
        assert set(result.raw_flow_df["proto"]) <= set(PROTO_CATEGORIES)

    def test_every_row_service_is_a_real_service_category(self, result):
        assert set(result.raw_flow_df["service"]) <= set(SERVICE_CATEGORIES)

    def test_tcp_flow_splits_on_clean_teardown_then_fresh_syn_within_timeout(self, result):
        """Flows A/B: same 4-tuple, B's SYN arrives only ~100s after A's
        FIN teardown (well under the 300s TCP inactivity timeout) --
        must still be 2 separate rows, not merged, because A closed."""
        df = result.raw_flow_df
        row_a, row_b = df.iloc[0], df.iloc[1]
        assert (row_a["fwd_pkts_tot"], row_a["bwd_pkts_tot"]) == (5, 3)
        assert row_a["flow_FIN_flag_count"] == 2
        assert row_a["service"] == "http"
        assert (row_b["fwd_pkts_tot"], row_b["bwd_pkts_tot"]) == (3, 1)
        assert row_b["flow_RST_flag_count"] == 1
        assert row_b["flow_FIN_flag_count"] == 0

    def test_tcp_flow_splits_on_rst_teardown_then_fresh_syn_within_timeout(self, result):
        """Flow C: fresh SYN 200s after B's RST (< 300s) -- must ALSO
        split, because B closed via RST, not FIN."""
        row_c = result.raw_flow_df.iloc[2]
        assert (row_c["fwd_pkts_tot"], row_c["bwd_pkts_tot"]) == (3, 1)
        assert row_c["flow_RST_flag_count"] == 0
        assert row_c["flow_FIN_flag_count"] == 0
        assert row_c["fwd_PSH_flag_count"] == 1

    def test_tcp_flow_splits_purely_on_inactivity_timeout_when_never_closed(self, result):
        """Flow D: fresh SYN 400s after C's last packet (> 300s). C was
        left OPEN (no FIN/RST) -- this split can ONLY be explained by
        the pure inactivity-timeout rule, isolating it from the
        closed+SYN rule the two tests above exercise."""
        row_d = result.raw_flow_df.iloc[3]
        assert (row_d["fwd_pkts_tot"], row_d["bwd_pkts_tot"]) == (1, 1)
        assert row_d["flow_duration"] == pytest.approx(0.01)

    def test_tcp_inactivity_timeout_constant_matches_zeek_default(self):
        assert INACTIVITY_TIMEOUT_TCP_SEC == 300.0

    def test_udp_inactivity_timeout_constant_matches_zeek_default(self):
        assert INACTIVITY_TIMEOUT_UDP_SEC == 60.0

    def test_udp_flow_splits_on_inactivity_timeout(self, result):
        """The dedicated UDP timeout-split fixture flow: 2 packets,
        then a >60s gap, then 2 more packets on the same 4-tuple --
        must be 2 separate rows of 2 packets each, not 1 of 4."""
        df = result.raw_flow_df
        udp_dash_rows = df[(df["proto"] == "udp") & (df["service"] == "-")]
        assert len(udp_dash_rows) == 3  # 1 generic-UDP flow + 2 split instances
        for _, row in udp_dash_rows.iterrows():
            assert (row["fwd_pkts_tot"], row["bwd_pkts_tot"]) == (1, 1)

    def test_active_idle_burst_split(self, result):
        """3 packets within 1s (one active burst), then a 9s gap (>
        ACTIVE_IDLE_SPLIT_SEC), then 2 more packets (a second 1s
        burst): active.avg must be the mean of the two ~1s bursts
        (1,000,000 microseconds), idle.avg the single ~9s gap
        (9,000,000 microseconds) -- see module docstring item 3 for
        why these are in microseconds, not seconds."""
        assert ACTIVE_IDLE_SPLIT_SEC == 5.0
        row = result.raw_flow_df.iloc[4]
        assert (row["fwd_pkts_tot"], row["bwd_pkts_tot"]) == (3, 2)
        assert row["flow_duration"] == pytest.approx(11.0)
        assert row["active.avg"] == pytest.approx(1_000_000.0)
        assert row["idle.avg"] == pytest.approx(9_000_000.0)

    def test_dns_flow_maps_to_dns_service(self, result):
        row = result.raw_flow_df[result.raw_flow_df["service"] == "dns"].iloc[0]
        assert row["proto"] == "udp"
        assert (row["fwd_pkts_tot"], row["bwd_pkts_tot"]) == (1, 1)

    def test_tls_flow_maps_to_ssl_service_despite_handshake_packets_outnumbering_it(self, result):
        """Real bug this session caught in its own fixture testing: a
        plain majority vote over ALL packets' labels would pick `-`
        here (2 generic-TCP handshake packets vs. 1 TLSv1.2-identified
        packet) -- the fixed policy (majority among only
        confidently-identified packets) correctly reports `ssl`."""
        row = result.raw_flow_df[result.raw_flow_df["service"] == "ssl"].iloc[0]
        assert row["proto"] == "tcp"
        assert (row["fwd_pkts_tot"], row["bwd_pkts_tot"]) == (2, 1)

    def test_timing_columns_use_the_documented_mixed_unit_convention(self, result):
        """flow_duration in seconds, flow_iat.avg in microseconds --
        see module docstring item 3's empirical cross-check against
        the real RT-IoT2022 training CSV. Flow A's packets are spaced
        exactly 0.01s apart; if flow_iat.avg were (wrongly) also in
        seconds this would read ~0.01, not ~10,000."""
        row_a = result.raw_flow_df.iloc[0]
        assert row_a["flow_duration"] == pytest.approx(0.07)
        assert row_a["flow_iat.avg"] == pytest.approx(10_000.0)

    def test_header_and_window_and_flag_counts_are_real_not_placeholder(self, result):
        row_a = result.raw_flow_df.iloc[0]
        assert row_a["fwd_header_size_tot"] == pytest.approx(200.0)  # 5 pkts * (20 IP + 20 TCP)
        assert row_a["bwd_header_size_tot"] == pytest.approx(120.0)  # 3 pkts * 40
        assert row_a["fwd_init_window_size"] == pytest.approx(64240.0)
        assert row_a["bwd_init_window_size"] == pytest.approx(65535.0)
        assert row_a["fwd_PSH_flag_count"] == 1
        assert row_a["down_up_ratio"] == pytest.approx(3 / 5)

    def test_non_tcp_flows_report_zero_window_size_matching_real_rt_iot2022_convention(self, result):
        """Confirmed against the real training CSV this session: every
        real `proto == 'udp'` row has `fwd_init_window_size ==
        bwd_init_window_size == 0.0` (TCP-only concept)."""
        udp_rows = result.raw_flow_df[result.raw_flow_df["proto"] == "udp"]
        assert (udp_rows["fwd_init_window_size"] == 0.0).all()
        assert (udp_rows["bwd_init_window_size"] == 0.0).all()


@requires_tshark
def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        extract_pcap_flow_features("does_not_exist.pcap")


# ---------------------------------------------------------------------------
# Real captured pcap (Step 3 validation) -- skipped if not present, since
# scratchpad/ is never committed (repo convention).
# ---------------------------------------------------------------------------


@requires_tshark
@requires_real_sample
def test_real_captured_sample_produces_structurally_sane_output():
    """End-to-end run against a real ~10s/1,716-packet own-traffic
    capture (2026-09-18 feasibility session's sample -- QUIC/TLS1.3/
    DNS/HTTP mix). Not a golden-value test (a real capture isn't
    reproducible byte-for-byte) -- checks structural sanity only, per
    this session's own Step 3 instruction: finite, non-negative where
    the raw CSV's own columns are documented non-negative, and roughly
    within the real training data's observed order of magnitude."""
    result = extract_pcap_flow_features(REAL_SAMPLE)
    assert result.n_flows > 0
    assert np.isfinite(result.features).all()

    df = result.raw_flow_df
    non_negative_cols = [c for c in df.columns if c not in ("proto", "service", "down_up_ratio")]
    for col in non_negative_cols:
        assert (df[col] >= 0).all(), f"{col} had a negative value"

    # real training data's own observed max order of magnitude (see
    # rt_iot_features.py's module comment: "flow_iat.std ranges
    # 0..1.34e8") -- a real capture should not blow wildly past it.
    assert (df["flow_iat.avg"] < 1.34e8).all()
    assert (df["flow_duration"] < 21728.34 * 10).all()

    # real, honestly-observed finding this session (not asserted as
    # universal): this particular capture's own DNS traffic happened to
    # be IPv6 (a link-local resolver), out of this extractor's
    # documented IPv4-only scope -- so no `dns` bucket is expected here,
    # confirmed by inspecting the raw capture directly, not a mapping
    # bug. `ssl`/`http`/`-` are the buckets this real file exercises.
    assert set(df["service"]) <= set(SERVICE_CATEGORIES)
    assert result.n_packets_skipped_non_ipv4 > 0  # the real ARP/IPv6 packets in this capture
