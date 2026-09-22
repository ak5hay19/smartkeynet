"""
forecaster/pcap_flow_features.py

Real per-flow feature extractor for arbitrary `.pcap` files (v2 Phase
1 -- docs/v2_gameplan.md; built on PROGRESS.md/SESSION_LOG.md's tshark
feasibility investigation, commit 950d8a8, which confirmed tshark's
own dissector suite supplies the `service` field and `tcp.len` gives
genuine payload-only byte counts). tshark is an external binary
(confirmed on PATH, not a pip package -- NOT added to requirements.txt,
same convention as that investigation's own `pyshark`/`cicflowmeter`
notes).

This module is the SECOND caller of `forecaster.rt_iot_features
.extract_flow_features` (imported here UNMODIFIED -- PLAN2.md Addition
D's "one implementation, two callers"): the first caller is
`rt_iot_features.run_pipeline`'s offline RT-IoT2022 CSV training path;
this module's `extract_pcap_flow_features` is the live/uploaded-.pcap
inference path. Both produce a DataFrame shaped exactly like
`REQUIRED_RAW_COLUMNS` and hand it to the SAME extraction function, so
a pcap-derived feature row is structurally identical (same columns,
same units) to a training row -- never a parallel, drifting
re-implementation.

tshark gives packets, not flows -- the new work here is grouping
packets into RT-IoT2022-shaped flow rows. Three design decisions were
made explicitly, investigated (not assumed), and are documented below
with their evidence:

1. FLOW BOUNDARY / TIMEOUT CONVENTION. RT-IoT2022's own paper
   ("Comparative Analysis of Machine Learning Models for Intrusion
   Detection ... Using the RT-IoT2022 Dataset" and the dataset's UCI
   repository description) states its bidirectional flow features were
   captured with "the Zeek network monitoring tool and the Flowmeter
   plugin" (the `zeek-flowmeter` Zeek script,
   github.com/zeek-flowmeter/zeek-flowmeter) -- NOT CICFlowMeter, which
   the earlier cicflowmeter-based feasibility attempt (2026-09-16
   SESSION_LOG entry) had assumed by default and which was the actual
   reason that attempt's gaps existed. This session looked up Zeek's
   OWN real connection-tracking defaults (docs.zeek.org /
   `scripts/base/protocols/conn/inactivity.zeek`) rather than assuming
   CICFlowMeter's unrelated 600s default:
     - TCP: a flow ends on a real FIN (both directions) or RST
       teardown; absent a clean teardown, Zeek's own
       `tcp_inactivity_timeout` default (5 minutes = 300s) closes it.
     - UDP / ICMP: Zeek's own `udp_inactivity_timeout` /
       `icmp_inactivity_timeout` default (1 minute = 60s) closes it
       (no FIN/RST concept for these protocols).
   This extractor mirrors both halves of that real convention: TCP
   flows split on FIN/RST-then-next-SYN as well as the 300s idle
   backstop; UDP/ICMP flows split purely on the 60s idle backstop.

2. ACTIVE/IDLE SUBFLOW SPLIT (`active.avg`/`idle.avg`). Separate from
   (1) -- this is the threshold `zeek-flowmeter`'s own script
   (`scripts/flowmeter.zeek`) uses to split a flow's packets into
   alternating "active burst" / "idle gap" periods for those two
   columns specifically. Read directly from that script's source:
   `const active_timeout = 5 sec &redef;`. Used verbatim below
   (`ACTIVE_IDLE_SPLIT_SEC = 5.0`).

3. TIMING-COLUMN UNIT CONVENTION -- a real, non-obvious finding, not
   assumed. RT-IoT2022's own CSV documents no units for its timing
   columns. Cross-checked empirically against the real training CSV
   (`data/raw/rt_iot2022/RT_IOT2022.csv`) rather than guessed:
   `fwd_pkts_per_sec * flow_duration == fwd_pkts_tot` holds exactly,
   confirming `flow_duration` is in SECONDS; but
   `flow_iat.avg / 1e6 * (n_pkts - 1) == flow_duration` ALSO holds
   (checked across 20+ sampled rows, including near-zero and
   62-second-duration flows) -- confirming `flow_iat.avg`/`.std`,
   `active.avg`, and `idle.avg` are in MICROSECONDS, a DIFFERENT unit
   than `flow_duration`'s seconds. This extractor reproduces that
   exact mixed-unit convention (`flow_duration` in seconds;
   `flow_iat.*`/`active.avg`/`idle.avg` multiplied by
   `MICROSECONDS_PER_SECOND` below) so a pcap-derived row is scaled by
   the checkpoint's persisted z-score scaler the same way a real
   training row would be -- silently leaving these in seconds would
   make every timing feature ~1e6x smaller than what the scaler
   expects, corrupting every downstream `threat_score`.

4. SERVICE-MAPPING POLICY (`SERVICE_MAP` below) -- see that constant's
   own docstring-comment for the full policy and rationale.

SCOPE, explicitly bounded (this module's own decision, not a project
Hard Rule): IPv4 TCP/UDP/ICMP only, matching
`rt_iot_features.PROTO_CATEGORIES` exactly. Non-IPv4 traffic (ARP,
IPv6, IPv6-ICMP, ...) is out of scope and explicitly SKIPPED, not
silently dropped -- `PCAPExtractionResult.n_packets_skipped_non_ipv4`
reports the real count so a caller can see what was excluded.
"""

from __future__ import annotations

import io
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from forecaster.rt_iot_features import FEATURE_DIM, REQUIRED_RAW_COLUMNS, extract_flow_features

# ---------------------------------------------------------------------------
# Flow-boundary / timeout constants -- see module docstring items 1-2 for
# the real, cited evidence behind each value.
# ---------------------------------------------------------------------------

INACTIVITY_TIMEOUT_TCP_SEC = 300.0  # Zeek `tcp_inactivity_timeout` default (5 min)
INACTIVITY_TIMEOUT_UDP_SEC = 60.0  # Zeek `udp_inactivity_timeout` default (1 min)
INACTIVITY_TIMEOUT_ICMP_SEC = 60.0  # Zeek `icmp_inactivity_timeout` default (1 min)
ACTIVE_IDLE_SPLIT_SEC = 5.0  # zeek-flowmeter's own `active_timeout` (flowmeter.zeek)
MICROSECONDS_PER_SECOND = 1_000_000.0  # see module docstring item 3

PROTO_NUM_TO_NAME: dict[float, str] = {1.0: "icmp", 6.0: "tcp", 17.0: "udp"}

# ---------------------------------------------------------------------------
# Service-mapping policy (module docstring item 4). Explicit CODE, not just
# a comment -- exercised directly by `tests/test_pcap_flow_features.py`.
#
# RT-IoT2022's `service` schema (`rt_iot_features.SERVICE_CATEGORIES`) has
# exactly 10 buckets: `-` (no identified service) plus 9 named application
# protocols, built ~2021-2022 from IoT device traffic. Modern general
# traffic (confirmed in the 2026-09-18 feasibility session on a real
# capture: QUIC 497/1,716 packets, TLSv1.3 74) does not map cleanly onto
# that schema. Policy, decided explicitly rather than left implicit:
#   - Any protocol tshark identifies that has a REAL, direct RT-IoT2022
#     analog (its own dissector name, confirmed via `tshark -G protocols`
#     in the feasibility session) maps to that bucket -- e.g. all TLS
#     versions and bare SSL map to the single `ssl` bucket (RT-IoT2022
#     itself does not distinguish TLS versions in its `service` column
#     either -- confirmed via the real CSV's `service` value_counts,
#     which has one `ssl` bucket, not per-version ones).
#   - Everything else -- including QUIC (no RT-IoT2022-era analog: the
#     dataset predates QUIC's mainstream deployment), MDNS, ARP-adjacent
#     protocols, PKIX-CRL, or any other tshark identification with no
#     schema slot -- maps to `-`, RT-IoT2022's OWN real "no identified
#     service" bucket (83.5% of the real training CSV). This is the
#     explicit "unmapped/other" bucket the task calls for: `-` already
#     means exactly that in RT-IoT2022's own schema, so reusing it here
#     is honest (it says "no confidently-identified service"), whereas
#     forcing QUIC into e.g. `ssl` (same transport-layer neighborhood)
#     would silently claim a specific identification tshark never made.
# ---------------------------------------------------------------------------

SERVICE_MAP: dict[str, str] = {
    "DNS": "dns",
    "MDNS": "-",  # multicast DNS: NOT RT-IoT2022's `dns` (regular unicast DNS) bucket -- explicitly unmapped
    "MQTT": "mqtt",
    "HTTP": "http",
    "HTTP2": "http",
    "SSL": "ssl",
    "TLSV1": "ssl",
    "TLSV1.1": "ssl",
    "TLSV1.2": "ssl",
    "TLSV1.3": "ssl",
    "NTP": "ntp",
    "DHCP": "dhcp",
    "BOOTP": "dhcp",  # tshark's DHCP dissector is layered on its BOOTP protocol name
    "IRC": "irc",
    "SSH": "ssh",
    "SSHV2": "ssh",
    "RADIUS": "radius",
}


def map_service(raw_protocol: str) -> str:
    """Maps one tshark `_ws.col.Protocol` value to an RT-IoT2022
    `service` category, per the explicit policy documented above.
    Unrecognized/no-schema-slot protocols (QUIC, MDNS, ARP, TCP/UDP
    with no identified upper layer, ...) map to `-`, never guessed
    into an unrelated named bucket."""
    return SERVICE_MAP.get(raw_protocol.strip().upper(), "-")


def _majority_service(raw_protocols: list[str]) -> str:
    """A flow's service label: majority vote among only its
    CONFIDENTLY-IDENTIFIED packets (mapped service != `-`), falling
    back to `-` only if none exist.

    Real bug caught during this session's own fixture testing, not
    theoretical: a plain majority vote over EVERY packet's mapped
    label badly under-counts real identifications on any flow whose
    3-way handshake outnumbers its one identified data packet (e.g. a
    3-packet TLS flow -- SYN/SYN-ACK labeled generic `TCP`, one PSH
    packet labeled `TLSv1.2` -- voted `-` 2-to-1 under a plain
    majority vote, even though the flow's application protocol WAS
    genuinely identified). This mirrors how Zeek's own `service` field
    actually works: once a connection's protocol is confidently
    detected on ANY packet, that identification stands for the whole
    connection, not a packet-count majority. Ties among multiple
    distinct real identifications (broken alphabetically-first, same
    `pandas.Series.mode()` convention `rt_iot_features.build_windows`
    uses) are a genuine edge case (e.g. a flow tshark flags with two
    different upper-layer names); `-` is used only when NOTHING in the
    flow was ever confidently identified."""
    mapped = pd.Series([map_service(p) for p in raw_protocols])
    identified = mapped[mapped != "-"]
    if identified.empty:
        return "-"
    return str(identified.mode().iloc[0])


# ---------------------------------------------------------------------------
# tshark invocation
# ---------------------------------------------------------------------------

TSHARK_FIELDS: tuple[str, ...] = (
    "frame.time_epoch",
    "ip.proto",
    "ip.src",
    "ip.dst",
    "ip.hdr_len",
    "tcp.srcport",
    "tcp.dstport",
    "tcp.len",
    "tcp.hdr_len",
    "tcp.flags.syn",
    "tcp.flags.ack",
    "tcp.flags.fin",
    "tcp.flags.reset",
    "tcp.flags.push",
    "tcp.flags.urg",
    "tcp.window_size_value",
    "udp.srcport",
    "udp.dstport",
    "udp.length",
    "_ws.col.protocol",
)


def _run_tshark_fields(pcap_path: Path) -> pd.DataFrame:
    """Shells out to tshark's own `-T fields` mode (confirmed working
    on PATH -- 2026-09-18 feasibility session) to get one row per
    packet with the raw fields this module needs. Returns a DataFrame
    of raw strings (parsed/typed by `_normalize_packets`); `|` is used
    as the field separator (not `,`) since IPv4 addresses/protocol
    names never contain it, avoiding any CSV-quoting edge case."""
    args = [
        "tshark",
        "-r",
        str(pcap_path),
        "-T",
        "fields",
        "-E",
        "separator=|",
        "-E",
        "header=y",
        "-E",
        "occurrence=f",
    ]
    for field in TSHARK_FIELDS:
        args += ["-e", field]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "tshark not found on PATH -- required for pcap flow extraction "
            "(external binary, not a pip package; see module docstring)."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"tshark failed reading {pcap_path}: {exc.stderr}") from exc

    return pd.read_csv(io.StringIO(proc.stdout), sep="|", dtype=str, keep_default_na=False)


# ---------------------------------------------------------------------------
# Per-packet normalization
# ---------------------------------------------------------------------------


def _normalize_packets(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Filters to IPv4 tcp/udp/icmp packets (module docstring SCOPE)
    and computes the per-packet derived fields flow aggregation
    needs: direction-neutral endpoints, payload length (tshark's
    `tcp.len` / `udp.length - 8` -- confirmed payload-only, not
    full-frame length, in the feasibility session), header length,
    TCP flags/window, and the raw protocol string for service
    mapping."""

    def _num(col: str) -> pd.Series:
        return pd.to_numeric(raw_df[col], errors="coerce")

    ip_proto = _num("ip.proto")
    keep = ip_proto.isin(PROTO_NUM_TO_NAME.keys())
    df = raw_df.loc[keep].copy()
    ip_proto = ip_proto.loc[keep]
    proto_name = ip_proto.map(PROTO_NUM_TO_NAME)

    time = pd.to_numeric(df["frame.time_epoch"], errors="coerce")
    ip_hdr_len = pd.to_numeric(df["ip.hdr_len"], errors="coerce").fillna(20.0)

    is_tcp = (proto_name == "tcp").to_numpy()
    is_udp = (proto_name == "udp").to_numpy()

    tcp_srcport = pd.to_numeric(df["tcp.srcport"], errors="coerce")
    tcp_dstport = pd.to_numeric(df["tcp.dstport"], errors="coerce")
    udp_srcport = pd.to_numeric(df["udp.srcport"], errors="coerce")
    udp_dstport = pd.to_numeric(df["udp.dstport"], errors="coerce")

    src_port = np.where(is_tcp, tcp_srcport, np.where(is_udp, udp_srcport, -1.0))
    dst_port = np.where(is_tcp, tcp_dstport, np.where(is_udp, udp_dstport, -1.0))
    src_port = pd.Series(src_port, index=df.index).fillna(-1.0).astype(int)
    dst_port = pd.Series(dst_port, index=df.index).fillna(-1.0).astype(int)

    tcp_len = pd.to_numeric(df["tcp.len"], errors="coerce").fillna(0.0)
    tcp_hdr_len = pd.to_numeric(df["tcp.hdr_len"], errors="coerce").fillna(20.0)
    udp_length = pd.to_numeric(df["udp.length"], errors="coerce").fillna(8.0)
    udp_payload = (udp_length - 8.0).clip(lower=0.0)

    payload_len = np.where(is_tcp, tcp_len, np.where(is_udp, udp_payload, 0.0))
    # UDP's header is a fixed 8 bytes; ICMP echo request/reply's is also a
    # fixed 8 bytes (type+code+checksum+id+seq) -- both fall through to the
    # same "+8.0" branch below, a real coincidence rather than an ICMP
    # special-case being silently skipped.
    header_len = np.where(is_tcp, ip_hdr_len + tcp_hdr_len, ip_hdr_len + 8.0)

    window = pd.to_numeric(df["tcp.window_size_value"], errors="coerce").fillna(0.0)
    window = np.where(is_tcp, window, 0.0)

    def _flag(col: str) -> np.ndarray:
        return df[col].astype(str).str.strip().str.lower().eq("true").to_numpy() & is_tcp

    out = pd.DataFrame(
        {
            "time": time.to_numpy(),
            "proto_name": proto_name.to_numpy(),
            "src_ip": df["ip.src"].to_numpy(),
            "dst_ip": df["ip.dst"].to_numpy(),
            "src_port": src_port.to_numpy(),
            "dst_port": dst_port.to_numpy(),
            "payload_len": payload_len,
            "header_len": header_len,
            "window": window,
            "syn": _flag("tcp.flags.syn"),
            "ack": _flag("tcp.flags.ack"),
            "fin": _flag("tcp.flags.fin"),
            "rst": _flag("tcp.flags.reset"),
            "psh": _flag("tcp.flags.push"),
            "urg": _flag("tcp.flags.urg"),
            "protocol": df["_ws.col.protocol"].to_numpy(),
        }
    )
    return out.dropna(subset=["time"]).reset_index(drop=True)


def _flow_key(row: pd.Series) -> str:
    """Direction-neutral flow identity: proto + the two (host, port)
    endpoints in a canonical (sorted) order, so both directions of a
    bidirectional flow share one key. ICMP has no ports, so its key is
    just the host pair."""
    proto = row["proto_name"]
    if proto == "icmp":
        a, b = sorted([row["src_ip"], row["dst_ip"]])
        return f"icmp|{a}|{b}"
    a, b = sorted([f"{row['src_ip']}:{row['src_port']}", f"{row['dst_ip']}:{row['dst_port']}"])
    return f"{proto}|{a}|{b}"


# ---------------------------------------------------------------------------
# Flow-instance splitting (module docstring item 1)
# ---------------------------------------------------------------------------


def _split_tcp_flow_instances(times: np.ndarray, syn: np.ndarray, fin: np.ndarray, rst: np.ndarray) -> np.ndarray:
    """One flow-instance id per packet (already time-sorted) for a
    single TCP 5-tuple's packets. A new instance starts when either:
    (a) the inactivity gap exceeds `INACTIVITY_TIMEOUT_TCP_SEC`, or
    (b) the previous instance was already closed (a real FIN or RST
    was seen) and this packet is a fresh SYN -- mirroring Zeek's own
    real connection-teardown-then-new-connection behavior, not just
    its idle-timeout backstop."""
    n = len(times)
    instances = np.zeros(n, dtype=int)
    current = 0
    closed = bool(fin[0] or rst[0])
    for i in range(1, n):
        gap = times[i] - times[i - 1]
        new_flow = gap > INACTIVITY_TIMEOUT_TCP_SEC or (closed and bool(syn[i]))
        if new_flow:
            current += 1
            closed = False
        if fin[i] or rst[i]:
            closed = True
        instances[i] = current
    return instances


def _split_by_inactivity(times: np.ndarray, timeout: float) -> np.ndarray:
    """UDP/ICMP flow splitting: purely the inactivity backstop (no
    FIN/RST concept for these protocols)."""
    n = len(times)
    instances = np.zeros(n, dtype=int)
    current = 0
    for i in range(1, n):
        if times[i] - times[i - 1] > timeout:
            current += 1
        instances[i] = current
    return instances


def _active_idle_periods(times: np.ndarray) -> tuple[list[float], list[float]]:
    """Splits one flow's sorted packet timestamps into alternating
    active-burst / idle-gap periods using `ACTIVE_IDLE_SPLIT_SEC`
    (module docstring item 2): a gap between consecutive packets
    exceeding the threshold ends the current active burst (its
    duration recorded) and starts an idle period (the gap's duration
    recorded); a gap at or below the threshold keeps extending the
    current burst. Returns `([], [])` for a single-packet flow (no gaps
    exist to split on)."""
    if len(times) < 2:
        return [], []
    active_periods: list[float] = []
    idle_periods: list[float] = []
    burst_start = times[0]
    for i in range(1, len(times)):
        gap = times[i] - times[i - 1]
        if gap > ACTIVE_IDLE_SPLIT_SEC:
            active_periods.append(float(times[i - 1] - burst_start))
            idle_periods.append(float(gap))
            burst_start = times[i]
    active_periods.append(float(times[-1] - burst_start))
    return active_periods, idle_periods


# ---------------------------------------------------------------------------
# Per-flow aggregation into RT-IoT2022-shaped raw columns
# ---------------------------------------------------------------------------


def _aggregate_flow(group: pd.DataFrame, proto_name: str, is_fwd: np.ndarray) -> dict:
    """Turns one flow's (time-sorted) packet rows into one raw row
    shaped exactly like `rt_iot_features.REQUIRED_RAW_COLUMNS`. See
    module docstring for the unit conventions (`flow_duration` in
    seconds; `flow_iat.*`/`active.avg`/`idle.avg` in microseconds) and
    the direction convention (`is_fwd` = same (host, port) as this
    flow's first packet)."""
    times = group["time"].to_numpy(dtype=float)
    payload = group["payload_len"].to_numpy(dtype=float)
    header = group["header_len"].to_numpy(dtype=float)
    fwd, bwd = is_fwd, ~is_fwd

    n_pkts = len(group)
    fwd_pkts_tot = int(fwd.sum())
    bwd_pkts_tot = int(bwd.sum())
    duration = float(times[-1] - times[0]) if n_pkts > 1 else 0.0

    fwd_pkts_per_sec = fwd_pkts_tot / duration if duration > 0 else 0.0
    bwd_pkts_per_sec = bwd_pkts_tot / duration if duration > 0 else 0.0
    flow_pkts_per_sec = n_pkts / duration if duration > 0 else 0.0
    payload_bytes_per_second = float(payload.sum()) / duration if duration > 0 else 0.0

    fwd_data_pkts_tot = int((payload[fwd] > 0).sum())
    bwd_data_pkts_tot = int((payload[bwd] > 0).sum())

    if n_pkts > 1:
        iats = np.diff(times)
        flow_iat_avg = float(iats.mean()) * MICROSECONDS_PER_SECOND
        flow_iat_std = float(iats.std(ddof=0)) * MICROSECONDS_PER_SECOND
    else:
        flow_iat_avg = 0.0
        flow_iat_std = 0.0

    active_periods, idle_periods = _active_idle_periods(times)
    active_avg = float(np.mean(active_periods)) * MICROSECONDS_PER_SECOND if active_periods else 0.0
    idle_avg = float(np.mean(idle_periods)) * MICROSECONDS_PER_SECOND if idle_periods else 0.0

    if proto_name == "tcp":
        window = group["window"].to_numpy(dtype=float)
        syn_arr = group["syn"].to_numpy()
        rst_arr = group["rst"].to_numpy()
        fin_arr = group["fin"].to_numpy()
        ack_arr = group["ack"].to_numpy()
        psh_arr = group["psh"].to_numpy()
        urg_arr = group["urg"].to_numpy()

        fwd_init_window_size = float(window[fwd][0]) if fwd.any() else 0.0
        bwd_init_window_size = float(window[bwd][0]) if bwd.any() else 0.0
        flow_syn_flag_count = int(syn_arr.sum())
        flow_rst_flag_count = int(rst_arr.sum())
        flow_fin_flag_count = int(fin_arr.sum())
        flow_ack_flag_count = int(ack_arr.sum())
        fwd_psh_flag_count = int(psh_arr[fwd].sum())
        fwd_urg_flag_count = int(urg_arr[fwd].sum())
    else:
        fwd_init_window_size = 0.0
        bwd_init_window_size = 0.0
        flow_syn_flag_count = flow_rst_flag_count = flow_fin_flag_count = flow_ack_flag_count = 0
        fwd_psh_flag_count = fwd_urg_flag_count = 0

    down_up_ratio = (bwd_pkts_tot / fwd_pkts_tot) if fwd_pkts_tot > 0 else 0.0
    service = _majority_service(group["protocol"].tolist())

    return {
        "fwd_pkts_per_sec": fwd_pkts_per_sec,
        "bwd_pkts_per_sec": bwd_pkts_per_sec,
        "flow_pkts_per_sec": flow_pkts_per_sec,
        "payload_bytes_per_second": payload_bytes_per_second,
        "fwd_pkts_tot": fwd_pkts_tot,
        "bwd_pkts_tot": bwd_pkts_tot,
        "fwd_data_pkts_tot": fwd_data_pkts_tot,
        "bwd_data_pkts_tot": bwd_data_pkts_tot,
        "flow_pkts_payload.avg": float(payload.mean()),
        "flow_pkts_payload.std": float(payload.std(ddof=0)),
        "flow_duration": duration,
        "flow_iat.avg": flow_iat_avg,
        "flow_iat.std": flow_iat_std,
        "active.avg": active_avg,
        "idle.avg": idle_avg,
        "fwd_header_size_tot": float(header[fwd].sum()),
        "bwd_header_size_tot": float(header[bwd].sum()),
        "fwd_init_window_size": fwd_init_window_size,
        "bwd_init_window_size": bwd_init_window_size,
        "flow_SYN_flag_count": flow_syn_flag_count,
        "flow_RST_flag_count": flow_rst_flag_count,
        "flow_FIN_flag_count": flow_fin_flag_count,
        "flow_ACK_flag_count": flow_ack_flag_count,
        "fwd_PSH_flag_count": fwd_psh_flag_count,
        "fwd_URG_flag_count": fwd_urg_flag_count,
        "down_up_ratio": down_up_ratio,
        "proto": proto_name,
        "service": service,
    }


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PCAPExtractionResult:
    """Real extraction output + honest bookkeeping (Step 4: report,
    don't silently smooth over, what was skipped/how flows split)."""

    raw_flow_df: pd.DataFrame  # REQUIRED_RAW_COLUMNS-shaped, one row per flow, chronological by flow start
    features: np.ndarray  # extract_flow_features(raw_flow_df) output, shape (n_flows, FEATURE_DIM)
    n_packets_total: int
    n_packets_used: int  # IPv4 tcp/udp/icmp packets that made it into a flow
    n_packets_skipped_non_ipv4: int  # ARP/IPv6/other -- explicitly out of scope, see module docstring
    n_flows: int
    flow_packet_counts: tuple[int, ...]
    service_counts: dict[str, int]


def extract_pcap_flow_features(pcap_path: Path | str) -> PCAPExtractionResult:
    """Real, end-to-end pcap -> RT-IoT2022-shaped flow-feature
    pipeline: tshark packet fields -> per-packet normalization ->
    flow-instance splitting (module docstring item 1) -> per-flow
    aggregation (item 3's unit convention) -> `extract_flow_features`
    (imported, unmodified from `rt_iot_features.py`)."""
    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise FileNotFoundError(f"{pcap_path} not found")

    raw_df = _run_tshark_fields(pcap_path)
    n_packets_total = len(raw_df)
    pkt_df = _normalize_packets(raw_df)
    n_packets_skipped_non_ipv4 = n_packets_total - len(pkt_df)

    if pkt_df.empty:
        raise ValueError(
            f"{pcap_path}: zero IPv4 tcp/udp/icmp packets found after filtering "
            f"({n_packets_total} raw packets, {n_packets_skipped_non_ipv4} skipped as "
            f"non-IPv4/ARP/other -- out of this extractor's documented scope). "
            f"Cannot extract any flow from this file."
        )

    pkt_df["flow_key"] = pkt_df.apply(_flow_key, axis=1)
    pkt_df = pkt_df.sort_values(["flow_key", "time"], kind="stable").reset_index(drop=True)

    flow_instance = np.zeros(len(pkt_df), dtype=int)
    for _key, group in pkt_df.groupby("flow_key", sort=False):
        idx = group.index.to_numpy()
        times = group["time"].to_numpy(dtype=float)
        proto_name = group["proto_name"].iloc[0]
        if proto_name == "tcp":
            inst = _split_tcp_flow_instances(
                times, group["syn"].to_numpy(), group["fin"].to_numpy(), group["rst"].to_numpy()
            )
        else:
            timeout = INACTIVITY_TIMEOUT_UDP_SEC if proto_name == "udp" else INACTIVITY_TIMEOUT_ICMP_SEC
            inst = _split_by_inactivity(times, timeout)
        flow_instance[idx] = inst
    pkt_df["flow_instance"] = flow_instance

    grouped = list(pkt_df.groupby(["flow_key", "flow_instance"], sort=False))
    grouped.sort(key=lambda kv: kv[1]["time"].min())  # chronological by flow start, matching CSV row-order convention

    rows = []
    flow_packet_counts = []
    for _key, group in grouped:
        group = group.sort_values("time", kind="stable").reset_index(drop=True)
        proto_name = group["proto_name"].iloc[0]
        anchor_src_ip = group["src_ip"].iloc[0]
        anchor_src_port = group["src_port"].iloc[0]
        is_fwd = ((group["src_ip"] == anchor_src_ip) & (group["src_port"] == anchor_src_port)).to_numpy()
        rows.append(_aggregate_flow(group, proto_name, is_fwd))
        flow_packet_counts.append(len(group))

    raw_flow_df = pd.DataFrame(rows, columns=list(REQUIRED_RAW_COLUMNS))
    features = extract_flow_features(raw_flow_df)
    assert features.shape == (len(rows), FEATURE_DIM)

    return PCAPExtractionResult(
        raw_flow_df=raw_flow_df,
        features=features,
        n_packets_total=n_packets_total,
        n_packets_used=int(len(pkt_df)),
        n_packets_skipped_non_ipv4=int(n_packets_skipped_non_ipv4),
        n_flows=len(rows),
        flow_packet_counts=tuple(flow_packet_counts),
        service_counts=dict(raw_flow_df["service"].value_counts()),
    )
