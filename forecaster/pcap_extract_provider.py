"""
forecaster/pcap_extract_provider.py

Live-.pcap `ForecastProvider` (v2 Phase 1 -- docs/v2_gameplan.md;
built on `forecaster.pcap_flow_features`'s tshark-backed real per-flow
extractor). Extracts real per-flow features from an arbitrary `.pcap`
file and replays them through the trained LSTM threat forecaster, one
flow per `update()` tick.

`PCAPExtractProvider` SUBCLASSES `forecaster.rt_iot_replay
.RTIoT2022ReplayProvider` directly rather than re-implementing its
window-buffer warm-start/replay mechanics: that class's own docstring
documents a real cold-start false-positive bug it fixes (an empty/
zero-padded buffer spiking `threat_score` to ~0.92-0.97 regardless of
true class at episode start) -- reusing it verbatim avoids
re-introducing that bug in a parallel copy. Only the data SOURCE
differs: real held-out RT-IoT2022 CSV rows there, real tshark-extracted
pcap flow rows here. Both are raw (unscaled) `FEATURE_DIM`-wide rows;
`RTIoT2022ReplayProvider.__init__`'s existing shape guard already
refuses a mismatched width rather than silently truncating/padding it.

Injected via the same `forecast_provider_factory` seam
(`env/environment.py` design decision 17) `RTIoT2022ReplayProvider`
and `attack/attacking_provider.py` already use -- zero `env/`,
`agents/`, reward, action-space, floor-table, or checkpoint changes
(this session's explicit scope boundary).
"""

from __future__ import annotations

from pathlib import Path

from forecaster.pcap_flow_features import PCAPExtractionResult, extract_pcap_flow_features
from forecaster.rt_iot_replay import RTIoT2022ReplayProvider


class PCAPExtractProvider(RTIoT2022ReplayProvider):
    """`RTIoT2022ReplayProvider` sourced from a live-extracted `.pcap`
    file's real per-flow features instead of held-out RT-IoT2022 CSV
    rows. Runtime mechanics (window-buffer warm start, per-tick
    replay, checkpoint-scaler application via the inherited
    `_prepare_step_features`) are identical by inheritance -- only the
    data source differs.

    Real, honest caveats (see PROGRESS.md/SESSION_LOG.md, this
    session -- carried forward from `docs/v2_gameplan.md`'s standing
    list):
      - A live/uploaded `.pcap`'s flows are THIS extractor's own
        tshark-parsing + flow-timeout + service-mapping policy
        (`forecaster/pcap_flow_features.py`'s module docstring),
        structurally comparable to RT-IoT2022's own Zeek-Flowmeter
        pipeline (same column definitions, same units -- verified
        empirically, see that module's docstring item 3) but not
        byte-identical to what actually produced the training CSV.
      - Live/uploaded traffic is real distribution shift vs.
        RT-IoT2022's IoT-device training traffic (protocol mix,
        content), independent of extraction fidelity.
      - If the pcap contains zero real IPv4 flows in its held-out
        window (e.g. a very short or entirely non-IPv4 capture),
        `extract_pcap_flow_features` raises rather than fabricating
        placeholder flows -- this constructor lets that propagate.
    """

    def __init__(
        self,
        checkpoint_path: str,
        pcap_path: str | Path,
        start_offset: int = 0,
        decision_threshold: float | None = None,
    ) -> None:
        self.extraction: PCAPExtractionResult = extract_pcap_flow_features(pcap_path)
        super().__init__(
            checkpoint_path,
            self.extraction.features,
            start_offset=start_offset,
            decision_threshold=decision_threshold,
        )
