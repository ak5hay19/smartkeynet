"""
dashboard/upload_demo.py

v2 Phase 2 (docs/v2_gameplan.md) -- the real upload-and-run demo built
on top of Phase 1's tested, unmodified extractor/provider
(`forecaster/pcap_flow_features.py`, `forecaster/pcap_extract_provider.py`
-- neither is touched by this module, per this session's scope
boundary). A person uploads a real `.pcap`; this module extracts real
per-flow features from it, replays them through the real trained LSTM
threat forecaster via `PCAPExtractProvider`, and steps a real masked
`DQNAgent` checkpoint through a live `SmartKeyNetEnv` episode -- every
decision shown is computed at request time, not pre-rendered (unlike
every other `dashboard/samples/*.html` panel, which this module does
NOT touch or reuse the pattern of beyond `dashboard/explain.py`'s
existing `DecisionTrace` assembler).

**Demo agent/config choice, documented here (this session's decision,
not prescribed elsewhere):** `configs/default.yaml` (S1, real
`use_foresight: ewma` so `flatten_state`'s forecast-dim branch is
active -- the same config `checkpoints/dqn_s1.pt` was retrained under,
Addition E Phase 4b) + `checkpoints/dqn_s1.pt` (the canonical S1 masked
DQN checkpoint -- same one `dashboard/render_living_system_demo.py`'s
"Living System -- S1, trained DQN" card and
`dashboard/render_migration_wave_demo.py` already reload) +
`checkpoints/lstm_threat_forecaster.pt` (the LSTM forecaster
`PCAPExtractProvider` composes with by inheritance from
`RTIoT2022ReplayProvider`/`LSTMForecastProvider` -- the same checkpoint
every other real-forecaster dashboard driver already reloads). This
mirrors the established `_load_masked_policy` reload pattern exactly
(reconstruct `DQNAgent` from the config's own `state_dim`/
`has_forecast`, load checkpoint weights, wrap in `GreedyDQNPolicy`,
eval-only -- no training call anywhere in this module).

**Per-decision "true service label" provenance (a design decision made
this session, documented since it isn't obvious):** the uploaded pcap's
real extracted flows drive the LSTM's *threat* signal via
`PCAPExtractProvider.update()` (one real flow row consumed per internal
env tick -- inherited, unmodified, from `RTIoT2022ReplayProvider`), but
the *request* being decided (tenant/service/sensitivity_class) comes
from `env/environment.py`'s separate, synthetic `random_request_generator`
stream -- these are two structurally independent channels
(`env/masking.py`'s Hard Rule 2 requires the pool-forecast head and the
threat head never cross-contaminate; the request stream is a third,
independent channel again). To show the real pcap evidence behind each
decision's `threat_score` without conflating it with the synthetic
request, each decision row reports BOTH, clearly separated: the
synthetic demo tenant/service (labelled as such) and the real pcap flow
whose features were most recently consumed into the LSTM's buffer
before this decision (`env._forecaster._replay_pos - 1`, the same class
of private-attribute reach `dashboard/explain.py::explain_decision_from_env`
and `render_living_system_demo.py` already establish as precedent, for
the same reason: the public Gym API doesn't surface this).

Per-decision reasoning (`floor`, `legal actions`, `chosen action`, and
*why* it was cheapest-legal) is `dashboard/explain.py`'s real
`DecisionTrace` (`explain_decision_from_env`), reused verbatim -- not
reimplemented -- so this panel is structurally unable to drift from the
real masking/cost layer, same guarantee Hard Rule 10 gives the Explain
Decision panel.

Honesty caveat (Step 3 of this session's brief): domain mismatch
between an arbitrary uploaded pcap and RT-IoT2022's IoT-device training
distribution is real (Phase 1/the 2026-09-18 feasibility session both
confirmed modern protocol mix, e.g. QUIC, has no RT-IoT2022-era
analog) -- `UPLOAD_CAVEAT_TEXT` below is rendered directly in the
result page, not just documented here.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agents.dqn import DQNAgent, flatten_state, load_dqn_config
from dashboard.explain import DecisionTrace, explain_decision_from_env
from env.contracts import ForecastProvider
from env.environment import SmartKeyNetEnv
from experiments.train import GreedyDQNPolicy, load_full_config
from forecaster.pcap_extract_provider import PCAPExtractProvider
from forecaster.pcap_flow_features import PCAPExtractionResult, extract_pcap_flow_features

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"
_DQN_CHECKPOINT_PATH = _REPO_ROOT / "checkpoints" / "dqn_s1.pt"
_LSTM_CHECKPOINT_PATH = _REPO_ROOT / "checkpoints" / "lstm_threat_forecaster.pt"

DEFAULT_MAX_DECISIONS = 30
"""Matches Phase 1's own fixture-pcap validation episode length
(`tests/test_pcap_extract_provider.py`) -- a demo-sized episode, real
end to end, small enough to read as a table."""

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
"""A plain, stated limit for this local demo server, not a tuned
production value -- large enough for any reasonable short demo capture,
small enough that a mistaken upload doesn't hang tshark for a
rehearsal-time demo."""

UPLOAD_CAVEAT_TEXT = (
    "Features extracted from uploaded traffic in real time. Protocol mix "
    "and traffic patterns may differ from the network security context "
    "the model was trained on (RT-IoT2022, IoT-device traffic, ~2021-2022) "
    "-- treat these results as a live demonstration of the real pipeline, "
    "not a validated detection claim on arbitrary traffic."
)


class InvalidPcapError(Exception):
    """A real, user-facing failure -- an uploaded file wasn't a valid
    pcap, produced zero extractable flows, or otherwise couldn't be
    processed. Never caught silently into a fabricated result."""


@dataclass(frozen=True)
class DecisionRow:
    decision_num: int
    demo_tenant: str
    demo_service: str
    demo_sensitivity_class: str
    pcap_flow_index: int
    pcap_flow_proto: str
    pcap_flow_service: str
    threat_score: float
    posture: str
    floor: str
    legal_actions: tuple[str, ...]
    chosen_action: str
    final_text: str


@dataclass(frozen=True)
class UploadDemoResult:
    pcap_filename: str
    n_packets_total: int
    n_packets_used: int
    n_packets_skipped_non_ipv4: int
    n_flows: int
    service_counts: dict[str, int]
    checkpoint_name: str
    config_name: str
    decisions: tuple[DecisionRow, ...]


def _load_masked_policy(
    config: dict,
    checkpoint_path: Path,
    forecast_provider_factory: Callable[[int | None], ForecastProvider],
) -> GreedyDQNPolicy:
    """Reconstruct the real `DQNAgent` + load real checkpoint weights.
    Mirrors `dashboard/render_living_system_demo.py::_load_masked_policy`
    exactly (same established reload pattern) -- eval only, no training
    call."""
    has_forecast = config.get("use_foresight", "off") != "off"
    probe_env = SmartKeyNetEnv({**config, "seed": 0}, forecast_provider_factory=forecast_provider_factory)
    state, _info = probe_env.reset(seed=0)
    state_dim = flatten_state(state, has_forecast).shape[0]
    agent = DQNAgent(state_dim=state_dim, has_forecast=has_forecast, config=load_dqn_config())
    agent.load(str(checkpoint_path))
    return GreedyDQNPolicy(agent)


def run_pcap_upload_demo(
    pcap_bytes: bytes,
    filename: str,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
    seed: int = 0,
) -> UploadDemoResult:
    """Real, end-to-end: uploaded pcap bytes -> real tshark-backed
    extraction -> real `PCAPExtractProvider` -> a real, live-stepped
    `SmartKeyNetEnv` episode under a real trained masked-DQN checkpoint.

    Raises `InvalidPcapError` on a real, honest failure (not a valid
    pcap tshark can read, or zero extractable IPv4 tcp/udp/icmp flows)
    -- the caller is expected to catch this and show an honest error,
    never to fall back to fabricated/placeholder output."""
    if len(pcap_bytes) == 0:
        raise InvalidPcapError(f"{filename!r} is empty -- nothing to extract.")
    if len(pcap_bytes) > MAX_UPLOAD_BYTES:
        raise InvalidPcapError(
            f"{filename!r} is {len(pcap_bytes)} bytes, over this demo's "
            f"{MAX_UPLOAD_BYTES} byte limit. Try a shorter capture."
        )

    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
        tmp.write(pcap_bytes)
        tmp_path = Path(tmp.name)

    try:
        try:
            extraction: PCAPExtractionResult = extract_pcap_flow_features(tmp_path)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            # Real, named failure modes from forecaster/pcap_flow_features.py
            # (not a valid/readable pcap; zero IPv4 tcp/udp/icmp packets
            # after filtering) -- re-raised as this module's own honest,
            # user-facing error type rather than silently producing a
            # fabricated result.
            raise InvalidPcapError(f"Could not extract flow features from {filename!r}: {exc}") from exc

        def provider_factory(_episode_seed: int | None) -> ForecastProvider:
            return PCAPExtractProvider(str(_LSTM_CHECKPOINT_PATH), tmp_path)

        config = load_full_config(_CONFIG_PATH)
        config["max_steps"] = max_decisions

        policy = _load_masked_policy(config, _DQN_CHECKPOINT_PATH, provider_factory)
        env = SmartKeyNetEnv(config, forecast_provider_factory=provider_factory)
        state, info = env.reset(seed=seed)

        n_flows = extraction.n_flows
        decisions: list[DecisionRow] = []
        truncated = False
        while not truncated and len(decisions) < max_decisions:
            mask = info["action_mask"]

            # The real pcap flow most recently consumed into the LSTM's
            # buffer before this decision -- see module docstring's
            # "true service label provenance" note.
            replay_pos = env._forecaster._replay_pos  # noqa: SLF001 -- established precedent, see docstring
            flow_idx = (replay_pos - 1) % n_flows
            flow_row = extraction.raw_flow_df.iloc[flow_idx]

            request = env._current_request  # noqa: SLF001 -- same precedent as dashboard/explain.py
            chosen = policy.act(state, mask)
            trace: DecisionTrace = explain_decision_from_env(env, state, chosen)

            decisions.append(
                DecisionRow(
                    decision_num=len(decisions) + 1,
                    demo_tenant=str(request["tenant"]),
                    demo_service=str(request["service"]),
                    demo_sensitivity_class=trace.sensitivity_class.name,
                    pcap_flow_index=int(flow_idx),
                    pcap_flow_proto=str(flow_row["proto"]),
                    pcap_flow_service=str(flow_row["service"]),
                    threat_score=float(trace.threat_score),
                    posture=trace.resolved_posture.name,
                    floor=trace.floor.name,
                    legal_actions=tuple(e.action.name for e in trace.mask if e.legal),
                    chosen_action=trace.chosen_action.name,
                    final_text=trace.final_text,
                )
            )
            state, _reward, _terminated, truncated, info = env.step(chosen)

        return UploadDemoResult(
            pcap_filename=filename,
            n_packets_total=extraction.n_packets_total,
            n_packets_used=extraction.n_packets_used,
            n_packets_skipped_non_ipv4=extraction.n_packets_skipped_non_ipv4,
            n_flows=extraction.n_flows,
            service_counts=dict(extraction.service_counts),
            checkpoint_name="checkpoints/dqn_s1.pt (masked DQN, trained on S1)",
            config_name="configs/default.yaml (S1)",
            decisions=tuple(decisions),
        )
    finally:
        tmp_path.unlink(missing_ok=True)
