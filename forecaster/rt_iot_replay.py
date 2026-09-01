"""
forecaster/rt_iot_replay.py

RT-IoT2022 held-out REPLAY provider (PLAN2.md Addition E follow-up,
"RT-IoT2022 replay wiring" session). Closes the gap the Addition E
Phases 2-4 session left explicit in `LSTMForecastProvider`'s own class
docstring: that session's live sanity check fed the trained LSTM
`env/environment.py::_threat_features_placeholder`'s synthetic 2-dim
`[qber, load]` pair, zero-padded up to `feature_dim=39` -- it proved the
WIRING works, not that the model consumes real, trained-on-shaped input.

This module replays REAL, already-extracted RT-IoT2022 window features
(held out from training, never seen during Addition E's training run)
one raw flow-row at a time into the trained LSTM, driving live episodes
with genuine network-derived features instead of the placeholder.

SCOPE (this session's instruction, not a project Hard Rule): this is
replay of a captured/extracted file, not live packet capture off a
real interface -- Hard Rule 11 (PLAN2.md §5.5) stays in force
unchanged. `env/environment.py`, `env/masking.py`, the reward
function, the action space, and the DQN are untouched; `RTIoT2022ReplayProvider`
is injected purely through the existing `forecast_provider_factory`
seam (design decision 17), the same seam Addition E Phase 4a and
`attack/attacking_provider.py` already use.

**Held-out disjointness, verified directly, not assumed**:
`held_out_replay_rows()` below reuses `forecaster.train_threat_lstm`'s
own `leakage_free_split()` (the exact split `train_threat_lstm.py`
used to produce the checkpoint's val metrics) and re-runs
`assert_no_row_overlap()` against it before returning anything -- the
replay rows this module hands out are the SAME val windows' raw rows
that produced the checkpoint's real held-out val metrics (benign
recall 1.000, attack recall 0.994), not a freshly-invented split that
could silently leak train rows into "held-out" replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from env.contracts import ForecastObservation
from forecaster.model import LSTMForecastProvider
from forecaster.rt_iot_features import (
    FEATURE_DIM,
    extract_flow_features,
    load_rt_iot2022,
)
from forecaster.train_threat_lstm import (
    DEFAULT_NPZ,
    assert_no_row_overlap,
    leakage_free_split,
)

DEFAULT_CSV = Path("data/raw/rt_iot2022/RT_IOT2022.csv")


@dataclass(frozen=True)
class HeldOutReplayRows:
    """Real, held-out (never used in training) RT-IoT2022 raw-row
    feature vectors, RAW/unscaled (`RTIoT2022ReplayProvider` applies
    the checkpoint's own persisted scaler per-row, exactly like
    `LSTMForecastProvider._prepare_step_features` already does for a
    live step -- scaling here would double-apply it).

    `benign_features`/`attack_features` are each in real chronological
    raw-CSV-row order (RT-IoT2022 ships no interleaving between labels
    -- see `train_threat_lstm.py`'s module docstring -- so within a
    class there is exactly one real contiguous held-out span, not a
    shuffled sample)."""

    benign_features: np.ndarray  # (n_benign_rows, FEATURE_DIM), raw/unscaled
    attack_features: np.ndarray  # (n_attack_rows, FEATURE_DIM), raw/unscaled
    benign_row_range: tuple[int, int]  # [start, end) into the raw CSV
    attack_row_range: tuple[int, int]
    n_train_windows: int
    n_val_windows: int
    window: int
    stride: int


def _contiguous_index_runs(sorted_idx: np.ndarray) -> list[tuple[int, int]]:
    """`(start, end_inclusive)` for each maximal run of consecutive
    integers in a sorted index array. `leakage_free_split` assigns each
    binary-label run's val slice as one contiguous window-index range
    (`range(val_start, end + 1)`, see its docstring), so this recovers
    those per-run val spans from the flat `val_idx` array it returns."""
    if len(sorted_idx) == 0:
        return []
    runs = []
    start = int(sorted_idx[0])
    prev = start
    for v in sorted_idx[1:]:
        v = int(v)
        if v != prev + 1:
            runs.append((start, prev))
            start = v
        prev = v
    runs.append((start, prev))
    return runs


def held_out_replay_rows(
    npz_path: Path | str = DEFAULT_NPZ,
    csv_path: Path | str = DEFAULT_CSV,
) -> HeldOutReplayRows:
    """Real held-out replay rows, reusing (not re-deriving)
    `train_threat_lstm.leakage_free_split` + `assert_no_row_overlap` --
    see module docstring."""
    data = np.load(npz_path, allow_pickle=True)
    y_binary = data["y_binary"].astype(np.int64)
    window = int(data["window"])
    stride = int(data["stride"])

    train_idx, val_idx = leakage_free_split(y_binary, window=window, stride=stride)
    assert_no_row_overlap(train_idx, val_idx, window=window, stride=stride)

    df = load_rt_iot2022(csv_path)

    spans_by_label: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
    for start, end in _contiguous_index_runs(np.sort(val_idx)):
        label = int(y_binary[start])
        row_start = start * stride
        row_end = end * stride + window  # exclusive
        spans_by_label[label].append((row_start, row_end))

    def extract_spans(spans: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
        if not spans:
            return np.empty((0, FEATURE_DIM)), (0, 0)
        chunks = [extract_flow_features(df.iloc[s:e]) for s, e in spans]
        combined_range = (spans[0][0], spans[-1][1])
        return np.concatenate(chunks, axis=0), combined_range

    benign_features, benign_range = extract_spans(spans_by_label[0])
    attack_features, attack_range = extract_spans(spans_by_label[1])

    return HeldOutReplayRows(
        benign_features=benign_features,
        attack_features=attack_features,
        benign_row_range=benign_range,
        attack_row_range=attack_range,
        n_train_windows=int(len(train_idx)),
        n_val_windows=int(len(val_idx)),
        window=window,
        stride=stride,
    )


class RTIoT2022ReplayProvider(LSTMForecastProvider):
    """`LSTMForecastProvider` with its per-tick threat-feature source
    replaced by a real, held-out RT-IoT2022 raw-row replay stream
    instead of whatever `env/environment.py` passes as
    `observation["threat_features"]` (still the synthetic `[qber,
    load]` placeholder -- untouched, per this session's scope
    boundary). Pool telemetry (`qber`/`skr`/`pool_fill`) IS real (from
    `pool_sim`), so that half of `observation` is still consumed
    normally, via the inherited `_pool_ewma` -- see Addition E's own
    prior-session decision that pool forecasting stays on the EWMA/
    synthetic pool trace, unaffected by this replay.

    One real raw row is consumed per `update()` call (matching
    `MovingAverageForecaster`/`LSTMForecastProvider`'s own per-internal-
    tick cadence -- `env/environment.py` calls `update()` once per
    internal tick, not once per decision). `_prepare_step_features`
    (inherited, unmodified) applies the checkpoint's persisted scaler
    to each real row exactly as it would to a live step's features;
    since `replay_features` is asserted to already be exactly
    `feature_dim` wide, no zero-padding/truncation occurs -- unlike the
    Phase 4a plumbing check, which fed a 2-dim placeholder that
    THIS class's shape guard (below) exists specifically to refuse.
    Once `self._window` (64) real rows have been consumed, the
    provider's buffer holds a genuine 64-row RT-IoT2022 window,
    structurally identical to what `train_threat_lstm.py` trained on
    (real raw-row order preserved; a sliding stride-1 buffer rather
    than the offline pipeline's stride-32 windowing, which does not
    change what the LSTM was trained to consume -- 64 consecutive real
    rows in order).

    `start_offset` lets different callers (e.g. different episode
    seeds in a replay campaign) start at a different real point in the
    same held-out span rather than always replaying the identical
    first 64 rows -- still real data, just a different real slice of
    it. Wraps (via modulo) if a caller consumes more rows than the
    span contains; flagged here rather than silently extending with
    fabricated rows."""

    def __init__(
        self,
        checkpoint_path: str,
        replay_features: np.ndarray,
        start_offset: int = 0,
        decision_threshold: float | None = None,
    ) -> None:
        super().__init__(checkpoint_path, decision_threshold)
        replay_features = np.asarray(replay_features, dtype=np.float64)
        if replay_features.ndim != 2 or replay_features.shape[1] != self._feature_dim:
            raise ValueError(
                f"replay_features must have shape (n_rows, {self._feature_dim}) to match "
                f"this checkpoint's trained feature_dim -- got {replay_features.shape}. "
                f"Refusing to silently truncate/zero-pad real replay data to fit."
            )
        if len(replay_features) == 0:
            raise ValueError("replay_features must contain at least one real row")

        self._replay_features = replay_features
        self._replay_pos = start_offset % len(replay_features)

    def update(self, observation: ForecastObservation) -> None:
        self._pool_ewma.update(observation)  # real pool telemetry, unaffected by replay
        row = self._replay_features[self._replay_pos % len(self._replay_features)]
        self._replay_pos += 1
        self._buffer.append(self._prepare_step_features(row))
