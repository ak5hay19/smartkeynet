"""
forecaster/dataset.py

Dataset ingestion for the dual-head forecaster (PLAN.md Addition A;
PLAN2 §6 "Slot-to-source map", §8 Addition D). Owned by Person A
(split.md §1).

Three responsibilities, in dependency order:

1. `extract_flow_features()` -- the **single** feature-extraction
   implementation, with two callers (PLAN2 §8: "one implementation, two
   callers"; Hard Rule 11: "the exact same feature-extraction path used
   for offline RT-IoT2022 training data ... no second, ad hoc pipeline
   for 'live' data"). It turns a sequence of flow records into a fixed
   feature matrix, and it does not know or care whether those records
   came from RT-IoT2022's CSV or from a `.pcap` parsed into flows.
2. `RTIoT2022Dataset` -- loads the real capture, standardizes it, and
   serves benign/attack feature *windows* plus their labels.
3. `build_rollout_dataset()` -- rolls a baseline policy through the real
   environment with a real RT-IoT2022 threat trace injected, recording
   per-step observations, threat labels and *future* pool targets. This
   is what makes a genuinely shared-trunk dual-head model trainable:
   every training step carries both a threat label and a pool target,
   from the same sequence.

Provenance rules this module enforces (PLAN2 §6 "Golden rules"):
  * RT-IoT2022 is the ONE real-network slot. Nothing else is loaded.
  * `rl_experiment_*` / `context_dataset_*` are never read here -- they
    are outputs of the design being critiqued, and training on them
    would be imitation-learning the flawed policy (README anti-pattern).
  * The label distribution is checked on load against the DO-NOT-USE
    degeneracy rule ("near-zero label variety").
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# The shared feature set
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: tuple[str, ...] = (
    # rate features -- PLAN2 §7.1's "packets/sec, bytes/sec"
    "fwd_pkts_per_sec",
    "bwd_pkts_per_sec",
    "flow_pkts_per_sec",
    "payload_bytes_per_second",
    # TCP flag counts -- PLAN2 §7.1's "SYN ratio" and its relatives; these
    # are what separate a SYN flood or a stealth scan from ordinary traffic
    "flow_SYN_flag_count",
    "flow_RST_flag_count",
    "flow_FIN_flag_count",
    "flow_ACK_flag_count",
    # shape of the conversation
    "down_up_ratio",
    "flow_duration",
    "fwd_pkts_tot",
    "bwd_pkts_tot",
    # timing
    "flow_iat.avg",
    "flow_iat.std",
    # payload size
    "fwd_pkts_payload.avg",
    "bwd_pkts_payload.avg",
)
"""The 16 flow features the threat head consumes.

Chosen to be (a) computable identically from an RT-IoT2022 CSV row and
from a flow record reconstructed out of a `.pcap` -- Hard Rule 11's
whole point -- and (b) the quantities PLAN2 §7.1 says the Threat Input
panel should display ("packets/sec, bytes/sec, unique ports touched,
SYN ratio, or equivalent"). Nothing here is a security constant; these
are traffic measurements.
"""

N_FEATURES = len(FEATURE_COLUMNS)

_LOG_SCALED_COLUMNS: frozenset[str] = frozenset(
    {
        "fwd_pkts_per_sec",
        "bwd_pkts_per_sec",
        "flow_pkts_per_sec",
        "payload_bytes_per_second",
        "flow_duration",
        "fwd_pkts_tot",
        "bwd_pkts_tot",
        "flow_iat.avg",
        "flow_iat.std",
        "fwd_pkts_payload.avg",
        "bwd_pkts_payload.avg",
    }
)
"""Columns that span many orders of magnitude get `log1p` before
standardization. Without it a single DOS_SYN_Hping flow at ~10^6
packets/sec dominates the column's mean and standard deviation and
every benign flow standardizes to approximately the same value."""

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

BENIGN_ATTACK_TYPES: frozenset[str] = frozenset(
    {"Thing_Speak", "MQTT_Publish", "Wipro_bulb"}
)
"""RT-IoT2022's normal-operation patterns: a ThingSpeak-LED device, an
MQTT publish workload, and a Wipro smart bulb. Everything else in the
`Attack_type` column is an attack pattern (SYN flood, ARP poisoning,
the NMAP scan family, Slowloris, SSH brute force). Taken from the
dataset's own class names -- not a threshold anyone chose here."""


def is_attack(attack_type: str) -> bool:
    return attack_type not in BENIGN_ATTACK_TYPES


# ---------------------------------------------------------------------------
# 1. The shared feature extractor (one implementation, two callers)
# ---------------------------------------------------------------------------


def extract_flow_features(records: Iterable[dict[str, Any]]) -> np.ndarray:
    """Turn flow records into an `(n_records, N_FEATURES)` float matrix.

    A "flow record" is any mapping with `FEATURE_COLUMNS` keys. Two
    callers exist and must stay the only two (Hard Rule 11):

      * `RTIoT2022Dataset` -- rows of the offline training CSV.
      * the pcap ingestion path (`api/main.py`'s upload/replay modes) --
        flow records reconstructed from packets.

    Missing or unparseable values become 0.0 rather than raising: a
    pcap-derived flow legitimately may not carry every field a Zeek-
    derived CSV row does, and a hole in one feature must not take down
    the whole pipeline at inference time.
    """
    rows: list[list[float]] = []
    for record in records:
        row: list[float] = []
        for column in FEATURE_COLUMNS:
            try:
                value = float(record.get(column, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            if not np.isfinite(value):
                value = 0.0
            if column in _LOG_SCALED_COLUMNS:
                value = float(np.log1p(max(0.0, value)))
            row.append(value)
        rows.append(row)

    if not rows:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


@dataclass(frozen=True)
class FeatureStandardizer:
    """Per-column mean/std, fitted once on the training split and then
    frozen alongside the model.

    Standardization is not cosmetic here: `env/forecast_provider.py`'s
    threat squash is calibrated for standardized input (benign ~ 0), and
    `env/environment.py`'s scenario threat levels are stated in the same
    units. A model shipped without its standardizer would silently
    change what "benign" means.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureStandardizer":
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)  # constant columns pass through unscaled
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return features.astype(np.float32)
        return ((features - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Sequence[float]]) -> "FeatureStandardizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# 2. RT-IoT2022
# ---------------------------------------------------------------------------

_DEFAULT_DATASET_PATHS: tuple[str, ...] = (
    "data/raw/rt_iot2022/RT_IOT2022.csv",
    "data/raw/RT_IOT2022.csv",
)
"""Both locations are checked because the file is gitignored and lands
wherever the operator put it -- PLAN2 §6 documents the dataset as
operator-supplied, and a hard-coded single path just produces a
confusing FileNotFoundError."""

_MIN_MINORITY_CLASS_FRACTION = 0.01
"""DO-NOT-USE degeneracy guard (PLAN2 §6): "any dataset file found to
have near-zero label variety ... must not be used as training data --
verify before loading anything new". Enforced on load rather than
trusted."""


def resolve_dataset_path(path: str | Path | None = None) -> Path:
    """Locate RT-IoT2022, raising a message that says what to do."""
    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"RT-IoT2022 not found at {resolved}")
        return resolved

    repo_root = Path(__file__).resolve().parent.parent
    for candidate in _DEFAULT_DATASET_PATHS:
        resolved = repo_root / candidate
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        "RT-IoT2022 not found. Place RT_IOT2022.csv at one of "
        f"{_DEFAULT_DATASET_PATHS} (it is gitignored -- see data/README.md)."
    )


@dataclass
class RTIoT2022Dataset:
    """Standardized RT-IoT2022 flow features, split benign vs attack.

    Holds the *whole* usable capture in memory as two float32 matrices
    (~123k x 16 floats, well under 10 MB). `max_rows` exists for tests
    and for the dashboard's quick-start path, not for training.
    """

    benign: np.ndarray
    attack: np.ndarray
    standardizer: FeatureStandardizer
    label_counts: dict[str, int]

    @property
    def attack_fraction(self) -> float:
        total = len(self.benign) + len(self.attack)
        return len(self.attack) / total if total else 0.0

    def sample_window(
        self, rng: np.random.Generator, window_size: int, attack: bool
    ) -> np.ndarray:
        """Draw one contiguous `(window_size, N_FEATURES)` window.

        Contiguous, not i.i.d.: the threat head is an LSTM and the
        thing it is meant to learn is the temporal signature of an
        attack (a scan ramping, a flood sustaining), which independent
        row draws would destroy.
        """
        source = self.attack if attack else self.benign
        if len(source) == 0:
            raise ValueError(f"no {'attack' if attack else 'benign'} rows loaded")
        if len(source) <= window_size:
            reps = int(np.ceil(window_size / len(source)))
            source = np.tile(source, (reps, 1))
        start = int(rng.integers(0, len(source) - window_size + 1))
        return source[start : start + window_size]


_DATASET_CACHE: dict[tuple[str, int | None], "RTIoT2022Dataset"] = {}
"""Parsed captures, keyed by (path, max_rows).

`env/environment.py` builds a `ThreatTraceSampler` on every `reset()`,
and an evaluation campaign resets thousands of times. Re-parsing a
123k-row CSV each time dominated everything else in the run. The data is
immutable once loaded, so caching it is free of surprises -- callers
that need a private copy pass their own `standardizer`, which bypasses
the cache."""


def load_rt_iot2022(
    path: str | Path | None = None,
    max_rows: int | None = None,
    standardizer: FeatureStandardizer | None = None,
) -> RTIoT2022Dataset:
    """Read RT-IoT2022 into standardized benign/attack feature matrices.

    `standardizer=None` fits a fresh one on the loaded rows; pass a
    frozen one to transform inference-time data with the exact
    statistics the model was trained under.
    """
    resolved = resolve_dataset_path(path)

    cache_key = (str(resolved), max_rows)
    if standardizer is None and cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    raw_rows: list[dict[str, Any]] = []
    labels: list[str] = []
    with open(resolved, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in FEATURE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{resolved} is missing expected feature columns: {missing}")
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            raw_rows.append(row)
            labels.append(row.get("Attack_type", ""))

    if not raw_rows:
        raise ValueError(f"{resolved} contained no data rows")

    label_counts: dict[str, int] = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    features = extract_flow_features(raw_rows)
    fitted_here = standardizer is None
    standardizer = standardizer if standardizer is not None else FeatureStandardizer.fit(features)
    standardized = standardizer.transform(features)

    is_attack_mask = np.array([is_attack(label) for label in labels], dtype=bool)
    benign = standardized[~is_attack_mask]
    attack = standardized[is_attack_mask]

    total = len(standardized)
    minority = min(len(benign), len(attack)) / total if total else 0.0
    if minority < _MIN_MINORITY_CLASS_FRACTION:
        raise ValueError(
            f"{resolved} has near-zero label variety (minority class {minority:.4%} of rows) -- "
            "refusing to use it as training data per PLAN2 §6's DO-NOT-USE rule"
        )

    dataset = RTIoT2022Dataset(
        benign=benign,
        attack=attack,
        standardizer=standardizer,
        label_counts=label_counts,
    )
    if fitted_here:
        _DATASET_CACHE[cache_key] = dataset
    return dataset


class ThreatTraceSampler:
    """Turns a scenario's threat *level* into real RT-IoT2022 traffic.

    This is the join between the scenario grid and the real capture, and
    it is deliberately the only one. `env/environment.py` asks for the
    feature window at step `t`; this sampler decides, from that step's
    scenario threat level, how likely the window is to be drawn from
    attack traffic rather than benign traffic, and returns real rows
    either way.

    Why a *mixture* rather than a synthetic signal: it makes the threat
    head do genuine work. The forecaster is never told the scenario's
    threat level -- it only ever sees flow features and has to infer
    posture from them. That is what makes S2's "threat elevates ->
    floors ratchet up" a detection result instead of a relabelling of a
    number the environment already knew.

    `attack_probability` maps a standardized threat level onto [0, 1] by
    a logistic with the same gain/bias `env/forecast_provider.py` uses,
    so "level 0.0 is benign" means the same thing in both places.
    """

    _GAIN = 1.7
    _BIAS = -2.94

    def __init__(
        self,
        dataset: RTIoT2022Dataset,
        window_size: int = 16,
        seed: int | None = None,
    ) -> None:
        self._dataset = dataset
        self._window_size = window_size
        self._rng = np.random.default_rng(seed)

    @property
    def window_size(self) -> int:
        return self._window_size

    @classmethod
    def attack_probability(cls, threat_level: float) -> float:
        return float(1.0 / (1.0 + np.exp(-(cls._GAIN * threat_level + cls._BIAS))))

    def window_for_level(self, threat_level: float) -> np.ndarray:
        """One `(window_size, N_FEATURES)` window of real flow features."""
        attack = bool(self._rng.random() < self.attack_probability(threat_level))
        return self._dataset.sample_window(self._rng, self._window_size, attack=attack)

    def labelled_window_for_level(self, threat_level: float) -> tuple[np.ndarray, int]:
        """As `window_for_level`, but also returns the ground-truth label
        (1 == the window was drawn from attack traffic). Training only --
        `env/environment.py` never sees this."""
        attack = bool(self._rng.random() < self.attack_probability(threat_level))
        window = self._dataset.sample_window(self._rng, self._window_size, attack=attack)
        return window, int(attack)


# ---------------------------------------------------------------------------
# 3. Rollout dataset (shared-trunk training data)
# ---------------------------------------------------------------------------

POOL_HORIZONS: tuple[int, int, int] = (10, 25, 50)
"""H in {10, 25, 50}, matching `env/contracts.py`'s PoolForecast."""


@dataclass
class RolloutDataset:
    """Per-step training tensors from baseline rollouts.

    `threat_windows[t]` is the real flow-feature window the environment
    showed the forecaster at step t; `threat_labels[t]` is a vector of
    `1 + THREAT_HORIZON_STEPS` binary labels -- whether the window's
    last step was attack traffic, followed by the same for the next k
    steps, so the horizon head is genuinely forecasting rather than
    reading its own input; `pool_signals[t]` is the pool-side part of
    the same `ForecastObservation`; `pool_targets[t]` is the *future*
    pool state the pool head has to predict, read off the same rollout.
    """

    threat_windows: np.ndarray  # (n, window, N_FEATURES)
    threat_labels: np.ndarray  # (n, 1 + THREAT_HORIZON_STEPS) in {0, 1}
    pool_signals: np.ndarray  # (n, window, N_POOL_SIGNALS)
    pool_targets: np.ndarray  # (n, 3 * len(POOL_HORIZONS))

    def __len__(self) -> int:
        return len(self.threat_labels)


N_POOL_SIGNALS = 4
"""`[qber, skr, pool_fill, hybrid_serves]` -- the pool-side channels of
`ForecastObservation`, in that fixed order. `arrivals_per_class` is
deliberately excluded: it is per-class arrival *counts*, which encode
the tenant mix, and routing that into a model whose threat head feeds
the policy table would blur Hard Rule 3's separation between "which
requests arrive" and "what the agent sees"."""


def pool_signal_vector(observation: dict[str, Any]) -> list[float]:
    """The fixed-order pool-side channels of one `ForecastObservation`."""
    return [
        float(observation["qber"]),
        float(observation["skr"]),
        float(observation["pool_fill"]),
        float(observation["hybrid_serves"]),
    ]


def build_rollout_dataset(
    config: dict[str, Any],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    window_steps: int,
    max_steps: int,
    dataset: RTIoT2022Dataset | None = None,
) -> RolloutDataset:
    """Roll baseline policies through the real environment and record
    per-step training tensors for the dual-head model.

    Why rollouts and not a static file: the pool head has to predict the
    *environment's own* pool trajectory, which only exists inside a
    rollout, and the trunk is shared, so the threat head has to be
    trained on sequences where the pool channels are the real ones.
    Injecting real RT-IoT2022 windows into those same rollouts is what
    gives every step both a threat label and a pool target.

    Which policies: the mandatory non-RL baselines, never the DQN
    (README anti-pattern: never train on `rl_experiment_*` logs -- the
    same reasoning applies to the agent's own trajectories, which would
    make the forecaster's input distribution depend on the policy it is
    about to be used to improve). Baseline rollouts also give a wider
    spread of pool trajectories than any single policy would.

    Hard Rule 8 is respected by the caller passing training scenarios
    only; `forecaster/train.py` refuses S6.
    """
    from agents.baselines import AlwaysHybridPolicy, AlwaysPQCPolicy, RandomPolicy, StaticThresholdPolicy
    from env.environment import SmartKeyNetEnv

    dataset = dataset if dataset is not None else load_rt_iot2022()

    policy_factories = (
        lambda seed: AlwaysPQCPolicy(),
        lambda seed: AlwaysHybridPolicy(),
        lambda seed: StaticThresholdPolicy(0.5),
        lambda seed: RandomPolicy(seed=seed),
    )

    threat_windows: list[np.ndarray] = []
    threat_labels: list[np.ndarray] = []
    pool_signals: list[np.ndarray] = []
    pool_targets: list[np.ndarray] = []

    for scenario in scenarios:
        for seed in seeds:
            for factory in policy_factories:
                env_config = {
                    **config,
                    "scenario": scenario,
                    "seed": seed,
                    "max_steps": max_steps,
                    # force the real-capture threat source for training data,
                    # whatever the ambient config says
                    "threat_input": {**(config.get("threat_input") or {}), "source": "rt_iot2022"},
                }
                env = SmartKeyNetEnv(env_config)
                state, info = env.reset(seed=seed)
                policy = factory(seed)

                step_features: list[list[float]] = []
                step_labels: list[int] = []
                step_pool: list[list[float]] = []
                pool_track: list[tuple[float, float, float]] = []

                truncated = False
                while not truncated:
                    observation = env.last_forecast_observation()
                    if observation is not None:
                        features = np.asarray(observation["threat_features"], dtype=np.float32)
                        if features.size == N_FEATURES:
                            step_features.append(features.tolist())
                            step_labels.append(int(env.last_threat_label()))
                            step_pool.append(pool_signal_vector(observation))
                            pool_track.append(
                                (
                                    float(observation["pool_fill"]),
                                    float(observation["skr"]),
                                    float(observation["hybrid_serves"]),
                                )
                            )
                    action = policy.act(state, info["action_mask"])
                    state, _reward, _terminated, truncated, info = env.step(action)

                _accumulate_windows(
                    step_features,
                    step_labels,
                    step_pool,
                    pool_track,
                    window_steps,
                    threat_windows,
                    threat_labels,
                    pool_signals,
                    pool_targets,
                )

    if not threat_windows:
        raise ValueError(
            "rollouts produced no usable windows -- check that threat_input.source is "
            "'rt_iot2022' and that max_steps exceeds window_steps + max(POOL_HORIZONS)"
        )

    return RolloutDataset(
        threat_windows=np.asarray(threat_windows, dtype=np.float32),
        threat_labels=np.asarray(threat_labels, dtype=np.float32),
        pool_signals=np.asarray(pool_signals, dtype=np.float32),
        pool_targets=np.asarray(pool_targets, dtype=np.float32),
    )


def _accumulate_windows(
    step_features: list[list[float]],
    step_labels: list[int],
    step_pool: list[list[float]],
    pool_track: list[tuple[float, float, float]],
    window_steps: int,
    out_threat_windows: list[np.ndarray],
    out_threat_labels: list[np.ndarray],
    out_pool_signals: list[np.ndarray],
    out_pool_targets: list[np.ndarray],
) -> None:
    """Slice one rollout into `(window, label vector, pool target)` triples.

    The label vector is `[label_now, label_{t+1}, ..., label_{t+k}]`
    where "now" is the window's **last** step. The k future entries are
    what make the horizon head a forecaster rather than a classifier
    reading its own input. Pool targets are the mean of each tracked
    quantity over the next H steps, for H in `POOL_HORIZONS`.
    """
    from forecaster.model import THREAT_HORIZON_STEPS  # local: avoids a circular import

    horizon_max = max(POOL_HORIZONS)
    n = len(step_features)
    last_start = n - window_steps - max(horizon_max, THREAT_HORIZON_STEPS)
    for start in range(0, max(0, last_start)):
        end = start + window_steps
        out_threat_windows.append(np.asarray(step_features[start:end], dtype=np.float32))
        out_pool_signals.append(np.asarray(step_pool[start:end], dtype=np.float32))

        now_index = end - 1
        labels = [step_labels[now_index]] + [
            step_labels[min(now_index + offset, n - 1)]
            for offset in range(1, THREAT_HORIZON_STEPS + 1)
        ]
        out_threat_labels.append(np.asarray(labels, dtype=np.float32))

        target: list[float] = []
        for quantity_index in range(3):
            for horizon in POOL_HORIZONS:
                future = pool_track[end : end + horizon]
                target.append(
                    float(np.mean([row[quantity_index] for row in future])) if future else 0.0
                )
        out_pool_targets.append(np.asarray(target, dtype=np.float32))
