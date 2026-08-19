"""
env/threat_source.py

Real threat features from RT-IoT2022 (PLAN.md "Datasets & Provenance": the
one real-network slot in the project; SMARTKEYNET_BUILD_SPEC.md §S9 lists
`threat_features_from_RT_IoT2022(d)` among the forecaster's inputs).

Replaces the synthetic `[qber, load, scenario_boost]` placeholder that the
environment fed the forecaster before 2026-08-18.

---------------------------------------------------------------------
What this dataset does and does not provide -- read before using it
---------------------------------------------------------------------
RT-IoT2022 is 123,117 labelled bidirectional flow records with 82 numeric
features and a 12-class `Attack_type` label. Two properties of the file
determine how it can honestly be used, and both were measured before this
module was written:

  1. **It is sorted by attack type.** There are only **12 label changes**
     across 123,117 rows -- each class is one contiguous block averaging
     ~10,260 rows.
  2. **There is no timestamp column.** The flows carry durations and
     inter-arrival statistics, but no absolute time, so the file has no
     recoverable chronological order.

Together those mean the file is **not a traffic timeline**. Sliding a window
along it in row order yields blocks of a single class, and a forecaster
trained that way learns "predict whatever the last 64 rows were" -- ~99.9%
accurate and completely uninformative. That is the same degenerate-label trap
that made the earlier ratcheted-posture forecaster score exactly the
majority-class rate, and it is worth naming twice.

So this module uses the dataset for what it genuinely supplies -- **real flow
feature distributions and real attack labels** -- and constructs the temporal
arrangement itself. The scenario decides *when* an escalation happens; the
dataset decides *what the network looks like* during one. That division is
stated plainly in the report rather than glossed: the timeline is synthetic,
the observations are real.

---------------------------------------------------------------------
A caveat on how strongly the held-out split can be read
---------------------------------------------------------------------
The dataset is extremely repetitive on the eight selected features.
Measured unique-row fractions within each pool:

    CALM (benign IoT)      8,754 rows ->  8,349 unique (95.4%)
    ELEVATED (NMAP scans)  5,341 rows ->    208 unique ( 3.9%)
    HIGH (attack)         72,086 rows ->  3,671 unique ( 5.1%)

The 70/30 split partitions *rows*, so no row is in both pools -- but with
only 208 distinct reconnaissance vectors, an evaluation row is very often a
value-level duplicate of a training row. The scorer's train/eval agreement
(0.10 vs 0.10, 0.52 vs 0.52, 0.80 vs 0.81) is therefore partly explained by
duplication rather than purely by generalisation, and should be quoted with
that qualification.

This is a genuine property of the traffic, not a flaw in the split: a SYN
flood really is tens of thousands of near-identical flows. It is recorded
here because it bounds what the held-out numbers prove.

---------------------------------------------------------------------
Mapping attack classes onto threat postures
---------------------------------------------------------------------
The 12 classes map onto the three postures along the actual intrusion
lifecycle, which is what makes the resulting signal forecastable rather than
arbitrary:

  CALM     normal IoT telemetry -- Thing_Speak, MQTT_Publish, Wipro_bulb
  ELEVATED reconnaissance -- the NMAP_* scan family
  HIGH     active attack -- DOS_SYN_Hping, DDOS_Slowloris, ARP_poisioning,
           Metasploit_Brute_Force_SSH

**Reconnaissance genuinely precedes exploitation**, so an ELEVATED window
carries observable precursors of a HIGH one. That is the leading indicator
the forecaster exists to detect, and here it comes from real captured traffic
rather than from a synthetic ramp.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from env.contracts import ThreatPosture

DEFAULT_CSV = Path("data/raw/rt_iot2022/RT_IOT2022.csv")

_SHUFFLE_SEED_BASE = 90_210
"""Fixed base for the per-posture shuffle seed. Any constant works; what
matters is that it is a constant and that the per-posture offset is the
posture's integer value, so the split is identical on every machine and every
process. See the comment at its use site for the bug this replaced."""


POSTURE_CLASSES: dict[ThreatPosture, tuple[str, ...]] = {
    ThreatPosture.CALM: ("Thing_Speak", "MQTT_Publish", "Wipro_bulb"),
    ThreatPosture.ELEVATED: (
        "NMAP_UDP_SCAN",
        "NMAP_XMAS_TREE_SCAN",
        "NMAP_OS_DETECTION",
        "NMAP_TCP_scan",
        "NMAP_FIN_SCAN",
    ),
    ThreatPosture.HIGH: (
        "DOS_SYN_Hping",
        "DDOS_Slowloris",
        "ARP_poisioning",
        "Metasploit_Brute_Force_SSH",
    ),
}

FEATURE_COLUMNS: tuple[str, ...] = (
    "fwd_pkts_per_sec",
    "flow_pkts_per_sec",
    "flow_SYN_flag_count",
    "flow_RST_flag_count",
    "flow_duration",
    "fwd_pkts_tot",
    "down_up_ratio",
    "flow_iat.avg",
)
"""Eight features, chosen for discriminative power and interpretability
rather than by an automated selector.

Measured class means show the separation is real, not hoped for: benign IoT
traffic runs at 0.65-105 forward packets/sec while DOS_SYN_Hping runs at
444,796. Scan traffic sits in between and carries distinctive flag patterns
(NMAP_XMAS/OS_DETECTION show SYN counts near zero with RST near one).

Kept deliberately small. The state vector this feeds is consumed by the DQN,
and widening it changes `flatten_state`'s output length -- a checkpoint
compatibility break. Eight is enough to separate the three postures and cheap
enough to standardise reliably.

All eight are heavy-tailed (packets/sec spans six orders of magnitude), so
`log1p` is applied before standardisation; without it a single DoS flow would
dominate every batch statistic.
"""

N_THREAT_FEATURES = len(FEATURE_COLUMNS)

_TRAIN_FRACTION = 0.7
"""Rows are split per class into a training pool and an evaluation pool.

Splitting *within* each class matters: the file is sorted by class, so a
naive head/tail split would put entire attack types on one side and the
forecaster would be evaluated on classes it had never seen. Sampling is
always drawn from the pool matching the run's role, so evaluation episodes
observe flows the forecaster was never trained on.
"""


@dataclass
class ThreatFeatureStats:
    """Per-feature mean/std from the TRAINING pool only, used to standardise
    every sample. Computed on train and applied to eval, never recomputed at
    evaluation time -- recomputing would leak evaluation statistics into the
    features and quietly flatter the model."""

    mean: np.ndarray
    std: np.ndarray

    def standardise(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.mean) / self.std


class RTIoT2022ThreatSource:
    """Samples real flow-feature vectors conditioned on a threat posture.

    The environment asks for "what does the network look like right now,
    given the scenario says the posture is ELEVATED?" and gets back a
    standardised feature vector drawn from real reconnaissance traffic.
    """

    def __init__(
        self,
        csv_path: str | Path = DEFAULT_CSV,
        split: str = "train",
        seed: int | None = None,
    ) -> None:
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")

        self.split = split
        self._rng = np.random.default_rng(seed)
        pools, stats = _load_pools(str(Path(csv_path).resolve()))
        self._pools = {posture: pool[split] for posture, pool in pools.items()}
        self._stats = stats

        for posture, pool in self._pools.items():
            if len(pool) == 0:
                raise ValueError(f"no {split} rows for posture {posture.name}")

    def sample(self, posture: ThreatPosture) -> np.ndarray:
        """One standardised feature vector drawn from `posture`'s class pool."""
        pool = self._pools[ThreatPosture(int(posture))]
        row = pool[self._rng.integers(0, len(pool))]
        return self._stats.standardise(row)

    def sample_mixture(
        self, calm_weight: float, elevated_weight: float, high_weight: float
    ) -> np.ndarray:
        """Draw from a *mixture* of postures.

        This is what the environment actually calls. A real escalation is not
        a clean switch from benign to attack traffic -- attack flows appear
        alongside ongoing normal traffic and grow as a share of it. Mixing
        lets the scenario's ramp express that gradual change, which is
        precisely the structure the forecaster has to learn to anticipate.
        """
        weights = np.array(
            [max(0.0, calm_weight), max(0.0, elevated_weight), max(0.0, high_weight)],
            dtype=np.float64,
        )
        total = weights.sum()
        if total <= 0.0:
            return self.sample(ThreatPosture.CALM)
        weights /= total

        drawn = self._rng.choice(3, p=weights)
        return self.sample(ThreatPosture(int(drawn)))

    @property
    def pool_sizes(self) -> dict[str, int]:
        return {posture.name: len(pool) for posture, pool in self._pools.items()}


@functools.lru_cache(maxsize=2)
def _load_pools(
    csv_path: str,
) -> tuple[dict[ThreatPosture, dict[str, np.ndarray]], ThreatFeatureStats]:
    """Load and bucket the CSV once per process.

    Cached because the file is ~55 MB and every environment reset would
    otherwise re-read it; a training run constructs thousands of environments.
    """
    import pandas as pd

    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"RT-IoT2022 not found at {csv_path}. Download it from the UCI "
            "repository and place it there; see data/get_data.py."
        )

    frame = pd.read_csv(csv_path, usecols=[*FEATURE_COLUMNS, "Attack_type"])

    pools: dict[ThreatPosture, dict[str, np.ndarray]] = {}
    train_blocks: list[np.ndarray] = []

    for posture, class_names in POSTURE_CLASSES.items():
        subset = frame[frame["Attack_type"].isin(class_names)]
        raw = subset[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        # heavy-tailed by orders of magnitude -- compress before standardising
        raw = np.log1p(np.clip(raw, 0.0, None))

        # Shuffle before splitting: the file is grouped by class, so
        # contiguous rows are near-duplicates from the same capture segment.
        # Seeded from the posture's ORDINAL, never from `hash(posture.name)`.
        #
        # Python randomises string hashing per process (PYTHONHASHSEED), so the
        # previous form drew a different shuffle seed on every interpreter
        # launch -- which meant the train/eval split of the threat data, and
        # therefore every threat-driven number in this project, differed
        # between runs of identical code with identical seeds. Two identical
        # 300-step S3 rollouts produced 41 and 13 regret events.
        #
        # Nothing caught it: `test_seed_reproducibility` compares two envs
        # inside ONE process, where the hash seed is fixed for the process's
        # lifetime, so the property it checks was real but strictly weaker than
        # the property that matters. The golden fixture caught it immediately,
        # because a fixture is compared across processes by construction.
        shuffle_rng = np.random.default_rng(_SHUFFLE_SEED_BASE + int(posture))
        shuffle_rng.shuffle(raw)

        cut = int(len(raw) * _TRAIN_FRACTION)
        pools[posture] = {"train": raw[:cut], "eval": raw[cut:]}
        train_blocks.append(raw[:cut])

    all_train = np.concatenate(train_blocks, axis=0)
    stats = ThreatFeatureStats(
        mean=all_train.mean(axis=0),
        std=np.maximum(all_train.std(axis=0), 1e-6),
    )
    return pools, stats


# ---------------------------------------------------------------------------
# Scalar threat score (Fisher linear discriminant)
# ---------------------------------------------------------------------------


@dataclass
class GradedThreatScorer:
    """Maps a flow vector to a graded threat score: ~0 benign, ~0.5
    reconnaissance, ~1.0 active attack.

    A single benign-vs-rest discriminant is not enough, and measuring that
    was necessary rather than obvious: fitted binary, reconnaissance scored
    0.865 and active attack 0.873 -- indistinguishable. The posture ladder has
    three rungs, so the scorer needs two decisions, not one.

    Two Fisher discriminants, composed:

        w_threat  separates benign from (recon + attack)
        w_attack  separates recon from attack, fitted on non-benign flows only

        score = P(threat) * (0.5 + 0.5 * P(attack | threat))

    which lands benign near 0, reconnaissance near 0.5 and attack near 1.0 --
    monotone along the intrusion lifecycle, which is what the policy table's
    posture anchors {0.0, 0.5, 1.0} expect.
    """

    threat_weights: np.ndarray
    threat_midpoint: float
    threat_scale: float
    attack_weights: np.ndarray
    attack_midpoint: float
    attack_scale: float

    @staticmethod
    def _sigmoid(projection: float, midpoint: float, scale: float) -> float:
        return float(1.0 / (1.0 + np.exp(-(projection - midpoint) / scale)))

    def score(self, features: np.ndarray) -> float:
        p_threat = self._sigmoid(
            float(np.dot(features, self.threat_weights)), self.threat_midpoint, self.threat_scale
        )
        p_attack = self._sigmoid(
            float(np.dot(features, self.attack_weights)), self.attack_midpoint, self.attack_scale
        )
        return float(p_threat * (0.5 + 0.5 * p_attack))


@dataclass
class ThreatScorer:
    """Maps an 8-feature flow vector to a scalar threat score in [0, 1].

    WHY A SCALAR IS NEEDED AT ALL. `MovingAverageForecaster` -- the EWMA
    fallback, and the middle rung of the E-A ablation -- collapses the threat
    feature vector to its arithmetic mean before squashing it. That works for
    a signal that is already scalar-ish and monotone in threat. It does not
    work for these features: measured on standardised RT-IoT2022 samples the
    per-posture means come out CALM 0.29, ELEVATED -0.52, HIGH -0.01, which is
    **not monotone**. Feeding those to the EWMA would make ELEVATED read as
    *calmer* than CALM, and every floor derived from it would be wrong.

    So the scalar is computed properly rather than by averaging. A Fisher
    linear discriminant is fitted once on the training pool, separating benign
    flows from non-benign (reconnaissance and attack together):

        w      = Sigma^-1 (mu_threat - mu_benign)
        score  = sigmoid((x . w - midpoint) / scale)

    Closed-form, no iteration, no extra dependency, and interpretable -- `w`
    is directly inspectable as "which flow features indicate threat".

    This is a deliberately *simple* supervised model, and that is the point:
    the LSTM threat head (which consumes the full 8-vector) is the project's
    real classifier, and this scalar exists so the EWMA fallback has an honest
    signal to smooth. Making the fallback a neural network too would destroy
    the ablation's meaning.
    """

    weights: np.ndarray
    midpoint: float
    scale: float

    def score(self, features: np.ndarray) -> float:
        projection = float(np.dot(features, self.weights))
        return float(1.0 / (1.0 + np.exp(-(projection - self.midpoint) / self.scale)))


def _fit_fisher(negative: np.ndarray, positive: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Closed-form Fisher discriminant plus a sigmoid calibration."""
    mean_negative = negative.mean(axis=0)
    mean_positive = positive.mean(axis=0)

    covariance = np.cov(np.vstack([negative, positive]), rowvar=False)
    # ridge: fwd_pkts_per_sec and flow_pkts_per_sec are very nearly collinear
    covariance += np.eye(covariance.shape[0]) * 1e-3
    weights = np.linalg.solve(covariance, mean_positive - mean_negative)

    projected_negative = float((negative @ weights).mean())
    projected_positive = float((positive @ weights).mean())
    midpoint = (projected_negative + projected_positive) / 2.0
    scale = max(1e-6, (projected_positive - projected_negative) / 4.0)
    return weights, midpoint, scale


def fit_graded_threat_scorer(
    source: RTIoT2022ThreatSource, n_samples: int = 4000
) -> GradedThreatScorer:
    """Fit both discriminants on `source`'s TRAINING pools."""
    benign = np.array([source.sample(ThreatPosture.CALM) for _ in range(n_samples)])
    recon = np.array([source.sample(ThreatPosture.ELEVATED) for _ in range(n_samples)])
    attack = np.array([source.sample(ThreatPosture.HIGH) for _ in range(n_samples)])

    threat_w, threat_m, threat_s = _fit_fisher(benign, np.vstack([recon, attack]))
    attack_w, attack_m, attack_s = _fit_fisher(recon, attack)

    return GradedThreatScorer(
        threat_weights=threat_w,
        threat_midpoint=threat_m,
        threat_scale=threat_s,
        attack_weights=attack_w,
        attack_midpoint=attack_m,
        attack_scale=attack_s,
    )


def fit_threat_scorer(source: RTIoT2022ThreatSource, n_samples: int = 4000) -> ThreatScorer:
    """Fit the discriminant on `source`'s training pools.

    Benign = CALM class flows. Threat = ELEVATED and HIGH pooled, because the
    discriminant's job is "is this normal traffic or not"; distinguishing
    reconnaissance from active attack is the LSTM threat head's job, and it
    has the full feature vector plus temporal context to do it with.
    """
    rng = np.random.default_rng(0)
    benign = np.array([source.sample(ThreatPosture.CALM) for _ in range(n_samples)])
    threat = np.array(
        [
            source.sample(ThreatPosture.ELEVATED if rng.random() < 0.5 else ThreatPosture.HIGH)
            for _ in range(n_samples)
        ]
    )

    mean_benign = benign.mean(axis=0)
    mean_threat = threat.mean(axis=0)

    pooled_covariance = np.cov(np.vstack([benign, threat]), rowvar=False)
    # ridge term keeps the inverse well-conditioned if two features are
    # near-collinear (fwd_pkts_per_sec and flow_pkts_per_sec very nearly are)
    pooled_covariance += np.eye(pooled_covariance.shape[0]) * 1e-3
    weights = np.linalg.solve(pooled_covariance, mean_threat - mean_benign)

    projected_benign = benign @ weights
    projected_threat = threat @ weights
    midpoint = float((projected_benign.mean() + projected_threat.mean()) / 2.0)
    scale = float(max(1e-6, (projected_threat.mean() - projected_benign.mean()) / 4.0))

    return ThreatScorer(weights=weights, midpoint=midpoint, scale=scale)
