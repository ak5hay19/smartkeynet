"""
env/contracts.py

Frozen interface contracts for SmartKeyNet (PLAN.md §4, §4A; split.md §0).

This file is the single most important artifact in the repo (split.md
§0, "the critical hour"): it is the seam that lets four people build
in parallel behind stable interfaces. Treat every name in here as a
promise to the other three owners. Changes require a PR that pings the
whole team (split.md §3, "Editing env/contracts.py unilaterally" is
listed as a banned anti-pattern).

This module only defines shapes: enums, TypedDicts, dataclasses, and
abstract interfaces. It contains no environment logic, no agent logic,
and no forecaster logic. See:
  - PLAN.md §4   (state spec, reward spec, architecture diagram)
  - PLAN.md §4A  (Addition A: dual-head forecaster; Addition C: regret/churn)
  - PLAN.md Hard Rules 1-9 (security/masking/one-MDP invariants)
  - split.md §0  ("split by INTERFACE, not by file")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TypedDict

import numpy as np

try:  # numpy>=1.21 ships numpy.typing; fall back gracefully if not.
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    NDArray = np.ndarray  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Action space (PLAN.md §1 "What the agent literally does per request")
# ---------------------------------------------------------------------------


class Action(IntEnum):
    """The five actions the DQN chooses among per request (PLAN.md §1).

    Order is fixed and load-bearing: every ActionMask in the codebase
    is positionally aligned to this enum's integer values. Do not
    reorder members without updating every mask producer/consumer.
    """

    SERVE_CLASSICAL = 0  # T0 -- X25519 / AES-256-GCM class
    SERVE_PQC = 1  # T1 -- ML-KEM-768 (NIST Category 3), the free workhorse
    SERVE_HYBRID = 2  # T2/T3 -- ML-KEM-768 XOR QKD pool material. Consumes pool.
    REUSE = 3  # keep the existing session key if age limits allow
    REKEY_NOW = 4  # force refresh (e.g. ahead of a forecast threat spike)


N_ACTIONS = len(Action)


ActionMask = NDArray[np.bool_]
"""Boolean array of shape (N_ACTIONS,), positionally aligned to `Action`.

`mask[a]` is True iff action `a` is legal this step. Produced by
`env/masking.py` from the policy-table floor plus feasibility checks
(pool empty, key age exceeded) -- never by the agent (Hard Rule 2:
floors are enforced by masking, not by reward penalties).
"""


def valid_actions(mask: ActionMask) -> set[Action]:
    """Convenience view of a mask as the set of legal `Action`s."""
    return {Action(i) for i, ok in enumerate(mask) if ok}


class KeyType(IntEnum):
    """One-hot-encoded in the state as `key_type_onehot` (PLAN.md §4)."""

    CLASSICAL = 0
    PQC = 1
    HYBRID = 2


class SensitivityClass(IntEnum):
    """Tenant/request confidentiality class (PLAN.md §4 state spec;
    "Datasets & Provenance" table).

    Calibrated against Q-OPSEC's `synthetic_context_dataset` (6
    balanced classes) and `confidentiality_train`/`valid` (4-class).
    The exact class count/boundaries are a joint Person A / Person B
    calibration call -- this four-tier placeholder is a skeleton
    starting point, not final.
    """

    S0 = 0  # public / non-sensitive telemetry
    S1 = 1  # routine business data
    S2 = 2  # regulated / financial data
    S3 = 3  # e.g. patient records -- long data lifetime, highest floor


class ThreatPosture(IntEnum):
    """Discrete threat class predicted by the forecaster's threat head.

    Feeds `env/masking.py`'s PolicyTable (class x posture -> floor).
    Must never be derived from a signal path that could *lower* a
    floor (Hard Rule 2). Final class count/boundaries TBD by Person A;
    this three-tier placeholder is a skeleton starting point.
    """

    CALM = 0
    ELEVATED = 1
    HIGH = 2


# ---------------------------------------------------------------------------
# Request schema (co-owned by Person A [request_generator] and
# Person B [environment] -- split.md §1)
# ---------------------------------------------------------------------------


class Request(TypedDict):
    """One key request: emitted by `env/request_generator.py`, consumed
    by `env/environment.py` and `env/masking.py`.

    PLAN.md §1: "For each incoming key request (tenant, service,
    sensitivity class) the agent picks one of: ..."
    """

    request_id: str
    step: int
    tenant: str
    service: str
    sensitivity_class: int  # SensitivityClass value
    pqc_capable: bool  # False => legacy endpoint; SERVE_CLASSICAL may be the only option
    hybrid_mandatory: (
        bool  # True => must eventually be served >= SERVE_HYBRID, or deferred (Hard Rule 9)
    )


# ---------------------------------------------------------------------------
# State spec (PLAN.md §4 "State (per step)"; Addition A foresight fields)
# ---------------------------------------------------------------------------


class StateDict(TypedDict):
    """Per-step state vector handed from the env (Person B) to the agent
    (Person C) (PLAN.md §4).

    NOTE: the *flattened* state-vector length changes with
    `configs/default.yaml`'s `use_foresight` flag (off / ewma / lstm;
    Addition A). Foresight fields are always present as keys here (so
    downstream code doesn't need to branch on the flag) but are
    populated at different fidelity -- or zeroed -- depending on the
    flag. Cover the actual flattened length for each flag value with a
    unit test (Addition A, "Unit tests: ... state-length under each
    flag").

    All fields are scalars for the current step/request unless noted.
    """

    # --- threat (PLAN.md §4 state spec) ---
    threat_score: float
    threat_forecast: Sequence[
        float
    ]  # length k=5: next-k-step threat signal (Addition A threat head)
    posture_probs: Sequence[float]  # length 4: {normal, elevated, high, critical}

    # --- QKD pool physics (env/pool_sim.py) ---
    qber: float
    skr: float  # instantaneous secret-key rate driving pool refill
    pool_fill: float  # current pool level, normalized [0, 1]

    # --- traffic / load ---
    arrival_rate: float
    load: float  # normalized system load (e.g. queue depth / capacity)
    avg_latency: float  # rolling average latency (env accounting)

    # --- current session key (per request) ---
    key_age: float  # normalised by the SP 800-57 cap L
    key_type_onehot: Sequence[float]  # length 4: {none, classical, pqc, hybrid}

    # --- request context ---
    request_class_onehot: Sequence[float]  # length 4, order = SensitivityClass
    floor_onehot: Sequence[float]  # length 4: {T0, T1, T2, T3}
    pqc_capable: float  # 0/1
    queue_len_norm: float  # deferral queue length / queue_ref
    queue_head_wait_norm: float  # head-of-line wait / 100 steps
    steps_since_rekey_norm: float  # / lifetime cap L
    sensitivity_class: int  # SensitivityClass value (kept for baselines/harness)
    policy_floor: int  # Action value: minimum tier this request must clear

    # --- foresight features (Addition A; DQN state ONLY -- never the
    #     policy table, Hard Rule 2) ---
    pool_level_hat: Sequence[float]  # length 3, H in {10, 25, 50}
    skr_mean_hat: Sequence[float]  # length 3, H in {10, 25, 50}
    skr_trend: float  # signed normalised slope of the SKR forecast
    hybrid_demand_hat: Sequence[float]  # length 3, H in {10, 25, 50}
    regret_event_recent: bool  # Addition C: recent-regret-event flag


# ---------------------------------------------------------------------------
# ForecastProvider interface (PLAN.md Addition A; split.md §0 contract #2)
# ---------------------------------------------------------------------------


class ForecastObservation(TypedDict):
    """One step of raw signal fed into `ForecastProvider.update` (Addition A).

    Built by `env/environment.py` each step from `pool_sim` +
    `request_generator` output; consumed identically by
    `MovingAverageForecaster` and `LSTMForecastProvider` so the two
    stay interchangeable (Addition A unit test: "provider
    interchangeability").
    """

    qber: float
    skr: float
    pool_fill: float
    arrivals_per_class: Sequence[int]  # indexed by SensitivityClass
    hybrid_serves: int  # hybrid serves this step
    threat_features: Sequence[
        float
    ]  # raw network features -> LSTM threat head (RT-IoT2022-derived)


@dataclass
class ThreatForecast:
    """Output of a ForecastProvider's threat head. Feeds
    `env/masking.py`'s PolicyTable -- may only ever *raise* floors
    (Hard Rule 2)."""

    threat_score: float
    posture_probs: Sequence[float]  # class distribution, indexed by ThreatPosture
    horizon_scores: Sequence[float]  # next-k=5-step threat score/probability


@dataclass
class PoolForecast:
    """Output of a ForecastProvider's pool head. Feeds the DQN state
    only (see StateDict's foresight fields) -- must never be routed
    into `env/masking.py`'s floor computation (Hard Rule 2)."""

    pool_level_hat: Sequence[float]  # H in {10, 25, 50}
    skr_mean_hat: Sequence[float]  # H in {10, 25, 50}
    hybrid_demand_hat: Sequence[float]  # H in {10, 25, 50}: expected hybrid-mandatory arrivals


class ForecastProvider(ABC):
    """Dual-head forecaster interface (PLAN.md Addition A).

    Two implementations are required to exist behind this interface:
      - `env.forecast_provider.MovingAverageForecaster` -- EWMA
        fallback, lets the env run in month 1 before the LSTM exists.
      - `forecaster.model.LSTMForecastProvider` -- real, trained
        offline, frozen during DQN training.

    `configs/default.yaml`'s `use_foresight: {off, ewma, lstm}` selects
    which implementation `env/environment.py` constructs. `off` is
    handled by the env itself (zeroed foresight fields), not by a
    third `ForecastProvider` implementation.
    """

    @abstractmethod
    def update(self, observation: ForecastObservation) -> None:
        """Feed one step of real observations into the provider's window."""
        raise NotImplementedError

    @abstractmethod
    def get_threat_forecast(self) -> ThreatForecast:
        """Return the current threat-head forecast. Feeds the policy table."""
        raise NotImplementedError

    @abstractmethod
    def get_pool_forecast(self) -> PoolForecast:
        """Return the current pool-head forecast. Feeds the DQN state only."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Event-log schema (PLAN.md Addition C; implements Hard Rule 9)
# ---------------------------------------------------------------------------


class RegretEvent(TypedDict):
    """Logged once per deferral *onset* (Addition C). Implements Hard
    Rule 9: pool exhaustion never causes a downgrade -- it causes a
    deferral, logged here."""

    step: int
    request_id: str
    tenant: str
    sensitivity_class: int
    policy_floor: int
    pool_fill_at_onset: float


class DeferredCriticalStep(TypedDict):
    """Logged once per step a hybrid-mandatory request continues to wait
    in `env/deferral_queue.py` (Addition C)."""

    step: int
    request_id: str
    steps_waited: int


class ForcedRekey(TypedDict):
    """Logged when `key_age` hits the SP 800-57-derived cap `L` and
    REUSE is masked out, forcing an automatic rekey (Addition C
    "Staleness is NOT a reward term")."""

    step: int
    request_id: str
    key_age_at_rekey: float
    load_at_rekey: float
    cost: float  # c_rekey(load) paid this step
