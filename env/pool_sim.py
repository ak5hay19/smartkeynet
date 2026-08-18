"""
env/pool_sim.py

QKD pool simulator: trace-driven refill, draw-down, and exhaustion
(PLAN.md §4 architecture diagram "QKD POOL SIM"; PLAN.md §10 kickoff
step 2). Owned by Person B (split.md §1).

This is the scarcity engine the whole thesis rests on: the pool level
computed here is what `env/masking.py` checks for SERVE_HYBRID
feasibility, and what `env/deferral_queue.py` queues against when a
hybrid-mandatory request can't be covered (Hard Rule 9).

Real refill/drain/exhaustion arithmetic (PLAN.md §10 step 2). No real
CV-QKD trace was sourced for this capstone; per PLAN.md "Datasets &
Provenance" ("documented synthetic SKR process ... synthetic fallback
is fully acceptable for a capstone if the generation procedure is
stated + rate ranges cited"), `SyntheticSKRQBERTrace` below is the
documented fallback and states its generation procedure in its own
docstring.

Unit convention: `PoolState.fill`/`capacity` are in **bits**;
`SKRQBERTrace` yields `skr` in **kbps** (citable, kbps-scale, matching
PLAN.md's "refills slowly (kbps ...)"). `PoolSim.step()` converts kbps
to bits-refilled-this-step under the modeling assumption that one
simulator step represents one wall-clock second (`_SECONDS_PER_STEP`
below) -- this is a fixed simulator convention, not a tunable config
value, so it is documented here rather than added to
`configs/default.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np
import yaml


@dataclass
class PoolState:
    """Snapshot of pool physics at one step."""

    fill: float  # current pool level, bits
    capacity: float  # max pool capacity, bits
    skr: float  # instantaneous secret-key rate driving refill
    qber: float  # current quantum bit error rate


class SKRQBERTrace(Protocol):
    """Interface for the CV-QKD trace source driving pool refill.

    Implemented by Person A's trace loader (PLAN.md "Datasets &
    Provenance" -> "QKD pool refill (SKR/QBER)" row) or a documented
    synthetic generator. `PoolSim` only ever consumes this interface.
    """

    def __iter__(self) -> Iterator[tuple[float, float]]:
        """Yield (skr, qber) pairs, one per simulator step."""
        ...


class PoolExhaustedError(Exception):
    """Raised by `PoolSim.draw` if called without a prior `can_draw` check.

    Exhaustion itself is not an error condition for the environment --
    it is handled by `env/deferral_queue.py` (Hard Rule 9) -- but a
    caller that draws without checking feasibility first is a bug.
    """


class PoolSim:
    """Trace-driven QKD pool simulator (refill + drain + exhaustion).

    Implements the "QKD POOL SIM" box in PLAN.md §4's architecture
    diagram. Refill is driven by an `SKRQBERTrace`; drain happens when
    `env/environment.py` calls `draw()` for a SERVE_HYBRID action.

    `capacity` and `initial_fill_frac` are simulator parameters
    supplied by the caller (see `load_pool_config` below, which reads
    them from `configs/default.yaml`'s `pool:` block) -- nothing about
    their values is hardcoded here.
    """

    _SECONDS_PER_STEP: float = 1.0  # fixed sim convention: 1 step == 1 wall-clock second

    def __init__(
        self,
        capacity: float,
        trace: SKRQBERTrace,
        initial_fill_frac: float,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if not 0.0 <= initial_fill_frac <= 1.0:
            raise ValueError(f"initial_fill_frac must be in [0, 1], got {initial_fill_frac}")

        self.capacity = float(capacity)
        self._trace = trace
        self._initial_fill_frac = float(initial_fill_frac)

        self._fill: float = 0.0
        self._skr: float = 0.0
        self._qber: float = 0.0
        self._iterator: Iterator[tuple[float, float]] | None = None

        self.reset()

    @property
    def fill(self) -> float:
        """Current pool level, bits (read-only view; mutate only via step()/draw())."""
        return self._fill

    def reset(self) -> PoolState:
        """Reset pool to its initial fill level and rewind the trace."""
        self._iterator = iter(self._trace)
        self._fill = self.capacity * self._initial_fill_frac
        self._skr = 0.0
        self._qber = 0.0
        return self._state()

    def step(self) -> PoolState:
        """Advance one timestep: pull the next (skr, qber) from the trace
        and refill the pool accordingly."""
        if self._iterator is None:
            raise RuntimeError("PoolSim.step() called before reset()")

        skr_kbps, qber = next(self._iterator)
        self._skr = float(skr_kbps)
        self._qber = float(qber)

        bits_refilled = self._skr * 1000.0 * self._SECONDS_PER_STEP  # kbps -> bits/step
        self._fill = min(self.capacity, self._fill + bits_refilled)

        return self._state()

    def can_draw(self, bits: float) -> bool:
        """Feasibility check used by `env/masking.py` before allowing
        SERVE_HYBRID, and by `env/deferral_queue.py` to decide when a
        queued request can finally be served."""
        return 0.0 <= bits <= self._fill

    def draw(self, bits: float) -> None:
        """Consume `bits` from the pool for a SERVE_HYBRID action.

        Raises `PoolExhaustedError` if `bits` exceeds current fill --
        callers must check `can_draw()` first. Pool exhaustion is
        handled by `env/deferral_queue.py`, never by silently
        under-drawing (Hard Rule 9).
        """
        if bits < 0:
            raise ValueError(f"draw() bits must be non-negative, got {bits}")
        if not self.can_draw(bits):
            raise PoolExhaustedError(
                f"cannot draw {bits} bits from pool with {self._fill} bits available"
            )
        self._fill = max(0.0, self._fill - bits)

    def _state(self) -> PoolState:
        return PoolState(fill=self._fill, capacity=self.capacity, skr=self._skr, qber=self._qber)


def load_pool_config(path: str | Path | None = None) -> dict[str, float]:
    """Read the `pool:` block out of `configs/default.yaml`.

    Centralizes config access so `capacity_bits`/`initial_fill_frac`
    are never duplicated as literals in test code or (eventually)
    `env/environment.py`. Defaults to the repo's `configs/default.yaml`
    if no path is given.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config["pool"]


_DEFAULT_REFILL_BITS_PER_STEP = 15.0
"""Fallback for `slice_skr_kbps` when a caller passes a `pool:` block
predating the 2026-08-19 recalibration (e.g. a hand-built test config).
Same value `configs/default.yaml` states with its full derivation."""


def slice_skr_kbps(pool_config: dict[str, Any]) -> float:
    """Secret-key rate available to *this simulated tenant slice*, in kbps.

    `PoolSim.step()` converts kbps to bits/step by multiplying by 1000
    (one step == one second, `_SECONDS_PER_STEP`), so this is simply
    `refill_bits_per_step / 1000` -- the config states the quantity
    that actually matters (bits of key material per decision epoch)
    and this converts into the unit the trace interface speaks.

    Two accepted spellings, so a `pool:` block from either side of the
    2026-08-19 recalibration resolves:

      * `refill_bits_per_step` (current) -- the slice's share directly.
      * `link_skr_kbps` / `kms_requests_per_decision_epoch` (interim,
        2026-08-19 morning) -- the two-factor form, kept working
        because `tests/test_environment.py`'s extreme-scarcity fixtures
        use it to ask for a deliberately huge refill.

    Sizing this rate is the single most consequential modelling choice
    in the repo -- it is what decides whether QKD is scarce at all --
    so `configs/default.yaml`'s `pool:` block carries the measured
    demand bracket it was chosen inside, and
    `tests/test_pool_sim.py::test_configured_refill_sits_inside_the_
    measured_demand_bracket` pins it there.
    """
    if "link_skr_kbps" in pool_config or "kms_requests_per_decision_epoch" in pool_config:
        link_kbps = float(pool_config.get("link_skr_kbps", 200.0))
        requests_per_epoch = float(pool_config.get("kms_requests_per_decision_epoch", 1000.0))
        if requests_per_epoch <= 0:
            raise ValueError(
                f"kms_requests_per_decision_epoch must be positive, got {requests_per_epoch}"
            )
        return link_kbps / requests_per_epoch

    bits_per_step = float(pool_config.get("refill_bits_per_step", _DEFAULT_REFILL_BITS_PER_STEP))
    if bits_per_step <= 0:
        raise ValueError(f"refill_bits_per_step must be positive, got {bits_per_step}")
    return bits_per_step / 1000.0


@dataclass
class SyntheticSKRQBERTrace:
    """Documented synthetic SKR/QBER trace (PLAN.md "Datasets &
    Provenance" -> QKD pool refill row: "documented synthetic SKR
    process (mean kbps, QBER-driven dips for S3) ... synthetic
    fallback is fully acceptable for a capstone if the generation
    procedure is stated + rate ranges cited"). No real CV-QKD trace
    was sourced for this project; this is that documented fallback.

    Generation procedure:
      1. Baseline SKR per step is drawn i.i.d. Gaussian,
         `skr ~ N(mean_skr_kbps, (skr_noise_frac * mean_skr_kbps)^2)`,
         clipped to `[0, inf)`. `mean_skr_kbps` defaults to 200 kbps,
         loosely in line with published metro-scale CV-QKD field-trial
         secret-key rates (O(10-100s) kbps over tens of km of standard
         fibre) -- the exact figure is not load-bearing, only the
         order of magnitude and the fact that it is stated here.
      2. Baseline QBER per step is drawn i.i.d. Gaussian,
         `qber ~ N(baseline_qber, qber_noise_std^2)`, clipped to
         `[0, 0.999]`.
      3. Optionally, a QBER spike window `[spike_start,
         spike_start + spike_duration)` adds `spike_magnitude` to QBER
         before clipping -- this is the dial-in hook for the S3
         "QKD degradation" scenario (PLAN.md §5 S3: "QBER up, SKR
         down, pool refill collapses").
      4. Within the spike window, SKR for that step is additionally
         scaled by `(1 - min(qber_after_spike, 0.5))`: a higher QBER
         shrinks the fraction of exchanged bits that survive error
         correction / privacy amplification into secret key. This is
         a documented monotonic stand-in, not a fitted physical model.
         If `spike_skr_scale` is set, it *replaces* that stand-in for
         the spike window and scales SKR by exactly that factor
         instead. The stand-in floors at a 50% reduction by
         construction (`min(qber, 0.5)`), which is fine as a generic
         monotonic relationship but cannot express S3's "pool refill
         collapses" (PLAN2 §9) -- in real CV-QKD the secret-key rate
         falls to zero as QBER approaches the security threshold,
         rather than asymptoting at half. Rather than re-fit a
         physical QBER->SKR curve (which would be an invented
         constant dressed up as physics), the collapse depth is an
         explicit, config-stated scenario dial. `None` (the default)
         leaves every pre-existing trace byte-identical.
      5. Draws use `numpy.random.default_rng(seed)`, re-seeded fresh
         each time `__iter__` is called, so the same trace object
         yields an identical sequence every time it is iterated --
         this is what lets `PoolSim.reset()` "rewind the trace"
         deterministically.
    """

    n_steps: int
    mean_skr_kbps: float = 200.0
    skr_noise_frac: float = 0.1
    baseline_qber: float = 0.02
    qber_noise_std: float = 0.005
    spike_start: int | None = None
    spike_duration: int = 0
    spike_magnitude: float = 0.0
    spike_skr_scale: float | None = None
    seed: int = 0

    def __iter__(self) -> Iterator[tuple[float, float]]:
        rng = np.random.default_rng(self.seed)
        for t in range(self.n_steps):
            in_spike = (
                self.spike_start is not None
                and self.spike_start <= t < self.spike_start + self.spike_duration
            )

            qber = float(rng.normal(self.baseline_qber, self.qber_noise_std))
            if in_spike:
                qber += self.spike_magnitude
            qber = float(np.clip(qber, 0.0, 0.999))

            skr = float(rng.normal(self.mean_skr_kbps, self.skr_noise_frac * self.mean_skr_kbps))
            if in_spike:
                if self.spike_skr_scale is not None:
                    skr *= self.spike_skr_scale
                else:
                    skr *= max(0.0, 1.0 - min(qber, 0.5))
            skr = max(0.0, skr)

            yield skr, qber
