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


def load_qkd_config(path: str | Path | None = None) -> dict[str, float]:
    """Read the `qkd:` block out of `configs/default.yaml`.

    Sibling of `load_pool_config`. The `qkd:` block holds the
    calibrated SKR/QBER process parameters that drive
    `SyntheticSKRQBERTrace` -- previously these lived only as dataclass
    field defaults, which made the scarcity calibration invisible to
    anyone reading the config (see that class's "Choosing
    `mean_skr_kbps`" section).
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config["qkd"]


@dataclass(frozen=True)
class QberDriftSchedule:
    """Scripted QBER degradation ramp -- the mechanism behind scenario
    S3 ("QKD degradation": QBER up, SKR down, pool refill collapses;
    PLAN.md §5).

    SMARTKEYNET_BUILD_SPEC.md §S1 specifies S3 as "implemented purely
    as a `drift_t` schedule on QBER -- a ramp from `qber_base` to
    `0.9 * qber_abort` over the middle third of the episode, then
    partial recovery. This gives you a collapse in refill without
    touching pool code." That is exactly this dataclass's job: it
    produces `drift_t`, an additive excess on top of baseline QBER,
    and the SKR collapse follows from the reconciliation gate in
    `SyntheticSKRQBERTrace` rather than from any separate SKR
    schedule.

    Piecewise-linear shape of `drift(t)`, where
    `peak_excess = peak_qber - baseline_qber`:

        t <  ramp_start            -> 0
        ramp_start..ramp_end       -> linear 0 -> peak_excess
        ramp_end..recovery_start   -> peak_excess            (hold at peak)
        recovery_start..recovery_end -> linear peak_excess -> residual_frac*peak_excess
        t >= recovery_end          -> residual_frac * peak_excess   (partial recovery:
                                      the link does not fully return to baseline)

    `for_episode()` builds the shape the spec calls for from an episode
    length.

    WHY THERE IS A HOLD PHASE (the spec does not mention one). Taken
    literally -- ramp linearly across the whole middle third, then
    recover -- the spec's own shape fails the spec's own acceptance
    test. §S1 test 6 requires mean refill in the middle third to fall
    below 30% of the first third, but a linear ramp spends most of that
    window at low QBER: the mean of the reconciliation gate over the
    ramp works out to ~0.45, so refill only falls to ~45%. Splitting
    the middle third into a ramp and a hold at peak brings the window
    mean to ~0.25, which clears the threshold, and it is the more
    realistic shape anyway -- a degraded link stays degraded for a
    while rather than turning around the instant it bottoms out.
    """

    ramp_start: int
    ramp_end: int
    recovery_start: int
    recovery_end: int
    peak_qber: float
    residual_frac: float = 0.25

    @classmethod
    def for_episode(
        cls, n_steps: int, peak_qber: float, residual_frac: float = 0.25
    ) -> "QberDriftSchedule":
        """Build the canonical S3 shape: flat for the first third,
        degraded across the middle third (ramping to `peak_qber` over
        its first half, holding there for its second), then partial
        recovery across the final third."""
        third = max(1, n_steps // 3)
        return cls(
            ramp_start=third,
            ramp_end=third + max(1, third // 2),
            recovery_start=2 * third,
            recovery_end=n_steps,
            peak_qber=peak_qber,
            residual_frac=residual_frac,
        )

    def excess_at(self, t: int, baseline_qber: float) -> float:
        """Additive QBER excess above `baseline_qber` at step `t`.

        Never negative: a drift schedule must not be able to push QBER
        *below* baseline, which would raise SKR and turn S3 into a
        boon rather than a degradation.
        """
        peak_excess = max(0.0, self.peak_qber - baseline_qber)
        if t < self.ramp_start:
            return 0.0
        if t < self.ramp_end:
            span = max(1, self.ramp_end - self.ramp_start)
            return peak_excess * (t - self.ramp_start) / span
        if t < self.recovery_start:
            return peak_excess
        if t < self.recovery_end:
            span = max(1, self.recovery_end - self.recovery_start)
            progress = (t - self.recovery_start) / span
            return peak_excess * (1.0 - (1.0 - self.residual_frac) * progress)
        return peak_excess * self.residual_frac

    def peak_hold_window(self) -> tuple[int, int]:
        """The `[start, end)` window during which drift sits at its
        peak. This is where "does degradation actually bite" should be
        measured -- averaging across the ramp understates it."""
        return self.ramp_end, self.recovery_start


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
         clipped to `[0, inf)`. See "Choosing `mean_skr_kbps`" below --
         it is a *calibrated* quantity, and the calibration procedure,
         not the bare number, is what is defensible.
      2. Baseline QBER per step is drawn i.i.d. Gaussian,
         `qber ~ N(baseline_qber, qber_noise_std^2)`, clipped to
         `[0, 0.999]`.
      3. Two optional QBER perturbations stack additively on top of
         the baseline draw, both before clipping:
           - `drift` (a `QberDriftSchedule`): the S3 degradation ramp.
           - a legacy rectangular spike window `[spike_start,
             spike_start + spike_duration)` adding `spike_magnitude`.
         `drift` is the mechanism SMARTKEYNET_BUILD_SPEC.md §S1
         specifies for S3; the spike window predates it and is kept
         because existing tests and configs use it.
      4. SKR is scaled by the **reconciliation gate** described below,
         which is what turns a QBER rise into a refill collapse.
      5. Draws use `numpy.random.default_rng(seed)`, re-seeded fresh
         each time `__iter__` is called, so the same trace object
         yields an identical sequence every time it is iterated --
         this is what lets `PoolSim.reset()` "rewind the trace"
         deterministically.

    ---------------------------------------------------------------
    The reconciliation gate (step 4 above)
    ---------------------------------------------------------------
    Error correction and privacy amplification only distil a secret
    key from a fraction of the exchanged raw bits, and that fraction
    falls to zero as QBER approaches the reconciliation abort
    threshold `qber_abort` (above which no secret key can be
    extracted at all). SMARTKEYNET_BUILD_SPEC.md §S1 gives the shape:

        gate(q) = clip(1 - q / qber_abort, 0, 1) ** kappa

    This trace uses that gate expressed relative to the link's own
    baseline operating point, i.e. driven by the QBER *excess* rather
    than absolute QBER:

        excess   = max(0, q - baseline_qber)
        headroom = qber_abort - baseline_qber
        gate(q)  = clip(1 - excess / headroom, 0, 1) ** gate_kappa

    Two properties this buys, both deliberate:
      - `gate == 1.0` exactly whenever QBER sits at or below baseline,
        so a scenario with no drift and no spike is unaffected by the
        gate. The gate cannot silently re-scale the S1 baseline.
      - `gate == 0.0` at `q >= qber_abort`, monotonically decreasing
        in between, so SKR is non-increasing in QBER and vanishes at
        the abort threshold -- the two properties the spec's
        `test_qber_gate_monotone` asks for.

    The previous form of this gate (`skr *= 1 - min(qber, 0.5)`,
    applied only inside a spike window) was replaced 2026-08-15: it
    was far too weak to produce the collapse S3 needs. At the spec's
    S3 peak of `0.9 * qber_abort` it removed only ~10% of the SKR,
    where the reconciliation gate removes ~96%.

    ---------------------------------------------------------------
    Choosing `mean_skr_kbps` (read before changing it)
    ---------------------------------------------------------------
    `mean_skr_kbps` is **calibrated, not looked up**. Published CV-QKD
    secret-key rates span many orders of magnitude with distance --
    from Mbps over metro spans down to O(bps) over 200+ km -- so
    quoting any single figure would fix nothing that matters here.
    What matters, and what SMARTKEYNET_BUILD_SPEC.md §S1 test 11 and
    §11.2 both insist on, is the dimensionless **scarcity ratio**

        rho = keys demanded per step / keys refilled per step

    which must land in `[0.8, 1.3]` on S1/S2 so that the pool actually
    binds. `rho << 0.8` means the pool never binds, every policy looks
    identical, and the DQN ties the tuned threshold baseline -- which
    is exactly what this repo measured on 2026-08-15 (`rho = 0.0013`,
    pool pinned at 100% full for 1999 of 2000 steps, zero regret
    events ever recorded). The default here is the value that puts
    this environment's always-hybrid demand at `rho ~ 1.1`; see
    `configs/default.yaml`'s `qkd:` block for the worked arithmetic
    and `tests/test_pool_sim.py::test_scarcity_ratio_in_target_band`
    for the guard that keeps it there.
    """

    n_steps: int
    mean_skr_kbps: float = 0.22
    skr_noise_frac: float = 0.1
    baseline_qber: float = 0.02
    qber_noise_std: float = 0.005
    qber_abort: float = 0.11
    gate_kappa: float = 1.5
    drift: QberDriftSchedule | None = None
    spike_start: int | None = None
    spike_duration: int = 0
    spike_magnitude: float = 0.0
    seed: int = 0

    def reconciliation_gate(self, qber: float) -> float:
        """Fraction of raw SKR surviving reconciliation at this QBER.

        Monotonically non-increasing in `qber`; exactly 1.0 at or
        below `baseline_qber`; exactly 0.0 at or above `qber_abort`.
        See the class docstring for the derivation.
        """
        headroom = self.qber_abort - self.baseline_qber
        if headroom <= 0.0:
            return 0.0 if qber >= self.qber_abort else 1.0
        excess = max(0.0, qber - self.baseline_qber)
        surviving_fraction = float(np.clip(1.0 - excess / headroom, 0.0, 1.0))
        return surviving_fraction**self.gate_kappa

    def __iter__(self) -> Iterator[tuple[float, float]]:
        rng = np.random.default_rng(self.seed)
        for t in range(self.n_steps):
            in_spike = (
                self.spike_start is not None
                and self.spike_start <= t < self.spike_start + self.spike_duration
            )

            qber = float(rng.normal(self.baseline_qber, self.qber_noise_std))
            if self.drift is not None:
                qber += self.drift.excess_at(t, self.baseline_qber)
            if in_spike:
                qber += self.spike_magnitude
            qber = float(np.clip(qber, 0.0, 0.999))

            skr = float(rng.normal(self.mean_skr_kbps, self.skr_noise_frac * self.mean_skr_kbps))
            skr *= self.reconciliation_gate(qber)
            skr = max(0.0, skr)

            yield skr, qber
