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

Unit convention: the pool counts whole 256-bit ETSI keys (see `PoolSim`);
`PoolState.fill`/`capacity` expose the same quantity in **bits** for the
masking layer and state assembly. `SKRQBERTrace` yields `skr` in **kbps** (citable, kbps-scale, matching
PLAN.md's "refills slowly (kbps ...)"). `PoolSim.step()` converts kbps
to bits-refilled-this-step under the modeling assumption that one
simulator step represents one wall-clock second (`_SECONDS_PER_STEP`
below) -- this is a fixed simulator convention, not a tunable config
value, so it is documented here rather than added to
`configs/default.yaml`.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml


@dataclass
class PoolState:
    """Snapshot of pool physics at one step.

    `fill`/`capacity` are reported in **bits** for compatibility with the
    state assembly and the masking layer, but they are *derived* from the
    integer key counts below -- the pool's own arithmetic is entirely in
    whole 256-bit keys (see `PoolSim`).
    """

    fill: float  # current pool level, bits (== keys * key_bits)
    capacity: float  # max pool capacity, bits
    skr: float  # instantaneous secret-key rate driving refill
    qber: float  # current quantum bit error rate
    keys: int = 0  # current pool level, whole ETSI keys -- the authoritative unit
    capacity_keys: int = 0  # max pool capacity, whole keys
    overflow_keys: int = 0  # keys discarded this step because the pool was full
    expired_keys: int = 0  # keys discarded this step by age-out


@dataclass
class RefillBatch:
    """One step's worth of distilled key material, tracked as a unit so
    draws can be attributed back to the refill that produced them.

    SMARTKEYNET_BUILD_SPEC.md §S1 asks for the pool to be "a
    `collections.deque` of `RefillBatch(batch_id, refill_step,
    keys_remaining)`" with FIFO consumption, which makes both age-out and
    §S2's attribution ledger fall out for free: the oldest batch is always
    at the left end, and every draw reports exactly which batches it ate.
    """

    batch_id: int
    refill_step: int
    keys_remaining: int


@dataclass(frozen=True)
class RefillResult:
    """What one `refill()` did. `overflow_keys` is a *result*, not a
    nuisance: it is the quantum material the link produced and the pool
    was too full to hold, and §S1 calls it "a free extra axis of
    evidence" -- an always-PQC policy wastes the entire link output, and
    a good agent should show near-zero overflow *and* near-zero regret.
    """

    keys_added: int
    overflow_keys: int
    expired_keys: int
    pool_keys_after: int
    skr_kbps: float
    qber: float


@dataclass(frozen=True)
class DrawResult:
    """Outcome of a draw. `lineage` lists `(batch_id, keys_taken)` oldest
    first, which is what `metrics/regret.py` attributes regret against."""

    ok: bool
    lineage: tuple[tuple[int, int], ...] = ()


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

    ---------------------------------------------------------------
    Unit convention: whole keys, with a fractional carry
    ---------------------------------------------------------------
    The pool counts **integer 256-bit ETSI keys**, not bits. This is
    SMARTKEYNET_BUILD_SPEC.md §S1's explicit instruction -- "Work in
    integer 256-bit keys, not bits. Fractional bits create off-by-epsilon
    exhaustion bugs and make the attribution ledger painful" -- and it is
    also the physical truth: half a key cannot establish a session.

    This module held a float `fill` in bits until 2026-08-19. Nothing
    visibly broke, because every draw in this environment happens to be
    exactly one key, but the float representation meant `fill` could sit
    at 255.99999999999997 bits and report "cannot cover a 256-bit draw"
    for reasons no reader could see. It also made the two things §S1 and
    §S2 ask for impossible to express: there were no batches to attribute
    a regret event against, and no notion of a key being too old.

    A step's distilled bits rarely divide evenly into keys, so the
    remainder is banked in `_fractional_carry` and spent on later steps.
    That is what keeps the long-run refill rate *exactly* equal to the
    trace's bit rate -- truncating each step independently would silently
    lose up to one key per step, which at this pool's 0.859 keys/step is
    most of the link's output.

    `capacity` and `initial_fill_frac` are simulator parameters supplied
    by the caller (see `load_pool_config` below, which reads them from
    `configs/default.yaml`'s `pool:` block) -- nothing about their values
    is hardcoded here. `capacity` is accepted in bits, because that is
    how the config expresses it and how the masking layer reads it back.
    """

    _SECONDS_PER_STEP: float = 1.0  # fixed sim convention: 1 step == 1 wall-clock second

    def __init__(
        self,
        capacity: float,
        trace: SKRQBERTrace,
        initial_fill_frac: float,
        key_bits: int = 256,
        max_key_age_steps: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if not 0.0 <= initial_fill_frac <= 1.0:
            raise ValueError(f"initial_fill_frac must be in [0, 1], got {initial_fill_frac}")
        if key_bits <= 0:
            raise ValueError(f"key_bits must be positive, got {key_bits}")

        self.capacity = float(capacity)
        self._key_bits = int(key_bits)
        self.capacity_keys = int(self.capacity // self._key_bits)
        if self.capacity_keys <= 0:
            raise ValueError(
                f"capacity {capacity} bits is smaller than one {key_bits}-bit key -- "
                "the pool could never hold anything"
            )
        self._trace = trace
        self._initial_fill_frac = float(initial_fill_frac)
        self._max_key_age_steps = max_key_age_steps

        self._batches: deque[RefillBatch] = deque()
        self._keys: int = 0
        self._fractional_carry: float = 0.0
        self._next_batch_id: int = 0
        self._now: int = 0
        self._overflow_keys_total: int = 0
        self._expired_keys_total: int = 0
        self._skr: float = 0.0
        self._qber: float = 0.0
        self._iterator: Iterator[tuple[float, float]] | None = None

        self.reset()

    # -----------------------------------------------------------------
    # Read-only views
    # -----------------------------------------------------------------

    @property
    def level(self) -> int:
        """Current pool level in whole keys -- the authoritative unit."""
        return self._keys

    @property
    def fill(self) -> float:
        """Current pool level in bits, derived from `level`.

        Kept because the state assembly and `env/masking.py` were written
        against bits. It is exact (`level * key_bits`), never fractional.
        """
        return float(self._keys * self._key_bits)

    @property
    def fill_fraction(self) -> float:
        return self._keys / self.capacity_keys if self.capacity_keys else 0.0

    @property
    def overflow_keys_total(self) -> int:
        """Keys the link produced that the pool was too full to hold, all
        episode. Reported per §3.3 as `pool_overflow_keys`."""
        return self._overflow_keys_total

    @property
    def expired_keys_total(self) -> int:
        return self._expired_keys_total

    def batches(self) -> tuple[RefillBatch, ...]:
        """Immutable snapshot of the lineage deque, oldest first."""
        return tuple(
            RefillBatch(b.batch_id, b.refill_step, b.keys_remaining) for b in self._batches
        )

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def reset(self) -> PoolState:
        """Reset pool to its initial fill level and rewind the trace."""
        self._iterator = iter(self._trace)
        self._batches = deque()
        self._fractional_carry = 0.0
        self._next_batch_id = 0
        self._now = 0
        self._overflow_keys_total = 0
        self._expired_keys_total = 0
        self._skr = 0.0
        self._qber = 0.0

        initial_keys = int(round(self.capacity_keys * self._initial_fill_frac))
        initial_keys = max(0, min(self.capacity_keys, initial_keys))
        self._keys = initial_keys
        if initial_keys > 0:
            # The starting stock is one batch refilled "before" step 0, so
            # age-out treats it like any other material rather than as
            # immortal seed keys.
            self._batches.append(RefillBatch(self._new_batch_id(), 0, initial_keys))
        return self._state()

    def step(self) -> PoolState:
        """Advance one timestep: pull the next (skr, qber) from the trace
        and refill the pool accordingly."""
        if self._iterator is None:
            raise RuntimeError("PoolSim.step() called before reset()")

        skr_kbps, qber = next(self._iterator)
        self._now += 1
        result = self.refill(float(skr_kbps), self._SECONDS_PER_STEP, self._now, qber=float(qber))
        return self._state(overflow_keys=result.overflow_keys, expired_keys=result.expired_keys)

    def refill(
        self, skr_kbps: float, step_seconds: float, now: int, qber: float | None = None
    ) -> RefillResult:
        """Distil `skr_kbps` into whole keys and add them to the pool.

        Implements §S1's refill block exactly: bank the remainder in the
        fractional carry so no distilled bit is ever lost, clamp to
        capacity, and report the clamped remainder as overflow.
        """
        self._skr = float(skr_kbps)
        if qber is not None:
            self._qber = float(qber)

        expired_keys = self._age_out(now)

        bits_this_step = self._skr * 1000.0 * step_seconds  # kbps -> bits
        keys_available = bits_this_step / self._key_bits + self._fractional_carry
        keys_added_raw = int(keys_available)  # floor; keys_available >= 0 always
        self._fractional_carry = keys_available - keys_added_raw

        space = self.capacity_keys - self._keys
        keys_added = max(0, min(keys_added_raw, space))
        overflow_keys = keys_added_raw - keys_added

        if keys_added > 0:
            self._batches.append(RefillBatch(self._new_batch_id(), now, keys_added))
            self._keys += keys_added
        self._overflow_keys_total += overflow_keys

        return RefillResult(
            keys_added=keys_added,
            overflow_keys=overflow_keys,
            expired_keys=expired_keys,
            pool_keys_after=self._keys,
            skr_kbps=self._skr,
            qber=self._qber,
        )

    # -----------------------------------------------------------------
    # Feasibility and draws
    # -----------------------------------------------------------------

    def peek_can_cover(self, keys: int) -> bool:
        """Pure feasibility check: can the pool cover `keys` right now?

        Used by `env/masking.py` before allowing SERVE_HYBRID and by
        `env/deferral_queue.py` to decide when a queued request can
        finally be served. Must not mutate anything -- §S1 test 10
        checks that by hashing the pool before and after.
        """
        return 0 <= keys <= self._keys

    def can_draw(self, bits: float) -> bool:
        """Bits-denominated feasibility check.

        Compatibility wrapper over `peek_can_cover`. A partial key is
        useless, so a request for any fraction of a key needs a whole one
        -- hence the ceiling.
        """
        if bits < 0:
            return False
        return self.peek_can_cover(self._keys_for_bits(bits))

    def draw_keys(self, keys: int, now: int | None = None) -> DrawResult:
        """Consume `keys` whole keys, oldest batch first.

        Returns the lineage consumed rather than raising, so callers that
        want to branch on feasibility can. `draw()` is the raising
        variant the environment uses.
        """
        if keys < 0:
            raise ValueError(f"draw_keys() keys must be non-negative, got {keys}")
        if not self.peek_can_cover(keys):
            return DrawResult(ok=False)

        lineage: list[tuple[int, int]] = []
        remaining = keys
        while remaining > 0:
            batch = self._batches[0]  # FIFO: oldest material leaves first
            taken = min(remaining, batch.keys_remaining)
            batch.keys_remaining -= taken
            remaining -= taken
            lineage.append((batch.batch_id, taken))
            if batch.keys_remaining == 0:
                self._batches.popleft()
        self._keys -= keys
        return DrawResult(ok=True, lineage=tuple(lineage))

    def draw(self, bits: float) -> None:
        """Consume `bits` from the pool for a SERVE_HYBRID action.

        Raises `PoolExhaustedError` if the pool cannot cover it --
        callers must check `can_draw()` first. Pool exhaustion is handled
        by `env/deferral_queue.py`, never by silently under-drawing
        (Hard Rule 9).
        """
        if bits < 0:
            raise ValueError(f"draw() bits must be non-negative, got {bits}")
        keys = self._keys_for_bits(bits)
        result = self.draw_keys(keys)
        if not result.ok:
            raise PoolExhaustedError(
                f"cannot draw {keys} key(s) ({bits} bits) from pool holding {self._keys} key(s)"
            )
        self._last_lineage = result.lineage

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _keys_for_bits(self, bits: float) -> int:
        """Whole keys needed to cover `bits`. A partial key cannot
        establish a session, so this rounds up."""
        return int(math.ceil(bits / self._key_bits - 1e-9))

    def _age_out(self, now: int) -> int:
        """Discard key material older than `max_key_age_steps`.

        Real key stores do not hold material forever (§S1). Disabled when
        `max_key_age_steps` is None, which is the default so existing
        calibration is unaffected unless a config opts in.
        """
        if self._max_key_age_steps is None:
            return 0
        expired = 0
        while self._batches and now - self._batches[0].refill_step >= self._max_key_age_steps:
            batch = self._batches.popleft()  # FIFO order means the oldest is always leftmost
            expired += batch.keys_remaining
            self._keys -= batch.keys_remaining
        self._expired_keys_total += expired
        return expired

    def _new_batch_id(self) -> int:
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        return batch_id

    def _state(self, overflow_keys: int = 0, expired_keys: int = 0) -> PoolState:
        return PoolState(
            fill=self.fill,
            capacity=self.capacity,
            skr=self._skr,
            qber=self._qber,
            keys=self._keys,
            capacity_keys=self.capacity_keys,
            overflow_keys=overflow_keys,
            expired_keys=expired_keys,
        )


def load_pool_config(path: str | Path | None = None) -> dict[str, float]:
    """Read the `pool:` block out of `configs/default.yaml`.

    Centralizes config access so `capacity_bits`/`initial_fill_frac`
    are never duplicated as literals in test code or (eventually)
    `env/environment.py`. Defaults to the repo's `configs/default.yaml`
    if no path is given.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    pool_config: dict[str, float] = config["pool"]
    return pool_config


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
    with open(path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    qkd_config: dict[str, float] = config["qkd"]
    return qkd_config


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
    ) -> QberDriftSchedule:
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
      1. Baseline SKR follows a **log-space Ornstein-Uhlenbeck process**,
         which is the form SMARTKEYNET_BUILD_SPEC.md §S1 specifies:

             x_{t+1}  = x_t + theta * (mu_x - x_t) * dt
                            + sigma * sqrt(dt) * eps_t,   eps_t ~ N(0,1)
             skr_raw_t = exp(x_t)

         Working in log space keeps the rate strictly positive without a
         clip, and mean reversion makes consecutive steps *correlated* --
         which matters, because a forecaster has nothing to learn from an
         i.i.d. sequence. This process was i.i.d. Gaussian until
         2026-08-19, which quietly removed the only autocorrelation the
         pool head of the forecaster could have exploited.

         See "Choosing `mean_skr_kbps`" below -- it is a *calibrated*
         quantity, and the calibration procedure, not the bare number, is
         what is defensible. Note the log-normal correction applied to
         `mu_x` in `__iter__`: without it `mean_skr_kbps` would not be the
         process mean and the whole scarcity calibration would be
         describing a different link than the one being simulated.
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
    mean_skr_kbps: float = 0.10
    skr_noise_frac: float = 0.1
    skr_ou_theta: float = 0.02
    skr_ou_sigma: float = 0.02
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
        return float(surviving_fraction**self.gate_kappa)

    def _ou_log_mean(self) -> float:
        """`mu_x` for the log-space OU process, log-normal corrected.

        A stationary OU process in log space has variance
        `sigma^2 / (2 * theta)`, so `E[exp(x)] = exp(mu_x + sigma^2/(4*theta))`.
        Taking `mu_x = log(mean_skr_kbps)` literally -- as the spec's formula
        does -- therefore produces a process whose realised mean is *above*
        the configured value by that factor, and `mean_skr_kbps` stops being
        the mean.

        That matters here more than it usually would: this project's entire
        scarcity calibration (§11.2) is expressed as a ratio against the
        refill rate, so a supply figure that silently disagrees with its own
        config key invalidates the calibration. Subtracting the correction
        makes `E[skr] == mean_skr_kbps` exactly.
        """
        stationary_variance = (self.skr_ou_sigma**2) / (2.0 * max(1e-9, self.skr_ou_theta))
        return float(np.log(max(1e-12, self.mean_skr_kbps)) - stationary_variance / 2.0)

    def __iter__(self) -> Iterator[tuple[float, float]]:
        rng = np.random.default_rng(self.seed)
        log_mean = self._ou_log_mean()
        # Start at the stationary mean so episodes do not all begin with the
        # same transient ramp, which would be a spurious signal every policy
        # could exploit.
        log_skr = log_mean
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

            # log-space OU step (dt == 1 simulator step)
            log_skr += self.skr_ou_theta * (log_mean - log_skr) + self.skr_ou_sigma * float(
                rng.normal()
            )
            skr = float(np.exp(log_skr))
            skr *= self.reconciliation_gate(qber)
            skr = max(0.0, skr)

            yield skr, qber


@dataclass
class TraceSKRQBERSource:
    """CV-QKD SKR/QBER trace loaded from CSV, with a train/eval split.

    SMARTKEYNET_BUILD_SPEC.md §S1 specifies two interchangeable SKR sources
    behind one interface: this trace mode and the documented synthetic
    process above. It also specifies the rule that matters most about it:

        "Split the trace: first 70% for training scenarios, last 30%
        reserved for evaluation. Reusing the same trace segment for train
        and eval is a silent leak that a reviewer will find."

    That rule is why this class exists rather than a bare `pandas.read_csv`.
    The split is enforced in the constructor -- `split="train"` cannot see
    the evaluation tail, whatever offset is drawn -- so a leak requires
    deliberately passing the wrong split rather than merely forgetting.

    **Provenance.** No real published CV-QKD trace was sourced for this
    capstone; PLAN.md's "Datasets & Provenance" section explicitly permits a
    documented synthetic fallback, which is what `SyntheticSKRQBERTrace` is
    and what every result in this repo uses. This loader exists so that
    dropping in a real trace is a config change rather than a code change,
    and `scripts/get_data.py` writes a synthetic CSV in the right shape so
    the path is exercised. Do not describe results from this loader as
    coming from a published testbed unless a published testbed CSV is
    actually what was loaded.

    Each episode cycles the trace from a random start offset inside its own
    split, so episodes see different segments without ever crossing the
    boundary.
    """

    trace_path: str | Path
    n_steps: int
    split: str = "train"
    train_fraction: float = 0.70
    seed: int = 0

    _TRAIN_SPLIT: str = "train"
    _EVAL_SPLIT: str = "eval"

    def __post_init__(self) -> None:
        if self.split not in (self._TRAIN_SPLIT, self._EVAL_SPLIT):
            raise ValueError(
                f"split must be {self._TRAIN_SPLIT!r} or {self._EVAL_SPLIT!r}, got {self.split!r}"
            )
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError(f"train_fraction must be in (0, 1), got {self.train_fraction}")

        rows = self._read_rows(Path(self.trace_path))
        if not rows:
            raise ValueError(f"trace {self.trace_path} contains no usable rows")

        boundary = int(len(rows) * self.train_fraction)
        if boundary <= 0 or boundary >= len(rows):
            raise ValueError(
                f"trace {self.trace_path} has only {len(rows)} rows -- too few to split "
                f"{self.train_fraction:.0%}/{1 - self.train_fraction:.0%} without one side "
                "being empty"
            )
        self._segment = rows[:boundary] if self.split == self._TRAIN_SPLIT else rows[boundary:]

    @staticmethod
    def _read_rows(path: Path) -> list[tuple[float, float]]:
        """Parse `skr_kbps,qber` pairs. Kept to the stdlib `csv` module: this
        is the only place the project would otherwise need pandas, and a
        two-column reader is not worth the dependency."""
        import csv

        if not path.exists():
            raise FileNotFoundError(
                f"SKR/QBER trace not found: {path}. Generate a synthetic one in the "
                "expected shape with `python -m data.get_data`, or point "
                "`qkd.trace_path` at a real CV-QKD CSV."
            )
        rows: list[tuple[float, float]] = []
        with open(path, encoding="utf-8", newline="") as handle:
            # Skip `#` provenance headers before the CSV header row. The
            # generated trace carries a five-line banner marking it synthetic,
            # and real published traces routinely carry their own preamble --
            # `DictReader` would otherwise read the first comment as the header.
            uncommented = (line for line in handle if not line.lstrip().startswith("#"))
            for record in csv.DictReader(uncommented):
                try:
                    rows.append((float(record["skr_kbps"]), float(record["qber"])))
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"trace {path} must have numeric `skr_kbps` and `qber` columns"
                    ) from error
        return rows

    @property
    def segment_length(self) -> int:
        return len(self._segment)

    def __iter__(self) -> Iterator[tuple[float, float]]:
        rng = np.random.default_rng(self.seed)
        offset = int(rng.integers(0, len(self._segment)))
        for step in range(self.n_steps):
            yield self._segment[(offset + step) % len(self._segment)]
