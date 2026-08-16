"""
env/scenarios.py

Scenario dispatch: turns `config["scenario"]` (S1..S6, PLAN.md §5) into
the concrete, exogenous perturbations `env/environment.py` applies to an
episode.

Before this module existed, `config["scenario"]` was read but never
acted on -- only S1 (the benign baseline) was ever actually simulated,
which is why PROGRESS.md recorded Gate W3 as "not attemptable for real
-- S3 doesn't exist as a scenario yet".

---------------------------------------------------------------------
Design: scenarios are DATA, not code paths
---------------------------------------------------------------------
`build_scenario()` returns a frozen `ScenarioSpec` describing three
exogenous perturbations, and `env/environment.py` reads that one object.
No scenario gets its own branch in `step()`, and no scenario is allowed
to touch agent code. This is what keeps Hard Rule 3 ("one agent, one
MDP") true as scenarios multiply: S1 and S6 are the same MDP with
different exogenous inputs, exactly as PLAN.md's "The migration wave is
a **scripted, exogenous schedule**" requires.

The three perturbation channels, and the module that consumes each:

  1. `qber_drift`   -> env/pool_sim.py's SyntheticSKRQBERTrace   (S3)
  2. `tenant_flood` -> env/request_generator.py's RequestGenerator (S4)
  3. `threat_windows` -> the forecaster's threat-feature input     (S2)

---------------------------------------------------------------------
Hard Rule 2 note on `threat_windows`
---------------------------------------------------------------------
A threat window raises the *input* to the forecaster's threat head. It
can therefore only push the derived posture up, and `PolicyTable`'s
sticky ratchet (env/masking.py) means a posture increase never reverses
within an episode. Nothing here can lower a floor, and there is
deliberately no negative-intensity window: `ThreatWindow.intensity` is
validated non-negative at construction.

This is also the injection point scenario S5 (the steering attack) will
later use -- an adversarial trace is just a different threat-feature
signal arriving through the same channel. That the channel can only
raise floors is precisely the property S5 exists to demonstrate.

---------------------------------------------------------------------
Hard Rule 8: S5 and S6 are eval-only
---------------------------------------------------------------------
`ScenarioSpec.eval_only` marks them, and `require_trainable()` raises
`ScenarioError` when a training entry point is handed one. Training on
the S6 migration schedule would be memorising the timeline, which is
what makes the held-out evaluation meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from env.contracts import Action
from env.pool_sim import QberDriftSchedule
from env.request_generator import TenantFlood


class ScenarioError(Exception):
    """Raised for an unknown scenario name, or when an eval-only
    scenario is used for training (Hard Rule 8)."""


@dataclass(frozen=True)
class ThreatWindow:
    """A scripted window of elevated threat signal (scenario S2).

    `intensity` is an additive term on the raw threat-feature vector
    handed to the forecaster, in the same units as those features. It
    must be non-negative -- see the module docstring's Hard Rule 2
    note.

    ---------------------------------------------------------------
    `ramp_steps`: why the escalation is gradual (added 2026-08-15)
    ---------------------------------------------------------------
    These windows used to be **rectangular**: the signal jumped 0 -> 3.0
    in a single step and back again, four transitions across a
    2,500-step episode. That is not a forecasting problem, it is a
    surprise -- and because absolute episode time is deliberately
    excluded from the state (so the agent cannot memorise the S6
    timeline), *nothing observable predicted the onset*.

    Two of this project's null results trace to that one fact:

      * The LSTM threat head scored 0.8719 -- exactly the majority-class
        rate. It was not underfitting; there was no signal to fit.
      * The E-A foresight ablation came out flat, and Gate W3 failed to
        a static rule. With no way to anticipate an escalation, there is
        nothing for anticipation to be worth, and a threshold is
        genuinely optimal.

    Real threat escalation is not a step function. Reconnaissance
    precedes exploitation; indicators build before an incident. A
    forecaster is only a sensible component *because* that is true, so
    modelling escalation as instantaneous quietly removed the premise
    of the whole forecasting story. Ramping the signal in over
    `ramp_steps` restores it: the build-up is observable, so a
    forecaster has something real to learn and an agent has time to
    pre-provision key material before floors ratchet up.

    This raises the floor no faster than the rectangular version did --
    the ramp only *delays* reaching full intensity, so protection still
    arrives no later than the underlying schedule dictates, and the
    Hard Rule 2 non-negativity guarantee is unchanged.
    """

    start_step: int
    end_step: int
    intensity: float
    ramp_steps: int = 120

    def __post_init__(self) -> None:
        if self.intensity < 0.0:
            raise ValueError(
                f"ThreatWindow.intensity must be non-negative (Hard Rule 2: threat "
                f"signals may only raise floors), got {self.intensity}"
            )
        if self.end_step <= self.start_step:
            raise ValueError(f"empty ThreatWindow: [{self.start_step}, {self.end_step})")
        if self.ramp_steps < 0:
            raise ValueError(f"ramp_steps must be non-negative, got {self.ramp_steps}")

    def contains(self, step: int) -> bool:
        return self.start_step <= step < self.end_step

    def intensity_at(self, step: int) -> float:
        """Signal contributed at `step`, ramping in and out.

        Never negative, and never above `intensity`, so this cannot
        widen the Hard Rule 2 guarantee in either direction.
        """
        if not self.contains(step):
            return 0.0
        if self.ramp_steps == 0:
            return self.intensity

        into_window = step - self.start_step
        out_of_window = self.end_step - step
        progress = min(into_window, out_of_window) / self.ramp_steps
        return self.intensity * min(1.0, max(0.0, progress))


@dataclass(frozen=True)
class FloorChange:
    """One scripted, timed floor increase for a tenant cohort (S6).

    PLAN.md §5 S6: "Scripted CNSA-2.0-style timeline ratchets tenant
    cohorts' floors in phases". Hard Rule 3 is what makes this a
    *dataclass* and not an action: "The migration wave is a **scripted,
    exogenous schedule** (a config file of timed floor changes). The
    agent never chooses migration order."
    """

    step: int
    tenant_cohort: str
    new_floor: Action


@dataclass(frozen=True)
class ScenarioSpec:
    """The complete exogenous description of one scenario."""

    name: str
    eval_only: bool
    qber_drift: QberDriftSchedule | None = None
    tenant_flood: TenantFlood | None = None
    threat_windows: tuple[ThreatWindow, ...] = ()
    migration_schedule: tuple[FloorChange, ...] = ()

    def floor_overrides_at(self, step: int) -> dict[str, Action]:
        """Cohort -> floor, for every schedule entry effective by `step`.

        Later entries win, and the schedule is validated to be
        non-decreasing per cohort, so this can only ever raise a floor
        as the episode advances.
        """
        overrides: dict[str, Action] = {}
        for change in self.migration_schedule:
            if change.step <= step:
                overrides[change.tenant_cohort] = change.new_floor
        return overrides

    def threat_boost_at(self, step: int) -> float:
        """Additive threat-feature signal at `step`.

        Windows are additive when they overlap, and the result is
        clamped at zero from below by `ThreatWindow`'s own validation,
        so this can never return a negative boost.
        """
        return sum(window.intensity_at(step) for window in self.threat_windows)


# ---------------------------------------------------------------------------
# Scenario parameters
#
# These are simulation *scenario* parameters -- how hard to shake the
# environment -- not security constants, so Hard Rule 4's citation
# requirement does not apply to them. Where SMARTKEYNET_BUILD_SPEC.md
# §S6's scenario table gives a value, that value is used and cited in
# the comment.
# ---------------------------------------------------------------------------

_S4_FLOOD_TENANT = "telemetry"
"""S4's noisy neighbour. Must be a low-sensitivity tenant, so the flood
is *not* itself hybrid-mandatory demand -- the question S4 asks is
whether a critical tenant's pool share survives a neighbour's load,
not whether the pool survives a burst of critical requests (that is
S3's job). `telemetry` carries no S2/S3 flows at all under
`env/request_generator.py`'s profiles, which makes it the right target.
"""

_S4_FLOOD_START = 600
_S4_FLOOD_END = 1200
_S4_FLOOD_MULTIPLIER = 50.0
"""Spec §S6 scenario table: "one telemetry tenant's `traffic_rate` x 50
for steps 600-1200"."""

_S2_THREAT_WINDOWS = (
    (500, 1100, 3.0),
    (1400, 2100, 8.0),
)
"""S2 (HNDL posture) elevation windows as `(start, end, intensity)`.

The two intensities are calibrated to step the posture up one level at
a time, so S2 exercises the full **CALM -> ELEVATED -> HIGH**
progression rather than jumping straight to the top. Measured against
`MovingAverageForecaster` at the environment's typical feature scale
(QBER ~0.02, normalised load ~0.1), once the EWMA has settled:

    boost 0.0 -> threat_score 0.02 -> CALM      (the S1 baseline)
    boost 3.0 -> threat_score 0.48 -> ELEVATED  (first window)
    boost 8.0 -> threat_score 0.88 -> HIGH      (second window)

`PolicyTable`'s sticky ratchet means the posture never falls back when
a window ends -- the floors reached stay reached for the rest of the
episode, which is the intended Hard Rule 2 behaviour and is asserted in
`tests/test_scenarios.py`.

Recalibrated 2026-08-15 alongside the fix to
`env/forecast_provider.py`'s squashing function. Before that fix the
raw threat features (all non-negative) went through a plain sigmoid
that could never return below 0.5, making `ThreatPosture.CALM`
unreachable and pinning even the benign S1 baseline at ELEVATED.
"""


_DEFAULT_MIGRATION_SCHEDULE = (
    {"step": 500, "tenant_cohort": "fintech", "new_floor": "SERVE_PQC"},
    {"step": 1000, "tenant_cohort": "hospital", "new_floor": "SERVE_PQC"},
    {"step": 1500, "tenant_cohort": "hospital", "new_floor": "SERVE_HYBRID"},
    {"step": 2000, "tenant_cohort": "fintech", "new_floor": "SERVE_HYBRID"},
)
"""The staged CNSA-2.0-style timeline S6 replays (PLAN.md §5 S6, §3).

Used when `configs/default.yaml`'s `migration_schedule:` is empty.
Phases ratchet cohort by cohort rather than all at once, which is the
point: the agent has never trained on this schedule (Hard Rule 8) and
has to keep budgeting as hybrid-mandatory demand arrives in waves.

Hospital before fintech, and to a higher final floor, reflects PLAN.md's
framing that data lifetime drives urgency -- patient records outlive
transaction records by decades, so they migrate first and further.
"""


def _assert_schedule_only_ratchets_up(schedule: tuple[FloorChange, ...]) -> None:
    """A migration schedule may only raise floors (Hard Rule 2).

    A schedule that lowered a cohort's floor mid-episode would be a
    downgrade dressed up as config -- exactly the thing the whole
    architecture exists to make impossible. Rejected at build time
    rather than trusted.
    """
    highest_so_far: dict[str, int] = {}
    for change in sorted(schedule, key=lambda entry: entry.step):
        previous = highest_so_far.get(change.tenant_cohort, -1)
        if int(change.new_floor) < previous:
            raise ScenarioError(
                f"migration schedule lowers {change.tenant_cohort}'s floor at step "
                f"{change.step} ({Action(previous).name} -> {change.new_floor.name}). "
                "Floors may only ratchet up (Hard Rule 2)."
            )
        highest_so_far[change.tenant_cohort] = int(change.new_floor)


def build_scenario(
    name: str, config: dict[str, Any], episode_steps: int
) -> ScenarioSpec:
    """Build the `ScenarioSpec` for `name`.

    `episode_steps` sizes step-indexed schedules that are defined
    relative to episode length -- S3's drift ramp occupies the middle
    third of the episode, per SMARTKEYNET_BUILD_SPEC.md §S1.

    Raises `ScenarioError` for an unrecognised name; the environment
    should never silently fall back to S1, because a typo'd scenario
    name that silently ran the baseline would invalidate results
    without any visible failure.
    """
    normalised = str(name).strip().upper()
    qkd_config = config.get("qkd", {})

    if normalised == "S1":
        return ScenarioSpec(name="S1", eval_only=False)

    if normalised == "S2":
        windows = tuple(
            ThreatWindow(start_step=start, end_step=end, intensity=intensity)
            for start, end, intensity in _S2_THREAT_WINDOWS
        )
        return ScenarioSpec(name="S2", eval_only=False, threat_windows=windows)

    if normalised == "S3":
        qber_abort = float(qkd_config.get("qber_abort", 0.11))
        peak_frac = float(qkd_config.get("s3_peak_qber_frac", 0.9))
        residual_frac = float(qkd_config.get("s3_residual_frac", 0.25))
        drift = QberDriftSchedule.for_episode(
            n_steps=episode_steps,
            peak_qber=peak_frac * qber_abort,
            residual_frac=residual_frac,
        )
        return ScenarioSpec(name="S3", eval_only=False, qber_drift=drift)

    if normalised == "S4":
        flood = TenantFlood(
            tenant=_S4_FLOOD_TENANT,
            start_step=_S4_FLOOD_START,
            end_step=_S4_FLOOD_END,
            rate_multiplier=_S4_FLOOD_MULTIPLIER,
        )
        return ScenarioSpec(name="S4", eval_only=False, tenant_flood=flood)

    if normalised == "S5":
        # Eval-only. The adversarial trace itself lives in attack/
        # (not built yet); this entry exists so the eval-only guard and
        # the config surface are in place, and so a training run that
        # names S5 fails loudly rather than silently training on it.
        return ScenarioSpec(name="S5", eval_only=True)

    if normalised == "S6":
        # Eval-only (Hard Rule 8). The migration schedule is a scripted,
        # exogenous config file -- never agent-controlled (Hard Rule 3).
        raw_schedule = config.get("migration_schedule") or _DEFAULT_MIGRATION_SCHEDULE
        schedule = tuple(
            FloorChange(
                step=int(entry["step"]),
                tenant_cohort=str(entry["tenant_cohort"]),
                new_floor=Action[str(entry["new_floor"])],
            )
            for entry in raw_schedule
        )
        _assert_schedule_only_ratchets_up(schedule)
        return ScenarioSpec(name="S6", eval_only=True, migration_schedule=schedule)

    raise ScenarioError(
        f"unknown scenario {name!r} -- expected one of S1, S2, S3, S4, S5, S6 "
        "(PLAN.md §5). Refusing to fall back to S1 silently."
    )


def require_trainable(spec: ScenarioSpec) -> ScenarioSpec:
    """Guard for training entry points (Hard Rule 8).

    SMARTKEYNET_BUILD_SPEC.md §2.4 asks the config loader to "refuse to
    build a training run whose scenario list contains any scenario with
    `eval_only: true`". Call this from any code path that trains.
    """
    if spec.eval_only:
        raise ScenarioError(
            f"scenario {spec.name} is evaluation-only and must not be trained on "
            "(Hard Rule 8: training on the held-out schedule means memorising the "
            "timeline, and the experiment then proves nothing)"
        )
    return spec
