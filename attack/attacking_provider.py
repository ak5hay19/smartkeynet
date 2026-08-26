"""
attack/attacking_provider.py

AttackingForecastProvider -- the live, drop-in `ForecastProvider`
implementation for the steering attack (PLAN.md §5 S5; PLAN2.md §7.5
Panel 5; paper draft equation 7, already implemented in
`attack/trace_generator.py`). Owned by Person D (split.md §1).

Wraps an existing `ForecastProvider` (`base_provider`) and shapes every
incoming window via `attack.trace_generator.generate_adversarial_window`
before forwarding it -- satisfies the `ForecastProvider` interface
frozen in `env/contracts.py` exactly, so it is a genuine drop-in
replacement for `env/forecast_provider.py::MovingAverageForecaster`,
same as that class is for a not-yet-built `LSTMForecastProvider`.
Injected via `env/environment.py`'s new `forecast_provider_factory`
constructor parameter (design decision 17) -- zero other changes
needed to `env/environment.py`, `env/masking.py`, or the agents
(this session's Hard Rule 3).

Only `ForecastObservation["threat_features"]` is shaped -- `qber`,
`skr`, `pool_fill`, `arrivals_per_class`, and `hybrid_serves` all pass
through to the wrapped `base_provider` unchanged. Equation 7 is
specifically about shaping the threat-feature window `xt` fed to the
threat forecaster; PLAN.md Addition A's pool head must never be
influenced by a threat-steering signal in either direction (the same
Hard Rule 2 spirit that already keeps `PoolForecast` out of
`env/masking.py`'s floor computation) -- so this class shapes only the
one field equation 7 actually describes, not every field flowing
through `update()`.

Dual-tracking (Part 1's measurement requirement, PLAN.md's paper draft
equation 4 -- V(pi), below-floor service rate measured against TRUE
posture, not estimated posture): an optional `shadow_provider`
receives the TRUE, UNSHAPED observation on every `update()` call,
purely for later measurement. `shadow_provider`'s own output is never
read by this class and never influences `get_threat_forecast()`/
`get_pool_forecast()` below, which always report the wrapped
`base_provider`'s result on the SHAPED input -- i.e. exactly what the
live episode (and the agent making decisions in it) actually saw and
acted on. `update()` is the only place in the whole codebase ever
handed both the true window (as its own argument) and a place to fork
it: `env/environment.py` never exposes a raw `ForecastObservation` to
any caller outside its own `_advance_to_next_decision` loop, so this
is the only correct interception point for a parallel, measurement-only
true-posture track -- not a design convenience, a structural necessity
given Hard Rule 3 (this session: zero changes to `env/environment.py`
beyond the narrow, signed-off `forecast_provider_factory` addition).
"""

from __future__ import annotations

from attack.trace_generator import generate_adversarial_window
from env.contracts import ForecastObservation, ForecastProvider, PoolForecast, ThreatForecast


class AttackingForecastProvider(ForecastProvider):
    """Drop-in `ForecastProvider` implementing the equation-7 steering
    attack live, against whatever `base_provider` instance it wraps.

    `alpha=0.0` makes every method here behave identically to calling
    `base_provider` directly (unshaped window in, same output out) --
    verified by `tests/test_attacking_provider.py`'s exact-equality
    test. `alpha=1.0` fully replaces the true window with
    `attack.trace_generator.g`'s benign-region target every step.
    """

    def __init__(
        self,
        base_provider: ForecastProvider,
        alpha: float,
        shadow_provider: ForecastProvider | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._base_provider = base_provider
        self._alpha = alpha
        self._shadow_provider = shadow_provider

    def update(self, observation: ForecastObservation) -> None:
        shaped_features = generate_adversarial_window(observation["threat_features"], self._alpha)
        shaped_observation = ForecastObservation(
            qber=observation["qber"],
            skr=observation["skr"],
            pool_fill=observation["pool_fill"],
            arrivals_per_class=observation["arrivals_per_class"],
            hybrid_serves=observation["hybrid_serves"],
            threat_features=shaped_features,
        )
        self._base_provider.update(shaped_observation)

        if self._shadow_provider is not None:
            # The TRUE, unshaped observation -- measurement only, never
            # read by get_threat_forecast()/get_pool_forecast() below.
            self._shadow_provider.update(observation)

    def get_threat_forecast(self) -> ThreatForecast:
        return self._base_provider.get_threat_forecast()

    def get_pool_forecast(self) -> PoolForecast:
        return self._base_provider.get_pool_forecast()

    def get_true_threat_forecast(self) -> ThreatForecast:
        """The TRUE-posture-tracked threat forecast -- what
        `shadow_provider` (fed the unshaped window every step) reports.
        Not part of the `ForecastProvider` interface; a measurement-only
        extension. Raises if no `shadow_provider` was given, rather
        than silently returning the attacked forecast, since that would
        make a caller's dual-tracking measurement silently wrong."""
        if self._shadow_provider is None:
            raise ValueError(
                "no shadow_provider was given to this AttackingForecastProvider -- "
                "dual-tracking (true vs. attacked posture) is unavailable"
            )
        return self._shadow_provider.get_threat_forecast()
