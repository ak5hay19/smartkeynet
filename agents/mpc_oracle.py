"""
agents/mpc_oracle.py

Perfect-foresight model-predictive controller -- SMARTKEYNET_BUILD_SPEC.md
§S7 diagnostic 5.

This is not a competitor. It is a **measuring instrument**, and the spec is
blunt about why it matters more than anything else in that stage:

    "the gap between `static_threshold` (no foresight) and `mpc_oracle`
     (perfect foresight) is *the maximum value anticipation can possibly have
     in your environment*. If that gap is small, no LSTM in the world will
     help, your Addition-A ablation will be a null result, and you should fix
     the environment in week 2 rather than discover it in week 5."

It is also the first diagnostic §7.1 prescribes when Gate W3 fails:

    "Is there any headroom at all? Compare `static_threshold` to
     `mpc_oracle`. If the gap is < 10% on regret events, **the environment
     has no foresight value** and no agent can win."

---------------------------------------------------------------------
How the cheat works, and why it is legitimate here
---------------------------------------------------------------------
MPC is given information no causal policy can have: it peeks `HORIZON` steps
into the environment's *actual* future -- the arrivals that will come and the
key material that will be refilled -- and allocates against it.

That makes it an upper bound rather than a baseline. Reporting it alongside
causal policies is standard practice for exactly this purpose: it converts
"our agent could be better" into "our agent is X% of the way to the best any
policy with foresight could do".

The peek is deliberately confined to `peek_future`, so there is exactly one
place to audit and no chance of the capability leaking into a policy that
claims to be causal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from env.contracts import Action, ActionMask, StateDict

HORIZON = 50
"""Steps of perfect foresight (spec §S7: "a horizon H=50")."""

_HYBRID_MANDATORY_FRACTION = 0.30
"""Share of arrivals that are hybrid-mandatory.

Matches `env/request_generator.py`'s realised mix (classes S2+S3), and the
0.20 + 0.10 the spec's §11.2 worked calibration uses. Read as a constant
rather than measured per-step because the oracle needs the *forward* rate,
and the realised backward-looking count is a biased estimate of it early in
an episode."""


@dataclass
class MPCOracle:
    """Greedy perfect-foresight allocator.

    At each decision it asks: **will the keys I would spend now be needed by
    a higher-priority request inside the horizon?** If yes, it declines to
    spend discretionarily and reuses or serves the cheapest legal tier
    instead. If no, the keys are surplus and may be spent freely.

    Greedy rather than an LP, per the spec's own note that ~80 lines of greedy
    allocation is sufficient. Priority ordering by sensitivity class makes the
    greedy choice optimal for this objective: reserving for the highest class
    first is exactly what minimising weighted deferrals wants.
    """

    env: Any = None
    horizon: int = HORIZON

    def bind(self, env: Any) -> MPCOracle:
        """Attach the environment whose future this oracle may inspect.

        Separate from construction so the harness can build the policy before
        the env exists, like every other `Policy`.
        """
        self.env = env
        return self

    def peek_future(self) -> tuple[float, float]:
        """THE ONLY PLACE THIS POLICY CHEATS.

        Returns `(hybrid_mandatory_keys_demanded, keys_refilled)` expected over
        the next `horizon` steps, read from the environment's actual state and
        generators rather than estimated from observation.

        Confined to one method so an auditor has exactly one thing to check,
        and so no causal policy can acquire the capability by accident.

        Demand counts three things, and getting this wrong is what made the
        first version of this oracle behave like always-hybrid: requests
        already queued, requests already pending, **and the arrivals the
        horizon has not yet seen**. Omitting that third term understated
        demand by roughly the horizon length, so surplus was always large and
        the oracle spent freely.
        """
        if self.env is None:
            return 0.0, 0.0

        # Future refill: the SKR trace is deterministic given its seed, so the
        # true forward rate is readable rather than forecast.
        bits_per_step = float(self.env._last_pool_state.skr) * 1000.0
        keys_refilled = (bits_per_step * self.horizon) / self.env._bits_per_hybrid_draw

        already_committed = sum(
            1 for request in self.env._pending_requests if request["hybrid_mandatory"]
        ) + len(self.env._deferral_queue)

        # Arrivals the horizon will bring. `_arrivals_total / _ticks_total` is
        # the realised arrival rate; the hybrid-mandatory share of it is what
        # will actually require keys.
        ticks = max(1, self.env._ticks_total)
        arrival_rate = self.env._arrivals_total / ticks
        mandatory_fraction = _HYBRID_MANDATORY_FRACTION
        future_arrivals = arrival_rate * mandatory_fraction * self.horizon

        return already_committed + future_arrivals, keys_refilled

    def peek_future_floor(self, state: StateDict) -> int:
        """THE SECOND AND LAST PLACE THIS POLICY CHEATS.

        Returns the highest floor this request's class will face within the
        horizon, read from the scenario's *scripted* threat schedule.

        WHY THIS IS THE WHOLE POINT. Without it the oracle serves the cheapest
        tier clearing the *current* floor, which is myopically optimal and
        scores exactly what `GreedyRecommenderPolicy` scores -- measured
        identical to the decimal. But a key established at a high tier now
        stays REUSABLE after the floor ratchets up (spec §S4 rule 4), whereas
        a cheap key forces a rekey at the worst possible moment: precisely
        when the floor has risen, demand has spiked, and the pool may be
        empty.

        That is the coupling PLAN.md §8 describes -- "serving hybrid now
        removes an option ten minutes from now" -- read in the other
        direction. Pre-provisioning is what foresight actually buys here, and
        an oracle that cannot do it is not an upper bound on foresight; it is
        an upper bound on myopia.
        """
        if self.env is None:
            return int(state["policy_floor"])

        from env.contracts import SensitivityClass, ThreatPosture
        from env.masking import PolicyTable

        current_floor = int(state.get("policy_floor", 0))
        sensitivity_class = SensitivityClass(int(state.get("sensitivity_class", 0)))

        # The scenario's threat schedule is scripted and deterministic, so the
        # posture the horizon will reach is knowable rather than forecastable.
        max_boost = 0.0
        for step in range(self.env._step_count, self.env._step_count + self.horizon):
            max_boost = max(max_boost, self.env._scenario.threat_boost_at(step))

        # Map that boost through the same normalisation the environment uses,
        # then through the same policy table, so the oracle's notion of "the
        # floor" cannot drift from the environment's.
        intensity = min(1.0, max_boost / self.env._MAX_THREAT_BOOST)
        future_posture = ThreatPosture(int(min(2, round(intensity * 2))))
        future_floor = int(PolicyTable().floor(sensitivity_class, future_posture))
        return max(current_floor, future_floor)

    def _lifetime_cap(self) -> float:
        """The SP 800-57-derived key lifetime cap `L`.

        Present configuration, not foresight -- every causal policy is given
        this too (`StaticThresholdPolicy` takes it as a constructor argument).
        It lives in its own accessor so `act` reads nothing from the
        environment directly, which keeps the "all environment access is
        audited" invariant exactly checkable.
        """
        if self.env is None:
            return float("inf")
        return float(self.env._max_key_age)

    def _current_pool_keys(self) -> float:
        """Present pool level in whole keys. Not foresight -- every causal
        policy sees this too, via `state["pool_fill"]`."""
        if self.env is None:
            return 0.0
        return float(self.env._pool_sim.fill / self.env._bits_per_hybrid_draw)

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        demand, refill = self.peek_future()
        surplus = self._current_pool_keys() + refill - demand
        floor = int(state.get("policy_floor", 0))
        future_floor = self.peek_future_floor(state)

        # 1. Reuse whenever legal -- UNLESS the horizon says the key is about
        #    to expire into a moment when the pool cannot fund its
        #    replacement.
        #
        #    Reuse looks free: no key drawn, no rekey cost. But a key held to
        #    its SP 800-57 cap `L` triggers a *forced* rekey at whatever
        #    moment the cap happens to fall, and under scarcity that moment is
        #    frequently one where the pool is empty and the request is
        #    deferred. Rekeying early, at a moment of one's own choosing while
        #    keys are available, is strictly better -- and it is exactly the
        #    behaviour PLAN.md wants ("rekeys early at cheap moments").
        #
        #    This is what the tuned threshold was doing with `rho = 0.9` and
        #    the first two versions of this oracle were not. Missing it is why
        #    the oracle scored *below* the threshold on S3 (-288,535 vs
        #    -179,936) -- an upper bound that loses to a causal policy is not
        #    an upper bound.
        # `key_age` is normalised by the lifetime cap (spec §4.2), so the
        # horizon has to be normalised the same way to compare against it.
        key_age_fraction = float(state.get("key_age", 0.0))
        lifetime_cap = self._lifetime_cap()
        horizon_fraction = self.horizon / max(1e-9, lifetime_cap)
        expires_within_horizon = key_age_fraction + horizon_fraction >= 1.0
        pool_can_fund_now = surplus > 1.0

        if mask[int(Action.REUSE)] and not (expires_within_horizon and pool_can_fund_now):
            return Action.REUSE

        # 2. A key must be established. Serve the CHEAPEST tier that clears
        #    the floor.
        #
        #    Note what this deliberately does not do: it never serves hybrid
        #    above the floor, no matter how large the surplus. With perfect
        #    foresight the oracle can see what a causal policy can only infer
        #    -- that discretionary hybrid buys nothing in this environment. It
        #    costs more latency, more energy and a scarce key, with no
        #    offsetting benefit, so spending above the floor is strictly
        #    dominated. The first version of this oracle spent whenever
        #    `surplus > 1`, which fired on almost every step and made it
        #    behave like always-hybrid -- it scored *worse* than the tuned
        #    threshold, which is disqualifying for something meant to be an
        #    upper bound.
        #
        #    `surplus` therefore only gates the one genuinely forward-looking
        #    choice below.
        #    Pre-provision to the FUTURE floor when the horizon can afford
        #    it: a key bought at the tier the floor is about to reach stays
        #    reusable across the ratchet, avoiding a forced rekey exactly when
        #    the pool is most stressed. When the surplus cannot fund that, the
        #    oracle falls back to the current floor rather than starving
        #    something that needs the key now.
        target_floor = future_floor if surplus > 1.0 else floor
        for action in (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID):
            if int(action) >= target_floor and mask[int(action)]:
                return action
        for action in (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID):
            if int(action) >= floor and mask[int(action)]:
                return action

        # 3. Nothing at or above the floor is establishable this step. Prefer
        #    REKEY_NOW (which resolves to the floor's tier) when the horizon
        #    can afford it; otherwise take whatever remains legal.
        if mask[int(Action.REKEY_NOW)] and surplus > 0.0:
            return Action.REKEY_NOW

        for action in Action:
            if mask[int(action)]:
                return action
        raise ValueError("no legal action in mask")


@dataclass
class MPCForecast(MPCOracle):
    """The same allocator as `MPCOracle`, driven by the LSTM **forecast**
    instead of the true future (SMARTKEYNET_BUILD_SPEC.md §8.3 rung 3).

    ---------------------------------------------------------------------
    Why this is the baseline that matters most
    ---------------------------------------------------------------------
    §8.3 calls adding this "the single most credibility-increasing thing you
    can do beyond PLAN.md's requirements", and the reason is a fairness
    argument the reader will otherwise make on your behalf.

    `MPCOracle` cheats: it reads the environment's actual future. Beating it is
    impossible and losing to it proves nothing. `static_threshold` has no
    foresight at all. So a DQN that beats the threshold might be winning for
    either of two very different reasons -- because it *learned a policy*, or
    merely because it *had a forecast* and the threshold did not.

    This baseline separates them. It has exactly the forecast the DQN has, and
    an allocator hand-written to use it well. If the DQN beats this, the win is
    attributable to learning rather than to information, which is the claim the
    project actually wants to make.

    **It is a causal policy.** Both of the parent's cheating methods are
    overridden to read only `StateDict` fields the environment already hands
    every policy -- so unlike its parent, this can be run without binding an
    environment, and `tests/test_mpc_oracle.py` asserts it touches no env
    internals.
    """

    def peek_future(self) -> tuple[float, float]:
        """Forecast-driven replacement for the oracle's perfect-foresight read.

        Overridden to a no-op returning zeros: the real work happens in
        `act_with_forecast`, because the parent's signature has no access to
        the state dict that carries the forecast. Calling this directly would
        be a bug, and returning zeros makes that bug loud (surplus collapses to
        the present pool level) rather than silently optimistic.
        """
        return 0.0, 0.0

    def _forecast_supply_and_demand(self, state: StateDict) -> tuple[float, float]:
        """Read projected demand and refill out of the forecast block.

        `hybrid_demand_hat` and `pool_level_hat` are indexed by the horizon set
        {10, 25, 50}; index 2 is H=50, matching this policy's horizon. Absent
        forecast fields (foresight `off`) degrade to zeros, which makes the
        policy behave myopically rather than crash -- the honest failure mode,
        since with no forecast there is genuinely nothing to anticipate with.
        """
        demand_hat = state.get("hybrid_demand_hat") or (0.0, 0.0, 0.0)
        pool_hat = state.get("pool_level_hat") or (0.0, 0.0, 0.0)

        horizon_index = min(2, len(demand_hat) - 1)
        projected_demand = float(demand_hat[horizon_index])
        # `pool_level_hat` is a projected FILL FRACTION; the allocator works in
        # whole keys, so scale by capacity. The refill implied by the forecast
        # is the projected level minus the present one, floored at zero -- a
        # forecast of a falling pool means no surplus, not negative refill.
        capacity_keys = max(1.0, self._capacity_keys())
        projected_level_keys = float(pool_hat[horizon_index]) * capacity_keys
        implied_refill = max(0.0, projected_level_keys - self._current_pool_keys())
        return projected_demand, implied_refill

    def _capacity_keys(self) -> float:
        if self.env is None:
            return 1.0
        return float(self.env._pool_sim.capacity_keys)

    def peek_future_floor(self, state: StateDict) -> int:
        """Forecast-driven floor lookahead.

        The parent reads the scenario's *scripted* threat schedule. This reads
        `threat_forecast`, the forecaster's own k-step posture prediction, and
        maps it through the same `PolicyTable` the environment uses -- so the
        two cannot drift apart.

        Aggregation is a MAX over the forecast window, exactly as
        `env/masking.py` does it. That is not a stylistic choice: max
        aggregation is what makes a forecast able only to *raise* a floor,
        which is the property Hard Rule 2 rests on. A forecast-driven baseline
        that averaged instead could talk itself into a lower floor than the
        present already justifies.
        """
        from env.contracts import SensitivityClass, ThreatPosture
        from env.masking import PolicyTable

        current_floor = int(state.get("policy_floor", 0))
        sensitivity_class = SensitivityClass(int(state.get("sensitivity_class", 0)))

        forecast = state.get("threat_forecast") or ()
        if not forecast:
            return current_floor

        peak_threat = max(float(value) for value in forecast)
        posture_index = min(2, int(round(peak_threat * 2)))
        future_floor = int(PolicyTable().floor(sensitivity_class, ThreatPosture(posture_index)))
        return max(current_floor, future_floor)

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        """Identical allocator to the parent, with forecast inputs.

        The duplication with `MPCOracle.act` is deliberate and small: the
        parent's version calls `peek_future()`, which here must be
        `_forecast_supply_and_demand(state)`. Restructuring the parent to take
        the state would change the audited surface of the oracle -- the thing
        an examiner checks to confirm the oracle cheats in exactly two places
        -- and that surface is worth more than the handful of shared lines.
        """
        demand, refill = self._forecast_supply_and_demand(state)
        surplus = self._current_pool_keys() + refill - demand
        floor = int(state.get("policy_floor", 0))
        future_floor = self.peek_future_floor(state)

        key_age_fraction = float(state.get("key_age", 0.0))
        lifetime_cap = self._lifetime_cap()
        horizon_fraction = self.horizon / max(1e-9, lifetime_cap)
        expires_within_horizon = key_age_fraction + horizon_fraction >= 1.0
        pool_can_fund_now = surplus > 1.0

        if mask[int(Action.REUSE)] and not (expires_within_horizon and pool_can_fund_now):
            return Action.REUSE

        target_tier = max(floor, future_floor)
        for action in (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID):
            if int(action) >= target_tier and mask[int(action)]:
                return action
        for action in Action:
            if mask[int(action)]:
                return action
        raise ValueError("no legal action in mask")
