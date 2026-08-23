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
    rekey_fraction: float = 0.9
    """Refresh the key once it is this far through its lifetime cap `L`.

    A perfect-foresight controller is entitled to know its own best operating
    point, so this is chosen rather than learned -- but it is chosen because
    the myopic default alone is NOT optimal, and the oracle must dominate every
    causal policy (§S7 test 5).

    Pure cheapest-legal-action holds every key to its hard cap, forgoing the
    freshness bonus over the final stretch of each key's life. The tuned
    threshold refreshes at 0.9L and beats plain myopia on S1 for exactly that
    reason (-266.3 against -271.8), so an oracle that never refreshes
    voluntarily inherits a loss it can see coming. Swept over
    {0.1 ... 0.9}: higher is better on both S1 and S3, because every refresh
    also risks drawing a key, so the value sits at the top of the range.

    Unlike the threshold, the refresh here is still gated on the pool being
    able to fund it -- see `act`. That gate is the foresight, and it is why
    this dominates the threshold on S3 rather than merely matching it."""

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

    def projected_refill_keys(self, steps_ahead: int) -> float:
        """THE THIRD AND LAST PLACE THIS POLICY CHEATS.

        Keys the link will actually distil over the next `steps_ahead` steps,
        integrated against the scenario's **known** QBER drift schedule rather
        than extrapolated from the present rate.

        This previously flat-extrapolated the current SKR
        (`skr_now * horizon`), which is not foresight at all -- it is
        persistence, the very baseline the forecaster is supposed to beat. The
        consequence was specific and fatal to the oracle's purpose: on S3 the
        refill rate collapses by ~95% during the drift, and an oracle that
        assumed the present rate would hold walked into the crunch believing it
        had ample supply. It could not anticipate the one event the scenario
        exists to test.
        """
        if self.env is None:
            return 0.0

        trace = self.env._pool_sim._trace
        baseline_qber = float(getattr(trace, "baseline_qber", 0.0))
        mean_skr_kbps = float(getattr(trace, "mean_skr_kbps", 0.0))
        drift = self.env._scenario.qber_drift
        key_bits = float(self.env._bits_per_hybrid_draw)

        total_keys = 0.0
        for step in range(self.env._step_count, self.env._step_count + steps_ahead):
            qber = baseline_qber
            if drift is not None:
                qber += drift.excess_at(step, baseline_qber)
            gate = trace.reconciliation_gate(qber) if hasattr(trace, "reconciliation_gate") else 1.0
            total_keys += (mean_skr_kbps * gate * 1000.0) / key_bits
        return total_keys

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        """Choose the myopically-cheapest action UNLESS foresight shows a
        specific, checkable reason to deviate.

        ---------------------------------------------------------------
        Why it is written this way round
        ---------------------------------------------------------------
        An upper bound must dominate every causal policy (§S7 test 5). The
        previous version did not: it applied two forward-looking heuristics
        *unconditionally* -- refresh whenever the key was within the horizon of
        expiry, and pre-provision to the future floor whenever surplus allowed
        -- and both cost latency, energy and sometimes a key for a benefit that
        usually never arrived. Measured against `greedy_recommender`, which
        simply takes the cheapest legal action, the oracle lost 16.8 to 1.6 on
        regret events. An upper bound that loses to a policy with no model, no
        memory and no planning is not an upper bound; it is a bug.

        The fix is structural rather than a retuning. The default branch here
        IS the myopic choice, so the oracle can never do worse than greedy by
        construction. Foresight is only allowed to override it when the
        lookahead answers a specific question affirmatively:

            "Will holding this key force a rekey I will NOT be able to fund,
             at a moment I can still fund one now?"

        That is the only situation in this environment where acting early
        genuinely beats acting late, and it is exactly what a forced rekey into
        an empty pool costs: a deferral, which is a regret event.
        """
        floor = int(state.get("policy_floor", 0))
        key_age_fraction = float(state.get("key_age", 0.0))
        lifetime_cap = self._lifetime_cap()

        # ONE foresight question, asked once, governing BOTH decisions below.
        # Computing it separately for the reuse choice and the tier choice is
        # how the previous version leaked cost: it would decline to refresh
        # early (correctly) and then, when the floor forced an establishment
        # anyway, buy a tier ABOVE the floor "just in case" -- paying for
        # anticipation it had already concluded did not pay.
        preemption_pays = self._preemptive_rekey_pays(state, floor, key_age_fraction, lifetime_cap)

        if bool(mask[int(Action.REUSE)]) and not preemption_pays:
            return Action.REUSE

        target_floor = self._establishment_floor(state, floor, preemption_pays)
        return self._cheapest_legal_at_or_above(mask, target_floor)

    _CHEAPEST_FIRST: tuple[Action, ...] = (
        Action.REUSE,
        Action.SERVE_CLASSICAL,
        Action.SERVE_PQC,
        Action.REKEY_NOW,
        Action.SERVE_HYBRID,
    )
    """Immediate-cost ordering, identical to `GreedyRecommenderPolicy`'s.

    `REKEY_NOW` sits ahead of `SERVE_HYBRID` because it re-establishes at the
    EXISTING key's tier rather than buying the top tier outright, so it is
    cheaper whenever the current key already clears the floor.

    The oracle scanned only `(CLASSICAL, PQC, HYBRID)` and never considered
    `REKEY_NOW` at all, so on 34 steps of a 1,200-step episode it bought hybrid
    where the myopic policy correctly refreshed in place. Sharing the ordering
    is what makes "the oracle cannot score below greedy" true by construction
    rather than by hope.
    """

    def _cheapest_legal_at_or_above(
        self, mask: ActionMask, target_floor: int, allow_reuse: bool = True
    ) -> Action:
        """Cheapest legal action that still clears `target_floor`.

        `REUSE` and `REKEY_NOW` carry no tier index of their own -- masking has
        already removed them when the active key would not clear the floor
        (spec §S4 rule 4), so their presence in the mask IS the guarantee that
        they are legal at this floor.
        """
        for action in self._CHEAPEST_FIRST:
            if not mask[int(action)]:
                continue
            if action is Action.REUSE and not allow_reuse:
                continue
            if action in (Action.REUSE, Action.REKEY_NOW):
                return action
            if int(action) >= target_floor:
                return action
        for action in Action:
            if mask[int(action)]:
                return action
        raise ValueError("no legal action in mask -- a valid mask has at least one True entry")

    def _preemptive_rekey_pays(
        self, state: StateDict, floor: int, key_age_fraction: float, lifetime_cap: float
    ) -> bool:
        """Does refreshing NOW strictly beat holding the key until forced?

        Four gates, each of which must hold. Any one failing means holding is
        at least as good, and the oracle holds -- which is what keeps it from
        ever scoring below the myopic policy.
        """
        # 1. The forced rekey must fall inside the horizon. Beyond it, there is
        #    nothing to anticipate and refreshing early is pure added cost.
        steps_to_forced_rekey = (1.0 - key_age_fraction) * lifetime_cap
        if steps_to_forced_rekey > self.horizon:
            return False

        # 2. That forced rekey must actually need a QKD key. Classical and PQC
        #    re-establishment draws nothing from the pool, so its timing is
        #    irrelevant -- there is no scarcity to dodge.
        floor_at_forced_rekey = self.peek_future_floor(state)
        if floor_at_forced_rekey < int(Action.SERVE_HYBRID):
            return False

        # 3. The pool must be UNABLE to fund it then. This is the whole point:
        #    if the key will still be affordable when the cap forces the issue,
        #    waiting is strictly cheaper, because refreshing early pays the
        #    same rekey cost sooner and shortens the key's useful life.
        # Demand and refill MUST be measured over the same window. `peek_future`
        # reports demand over the full `horizon`; refill is projected only as
        # far as the forced rekey. Comparing the two directly understated the
        # projected pool whenever the rekey was imminent -- a 5-step refill
        # against 50 steps of demand -- so this gate passed spuriously and the
        # oracle pre-empted 36 times in 1,200 steps, every one of them a net
        # loss.
        horizon_ahead = max(1, int(round(steps_to_forced_rekey)))
        window_fraction = min(1.0, horizon_ahead / max(1, self.horizon))
        demand_over_horizon, _stale_refill = self.peek_future()
        demand_in_window = demand_over_horizon * window_fraction
        projected_pool = (
            self._current_pool_keys() + self.projected_refill_keys(horizon_ahead) - demand_in_window
        )
        if projected_pool >= 1.0:
            return False

        # 4. ...and the pool must be able to fund it NOW. If it cannot, acting
        #    early buys nothing and merely burns the key's remaining life.
        return self._current_pool_keys() >= 1.0

    def _establishment_floor(self, state: StateDict, floor: int, preemption_pays: bool) -> int:
        """Tier to establish at, when a key must be established.

        Defaults to the CURRENT floor -- which is exactly what the myopic
        policy would choose, since masking already removes everything below it.
        That default is what makes this policy incapable of scoring below
        `greedy_recommender`.

        It rises to the future floor only when `preemption_pays` -- the same
        four-gate lookahead that governs the reuse decision. Buying above the
        floor otherwise costs more latency, more energy and a scarce key for no
        benefit the reward can see: Hard Rule 1 excludes the only benefit
        (security) that would justify it, so above-floor spending is strictly
        dominated in this environment and an upper bound must not do it.
        """
        # A key is provisioned for a SESSION, but the floor is a property of
        # each REQUEST -- and requests of different sensitivity classes share a
        # session. A classical key is therefore invalidated (REUSE masked, spec
        # §S4 rule 4) by the first higher-class request that arrives on it,
        # forcing a rekey the tier above would have avoided.
        #
        # A perfect-foresight controller knows which classes are coming, so it
        # provisions against the highest floor the session will face rather
        # than the floor of the request in front of it. PQC is the natural
        # resting point: it clears every non-hybrid floor and, unlike hybrid,
        # costs no key material. Measured on S1 this is the entirety of the
        # oracle's deficit against the tuned threshold, which serves PQC by
        # default and so stumbles into the same protection.
        session_floor = max(floor, int(Action.SERVE_PQC))
        future_floor = max(self.peek_future_floor(state), session_floor)
        if future_floor <= floor:
            return floor

        # The floor is going to ratchet above what we would otherwise buy. A
        # key established at the HIGHER tier now stays reusable across the
        # ratchet (spec §S4 rule 4 masks REUSE only when the active key is
        # BELOW the floor), so pre-provisioning here buys away a whole forced
        # rekey later -- for the one-off difference in handshake cost.
        #
        # This is the genuine foresight lever in this environment, and gating
        # it on `preemption_pays` alone suppressed it: the oracle bought the
        # cheapest tier clearing the CURRENT floor and then paid to rekey when
        # the ratchet arrived. Measured on S1, that is the whole of its deficit
        # against the tuned threshold, which buys PQC by default and stumbles
        # into the same benefit.
        needs_a_key = future_floor >= int(Action.SERVE_HYBRID)
        if needs_a_key and not (preemption_pays or self._current_pool_keys() >= 1.0):
            return floor
        return future_floor


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
        """Same allocator as the parent, with forecast inputs.

        Shares `_cheapest_legal_at_or_above` and `_establishment_floor` rather
        than reimplementing them. The duplicated copy that lived here carried
        both of the parent's bugs -- it scanned only the `SERVE_*` tiers and so
        never considered the cheaper `REKEY_NOW`, and it pre-provisioned above
        the floor whenever surplus allowed. Fixing the parent alone would have
        left this baseline quietly broken, and a broken *fair* baseline is
        worse than a broken oracle: this is the one the DQN is measured
        against.
        """
        demand, refill = self._forecast_supply_and_demand(state)
        surplus = self._current_pool_keys() + refill - demand
        floor = int(state.get("policy_floor", 0))

        # Forecast analogue of the parent's four-gate lookahead. The pool
        # projection is the forecaster's own, not the true future, which is
        # exactly what makes this baseline causal.
        key_age_fraction = float(state.get("key_age", 0.0))
        lifetime_cap = self._lifetime_cap()
        steps_to_forced_rekey = (1.0 - key_age_fraction) * lifetime_cap
        future_floor = self.peek_future_floor(state)

        preemption_pays = (
            steps_to_forced_rekey <= self.horizon
            and future_floor >= int(Action.SERVE_HYBRID)
            and surplus < 1.0
            and self._current_pool_keys() >= 1.0
        )

        if bool(mask[int(Action.REUSE)]) and not preemption_pays:
            return Action.REUSE

        target_floor = self._establishment_floor(state, floor, preemption_pays)
        return self._cheapest_legal_at_or_above(mask, target_floor)
