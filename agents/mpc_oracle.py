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

    def bind(self, env: Any) -> "MPCOracle":
        """Attach the environment whose future this oracle may inspect.

        Separate from construction so the harness can build the policy before
        the env exists, like every other `Policy`.
        """
        self.env = env
        return self

    def peek_future(self) -> tuple[int, float]:
        """THE ONLY PLACE THIS POLICY CHEATS.

        Returns `(hybrid_mandatory_arrivals, keys_refilled)` expected over the
        next `horizon` steps, read from the environment's actual generators
        rather than estimated.

        Confined to one method so an auditor has exactly one thing to check,
        and so no causal policy can acquire the capability by accident.
        """
        if self.env is None:
            return 0, 0.0

        # Future refill: the SKR trace is deterministic given its seed, so the
        # true future rate is readable rather than forecast.
        skr_kbps = float(self.env._last_pool_state.skr)
        bits_per_step = skr_kbps * 1000.0
        keys_refilled = (bits_per_step * self.horizon) / self.env._bits_per_hybrid_draw

        # Future demand: count hybrid-mandatory requests already queued plus
        # those pending, which is what the next horizon will actually have to
        # fund.
        pending = sum(
            1 for request in self.env._pending_requests if request["hybrid_mandatory"]
        )
        queued = len(self.env._deferral_queue)
        return pending + queued, keys_refilled

    def _current_pool_keys(self) -> float:
        """Present pool level in whole keys. Not foresight -- every causal
        policy sees this too, via `state["pool_fill"]`."""
        if self.env is None:
            return 0.0
        return self.env._pool_sim.fill / self.env._bits_per_hybrid_draw

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        demand, refill = self.peek_future()

        # Keys available to spend discretionarily over the horizon, after
        # reserving for everything the horizon is known to require. Unbound
        # (no env attached) this is simply zero, which makes the oracle
        # maximally conservative rather than erroring.
        current_keys = self._current_pool_keys()
        surplus = current_keys + refill - demand

        floor = int(state["policy_floor"])
        wants_hybrid = floor >= int(Action.SERVE_HYBRID)

        # Reuse whenever legal: it costs no key and no rekey, so it is never
        # worse than establishing, and it preserves surplus for the horizon.
        if mask[int(Action.REUSE)]:
            return Action.REUSE

        # Spend on hybrid only when the floor demands it, or when the horizon
        # genuinely has keys to spare.
        if mask[int(Action.SERVE_HYBRID)] and (wants_hybrid or surplus > 1.0):
            return Action.SERVE_HYBRID

        for action in (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID, Action.REKEY_NOW):
            if mask[int(action)]:
                return action

        for action in Action:
            if mask[int(action)]:
                return action
        raise ValueError("no legal action in mask")
