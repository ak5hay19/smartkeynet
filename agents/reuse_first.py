"""
agents/reuse_first.py

Reuse-first heuristic baseline (camera-ready baselines session,
2026-11-16): at every decision take `REUSE` if the action mask says it
is legal, otherwise fall back to the lowest legal action in `Action`'s
fixed enum order -- the cheapest tier that still clears the floor,
exactly the fallback `agents/baselines.py` uses for its own policies.

Why it exists: the paper says a reuse-first heuristic "could close much
of the gap" between the static-threshold baseline and the masked DQN.
This is that heuristic, written down so the claim is measured rather
than asserted (see `docs/baselines_remeasured.md`).

Security is untouched, by construction (Hard Rule 2): this policy never
looks at the floor, the threat posture, or anything but the mask it is
handed, and it can only ever return an action whose mask entry is
`True`. `env/masking.py::compute_mask` is what forbids a `REUSE` that
would deliver key material below the current floor, so reuse-first
inherits that guarantee instead of re-implementing it.

Deliberately a NEW file: `agents/baselines.py` (the Hard-Rule-7
baselines) is not modified. The fallback is re-stated here rather than
imported from that module's private helper, so the two files stay
independently readable.
"""

from __future__ import annotations

from env.contracts import Action, ActionMask, StateDict


class ReuseFirstPolicy:
    """`REUSE` whenever it is legal, else the lowest legal action."""

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        if mask[int(Action.REUSE)]:
            return Action.REUSE
        for action in Action:
            if mask[int(action)]:
                return action
        raise ValueError("no legal action in mask -- a valid mask must have at least one True entry")
