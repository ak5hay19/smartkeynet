"""
dashboard/render_living_system_demo.py

Driver for `dashboard/render_living_system.py`: builds the real
`build_tenant_graph()`-backed tenant graph, runs a genuine S2 (HNDL
posture) episode through `SmartKeyNetEnv` under `StaticThresholdPolicy`
(the same established baseline `render_explain_demo.py` uses) with that
real graph injected via `request_stream_factory` (the same swap-test
injection point `tests/test_environment.py::
test_hard_rule_3_graph_driven_generator_is_a_drop_in_replacement`
already exercises, Hard Rule 3), then renders 3 real static snapshots
to `dashboard/samples/`.

**Why S2, not S1 (design decision, argued per instruction):** S2's
scripted `threat_schedule` (`elevate_at_step=50`) is the scenario whose
real threat/posture is designed to change mid-episode -- an S1
episode's floor never moves at all, so its snapshots would show
nearly-identical graphs, a much weaker demo opener.

**A real finding along the way, investigated and reported honestly
(Hard Rule 7), not glossed over:** on this graph-driven request stream,
S2's scripted `elevate_at_step=50` schedule turns out NOT to be what
actually first raises the floor. `PolicyTable`'s one-way ratchet
(`ratchet_up`) fires within the first 1-2 real decisions of every seed
checked (0, 1, 2, 3) -- long before step 50 -- because `env/
environment.py::_threat_features_placeholder`'s ordinary, pre-elevation
signal is `[qber, load]`, and `load` (queue backlog / 10) climbs fast
enough under this graph's real per-tenant arrival pattern that
`MovingAverageForecaster`'s `sigmoid(mean([qber, load]))` EWMA crosses
into ELEVATED-classified territory almost immediately -- a property of
the placeholder threat-feature formula mixing an ordinary congestion
metric into what's meant to represent threat, not a bug in this
session's own code, and not specific to the tenant graph (verified: the
same effect reproduces with `random_request_generator` too). Once
ratcheted, posture never lowers (Hard Rule 2), so for this real run the
scripted schedule is superseded by an earlier, organic escalation --
worth flagging for whoever next revisits `forecast_provider.py`'s
placeholder formula (see the standing `env/forecast_provider.py` open
item in PROGRESS.md), not fixed here (out of this session's scope,
would need sign-off). Because of this, `pick_snapshot_indices` below
does NOT assume the ratchet aligns with `elevate_at_step` -- it selects
real milestones instead (see its own docstring).

**Why static snapshots, not live animation (per the session's explicit
brief, restated here for the record):** a demo needs a clear, truthful
picture of real decisions, not a real-time simulation. Three snapshots
taken at different real, honestly-selected points in one real episode
(not evenly-spaced/arbitrary indices -- see `pick_snapshot_indices`)
show the graph's real state evolving -- from mostly-untouched, to every
real tenant node having been served at least once, to the final settled
state -- without any JS/streaming machinery.

Run directly: `python -m dashboard.render_living_system_demo`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agents.baselines import StaticThresholdPolicy
from dashboard.render_living_system import (
    LivingSystemSnapshot,
    RecentDecisionView,
    build_snapshot,
    write_living_system_html,
)
from env.contracts import Action, SensitivityClass, ThreatPosture
from env.environment import SmartKeyNetEnv
from env.request_generator import RequestGenerator, build_tenant_graph
from experiments.train import load_full_config

_SCENARIO_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "scenarios" / "s2_hndl.yaml"
_SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

_TENANT_GRAPH_SEED = 7
"""Dedicated structural seed for the tenant graph, deliberately decoupled
from the episode seed -- same rationale `env/environment.py`'s own S4
(`ddos.graph_seed`)/S6 (`migration_graph_seed`) dispatch already
established (design decisions 13/15): tenant identity must be a stable
fact of this demo, not redrawn per episode seed."""


@dataclass
class _CollectedEpisode:
    decisions: list[RecentDecisionView]
    pool_fill_by_index: list[float]
    posture_by_index: list[ThreatPosture]
    tenant_attrs: dict[str, dict]
    hub_id: str


def collect_real_episode(seed: int = 0, max_steps: int = 250) -> _CollectedEpisode:
    """Step a real S2 episode under `StaticThresholdPolicy(0.5)`,
    collecting one real `RecentDecisionView` per decision (tenant,
    service, action, and the real served tier `SmartKeyNetEnv.
    _resulting_key_type` resolves it to -- the exact same ground-truth
    function `_apply_action` itself uses, never re-derived by hand),
    plus the real `pool_fill`/resolved-posture pair in effect for that
    decision, parallel-indexed to `decisions`.

    Reaches into `env._current_request`/`env._sessions`/
    `env._resulting_key_type`/`env._forecaster` -- the same established
    precedent `dashboard/explain.py::explain_decision_from_env` already
    uses for exactly this reason (the public Gym API doesn't surface
    per-request tenant/session-key internals).
    """
    config = load_full_config(_SCENARIO_CONFIG_PATH)
    config["max_steps"] = max_steps
    n_nodes = config["tenant_graph"]["n_nodes"]

    graph = build_tenant_graph(n_nodes=n_nodes, seed=_TENANT_GRAPH_SEED)
    tenant_attrs = {
        node: dict(attrs) for node, attrs in graph.nodes(data=True) if attrs.get("kind") == "tenant"
    }
    hub_id = next(node for node, attrs in graph.nodes(data=True) if attrs.get("kind") == "hub")

    env = SmartKeyNetEnv(
        config,
        request_stream_factory=lambda s: iter(RequestGenerator(graph, seed=s)),
    )
    state, info = env.reset(seed=seed)
    policy = StaticThresholdPolicy(pool_fill_threshold=0.5)

    decisions: list[RecentDecisionView] = []
    pool_fill_by_index: list[float] = []
    posture_by_index: list[ThreatPosture] = []

    truncated = False
    while not truncated:
        mask = info["action_mask"]
        action = policy.act(state, mask)

        request = env._current_request
        tenant, service = request["tenant"], request["service"]
        floor = Action(state["policy_floor"])
        session = env._sessions[(tenant, service)]
        served_tier = env._resulting_key_type(action, session, floor)
        assert served_tier is not None  # a legal action always resolves to a real tier

        if env._forecaster is None:
            posture = ThreatPosture.CALM
        else:
            probs = env._forecaster.get_threat_forecast().posture_probs
            posture = ThreatPosture(int(np.argmax(probs)))

        decisions.append(
            RecentDecisionView(
                step=env._step_count,
                tenant=tenant,
                service=service,
                sensitivity_class=SensitivityClass(request["sensitivity_class"]),
                action=action,
                served_tier=served_tier,
            )
        )
        pool_fill_by_index.append(float(state["pool_fill"]))
        posture_by_index.append(posture)

        state, _reward, _terminated, truncated, info = env.step(action)

    return _CollectedEpisode(
        decisions=decisions,
        pool_fill_by_index=pool_fill_by_index,
        posture_by_index=posture_by_index,
        tenant_attrs=tenant_attrs,
        hub_id=hub_id,
    )


def pick_snapshot_indices(episode: _CollectedEpisode) -> dict[str, int]:
    """Select three genuinely different real indices into `episode.decisions`
    by real predicate, not a fixed/evenly-spaced position (mirrors
    `render_explain_demo.py::pick_demo_traces`'s own approach). Does
    NOT key off the resolved posture/S2's scripted schedule -- see this
    module's own docstring for why that predicate turned out to be
    unreliable (the real ratchet fires almost immediately on this
    stream, independent of `elevate_at_step`). Instead uses a milestone
    that's true regardless of exactly when the ratchet fires:

    - the first decision of the episode (mostly-unpopulated graph --
      only the one tenant this decision itself concerns has a real
      served tier yet);
    - the first decision by which every real tenant node in the graph
      has been served at least once (found by scanning for the real
      index where the running distinct-tenant-served count first
      reaches the graph's real tenant count -- never assumed to land
      at any particular index); and
    - the last decision of the episode (final settled state).

    Never fabricates an index -- if the graph never becomes fully
    populated within the episode, only the first/last snapshots are
    returned (flagged via the returned dict's missing key), same
    honesty convention as `pick_demo_traces`'s `if not None` guards.
    """
    # Prefixed "living_system_" -- dashboard/samples/ is a shared output
    # directory across every rendered panel's own demo driver;
    # render_explain_demo.py already writes a real "01_first_decision.html"
    # there for the Explain Decision panel, and an unprefixed name here
    # would silently clobber it (caught during this session -- see
    # SESSION_LOG.md).
    picks: dict[str, int] = {"living_system_01_first_decision": 0}

    n_tenants = len(episode.tenant_attrs)
    seen: set[str] = set()
    full_idx = None
    for i, decision in enumerate(episode.decisions):
        seen.add(decision.tenant)
        if len(seen) == n_tenants:
            full_idx = i
            break
    if full_idx is not None:
        picks["living_system_02_graph_fully_populated"] = full_idx

    picks["living_system_03_final_decision"] = len(episode.decisions) - 1
    return picks


def main() -> None:
    episode = collect_real_episode()
    picks = pick_snapshot_indices(episode)

    _SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, idx in picks.items():
        snapshot: LivingSystemSnapshot = build_snapshot(
            label=name,
            hub_id=episode.hub_id,
            tenant_attrs=episode.tenant_attrs,
            all_decisions=episode.decisions,
            snapshot_index=idx,
            pool_fill=episode.pool_fill_by_index[idx],
            posture=episode.posture_by_index[idx],
        )
        out_path = _SAMPLES_DIR / f"{name}.html"
        write_living_system_html(snapshot, out_path, title=f"SmartKeyNet -- Living System -- {name}")
        served_now = sum(1 for t in snapshot.tenants if t.last_served_tier is not None)
        print(
            f"wrote {out_path} "
            f"(t={snapshot.step}, posture={snapshot.posture.name}, pool_fill={snapshot.pool_fill:.3f}, "
            f"tenants_served_so_far={served_now}/{len(snapshot.tenants)}, "
            f"recent_decisions={len(snapshot.recent_decisions)})"
        )


if __name__ == "__main__":
    main()
