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

**Why S2, not S1, was built first (design decision, argued per
instruction in the original session):** S2's scripted `threat_schedule`
(`elevate_at_step=50`) is the scenario whose real threat/posture is
designed to change mid-episode -- an S1 episode's floor never moves at
all *from a scripted schedule*, so its snapshots looked like a weaker
demo opener on their own.

**S1 added alongside S2 (this session, not a replacement):** the S2
snapshot alone tells only half the story -- elevated posture ratchets
every tenant's floor up so high that decisions are almost uniformly
SERVE_HYBRID, a visually uniform graph that shows masking holding the
floor under threat but not the agent discriminating between requests
day-to-day. `collect_real_episode` now takes an explicit
`scenario_config_path`, and `main()` runs it twice -- once against
`configs/scenarios/s2_hndl.yaml` (unchanged call, same output
filenames as before) and once against `configs/default.yaml` (`scenario:
S1`, the real steady/calm-traffic config: no `threat_schedule`, no
`ddos`/`migration_schedule` dispatch -- see `env/environment.py`'s
`self._threat_schedule_cfg = config["threat_schedule"] if self._scenario
== "S2" else None`-style scenario dispatch). Both runs go through the
exact same real machinery (`SmartKeyNetEnv`, the same real
`build_tenant_graph`-backed request stream, the same `StaticThresholdPolicy`,
the same `render_living_system.py` renderer) -- only the scenario config
path differs. `pick_snapshot_indices` takes a `prefix` param so the two
runs write to distinct, non-colliding filenames (S2 keeps its original
unprefixed-scenario names; S1's carry a `living_system_s1_` prefix).

**Honesty caveat, read before trusting this to show variety (Hard Rule
7):** prior sessions found `env/forecast_provider.py`'s placeholder
`MovingAverageForecaster` mixes an ordinary `load` (queue-backlog) term
into the threat signal, and that `load` term alone is enough to ratchet
posture upward within the first 1-2 decisions of a run -- independent of
any scripted schedule. Whether S1 genuinely produces a varied tier mix
or *also* saturates to HYBRID for the same load-driven reason is a real
empirical question this module answers by actually running the episode,
never assumed in either direction -- see `main()`'s printed tier
distribution and SESSION_LOG.md's dated entry for the real, honest
result.

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

from collections import Counter
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

_S2_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "scenarios" / "s2_hndl.yaml"
_S1_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
"""`configs/default.yaml` sets `scenario: S1` and carries no
`threat_schedule` block -- the real steady/calm-traffic scenario, read
through the exact same `load_full_config` + `SmartKeyNetEnv` path as
S2, per `env/environment.py`'s own scenario dispatch (design decision
10: `self._threat_schedule_cfg` is only ever populated when `self.
_scenario == "S2"`)."""
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


def collect_real_episode(
    seed: int = 0, max_steps: int = 250, scenario_config_path: Path = _S2_CONFIG_PATH
) -> _CollectedEpisode:
    """Step a real episode (S2 by default; pass `scenario_config_path=
    _S1_CONFIG_PATH` for S1) under `StaticThresholdPolicy(0.5)`,
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
    config = load_full_config(scenario_config_path)
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


def pick_snapshot_indices(episode: _CollectedEpisode, *, prefix: str = "living_system") -> dict[str, int]:
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
    # `prefix` (default "living_system", matching the original S2 run's
    # unprefixed names byte-for-byte) -- dashboard/samples/ is a shared
    # output directory across every rendered panel's own demo driver;
    # render_explain_demo.py already writes a real "01_first_decision.html"
    # there for the Explain Decision panel, and an unprefixed name here
    # would silently clobber it (caught in the original S2 session -- see
    # SESSION_LOG.md). The S1 run below passes prefix="living_system_s1"
    # so its three files are additive, never colliding with S2's.
    picks: dict[str, int] = {f"{prefix}_01_first_decision": 0}

    n_tenants = len(episode.tenant_attrs)
    seen: set[str] = set()
    full_idx = None
    for i, decision in enumerate(episode.decisions):
        seen.add(decision.tenant)
        if len(seen) == n_tenants:
            full_idx = i
            break
    if full_idx is not None:
        picks[f"{prefix}_02_graph_fully_populated"] = full_idx

    picks[f"{prefix}_03_final_decision"] = len(episode.decisions) - 1
    return picks


def _write_snapshots(episode: _CollectedEpisode, *, prefix: str) -> dict[str, LivingSystemSnapshot]:
    picks = pick_snapshot_indices(episode, prefix=prefix)
    written: dict[str, LivingSystemSnapshot] = {}
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
        written[name] = snapshot
    return written


def _tier_distribution(episode: _CollectedEpisode) -> Counter[str]:
    """Real served-tier counts across every real decision of the episode
    (not just the three snapshot cutoffs) -- the honest headline number
    for whether a scenario's decisions actually vary by tier."""
    return Counter(decision.served_tier.name for decision in episode.decisions)


def main() -> None:
    _SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    pre_existing_html = {p.name for p in _SAMPLES_DIR.glob("*.html")}

    s2_episode = collect_real_episode(scenario_config_path=_S2_CONFIG_PATH)
    _write_snapshots(s2_episode, prefix="living_system")

    s1_episode = collect_real_episode(scenario_config_path=_S1_CONFIG_PATH)
    s1_picks = pick_snapshot_indices(s1_episode, prefix="living_system_s1")
    for name in s1_picks:
        filename = f"{name}.html"
        # Collision guard (explicit per instruction): the S1 run is
        # additive -- it must never silently overwrite a pre-existing
        # dashboard/samples/ file this run didn't itself just (re)write
        # (e.g. the S2 files above, or any other panel's samples).
        assert filename not in pre_existing_html, (
            f"refusing to write {filename} -- it collides with a pre-existing "
            "dashboard/samples/ file not produced by this S1 run"
        )
    _write_snapshots(s1_episode, prefix="living_system_s1")

    s1_tiers = _tier_distribution(s1_episode)
    s2_tiers = _tier_distribution(s2_episode)
    print(
        f"\nReal tier distribution -- S1 (steady/calm), "
        f"{len(s1_episode.decisions)} decisions: {dict(sorted(s1_tiers.items()))}"
    )
    print(
        f"Real tier distribution -- S2 (elevated HNDL), "
        f"{len(s2_episode.decisions)} decisions: {dict(sorted(s2_tiers.items()))}"
    )


if __name__ == "__main__":
    main()
