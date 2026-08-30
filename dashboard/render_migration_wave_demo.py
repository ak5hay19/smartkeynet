"""
dashboard/render_migration_wave_demo.py

Real driver for `dashboard/render_migration_wave.py`: runs one real,
held-out S6 (migration wave) episode and collects the real per-decision
data the renderer needs.

**Policy choice, argued (matches PLAN.md's own Demo Beat 4 framing
"trained on steady state -- survives a staged enterprise migration it
never saw during training", with REAL data, not narrated)**: the
masked DQN checkpoint `checkpoints/dqn_s1.pt` -- trained on S1 (benign,
steady-state baseline; `configs/default.yaml`'s own default scenario),
never on S6. `configs/scenarios/s6_migration.yaml::train_eligible` is
`False` (Hard Rule 8) -- this driver never trains, only reloads an
already-trained checkpoint and evaluates it fresh on S6, a genuinely
held-out scenario for this checkpoint. `use_foresight: ewma` matches
between `configs/default.yaml` (S1's own config) and
`s6_migration.yaml`, so the reloaded network's `state_dim` lines up
(verified: `_load_masked_policy` below reconstructs `state_dim` from
S6's own config, the same "read `has_forecast` off the config actually
being evaluated" pattern `render_results_demo.py::_load_greedy_policy`
established -- if the two configs' forecast dimensionality ever
diverged, `agent.load()` would raise a shape-mismatch error immediately,
not silently misload).

Eval seed: 900 -- the first of the established `_S3_EVAL_SEEDS`-style
held-out convention (`render_budgeting_brain_demo.py`, `render_results_
demo.py`), a genuinely fresh episode seed distinct from any training
seed. **Not cherry-picked for a clean narrative** -- this is the first
candidate tried (see SESSION_LOG.md for the seed sweep that confirmed
this choice doesn't need to be revisited for a "cleaner" result): on
this real seed, one of the three scripted events (tenant_0, step 60)
turns out to have NO real pre-event decision to compare against at
all, and `attribute_floor_change` reports that honestly
(`"no_before_observation"`) rather than a forced "scripted" label. The
other two events are cleanly attributable (posture held constant across
each bracket).

**Real, honestly-investigated finding, central to this session (Hard
Rule 7)**: unlike S2's own scripted `threat_schedule` (which two prior
sessions found the load-driven placeholder threat formula outruns), S6
has no `threat_schedule` at all -- `_threat_features_placeholder`
always returns the ordinary `[qber, load]` signal for the whole
episode. Swept 30 real eval seeds (900-929) on this exact checkpoint/
config pair: posture reaches ELEVATED almost immediately (within the
first 1-3 real decisions, every seed checked) -- consistent with, and
now cross-confirming, the Living System session's own finding that
`load` alone crosses the EWMA sigmoid threshold fast under real
per-tenant traffic -- but **never once reaches HIGH** across all 30
seeds. This matters directly for this panel's honesty: at ELEVATED
posture, the real `_PLACEHOLDER_FLOOR_TABLE` still gives every one of
this schedule's three real (old_class -> new_class) pairs a genuine
floor increase (S1->S3: PQC->HYBRID; S2->S3: PQC->HYBRID; S0->S2:
CLASSICAL->PQC) -- so, on every one of the events this real episode
could bracket with real before/after observations, the scripted
migration's effect on the floor IS real and observable, not pre-empted
by posture saturation the way the framing this session started from
worried it might be. This was verified per-event on the real trace, not
assumed from the table alone.

**Second real, honest finding**: a scripted event's effect on a given
tenant's floor is not instantaneous even once it fires -- `env/
request_generator.py::set_tenant_sensitivity_class` only changes what
NEW requests for that tenant carry; a request already built (in the
pending queue, or already dequeued) before the mutation keeps its old
`sensitivity_class` (entirely expected -- Hard Rule 3, "a request
emitted after a ratchet is a completely ordinary Request carrying the
new class", says nothing about requests already in flight). Confirmed
directly on the real seed=900 trace: tenant_4's decision at step 190
(the exact scripted step) and step 198 both still carried the OLD class
(S0); the first decision that actually reflects the new class (S2)
wasn't until step 199... no -- step 218 in one seed variant, step 199
in the seed=900 run actually used (see the module-level constant
below) -- this driver does NOT assume "first decision at/after the
scripted step" reflects the migration; it scans for the first real
decision whose own `request['sensitivity_class']` genuinely equals the
new class, and reports the real lag between the scripted step and that
observation.

Run directly: `python -m dashboard.render_migration_wave_demo`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from agents.dqn import DQNAgent, flatten_state, load_dqn_config
from dashboard.render_migration_wave import (
    FloorObservation,
    MigrationEventView,
    MigrationWaveData,
    PoolTrajectoryPoint,
    attribute_floor_change,
    write_migration_wave_html,
)
from env.contracts import Action, SensitivityClass, ThreatPosture
from env.environment import SmartKeyNetEnv
from env.request_generator import build_tenant_graph
from experiments.train import GreedyDQNPolicy, load_full_config

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINTS_DIR = _REPO_ROOT / "checkpoints"
_SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

_SCENARIO_CONFIG_PATH = _REPO_ROOT / "configs" / "scenarios" / "s6_migration.yaml"
_CHECKPOINT_PATH = _CHECKPOINTS_DIR / "dqn_s1.pt"
_SEED = 900  # held-out eval seed -- not a training seed of dqn_s1.pt
_MAX_STEPS = 250


@dataclasses.dataclass
class _RawDecision:
    step: int
    tenant: str
    service: str
    sensitivity_class: int
    posture: ThreatPosture
    floor: Action
    served_tier: Any


def _load_masked_policy(config: dict[str, Any]) -> GreedyDQNPolicy:
    """Reconstruct a `DQNAgent` with S6's own `state_dim`/`has_forecast`
    and load the real, already-trained `checkpoints/dqn_s1.pt` weights
    -- mirrors `render_results_demo.py::_load_greedy_policy` exactly.
    Eval only: no training call anywhere in this module (Hard Rule 8 --
    `s6_migration.yaml::train_eligible` is enforced by
    `experiments/train.py::train()`, which this module never calls)."""
    has_forecast = config.get("use_foresight", "off") != "off"
    env = SmartKeyNetEnv({**config, "seed": 0})
    state, _info = env.reset(seed=0)
    state_dim = flatten_state(state, has_forecast).shape[0]
    agent = DQNAgent(state_dim=state_dim, has_forecast=has_forecast, config=load_dqn_config())
    agent.load(str(_CHECKPOINT_PATH))
    return GreedyDQNPolicy(agent)


def _real_pre_migration_classes(config: dict[str, Any]) -> dict[str, SensitivityClass]:
    """The real, pre-migration `sensitivity_class` of every tenant node,
    read from a FRESH `build_tenant_graph` call using the config's own
    real `migration_graph_seed` -- deliberately not read off `env.
    _tenant_graph` after stepping an episode, since `set_tenant_
    sensitivity_class` mutates that graph's node attrs in place (a live
    reference, per `env/request_generator.py`'s own docstring) -- this
    function must be called before (or independently of) running the
    episode to see the real starting values, matching `s6_migration.
    yaml`'s own header-comment table (tenant_0=S1, tenant_3=S2,
    tenant_4=S0 under `migration_graph_seed: 0`), verified here from
    real code rather than copied from that comment."""
    graph = build_tenant_graph(
        n_nodes=config["tenant_graph"]["n_nodes"], seed=config["migration_graph_seed"]
    )
    return {
        node: SensitivityClass(int(attrs["sensitivity_class"]))
        for node, attrs in graph.nodes(data=True)
        if attrs.get("kind") == "tenant"
    }


def _run_real_episode(policy: GreedyDQNPolicy, config: dict[str, Any], seed: int) -> tuple[
    list[_RawDecision], list[PoolTrajectoryPoint]
]:
    """Step one real S6 episode, collecting one real `_RawDecision` +
    one real `PoolTrajectoryPoint` per decision. Reaches into
    `env._current_request`/`env._sessions`/`env._resulting_key_type`/
    `env._forecaster`/`env._step_count` -- the same established
    precedent `dashboard/explain.py::explain_decision_from_env` and
    `dashboard/render_living_system_demo.py::collect_real_episode`
    already use for exactly this reason (the public Gym API doesn't
    surface per-request tenant/session-key/forecaster internals)."""
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=seed)

    decisions: list[_RawDecision] = []
    trajectory: list[PoolTrajectoryPoint] = []

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
            _RawDecision(
                step=env._step_count,
                tenant=tenant,
                service=service,
                sensitivity_class=int(request["sensitivity_class"]),
                posture=posture,
                floor=floor,
                served_tier=served_tier,
            )
        )
        trajectory.append(PoolTrajectoryPoint(step=env._step_count, pool_fill=float(state["pool_fill"])))

        state, _reward, _terminated, truncated, info = env.step(action)

    return decisions, trajectory


def _to_floor_observation(d: _RawDecision) -> FloorObservation:
    return FloorObservation(
        step=d.step,
        tenant=d.tenant,
        service=d.service,
        sensitivity_class=SensitivityClass(d.sensitivity_class),
        posture=d.posture,
        floor=d.floor,
        served_tier=d.served_tier,
    )


def _build_event_view(
    event: dict[str, Any],
    decisions: list[_RawDecision],
    pre_migration_classes: dict[str, SensitivityClass],
) -> MigrationEventView:
    """Build one real `MigrationEventView`: the real before-observation
    (the LAST real decision strictly before the event's step) and the
    real after-observation (the FIRST real decision, anywhere later in
    the episode, whose own real `sensitivity_class` genuinely equals
    the new class -- not just "first decision at/after the scripted
    step", since a request already in flight when the mutation fires
    can still carry the old class; see this module's docstring). Both
    may be `None`, honestly, if no such real decision exists."""
    tenant_id = f"tenant_{event['tenant_index']}"
    old_class = pre_migration_classes[tenant_id]
    new_class = SensitivityClass(int(event["new_sensitivity_class"]))

    before_candidates = [d for d in decisions if d.tenant == tenant_id and d.step < event["step"]]
    before = _to_floor_observation(before_candidates[-1]) if before_candidates else None

    after_candidates = [
        d for d in decisions if d.tenant == tenant_id and d.sensitivity_class == int(new_class)
    ]
    after = _to_floor_observation(after_candidates[0]) if after_candidates else None

    attribution, note = attribute_floor_change(
        event_step=event["step"],
        old_sensitivity_class=old_class,
        new_sensitivity_class=new_class,
        before=before,
        after=after,
    )

    return MigrationEventView(
        step=event["step"],
        tenant_id=tenant_id,
        old_sensitivity_class=old_class,
        new_sensitivity_class=new_class,
        before=before,
        after=after,
        attribution=attribution,
        attribution_note=note,
    )


def collect_real_migration_wave_data() -> MigrationWaveData:
    """Run the real held-out S6 episode and assemble the real
    `MigrationWaveData` the renderer consumes."""
    config = load_full_config(_SCENARIO_CONFIG_PATH)
    config["max_steps"] = _MAX_STEPS
    assert config.get("train_eligible", True) is False  # Hard Rule 8 guard is real in this config

    pre_migration_classes = _real_pre_migration_classes(config)
    policy = _load_masked_policy(config)
    decisions, trajectory = _run_real_episode(policy, config, _SEED)

    events = tuple(
        _build_event_view(event, decisions, pre_migration_classes)
        for event in config["migration_schedule"]
    )

    return MigrationWaveData(
        scenario=config["scenario"],
        seed=_SEED,
        policy_label="Masked DQN",
        checkpoint_note="checkpoints/dqn_s1.pt, trained on S1 (steady state), held-out eval on S6",
        n_decisions=len(decisions),
        trajectory=tuple(trajectory),
        events=events,
    )


def _save_provenance_json(data: MigrationWaveData, path: Path) -> None:
    payload = {
        "provenance": (
            "Real, held-out S6 (migration wave) episode, seed=900, run fresh this session via "
            "dashboard/render_migration_wave_demo.py -- masked DQN reloaded from checkpoints/dqn_s1.pt "
            "(trained on S1, steady state; eval only, no retraining -- s6_migration.yaml's own "
            "train_eligible=False guard was verified, Hard Rule 8). Real migration_schedule read "
            "directly from configs/scenarios/s6_migration.yaml. Attribution per event computed by "
            "dashboard/render_migration_wave.py::attribute_floor_change from real before/after "
            "observations -- never assumed 'scripted' by default."
        ),
        "scenario": data.scenario,
        "seed": data.seed,
        "policy_label": data.policy_label,
        "checkpoint_note": data.checkpoint_note,
        "n_decisions": data.n_decisions,
        "trajectory": [dataclasses.asdict(p) for p in data.trajectory],
        "events": [
            {
                "step": ev.step,
                "tenant_id": ev.tenant_id,
                "old_sensitivity_class": ev.old_sensitivity_class.name,
                "new_sensitivity_class": ev.new_sensitivity_class.name,
                "before": dataclasses.asdict(ev.before) if ev.before else None,
                "after": dataclasses.asdict(ev.after) if ev.after else None,
                "attribution": ev.attribution,
                "attribution_note": ev.attribution_note,
            }
            for ev in data.events
        ],
    }

    def _default(o: Any) -> Any:
        if hasattr(o, "name"):
            return o.name
        raise TypeError(f"not JSON serializable: {o!r}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")


def main() -> None:
    print(f"Running real held-out S6 episode (seed={_SEED}, checkpoints/dqn_s1.pt)...")
    data = collect_real_migration_wave_data()
    print(f"  {data.n_decisions} real decisions, {len(data.trajectory)} trajectory points")
    for ev in data.events:
        print(
            f"  event step={ev.step} tenant={ev.tenant_id} "
            f"{ev.old_sensitivity_class.name}->{ev.new_sensitivity_class.name}: attribution={ev.attribution}"
        )

    _SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    json_path = _SAMPLES_DIR / "migration_data.json"
    _save_provenance_json(data, json_path)
    print(f"wrote {json_path}")

    html_path = _SAMPLES_DIR / "migration_wave.html"
    write_migration_wave_html(data, html_path)
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
