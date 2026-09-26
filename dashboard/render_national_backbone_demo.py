"""
dashboard/render_national_backbone_demo.py

Real driver for `dashboard/render_national_backbone.py`. Runs one real,
held-out episode of `configs/scenarios/s_narrative_national_backbone.
yaml` (S6's existing reclassification mechanism, unchanged) and hands
the renderer real data. Reuses `render_migration_wave_demo`'s episode
collection / policy loading / event bracketing verbatim -- same
checkpoint (`checkpoints/dqn_s1.pt`, trained on S1, never on this
schedule), same eval seed (900), same forecaster.

**Seed search (the honest-labeling choice)**: sensitivity classes are
NOT special-cased. `search_graph_seed` scans `build_tenant_graph` seeds
and takes the first whose real, untouched assignment gives
tenant_0=S3, tenant_1=S3, tenant_2=S2, tenant_3=S1 (S0 also accepted for
tenant_3). Result: seed 30 (14 of the first 2000 seeds qualify). Only
the four display names in `_AGENCIES` are applied on top; the panel
shows the underlying tenant id beside each name.

Run directly: `python -m dashboard.render_national_backbone_demo`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from dashboard.render_migration_wave_demo import (
    _CHECKPOINT_PATH,
    _MAX_STEPS,
    _SEED,
    _build_event_view,
    _load_masked_policy,
    _real_pre_migration_classes,
    _replay_provider_factory,
    _run_real_episode,
    _to_floor_observation,
)
from dashboard.render_national_backbone import (
    AgencyView,
    NationalBackboneData,
    write_national_backbone_html,
)
from env.contracts import SensitivityClass
from env.request_generator import build_tenant_graph
from experiments.train import load_full_config

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
_SCENARIO_CONFIG_PATH = _REPO_ROOT / "configs" / "scenarios" / "s_narrative_national_backbone.yaml"

# (display name, tenant id) -- the ONLY thing this scenario adds on top of the real system.
_AGENCIES: tuple[tuple[str, str], ...] = (
    ("Defense Ministry", "tenant_0"),
    ("Health Ministry", "tenant_1"),
    ("Tax Authority", "tenant_2"),
    ("Passport Office", "tenant_3"),
)
_RECLASS_AGENCY = "Passport Office"

# Story shape required of the real, untouched random assignment.
_REQUIRED_CLASSES = {"tenant_0": {3}, "tenant_1": {3}, "tenant_2": {2}, "tenant_3": {0, 1}}


def search_graph_seed(n_nodes: int = 10, max_seed: int = 2000) -> int | None:
    """First `build_tenant_graph` seed in `range(max_seed)` whose real
    assignment has the story's shape; `None` if none exists."""
    for seed in range(max_seed):
        graph = build_tenant_graph(n_nodes=n_nodes, seed=seed)
        if all(
            int(graph.nodes[t]["sensitivity_class"]) in allowed for t, allowed in _REQUIRED_CLASSES.items()
        ):
            return seed
    return None


def collect_real_national_backbone_data() -> NationalBackboneData:
    config = load_full_config(_SCENARIO_CONFIG_PATH)
    config["max_steps"] = _MAX_STEPS
    assert config.get("train_eligible", True) is False  # Hard Rule 8 guard is real in this config
    assert config["scenario"] == "S6"

    pre_classes = _real_pre_migration_classes(config)
    for tenant, allowed in _REQUIRED_CLASSES.items():  # the story's shape must really hold, not be assumed
        assert int(pre_classes[tenant]) in allowed, (tenant, pre_classes[tenant])
    assert len(config["migration_schedule"]) == 1

    factory = _replay_provider_factory()
    policy = _load_masked_policy(config, factory)
    decisions, trajectory = _run_real_episode(policy, config, _SEED, factory)

    agencies = tuple(
        AgencyView(
            display_name=name,
            tenant_id=tenant,
            initial_class=SensitivityClass(int(pre_classes[tenant])),
            observations=tuple(_to_floor_observation(d) for d in decisions if d.tenant == tenant),
        )
        for name, tenant in _AGENCIES
    )
    event = _build_event_view(config["migration_schedule"][0], decisions, pre_classes)
    n_below = sum(1 for d in decisions if int(d.served_tier) < int(d.floor))

    return NationalBackboneData(
        scenario=config["scenario"],
        seed=_SEED,
        graph_seed=config["migration_graph_seed"],
        policy_label="Masked DQN",
        checkpoint_note=f"checkpoints/{_CHECKPOINT_PATH.name}, trained on S1 (steady state), held-out eval",
        n_decisions=len(decisions),
        n_below_floor=n_below,
        agencies=agencies,
        reclass_agency=_RECLASS_AGENCY,
        reclass_event=event,
        trajectory=tuple(trajectory),
    )


def _save_provenance_json(data: NationalBackboneData, path: Path) -> None:
    def _default(o: Any) -> Any:
        if hasattr(o, "name"):
            return o.name
        raise TypeError(f"not JSON serializable: {o!r}")

    payload = {
        "provenance": (
            "Real held-out episode, seed=900, configs/scenarios/s_narrative_national_backbone.yaml "
            "(S6 mechanism, migration_graph_seed=30 found by seed search). Agency names are illustrative "
            "display labels over tenant_0..3; classes are the real, unmodified build_tenant_graph assignment."
        ),
        "graph_seed": data.graph_seed,
        "n_decisions": data.n_decisions,
        "n_below_floor": data.n_below_floor,
        "agencies": [
            {
                "display_name": a.display_name,
                "tenant_id": a.tenant_id,
                "initial_class": a.initial_class.name,
                "observations": [dataclasses.asdict(o) for o in a.observations],
            }
            for a in data.agencies
        ],
        "reclass_event": dataclasses.asdict(data.reclass_event),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")


def main() -> None:
    print(f"seed search -> first qualifying graph seed: {search_graph_seed()}")
    data = collect_real_national_backbone_data()
    ev = data.reclass_event
    print(f"  {data.n_decisions} decisions, {data.n_below_floor} below floor")
    print(f"  reclass step={ev.step} {ev.tenant_id} {ev.old_sensitivity_class.name}->{ev.new_sensitivity_class.name}: {ev.attribution}")
    _SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    _save_provenance_json(data, _SAMPLES_DIR / "national_backbone_data.json")
    print("wrote", write_national_backbone_html(data, _SAMPLES_DIR / "national_backbone.html"))


if __name__ == "__main__":
    main()
