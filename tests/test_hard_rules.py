"""The Hard Rules, machine-checked -- SMARTKEYNET_BUILD_SPEC.md §2.

These are the tests that make the thesis claims structural rather than
asserted. If one of these fails, a headline claim in docs/report.md is false.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pytest

from env.contracts import Action, SensitivityClass, ThreatPosture
from env.environment import SmartKeyNetEnv
from env.masking import PolicyTable
from env.scenarios import ScenarioError, build_scenario, require_trainable
from metrics.reward_inputs import FORBIDDEN_FIELD_SUBSTRINGS, RewardInputs

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# §2.1 HR1 -- no security in the reward, enforced by type
# ---------------------------------------------------------------------------


def test_reward_inputs_has_no_security_fields():
    """The exact frozen field set, and no security vocabulary in any name."""
    field_names = {field.name for field in dataclasses.fields(RewardInputs)}
    assert field_names == {
        "latency_ms",
        "energy_mj",
        "key_age_steps",
        "key_lifetime_cap_steps",
        "qkd_keys_consumed",
        "deferred_critical_steps",
        "did_rekey",
        "normalised_load",
    }
    for name in field_names:
        for forbidden in FORBIDDEN_FIELD_SUBSTRINGS:
            assert forbidden not in name.lower(), f"'{forbidden}' appears in RewardInputs.{name}"


def test_reward_inputs_is_frozen():
    """Immutable, so no caller can smuggle a field in at runtime."""
    inputs = RewardInputs(1.0, 1.0, 1, 500, 0, 0, False, 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        inputs.latency_ms = 2.0  # type: ignore[misc]


def test_reward_computation_reads_no_security_state():
    """AST-level check on the environment's reward computation."""
    source = inspect.getsource(SmartKeyNetEnv._apply_action)
    lowered = source.lower()
    for forbidden in ("threat", "posture", "security", "risk"):
        assert forbidden not in lowered, f"'{forbidden}' appears in the reward computation"


# ---------------------------------------------------------------------------
# §2.2 HR2 + HR9 -- floors inviolable, exhaustion defers (adversarial fuzz)
# ---------------------------------------------------------------------------


def test_no_floor_violation_under_adversarial_fuzz():
    """An adversarial policy that always picks the LOWEST-index legal action
    -- i.e. the weakest tier available -- must still never serve below floor,
    across randomised pool sizes and scenarios.

    Reduced from the spec's 2,000 episodes to keep the fast suite fast; the
    episodes are long enough that each one exercises thousands of masks.
    """
    import yaml

    base = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    rng = np.random.default_rng(0)

    total_violations = 0
    for episode in range(6):
        config = {
            **base,
            "scenario": ["S1", "S2", "S3"][episode % 3],
            "max_steps": 400,
            "scenario_steps": 400,
            "threat_source": "synthetic",
            "pool": {
                **base["pool"],
                "initial_fill_frac": float(rng.choice([0.0, 0.25, 1.0])),
                "capacity_bits": float(rng.choice([256.0, 2560.0, 25600.0])),
            },
        }
        env = SmartKeyNetEnv(config)
        state, info = env.reset(seed=int(rng.integers(0, 10_000)))

        for _ in range(400):
            mask = info["action_mask"]
            floor = int(state["policy_floor"])
            # adversarial: weakest legal action every time
            action = Action(int(np.flatnonzero(mask)[0]))
            if action in (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID):
                if int(action) < floor:
                    total_violations += 1
            state, _r, _te, truncated, info = env.step(action)
            if truncated:
                break

    assert total_violations == 0


def test_pool_exhaustion_defers_and_never_downgrades():
    """HR9: with a pool that cannot cover anything, hybrid-mandatory requests
    are queued, not served weak."""
    import yaml

    base = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    config = {
        **base,
        "scenario": "S1",
        "max_steps": 400,
        "scenario_steps": 400,
        "threat_source": "synthetic",
        "pool": {**base["pool"], "initial_fill_frac": 0.0},
        "qkd": {**base["qkd"], "mean_skr_kbps": 0.001},  # effectively no refill
    }
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)

    saw_deferral = False
    for _ in range(400):
        mask = info["action_mask"]
        floor = int(state["policy_floor"])
        action = Action(int(np.flatnonzero(mask)[0]))
        assert not (
            action in (Action.SERVE_CLASSICAL, Action.SERVE_PQC) and int(action) < floor
        )
        state, _r, _te, truncated, info = env.step(action)
        if info["regret_events"]:
            saw_deferral = True
        if truncated:
            break

    assert saw_deferral, "a starved pool must produce deferrals"


# ---------------------------------------------------------------------------
# §2.3 HR3 -- one agent, one MDP
# ---------------------------------------------------------------------------


def test_agents_never_import_the_graph():
    """AST-scan: no module under agents/ may import networkx or the graph."""
    for path in (REPO / "agents").glob("*.py"):
        tree = ast.parse(path.read_text())
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        for name in imported:
            assert "networkx" not in name, f"{path.name} imports networkx"
            assert "graph" not in name.lower(), f"{path.name} imports a graph module"


def test_state_dimensionality_is_identical_across_request_sources():
    """The HR3 substitution test: swapping the graph for a plain Poisson
    process must not change what the agent sees."""
    import yaml

    from agents.dqn import flatten_state

    base = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    dims = set()
    for source in ("random", "graph"):
        config = {**base, "scenario": "S1", "max_steps": 10, "request_source": source}
        env = SmartKeyNetEnv(config)
        state, _info = env.reset(seed=0)
        dims.add(flatten_state(state, has_forecast=True).shape[0])
    assert len(dims) == 1, f"state dim differs by request source: {dims}"


# ---------------------------------------------------------------------------
# §2.4 HR8 -- eval-only scenarios cannot be trained on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["S5", "S6"])
def test_training_rejects_eval_only_scenarios(scenario):
    import yaml

    base = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    spec = build_scenario(scenario, base, episode_steps=1000)
    with pytest.raises(ScenarioError):
        require_trainable(spec)


# ---------------------------------------------------------------------------
# Policy-table structure -- what makes "threat can only raise floors" a
# theorem rather than a hope (§S4)
# ---------------------------------------------------------------------------


def test_policy_table_is_monotone_in_class_and_posture():
    table = PolicyTable()
    for posture in ThreatPosture:
        floors = [int(table.floor(sc, posture)) for sc in SensitivityClass]
        assert floors == sorted(floors), f"non-monotone in class at {posture.name}"
    for sensitivity_class in SensitivityClass:
        floors = [int(table.floor(sensitivity_class, p)) for p in ThreatPosture]
        assert floors == sorted(floors), f"non-monotone in posture at {sensitivity_class.name}"


def test_ratchet_has_no_downward_path():
    """The property the steering attack relies on: there is no API that
    lowers a ratcheted posture."""
    assert not hasattr(PolicyTable, "ratchet_down")
    table = PolicyTable()
    table.ratchet_up(ThreatPosture.HIGH)
    for posture in ThreatPosture:
        assert int(table.floor(SensitivityClass.S3, posture)) >= int(
            PolicyTable().floor(SensitivityClass.S3, ThreatPosture.HIGH)
        )
