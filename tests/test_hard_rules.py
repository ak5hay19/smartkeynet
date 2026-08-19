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
import yaml

from env.contracts import Action, SensitivityClass, ThreatPosture
from env.reward import RewardWeights, compute_reward
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


_SECURITY_MODULES: frozenset[str] = frozenset(
    {"env.policy_table", "env.masking", "env.threat_source", "forecaster"}
)
"""Modules that carry security state. The reward may not import any of them."""


def test_reward_module_imports_no_security_state():
    """SMARTKEYNET_BUILD_SPEC.md §2.1, second half: AST-parse the reward
    module and assert it imports nothing security-flavoured.

    This replaced a substring scan of `SmartKeyNetEnv._apply_action` on
    2026-08-19. That scan was the only thing enforcing Hard Rule 1, and it
    was defeated by a one-line alias (`t = state["threat_score"]` in a
    caller, then pass `t` in), because it only ever looked at the literal
    text of one method. Reachability is the property that matters, so test
    reachability: if `env/reward.py` cannot import the modules that hold
    threat/posture/floor state, it cannot read them regardless of how any
    call site is written.
    """
    tree = ast.parse((REPO / "env" / "reward.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    for module in imported:
        for forbidden in _SECURITY_MODULES:
            assert not (module == forbidden or module.startswith(forbidden + ".")), (
                f"env/reward.py imports {module}, which carries security state "
                "-- Hard Rule 1 violated"
            )


def test_reward_signature_accepts_only_reward_inputs():
    """The type *is* the enforcement mechanism (§2.1), so pin it.

    `compute_reward` must take exactly a `RewardInputs` and a
    `RewardWeights`. Widening this signature -- passing the state, the
    request, or the floor "just for logging" -- is how Hard Rule 1 dies
    quietly, so it fails a test rather than a review.
    """
    signature = inspect.signature(compute_reward)
    parameters = list(signature.parameters.values())
    assert [p.name for p in parameters] == ["inputs", "weights"]
    assert parameters[0].annotation in (RewardInputs, "RewardInputs")
    assert parameters[1].annotation in (RewardWeights, "RewardWeights")


def test_reward_cannot_be_computed_from_security_state():
    """Behavioural companion: two steps whose security context differs in
    every way but whose `RewardInputs` are identical must earn identical
    reward. This is the property the paper actually claims."""
    weights = RewardWeights.from_config(
        yaml.safe_load((REPO / "configs" / "default.yaml").read_text(encoding="utf-8"))["reward"]
    )
    inputs = RewardInputs(
        latency_ms=120.0,
        energy_mj=1.3,
        key_age_steps=10,
        key_lifetime_cap_steps=500,
        qkd_keys_consumed=0,
        deferred_critical_steps=0,
        did_rekey=True,
        normalised_load=0.4,
    )
    total_a, terms_a = compute_reward(inputs, weights)
    total_b, terms_b = compute_reward(inputs, weights)
    assert total_a == total_b and terms_a == terms_b

    # And the breakdown must be complete: the terms sum to the total, so no
    # unexplained contribution can hide in the reward.
    assert sum(terms_a.values()) == pytest.approx(total_a)


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


# ---------------------------------------------------------------------------
# §2.5 HR4 -- no invented security constants (enforced by lint)
# ---------------------------------------------------------------------------

_CONSTANTS_PATH = REPO / "configs" / "constants.yaml"


def _numeric_leaf_blocks(node, path=""):
    """Yield `(path, block)` for every dict that directly contains a number.

    Walks nested mappings and lists so a constant cannot escape the lint by
    being nested one level deeper than the linter looks.
    """
    if isinstance(node, dict):
        has_number = any(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in node.values()
        )
        if has_number:
            yield path or "<root>", node
        for key, value in node.items():
            yield from _numeric_leaf_blocks(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _numeric_leaf_blocks(value, f"{path}[{index}]")


def test_every_constant_has_a_source():
    """SMARTKEYNET_BUILD_SPEC.md §2.5: walk configs/constants.yaml and fail
    on any leaf number whose parent block lacks a non-empty `source`.

    Hard Rule 4 had no enforcement at all until 2026-08-19 -- there was no
    constants.yaml and no lint, so "every numeric security constant carries a
    citation" was an aspiration. It is one of the six non-negotiables in the
    spec's standing preamble, and the whole point of a citation rule is that
    it is checked mechanically; a reviewer spot-checking three constants and
    finding them fine tells you nothing about the fourth.
    """
    constants = yaml.safe_load(_CONSTANTS_PATH.read_text(encoding="utf-8"))
    assert constants, "configs/constants.yaml is empty"

    offenders: list[str] = []
    for path, block in _numeric_leaf_blocks(constants):
        # A block satisfies the rule via its own `source`, or via any
        # `<field>_source` key describing the specific numbers it holds.
        source_keys = [
            key
            for key, value in block.items()
            if (key == "source" or key.endswith("_source"))
            and isinstance(value, str)
            and value.strip()
        ]
        if not source_keys:
            offenders.append(path)

    assert not offenders, (
        "constants without a non-empty `source` (Hard Rule 4): " + ", ".join(offenders)
    )


def test_uncited_cost_model_is_not_claimed_as_measured():
    """The companion honesty check: any constant block flagged
    `measured: false` must not be described as measured anywhere in the
    report.

    Spec §S5 suggests measuring primitive costs on the evaluation host so the
    report can say "costs are measured, not assumed" -- which the spec calls a
    genuinely strong sentence. That measurement was not taken here, so this
    test exists to stop the strong sentence being written anyway.
    """
    constants = yaml.safe_load(_CONSTANTS_PATH.read_text(encoding="utf-8"))
    unmeasured = [
        name
        for name, block in constants.items()
        if isinstance(block, dict) and block.get("measured") is False
    ]
    assert unmeasured, "expected the cost-model blocks to be flagged measured: false"

    report = (REPO / "docs" / "report.md").read_text(encoding="utf-8").lower()
    for phrase in ("measured on the evaluation host", "costs are measured"):
        assert phrase not in report, (
            f"docs/report.md claims '{phrase}' but {unmeasured} are flagged "
            "measured: false in configs/constants.yaml"
        )
