"""Import-smoke test for `agents.soft_reward_baseline` (CI baseline — PLAN.md §10, split.md
§3 "minimal CI"). Fails loudly if someone breaks the skeleton; gets
replaced by real behavioral tests as each module is implemented.
"""

import importlib
from pathlib import Path


def test_module_imports():
    module = importlib.import_module("agents.soft_reward_baseline")
    assert module is not None


# ---------------------------------------------------------------------------
# §S10 test 1 -- the victim must be ABLE to violate a floor
# ---------------------------------------------------------------------------


def _run_weakest_legal(masking_enabled: bool, seed: int = 7, steps: int = 800):
    """Drive the environment with the weakest legal action at every step.

    That is what a fully-steered soft-reward agent converges to, and it is the
    adversarial policy §2.2 specifies for the floor fuzz test -- so it is the
    strongest test of whether floors hold.
    """
    import numpy as np
    import yaml

    from env.contracts import Action
    from env.environment import SmartKeyNetEnv

    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = {
        **base,
        "scenario": "S3",
        "max_steps": steps,
        "scenario_steps": steps + 200,
        "seed": seed,
        "masking": {"enabled": masking_enabled},
    }
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=seed)
    for _ in range(steps):
        weakest = int(np.flatnonzero(info["action_mask"])[0])
        state, _reward, _terminated, truncated, info = env.step(Action(weakest))
        if truncated:
            break
    return env


def test_masking_disabled_permits_floor_violations():
    """§S10 test 1, and the sanity check the whole comparison rests on.

    "sanity: the victim *can* violate, otherwise the comparison is vacuous."

    This was unimplementable until 2026-08-19: there was no way to disable
    masking, so the soft-reward agent ran inside the same inviolable layer as
    ours and reported `floor_violations = 0` for the same structural reason we
    do. The headline contrast -- security-as-constraint holds where
    security-as-reward does not -- was therefore not being measured at all. A
    column of zeros in both rows is not evidence about the reward design.
    """
    advisory = _run_weakest_legal(masking_enabled=False)
    assert advisory.floor_violations > 0, (
        "with floors advisory, an adversarial policy must be able to serve below one -- "
        "otherwise the soft-reward comparison is vacuous"
    )


def test_masking_enabled_permits_none():
    """The other half, and the paper's actual claim: the same adversarial
    policy, the same seed, the same scenario -- zero violations."""
    enforced = _run_weakest_legal(masking_enabled=True)
    assert enforced.floor_violations == 0


def test_masking_is_enabled_by_default():
    """Advisory floors must require a deliberate config change.

    This is the project's central guarantee; it must never be switched off by
    a default, a merge, or a config someone forgot to set.
    """
    import yaml

    from env.environment import SmartKeyNetEnv

    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "masking" not in base or base.get("masking", {}).get("enabled", True) is True

    env = SmartKeyNetEnv({**base, "scenario": "S1", "max_steps": 5})
    assert env.masking_enabled is True


def test_advisory_mode_does_not_relax_physical_constraints():
    """Only the POLICY floor becomes advisory. A legacy endpoint still cannot
    negotiate PQC and an empty pool still cannot fund a hybrid draw -- those
    are facts about the world, not policy, and relaxing them would make the
    victim's numbers meaningless rather than merely worse."""
    import numpy as np

    from env.contracts import Action
    from env.masking import compute_mask

    legacy_request = {
        "request_id": "r",
        "tenant": "legacy_scada",
        "service": "svc",
        "sensitivity_class": 0,
        "pqc_capable": False,
        "hybrid_mandatory": False,
    }
    mask = compute_mask(
        request=legacy_request,
        floor=Action.SERVE_PQC,
        key_age=10.0,
        max_key_age=500.0,
        pool_can_draw=False,
        active_key_tier=Action.SERVE_CLASSICAL,
        enforce_floor=False,
    )
    assert not mask[int(Action.SERVE_PQC)], "a non-PQC endpoint cannot negotiate PQC"
    assert not mask[int(Action.SERVE_HYBRID)], "an empty pool cannot fund a hybrid draw"
    assert mask.any()
