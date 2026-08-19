"""Tests for `env.contracts` (PLAN.md §4, §4A; split.md §0).

Unlike the other `tests/test_*.py` files, `env.contracts` is real code
(not a stub), so this file checks its actual shapes rather than just
import success -- it's the frozen interface everyone else builds
against.
"""

import numpy as np

from env.contracts import (
    N_ACTIONS,
    Action,
    Request,
    StateDict,
    valid_actions,
)


def test_module_imports():
    import env.contracts as module

    assert module is not None


def test_action_enum_order_is_stable():
    # ActionMask arrays everywhere are positionally aligned to this
    # order (env/contracts.py docstring) -- reordering is a breaking
    # change to the whole codebase.
    assert [a.value for a in Action] == [0, 1, 2, 3, 4]
    assert Action.SERVE_CLASSICAL == 0
    assert Action.SERVE_PQC == 1
    assert Action.SERVE_HYBRID == 2
    assert Action.REUSE == 3
    assert Action.REKEY_NOW == 4
    assert N_ACTIONS == 5


def test_valid_actions_reads_boolean_mask():
    mask = np.array([True, False, True, False, False])
    assert valid_actions(mask) == {Action.SERVE_CLASSICAL, Action.SERVE_HYBRID}


def test_request_typed_dict_accepts_expected_keys():
    request: Request = {
        "request_id": "req-0",
        "step": 0,
        "tenant": "hospital",
        "service": "records-api",
        "sensitivity_class": 3,
        "pqc_capable": True,
        "hybrid_mandatory": True,
    }
    assert request["tenant"] == "hospital"


def test_state_dict_accepts_expected_keys():
    state: StateDict = {
        "threat_score": 0.1,
        "threat_forecast": [0.1, 0.1, 0.1, 0.1, 0.1],
        "qber": 0.02,
        "skr": 500.0,
        "pool_fill": 0.75,
        "arrival_rate": 12.0,
        "load": 0.3,
        "avg_latency": 5.0,
        "key_age": 10.0,
        "key_type_onehot": [0.0, 1.0, 0.0],
        "sensitivity_class": 2,
        "policy_floor": 1,
        "pool_level_hat": [0.7, 0.65, 0.6],
        "skr_mean_hat": [480.0, 460.0, 440.0],
        "hybrid_demand_hat": [2.0, 5.0, 9.0],
        "regret_event_recent": False,
    }
    assert state["pool_fill"] == 0.75
