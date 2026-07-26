"""Import-smoke test for `env.pool_sim` (CI baseline — PLAN.md §10, split.md
§3 "minimal CI"). Fails loudly if someone breaks the skeleton; gets
replaced by real behavioral tests as each module is implemented.
"""

import importlib


def test_module_imports():
    module = importlib.import_module("env.pool_sim")
    assert module is not None
