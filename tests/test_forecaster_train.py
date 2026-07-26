"""Import-smoke test for `forecaster.train` (CI baseline — PLAN.md §10, split.md
§3 "minimal CI"). Fails loudly if someone breaks the skeleton; gets
replaced by real behavioral tests as each module is implemented.
"""

import importlib


def test_module_imports():
    module = importlib.import_module("forecaster.train")
    assert module is not None
