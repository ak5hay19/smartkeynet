"""
dashboard/app.py

Live demo dashboard (PLAN.md §6, Demo Beats 1-4). Owned by Person D
(split.md §1).

Panels: tenant service graph (edges colored by key type served), QKD
pool gauge, threat forecast strip, live regret counter, latency chart.
Ships first as a static mockup from fake logs (split.md §1, Person D
"ships first") before wiring to a live env/agent.

Implementation choice (Plotly Dash vs. a small React app) is Person
D's call (PLAN.md tech stack: "negotiable"); this stub assumes Plotly
Dash.
"""

from __future__ import annotations

from typing import Any


def load_fake_logs() -> dict[str, Any]:
    """Load a static mock event log so the dashboard shape can be
    agreed on before the real env/agent exist (split.md §1)."""
    raise NotImplementedError


def build_app(logs: dict[str, Any] | None = None) -> Any:
    """Construct the Dash (or equivalent) app object.

    Returns the app object (rather than calling `.run()`) so tests can
    import this module without starting a server.
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
