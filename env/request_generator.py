"""
env/request_generator.py

NetworkX tenant/service graph and the request stream it emits
(PLAN.md §4 architecture diagram; §10 kickoff step 4). Owned by
Person A (split.md §1).

Hard Rule 3 test: deleting this graph and replacing it with a plain
arrival process must not change one line of agent code -- the graph
only shapes *which requests arrive*, never anything the agent sees
directly.
"""

from __future__ import annotations

from typing import Iterator

import networkx as nx
import numpy as np

from env.contracts import Request, SensitivityClass


def build_tenant_graph(n_nodes: int = 10, seed: int | None = None) -> nx.Graph:
    """Build the synthetic tenant/service graph (PLAN.md "Datasets &
    Provenance" -> "Tenant graph" row: NetworkX synthetic, documented
    generator).

    Edge attrs: `sensitivity_class`, `traffic_rate`, `pqc_capable`
    (legacy endpoints where classical is the only interoperable option
    -> masking makes classical mandatory there; S6 flips `pqc_capable`
    -> true as subsystems upgrade).

    Start with `n_nodes=10` (PLAN.md §10 step 4); scale toward ~50
    once the spine is solid (PLAN.md §4 architecture diagram).
    """
    raise NotImplementedError


class RequestGenerator:
    """Samples a request stream from a tenant graph (PLAN.md §4
    architecture diagram: "request stream <- sampled from graph").

    Ships a random/dummy generator first (split.md §1, Person A "ships
    a stub first") so `env/environment.py` and `agents/` aren't
    blocked on the real graph sampler.
    """

    def __init__(self, graph: nx.Graph, seed: int | None = None) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        """Rewind the request stream for a new episode."""
        raise NotImplementedError

    def step(self, step: int) -> list[Request]:
        """Return the requests that arrive at this step (possibly empty)."""
        raise NotImplementedError


_ARRIVAL_RATE_PER_STEP: float = 1.0
"""Mean arrivals per step (Poisson rate lambda) -- a documented
simulator constant, not a `configs/default.yaml` value (there's no
arrival-rate config key yet; `tenant_graph.n_nodes` is the only
request-generator-adjacent config, and it's `build_tenant_graph`'s
concern, not this stub's)."""

_TENANTS: tuple[str, ...] = ("hospital", "fintech", "logging", "iot-telemetry")
_SERVICES: tuple[str, ...] = ("auth", "billing", "ingest", "export", "notify")
_PQC_CAPABLE_PROB: float = 0.9  # most endpoints support PQC; the rest are legacy (pqc_capable=False)
_HYBRID_MANDATORY_PROB: float = 0.2


def random_request_generator(seed: int | None = None) -> Iterator[Request]:
    """Dummy stub stream (no graph) -- unblocks B/C on day 1-2 (split.md
    §1, Person A "ships a stub first").

    Emits synthetic `Request`s at a fixed rate with random tenant/
    sensitivity/pqc_capable fields. Must be swappable 1:1 with
    `RequestGenerator` behind the same call shape (Hard Rule 3 test).

    "Fixed rate" here means a stationary Poisson arrival process with
    mean `_ARRIVAL_RATE_PER_STEP` requests/step: an infinite generator
    that walks an internal step counter forward, drawing a Poisson
    number of arrivals for each step and yielding one `Request` per
    arrival (in step order) before advancing. Nothing graph-specific
    leaks into the output -- content fields (tenant/service/
    sensitivity_class/pqc_capable/hybrid_mandatory) are drawn
    independently per request from a seeded `numpy` generator, so the
    same `seed` reproduces the exact same stream and different seeds
    diverge.
    """
    rng = np.random.default_rng(seed)
    n_sensitivity_classes = len(SensitivityClass)

    step = 0
    request_index = 0
    while True:
        n_arrivals = int(rng.poisson(_ARRIVAL_RATE_PER_STEP))
        for _ in range(n_arrivals):
            request_index += 1
            yield Request(
                request_id=f"synthetic-{request_index}",
                step=step,
                tenant=str(rng.choice(_TENANTS)),
                service=str(rng.choice(_SERVICES)),
                sensitivity_class=int(rng.integers(0, n_sensitivity_classes)),
                pqc_capable=bool(rng.random() < _PQC_CAPABLE_PROB),
                hybrid_mandatory=bool(rng.random() < _HYBRID_MANDATORY_PROB),
            )
        step += 1
