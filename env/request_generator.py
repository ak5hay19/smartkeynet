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

from env.contracts import Request


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


def random_request_generator(seed: int | None = None) -> Iterator[Request]:
    """Dummy stub stream (no graph) -- unblocks B/C on day 1-2 (split.md
    §1, Person A "ships a stub first").

    Emits synthetic `Request`s at a fixed rate with random tenant/
    sensitivity/pqc_capable fields. Must be swappable 1:1 with
    `RequestGenerator` behind the same call shape (Hard Rule 3 test).
    """
    raise NotImplementedError
