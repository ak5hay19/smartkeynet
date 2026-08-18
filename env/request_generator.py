"""
env/request_generator.py

NetworkX tenant/service graph and the request stream it emits
(PLAN.md §4 architecture diagram; §10 kickoff step 4). Owned by
Person A (split.md §1).

Hard Rule 3 test: deleting this graph and replacing it with a plain
arrival process must not change one line of agent code -- the graph
only shapes *which requests arrive*, never anything the agent sees
directly. That property is what `random_request_generator` (the
original graph-free stub, kept below and still fully tested) and
`RequestGenerator` (the real graph sampler) being 1:1 swappable
behind the same `Iterator[Request]` shape actually demonstrates;
`tests/test_request_generator.py` asserts it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

import networkx as nx
import numpy as np

from env.contracts import Request, SensitivityClass


# ---------------------------------------------------------------------------
# Tenant profiles -- the documented generator PLAN.md's "Datasets &
# Provenance" table promises for the "Tenant graph" row ("NetworkX
# synthetic (documented generator)").
#
# These are *workload* descriptions (who sends what kind of traffic),
# not security constants: no floor, tier, or key lifetime is decided
# here. Floors are `env/masking.py`'s PolicyTable's job and depend only
# on (sensitivity_class x posture) -- Hard Rule 2/4 are untouched by
# anything in this module.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TenantProfile:
    """One tenant's workload character.

    `sensitivity_weights` is an unnormalized distribution over
    `SensitivityClass` (S0..S3) describing what kind of data this
    tenant's flows carry. `node_share` is its share of the graph's
    service nodes. `pqc_capable_prob` is the fraction of its endpoints
    that speak ML-KEM -- the rest are migration-era legacy endpoints
    (PLAN2 §12: "legacy endpoints flagged as not PQC-capable
    (migration-era realism)"). `traffic_scale` multiplies the traffic
    rate of every edge the tenant owns.
    """

    sensitivity_weights: tuple[float, float, float, float]
    node_share: float
    pqc_capable_prob: float
    traffic_scale: float


_TENANT_PROFILES: dict[str, TenantProfile] = {
    # Patient records: decades-long data lifetime (PLAN2 §3.2), so
    # S3-dominated. Medical devices are the canonical slow-to-migrate
    # endpoint class, hence the lowest pqc_capable_prob.
    "hospital": TenantProfile(
        sensitivity_weights=(0.05, 0.15, 0.30, 0.50),
        node_share=0.25,
        pqc_capable_prob=0.75,
        traffic_scale=0.8,
    ),
    # Regulated financial data: S2-dominated, modern stack, high volume.
    "fintech": TenantProfile(
        sensitivity_weights=(0.05, 0.25, 0.55, 0.15),
        node_share=0.25,
        pqc_capable_prob=0.95,
        traffic_scale=1.2,
    ),
    # Shared observability plane: mostly routine/public, very chatty.
    "logging": TenantProfile(
        sensitivity_weights=(0.55, 0.35, 0.10, 0.00),
        node_share=0.25,
        pqc_capable_prob=0.90,
        traffic_scale=1.5,
    ),
    # Constrained IoT devices: public telemetry, weakest migration story.
    "iot-telemetry": TenantProfile(
        sensitivity_weights=(0.70, 0.25, 0.05, 0.00),
        node_share=0.25,
        pqc_capable_prob=0.55,
        traffic_scale=1.0,
    ),
}

_SERVICES: tuple[str, ...] = ("auth", "billing", "ingest", "export", "notify")

_HYBRID_MANDATORY_BY_CLASS: tuple[float, float, float, float] = (0.00, 0.05, 0.25, 0.50)
"""P(hybrid_mandatory | sensitivity_class), indexed by `SensitivityClass`.

Replaces the graph-free stub's flat, class-blind 0.2. Rising with
sensitivity is the whole point -- "must eventually be served >=
SERVE_HYBRID or be deferred" (Hard Rule 9) is a property of long-lived
regulated data, not a coin flip. Calibrated so the *aggregate* rate
across the default graph stays near the stub's 0.2, which is the rate
the 2026-08-19 pool recalibration was frozen against
(`tests/test_request_generator.py::
test_graph_hybrid_mandatory_rate_matches_the_calibrated_pool_regime`).
"""

_INTER_TENANT_HUB = "logging"
"""Every tenant peers with the shared observability plane -- this is
what puts genuinely cross-tenant flows (and therefore cross-tenant
pool contention) in the graph rather than four isolated islands."""

_INTRA_TENANT_CHORDS_PER_NODE = 1
"""Beyond the ring that guarantees intra-tenant connectivity, each node
gets this many extra random chords, giving a few high-degree services
rather than a uniform ring."""

_TRAFFIC_RATE_LOGNORMAL_SIGMA = 0.6
"""Edge traffic rates are lognormal -- real service meshes are heavy-
tailed (a few flows carry most requests), and a uniform rate would make
every tenant's pool contention identical by construction."""

_ARRIVAL_RATE_PER_STEP: float = 1.0
"""Mean arrivals per step (Poisson rate lambda) -- a documented
simulator constant, not a `configs/default.yaml` value. Shared by both
generators in this module so the graph-driven stream and the graph-free
stub sit at the same aggregate load, making them comparable (and
1:1 swappable -- Hard Rule 3)."""


def build_tenant_graph(n_nodes: int = 10, seed: int | None = None) -> nx.Graph:
    """Build the synthetic tenant/service graph (PLAN.md "Datasets &
    Provenance" -> "Tenant graph" row: NetworkX synthetic, documented
    generator).

    Nodes are `(tenant, service)` pairs carrying `tenant` and `service`
    attributes. Edges are the flows between them and carry the three
    attributes the contract names: `sensitivity_class`, `traffic_rate`,
    `pqc_capable` (legacy endpoints where classical is the only
    interoperable option -> masking makes classical mandatory there;
    S6 flips `pqc_capable` -> true as subsystems upgrade).

    Generation procedure (stated here because PLAN.md requires the
    generator to be documented, not just synthetic):

      1. `n_nodes` service nodes are allocated across the four tenants
         in `_TENANT_PROFILES` by `node_share`, with every tenant
         guaranteed at least one node. Each node draws a service name
         from `_SERVICES`; collisions are disambiguated with a numeric
         suffix so node identity is always unique.
      2. Intra-tenant flows: each tenant's nodes are wired into a ring
         (guaranteeing the tenant's subgraph is connected), plus
         `_INTRA_TENANT_CHORDS_PER_NODE` random chords per node.
      3. Inter-tenant flows: every tenant's first node is peered with a
         node of the shared `_INTER_TENANT_HUB` tenant, so requests
         genuinely contend across tenant boundaries.
      4. Each edge draws `sensitivity_class` from the *stricter* of its
         two endpoints' tenant profiles -- a flow carrying hospital
         data is hospital-grade wherever it terminates. "Stricter" is
         resolved by the tenants' mean sensitivity weight, computed
         from the profiles, not hardcoded.
      5. `traffic_rate` is lognormal (`sigma =
         _TRAFFIC_RATE_LOGNORMAL_SIGMA`) scaled by the owning tenant's
         `traffic_scale`, then used as the sampling weight by
         `RequestGenerator`.
      6. `pqc_capable` is Bernoulli at the stricter endpoint's tenant's
         `pqc_capable_prob`.

    All draws come from a single `numpy.random.default_rng(seed)`, so
    the same seed reproduces the same graph exactly.

    Start with `n_nodes=10` (PLAN.md §10 step 4); scale toward ~50
    once the spine is solid (PLAN.md §4 architecture diagram).
    """
    if n_nodes < len(_TENANT_PROFILES):
        raise ValueError(
            f"n_nodes must be at least one per tenant ({len(_TENANT_PROFILES)}), got {n_nodes}"
        )

    rng = np.random.default_rng(seed)
    graph = nx.Graph()

    # --- 1. allocate nodes to tenants -------------------------------------
    tenants = list(_TENANT_PROFILES)
    counts = {tenant: 1 for tenant in tenants}  # floor of one node each
    remaining = n_nodes - len(tenants)
    if remaining > 0:
        shares = np.array([_TENANT_PROFILES[t].node_share for t in tenants], dtype=float)
        shares = shares / shares.sum()
        extra = _largest_remainder_allocation(remaining, shares)
        for tenant, n_extra in zip(tenants, extra):
            counts[tenant] += int(n_extra)

    nodes_by_tenant: dict[str, list[str]] = {}
    for tenant in tenants:
        tenant_nodes: list[str] = []
        for index in range(counts[tenant]):
            service = _SERVICES[index % len(_SERVICES)]
            suffix = index // len(_SERVICES)
            service_name = service if suffix == 0 else f"{service}-{suffix + 1}"
            node_id = f"{tenant}/{service_name}"
            graph.add_node(node_id, tenant=tenant, service=service_name)
            tenant_nodes.append(node_id)
        nodes_by_tenant[tenant] = tenant_nodes

    # --- 2/3. wire the flows ----------------------------------------------
    edges: list[tuple[str, str]] = []
    for tenant, tenant_nodes in nodes_by_tenant.items():
        if len(tenant_nodes) > 1:
            for i, node in enumerate(tenant_nodes):
                edges.append((node, tenant_nodes[(i + 1) % len(tenant_nodes)]))
            if len(tenant_nodes) > 2:
                for node in tenant_nodes:
                    for _ in range(_INTRA_TENANT_CHORDS_PER_NODE):
                        other = str(rng.choice(tenant_nodes))
                        if other != node:
                            edges.append((node, other))
        else:
            # A single-node tenant has no intra-tenant flow; its only
            # traffic is the inter-tenant peering added below. A
            # self-loop would be a fake flow, so it is deliberately not
            # added -- see the hub peering step.
            pass

    hub_nodes = nodes_by_tenant[_INTER_TENANT_HUB]
    for tenant, tenant_nodes in nodes_by_tenant.items():
        if tenant == _INTER_TENANT_HUB:
            continue
        hub = str(rng.choice(hub_nodes))
        edges.append((tenant_nodes[0], hub))

    # --- 4/5/6. edge attributes -------------------------------------------
    strictness = {
        tenant: float(
            np.dot(np.arange(len(SensitivityClass)), np.asarray(profile.sensitivity_weights))
        )
        for tenant, profile in _TENANT_PROFILES.items()
    }

    for u, v in edges:
        if graph.has_edge(u, v) or u == v:
            continue
        tenant_u = graph.nodes[u]["tenant"]
        tenant_v = graph.nodes[v]["tenant"]
        owner = tenant_u if strictness[tenant_u] >= strictness[tenant_v] else tenant_v
        profile = _TENANT_PROFILES[owner]

        weights = np.asarray(profile.sensitivity_weights, dtype=float)
        sensitivity_class = int(rng.choice(len(SensitivityClass), p=weights / weights.sum()))
        traffic_rate = float(
            rng.lognormal(mean=0.0, sigma=_TRAFFIC_RATE_LOGNORMAL_SIGMA) * profile.traffic_scale
        )
        pqc_capable = bool(rng.random() < profile.pqc_capable_prob)

        graph.add_edge(
            u,
            v,
            tenant=owner,
            sensitivity_class=sensitivity_class,
            traffic_rate=traffic_rate,
            pqc_capable=pqc_capable,
        )

    return graph


def _largest_remainder_allocation(total: int, shares: np.ndarray) -> list[int]:
    """Allocate `total` integer units across `shares` (which sum to 1)
    without drift -- floor everything, then hand the remainder to the
    largest fractional parts. Keeps `sum(result) == total` exactly,
    which naive rounding does not."""
    exact = shares * total
    floors = np.floor(exact).astype(int)
    remainder = total - int(floors.sum())
    if remainder > 0:
        order = np.argsort(-(exact - floors))
        for idx in order[:remainder]:
            floors[idx] += 1
    return floors.tolist()


class RequestGenerator:
    """Samples a request stream from a tenant graph (PLAN.md §4
    architecture diagram: "request stream <- sampled from graph").

    One `Request` per arrival; arrivals per step are Poisson with mean
    `_ARRIVAL_RATE_PER_STEP` (the same aggregate rate the graph-free
    stub uses, so the two are directly comparable). Each arrival picks
    an edge with probability proportional to that edge's
    `traffic_rate`, then copies the flow's `sensitivity_class` and
    `pqc_capable` straight off the edge. `hybrid_mandatory` is drawn
    per request from `_HYBRID_MANDATORY_BY_CLASS`.

    `tenant_rate_multipliers` is the scenario hook: a mapping of
    tenant -> multiplier applied to that tenant's edges' sampling
    weight *and* to the aggregate arrival rate. This is how S4 (one
    low-sensitivity tenant floods the API) is expressed -- as a
    property of the arrival process, exogenous to the agent, which
    never sees the graph or the multipliers (Hard Rule 3).

    Ships alongside (not instead of) `random_request_generator`, which
    stays the Hard Rule 3 swappability check.
    """

    def __init__(
        self,
        graph: nx.Graph,
        seed: int | None = None,
        tenant_rate_multipliers: Mapping[str, float] | None = None,
    ) -> None:
        if graph.number_of_edges() == 0:
            raise ValueError("tenant graph has no edges -- nothing to sample requests from")

        self._graph = graph
        self._seed = seed
        self._tenant_rate_multipliers = dict(tenant_rate_multipliers or {})

        self._edges = list(graph.edges(data=True))
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._request_index = 0
        self._recompute_weights()

    def _recompute_weights(self) -> None:
        rates = np.array(
            [
                data["traffic_rate"] * self._tenant_rate_multipliers.get(data["tenant"], 1.0)
                for _, _, data in self._edges
            ],
            dtype=float,
        )
        total = rates.sum()
        if total <= 0:
            raise ValueError("tenant graph edge traffic rates sum to zero")
        self._edge_probs = rates / total
        # The flood also raises aggregate load, not just its share of it:
        # a noisy neighbour sends *more* requests, it doesn't merely take
        # a bigger slice of a fixed budget.
        baseline = sum(data["traffic_rate"] for _, _, data in self._edges)
        self._arrival_rate = _ARRIVAL_RATE_PER_STEP * (total / baseline)

    @property
    def arrival_rate(self) -> float:
        """Current mean arrivals per step (after any rate multipliers)."""
        return self._arrival_rate

    def set_tenant_rate_multipliers(self, multipliers: Mapping[str, float]) -> None:
        """Replace the per-tenant arrival-rate multipliers mid-episode.

        Used by scenario dispatch (S4's flood window). Exogenous to the
        agent by construction -- nothing here is reachable from a
        `StateDict` or an `ActionMask`.
        """
        self._tenant_rate_multipliers = dict(multipliers)
        self._recompute_weights()

    def reset(self) -> None:
        """Rewind the request stream for a new episode."""
        self._rng = np.random.default_rng(self._seed)
        self._step = 0
        self._request_index = 0

    def step(self, step: int) -> list[Request]:
        """Return the requests that arrive at this step (possibly empty)."""
        n_arrivals = int(self._rng.poisson(self._arrival_rate))
        requests: list[Request] = []
        for _ in range(n_arrivals):
            edge_idx = int(self._rng.choice(len(self._edges), p=self._edge_probs))
            u, v, data = self._edges[edge_idx]
            sensitivity_class = int(data["sensitivity_class"])
            self._request_index += 1
            requests.append(
                Request(
                    request_id=f"graph-{self._request_index}",
                    step=step,
                    tenant=str(data["tenant"]),
                    # The flow is (u, v); the request is served *to* the
                    # initiating endpoint, taken as u. Deterministic
                    # rather than a coin flip so a given edge always maps
                    # to a stable (tenant, service) session key identity
                    # -- env/environment.py keys session state on exactly
                    # that pair.
                    service=str(self._graph.nodes[u]["service"]),
                    sensitivity_class=sensitivity_class,
                    pqc_capable=bool(data["pqc_capable"]),
                    hybrid_mandatory=bool(
                        self._rng.random() < _HYBRID_MANDATORY_BY_CLASS[sensitivity_class]
                    ),
                )
            )
        return requests

    def stream(self) -> Iterator[Request]:
        """Infinite `Iterator[Request]` in step order.

        This is the shape `env/environment.py` consumes (identical to
        `random_request_generator`'s), which is what keeps the graph
        sampler a 1:1 drop-in replacement for the graph-free stub --
        Hard Rule 3's "deleting this graph ... must not change one line
        of agent code", made literal.
        """
        while True:
            for request in self.step(self._step):
                yield request
            self._step += 1


_TENANTS: tuple[str, ...] = ("hospital", "fintech", "logging", "iot-telemetry")
_PQC_CAPABLE_PROB: float = 0.9  # most endpoints support PQC; the rest are legacy (pqc_capable=False)
_HYBRID_MANDATORY_PROB: float = 0.2


def random_request_generator(
    seed: int | None = None,
    load_spike: dict[str, float] | None = None,
) -> Iterator[Request]:
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

    `load_spike`, if given, is a **diagnostic stand-in for a temporary
    load surge -- explicitly NOT real S4 semantics** (PLAN.md §5's S4
    is a specific low-sensitivity tenant flooding the system, which
    needs the real tenant graph to target genuinely). As of 2026-08-19
    that graph exists (`build_tenant_graph`/`RequestGenerator` above)
    and real S4 is dispatched by `env/environment.py` through it, so
    this kwarg is retained only for reproducing the 2026-08-10
    diagnostic runs recorded in SESSION_LOG.md -- it is not the
    scenario, and nothing in the current experiment grid uses it.

    Expected shape: `{"period_steps": int, "spike_duration_steps": int,
    "spike_rate_multiplier": float, "low_rate_multiplier": float}`. The
    rate is `_ARRIVAL_RATE_PER_STEP * spike_rate_multiplier` whenever
    `step % period_steps < spike_duration_steps`, and
    `_ARRIVAL_RATE_PER_STEP * low_rate_multiplier` otherwise -- a
    *periodic* window (recurs, not a one-off): `env/environment.py`
    resets this stream's internal step counter to 0 on every `reset()`,
    and `experiments/train.py` trains on one long continuous episode
    (tens of thousands of ticks) while `experiments/harness.py`'s eval
    episodes are short (`max_steps`, typically 250) and freshly reset --
    a single absolute-step window could satisfy one of those and be
    invisible to the other, while a periodic window is observable by
    both (many repetitions during training; at least one full cycle
    inside a short eval episode).

    `low_rate_multiplier` genuinely needs to be **below 1.0**, not just
    "back to baseline" -- `env/environment.py`'s decision loop renders
    at most one decision per external `step()` call regardless of how
    many requests arrived that tick (see that module's
    `_advance_to_next_decision`), so the undecorated stationary stream
    (`_ARRIVAL_RATE_PER_STEP == 1.0`, one decision/tick) already sits at
    the queue's critical point (arrival rate == service rate) -- there
    is no slack to drain a backlog built up during the spike unless the
    "low" phase actually runs under that critical point. A window that
    reads "elevated in-window, unchanged baseline out-of-window" was
    tried and found to permanently saturate `load` at its cap after the
    first cycle instead of oscillating (verified empirically -- see
    SESSION_LOG.md 2026-08-10) -- this is why the shape has two
    multipliers, not one. Outside this diagnostic (`load_spike` is
    `None`), the stream is byte-for-byte identical to the undecorated
    stationary process (same `rng` draw sequence) -- this keeps the
    feature additive and backward-compatible.
    """
    rng = np.random.default_rng(seed)
    n_sensitivity_classes = len(SensitivityClass)

    step = 0
    request_index = 0
    while True:
        rate = _ARRIVAL_RATE_PER_STEP
        if load_spike is not None:
            period = load_spike["period_steps"]
            duration = load_spike["spike_duration_steps"]
            in_spike = (step % period) < duration
            multiplier = load_spike["spike_rate_multiplier"] if in_spike else load_spike["low_rate_multiplier"]
            rate = _ARRIVAL_RATE_PER_STEP * multiplier
        n_arrivals = int(rng.poisson(rate))
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
