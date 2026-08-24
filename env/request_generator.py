"""
env/request_generator.py

NetworkX tenant/service graph and the request stream it emits
(PLAN.md §4 architecture diagram; §10 kickoff step 4). Owned by
Person A (split.md §1).

Hard Rule 3 test: deleting this graph and replacing it with a plain
arrival process must not change one line of agent code -- the graph
only shapes *which requests arrive*, never anything the agent sees
directly. See `tests/test_environment.py::
test_hard_rule_3_graph_driven_generator_is_a_drop_in_replacement` for
the concrete swap test: an identical episode run against
`random_request_generator` and against `RequestGenerator` (built from
`build_tenant_graph`), through the exact same environment/agent-facing
code.

---------------------------------------------------------------------
Design decisions (2026-08-23 session, `build_tenant_graph`/
`RequestGenerator`):

1. **Tenant identity lives on the node, not the edge.** PLAN.md's
   "Datasets & Provenance" table describes the tenant graph's
   `sensitivity_class`/`traffic_rate`/`pqc_capable` as "edge attrs" --
   written before this session's design decision that tenants must be
   real, *persistent* entities (so S4 can target "flood tenant X" and
   S6 can target "raise tenant cohort Y's floor" -- neither means
   anything if those properties are redrawn per edge/request instead
   of belonging to a stable tenant). This session puts them on each
   tenant node instead, sampled once at graph-build time and read
   (never re-rolled) by every request `RequestGenerator` emits for
   that tenant. This intentionally supersedes the "edge attrs" phrasing.
2. **Topology: a single hub node (`_HUB_NODE_ID`) with one edge to
   each tenant node** (star / hub-and-spoke). Justified two ways: (a)
   it's a direct rendering of the architecture diagram (PLAN.md §4) --
   every tenant's traffic conceptually flows through one shared KMS/
   pool, which a hub node makes explicit and gives a future dashboard
   graph view something simple and meaningful to draw; (b) it keeps
   `build_tenant_graph` honest about what this session actually models
   -- tenants as independent, un-clustered traffic sources -- rather
   than inventing an inter-tenant topology (e.g. tenant-to-tenant
   edges) with no grounding in PLAN.md and no present use.
3. **Services are a small, persistent set per tenant** (1-2, drawn
   without replacement from the existing `_SERVICES` vocabulary
   `random_request_generator` already uses -- reusing it, not inventing
   a second naming scheme, keeps episodes comparable across
   generators). `contracts.py`'s `Request` field is named `service`,
   not `service_id` -- the session brief's `service_id` phrasing does
   not match the frozen contract; `service` is what's actually
   implemented here.
4. **`data_lifetime_days` does not exist.** The session brief assumed
   `env/contracts.py`'s `Request` has a `data_lifetime_days` field it
   could vary per request; the frozen `Request` TypedDict (verified by
   reading the file this session) has no such field --
   `request_id`/`step`/`tenant`/`service`/`sensitivity_class`/
   `pqc_capable`/`hybrid_mandatory` only. Nothing was added to
   `contracts.py` to manufacture one (that file is frozen -- would need
   a flagged, sign-off'd change, not a unilateral addition); this
   design point from the brief is simply inapplicable to the real
   contract and is noted here rather than silently ignored.
5. **`hybrid_mandatory` stays a per-request, tenant-independent draw**
   (same `_HYBRID_MANDATORY_PROB` `random_request_generator` uses) --
   `contracts.py` and PLAN.md never describe it as a tenant property,
   and making it persistent per tenant would silently change Hard
   Rule 9 dynamics (every request from a tenant either always or never
   deferrable) with no request from PLAN.md/split.md to do so.
---------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import networkx as nx
import numpy as np
import yaml

from env.contracts import Request, SensitivityClass


_HUB_NODE_ID = "kms_hub"
"""The single hub node every tenant node connects to (design decision
2 above) -- represents the shared KMS/QKD pool every tenant's requests
conceptually route through."""

_MIN_TENANT_TRAFFIC_RATE = 0.2
_MAX_TENANT_TRAFFIC_RATE = 5.0
"""Bounds for each tenant's persistent `traffic_rate` (relative
weight, not requests/step directly -- `RequestGenerator` normalizes
these into sampling probabilities). A >20x spread between the
quietest and loudest possible tenant gives S4 (one tenant flooding)
real headroom to be dramatic later without this session needing to
build S4 itself."""

_MIN_SERVICES_PER_TENANT = 1
_MAX_SERVICES_PER_TENANT = 2
"""Each tenant gets a small, persistent subset of `_SERVICES` (design
decision 3 above) -- enough for >1 concurrent (tenant, service)
session per tenant (env/environment.py keys session state by that
pair) without every tenant touching every service."""


def load_tenant_graph_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the `tenant_graph:` block out of `configs/default.yaml`.

    Mirrors `env.pool_sim.load_pool_config` / `env.masking.
    load_key_lifetime_config` / `agents.dqn.load_dqn_config`'s existing
    convention -- a standalone, typed-by-caller-usage loader, not
    auto-invoked inside `build_tenant_graph` itself (same relationship
    those three loaders have to their own modules).
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config["tenant_graph"]


def build_tenant_graph(n_nodes: int = 10, seed: int | None = None) -> nx.Graph:
    """Build the synthetic tenant/service graph (PLAN.md "Datasets &
    Provenance" -> "Tenant graph" row: NetworkX synthetic, documented
    generator).

    Topology: one hub node (`_HUB_NODE_ID`) with an edge to each of
    `n_nodes` tenant nodes (design decision 2 above). Node attrs on
    each tenant node (`kind="tenant"`) -- persistent, sampled once
    here, never re-rolled per request:
      - `sensitivity_class` (int, a real `SensitivityClass` value --
        Hard Rule 4, no invented tier scheme)
      - `traffic_rate` (positive float, relative weight -- see
        `_MIN_TENANT_TRAFFIC_RATE`/`_MAX_TENANT_TRAFFIC_RATE`)
      - `pqc_capable` (bool; `False` reads as "legacy endpoint where
        classical is the only interoperable option," Hard Rule 6 --
        same interpretation `random_request_generator` already uses,
        same probability `_PQC_CAPABLE_PROB`)
      - `services` (tuple[str, ...], 1-2 names drawn from `_SERVICES`,
        design decision 3 above)
    The hub node (`kind="hub"`) carries no tenant attributes.

    `n_nodes` defaults to `10`, matching `configs/default.yaml`'s
    `tenant_graph.n_nodes` (PLAN.md §10 step 4: "10 nodes to start");
    pass `load_tenant_graph_config()["n_nodes"]` explicitly to read the
    config-driven value instead of this default. Deterministic under a
    fixed `seed`: identical node/edge set and attributes on repeated
    calls; a different seed changes the sampled attributes (and, since
    `_SERVICES` sampling can differ, sometimes the edge set's
    incidental service assignments) but never the node/edge topology
    itself, which depends only on `n_nodes`.
    """
    rng = np.random.default_rng(seed)
    n_sensitivity_classes = len(SensitivityClass)

    graph = nx.Graph()
    graph.add_node(_HUB_NODE_ID, kind="hub")

    for i in range(n_nodes):
        tenant_id = f"tenant_{i}"
        n_services = int(rng.integers(_MIN_SERVICES_PER_TENANT, _MAX_SERVICES_PER_TENANT + 1))
        services = tuple(str(s) for s in rng.choice(_SERVICES, size=n_services, replace=False))
        graph.add_node(
            tenant_id,
            kind="tenant",
            sensitivity_class=int(rng.integers(0, n_sensitivity_classes)),
            traffic_rate=float(rng.uniform(_MIN_TENANT_TRAFFIC_RATE, _MAX_TENANT_TRAFFIC_RATE)),
            pqc_capable=bool(rng.random() < _PQC_CAPABLE_PROB),
            services=services,
        )
        graph.add_edge(_HUB_NODE_ID, tenant_id)

    return graph


class RequestGenerator:
    """Samples a request stream from a tenant graph (PLAN.md §4
    architecture diagram: "request stream <- sampled from graph").

    Each step, which tenant(s) generate a request is sampled weighted
    by that tenant's persistent `traffic_rate` node attribute (higher-
    traffic tenants produce proportionally more requests); a sampled
    request's `sensitivity_class`/`pqc_capable` are read directly from
    that tenant's node attributes -- never redrawn -- which is what
    gives tenants real, stable identity (the whole point of this
    session, see the module docstring's design decisions). `service`
    is drawn per request from that tenant's small persistent
    `services` set; `hybrid_mandatory` is an independent per-request
    draw (design decision 5).

    Also directly iterable (`iter(generator_instance)`): yields the
    exact same `Request` objects `.step()` would, one at a time in
    step order, as an infinite generator -- the same `Iterator[Request]`
    protocol `random_request_generator` provides. This is the adapter
    that makes this class a genuine drop-in replacement at
    `env/environment.py`'s single injection point (its
    `request_stream_factory` constructor parameter) without that
    module needing to special-case which generator is in use (Hard
    Rule 3).

    ---------------------------------------------------------------------
    S4 (DDoS/noisy-neighbor) mechanism -- `flood_override` (2026-08-24
    session), read this before touching S4 config:

    `flood_override`, if given, is `{"tenant_id": str, "extra_rate":
    float}`. It does **not** mutate the base weighted-sampling loop
    above at all (that code is untouched, byte-for-byte, whether or not
    a flood is active). Instead, every `step()` call additionally draws
    an **independent** `Poisson(extra_rate)` batch of extra requests,
    all attributed to `tenant_id`, using its same persistent node
    attributes as any normal request from that tenant -- drawn from a
    **second, separate RNG stream** (`_flood_rng`) that the base loop's
    `_rng` never touches and is never touched by. This is deliberate,
    not incidental: this class's actual sampling model is a single
    shared `Poisson(_ARRIVAL_RATE_PER_STEP)` total, split across
    tenants by a weighted multinomial draw -- multiplying one tenant's
    `traffic_rate` attribute in that model would *mechanically* steal
    share from every other tenant (a fixed pie resliced), which fails
    this project's own requirement that a flood be *isolated* to the
    designated tenant (verified in `tests/test_environment.py`'s S4
    regression test: other tenants' request counts are measured
    byte-for-byte unaffected by flood activity, not just "close"). A
    second, independent Poisson stream, added on top of the unmodified
    base draw, is what actually delivers that isolation while still
    keeping the flood entirely upstream, in the arrival process, with
    zero visibility to anything downstream (Hard Rule 3) -- a flood
    request is a completely ordinary `Request`, built by the exact same
    field-construction logic (`_build_request`) as any other.
    `flood_override=None` (the default) leaves every line of this
    class's behavior identical to before this addition.
    ---------------------------------------------------------------------
    """

    def __init__(
        self,
        graph: nx.Graph,
        seed: int | None = None,
        flood_override: dict[str, Any] | None = None,
    ) -> None:
        self._graph = graph
        self._seed = seed

        self._tenants: list[tuple[str, dict[str, Any]]] = [
            (node, attrs) for node, attrs in graph.nodes(data=True) if attrs.get("kind") == "tenant"
        ]
        if not self._tenants:
            raise ValueError(
                "graph has no tenant nodes (kind='tenant') -- build it with build_tenant_graph(), "
                "not an empty or hand-built nx.Graph()"
            )
        self._tenant_attrs_by_id: dict[str, dict[str, Any]] = dict(self._tenants)

        traffic_rates = np.array([attrs["traffic_rate"] for _, attrs in self._tenants], dtype=float)
        self._tenant_weights = traffic_rates / traffic_rates.sum()

        if flood_override is not None and flood_override["tenant_id"] not in self._tenant_attrs_by_id:
            raise ValueError(
                f"flood_override tenant_id {flood_override['tenant_id']!r} is not a tenant node in this graph"
            )
        self._flood_override = flood_override

        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._flood_rng: np.random.Generator = np.random.default_rng(seed)
        self._request_index = 0

    def reset(self) -> None:
        """Rewind the request stream for a new episode -- re-seeds both
        RNG streams from the original `seed`, so a fresh `reset()`
        reproduces the exact same stream `__init__` would have
        produced."""
        self._rng = np.random.default_rng(self._seed)
        self._flood_rng = np.random.default_rng(self._seed)
        self._request_index = 0

    def step(self, step: int) -> list[Request]:
        """Return the requests that arrive at this step (possibly
        empty). Base arrival count is Poisson(`_ARRIVAL_RATE_PER_STEP`)
        -- same rate `random_request_generator` uses, so the two
        generators are comparable in overall traffic volume, not just
        interface shape. If `flood_override` is set, an additional,
        independent Poisson(`extra_rate`) batch for that one tenant is
        appended -- see the class docstring's S4 section."""
        n_arrivals = int(self._rng.poisson(_ARRIVAL_RATE_PER_STEP))
        requests: list[Request] = []
        for _ in range(n_arrivals):
            tenant_idx = int(self._rng.choice(len(self._tenants), p=self._tenant_weights))
            tenant_id, attrs = self._tenants[tenant_idx]
            requests.append(self._build_request(step, tenant_id, attrs, self._rng))

        if self._flood_override is not None:
            flood_tenant_id = self._flood_override["tenant_id"]
            flood_attrs = self._tenant_attrs_by_id[flood_tenant_id]
            n_flood_arrivals = int(self._flood_rng.poisson(self._flood_override["extra_rate"]))
            for _ in range(n_flood_arrivals):
                requests.append(self._build_request(step, flood_tenant_id, flood_attrs, self._flood_rng))

        return requests

    def _build_request(
        self, step: int, tenant_id: str, attrs: dict[str, Any], rng: np.random.Generator
    ) -> Request:
        service = str(rng.choice(attrs["services"]))
        self._request_index += 1
        return Request(
            request_id=f"graph-{self._request_index}",
            step=step,
            tenant=tenant_id,
            service=service,
            sensitivity_class=int(attrs["sensitivity_class"]),
            pqc_capable=bool(attrs["pqc_capable"]),
            hybrid_mandatory=bool(rng.random() < _HYBRID_MANDATORY_PROB),
        )

    def __iter__(self) -> Iterator[Request]:
        self.reset()
        step = 0
        while True:
            yield from self.step(step)
            step += 1

    def set_tenant_sensitivity_class(self, tenant_id: str, new_sensitivity_class: int) -> None:
        """Mutate one tenant's persistent `sensitivity_class` node
        attribute after construction -- S6 (migration wave)'s real
        mechanism (2026-08-24 session), read this before touching S6
        config.

        Writes directly into `_tenant_attrs_by_id[tenant_id]` -- the
        exact attribute dict object `nx.Graph.nodes(data=True)` handed
        back at `__init__` time (a live reference, not a copy), so the
        graph node itself is updated too, by construction, with no
        second write needed. Every subsequent request this generator
        emits for `tenant_id` (via the ordinary, completely unmodified
        `step()`/`_build_request` path, base or flood) reads the new
        class -- entirely upstream, in the request-arrival process
        (Hard Rule 3): `_build_request` reads `attrs["sensitivity_class"]`
        fresh on every call, and `env/masking.py`'s floor computation
        reads `sensitivity_class` fresh off each incoming `Request`
        every decision (confirmed by reading both before this method
        was written, not assumed) -- so this one mutation is
        sufficient on its own; no masking.py or environment.py
        floor-computation change was needed or made.

        Validates `new_sensitivity_class` is a real `SensitivityClass`
        value (Hard Rule 4: no invented tiers) -- raises `ValueError`
        via the enum constructor otherwise, same as any other real
        tier value in this codebase."""
        if tenant_id not in self._tenant_attrs_by_id:
            raise ValueError(f"tenant_id {tenant_id!r} is not a tenant node in this graph")
        self._tenant_attrs_by_id[tenant_id]["sensitivity_class"] = int(
            SensitivityClass(int(new_sensitivity_class))
        )


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
    needs the real tenant graph to target genuinely -- that graph
    doesn't exist yet, see `build_tenant_graph` above). This is a
    cruder, tenant-blind stand-in used only to test whether the
    arrival rate itself moving over time changes rekey behavior at
    all -- a prerequisite question before real S4 is worth building.

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
    first cycle instead of oscillating (verified empirically this
    session -- see SESSION_LOG.md 2026-08-10) -- this is why the shape
    has two multipliers, not one. Outside this diagnostic (`load_spike`
    is `None`), the stream is byte-for-byte identical to the
    undecorated stationary process (same `rng` draw sequence) -- this
    keeps the feature additive and backward-compatible.
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
