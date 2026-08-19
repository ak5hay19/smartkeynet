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

from collections.abc import Iterator
from dataclasses import dataclass

import networkx as nx
import numpy as np

from env.contracts import Request, SensitivityClass

_ARRIVAL_RATE_PER_STEP: float = 1.0
"""Mean arrivals per step (Poisson rate lambda) -- a documented
simulator constant, not a `configs/default.yaml` value.

Shared by both request sources: `random_request_generator` uses it
directly, and `build_tenant_graph` scales the whole graph's edge rates
so the graph source has the *same* long-run mean. Keeping the two
sources at equal mean load is what makes them a fair substitution for
the Hard Rule 3 test -- they must differ in burstiness and tenant
structure, not in how much work arrives."""

_TENANTS: tuple[str, ...] = ("hospital", "fintech", "logging", "iot-telemetry")
_SERVICES: tuple[str, ...] = ("auth", "billing", "ingest", "export", "notify")
_PQC_CAPABLE_PROB: float = (
    0.9  # most endpoints support PQC; the rest are legacy (pqc_capable=False)
)
_HYBRID_MANDATORY_PROB: float = 0.2


@dataclass(frozen=True)
class TenantFlood:
    """Scenario S4's mechanism: one tenant floods the API.

    PLAN.md §5 S4 ("DDoS / noisy neighbour": "One low-sensitivity
    tenant floods API"), implemented per SMARTKEYNET_BUILD_SPEC.md §S3
    as a per-tenant rate multiplier over a window ("S4 (DDoS)
    multiplies one low-sensitivity tenant's `traffic_rate` by 50 for a
    window").

    Lives here rather than in `env/scenarios.py` because it is a
    property of the request stream; `env/scenarios.py` imports it,
    which keeps the dependency one-directional.
    """

    tenant: str
    start_step: int
    end_step: int
    rate_multiplier: float


@dataclass(frozen=True)
class TenantProfile:
    """Per-tenant flow characteristics for `build_tenant_graph`.

    `class_weights` is the tenant's distribution over
    `SensitivityClass` (S0..S3), indexed positionally. Class assignment
    is deliberately **tenant-conditioned rather than uniform**, per
    SMARTKEYNET_BUILD_SPEC.md §S3: "hospital edges skew
    confidential/secret, telemetry skews public/internal. This
    heterogeneity is what makes budgeting non-trivial -- if every
    tenant looked the same, a threshold rule would be optimal."

    `legacy_fraction` is the share of this tenant's flows terminating
    on endpoints that cannot negotiate PQC (`pqc_capable=False`) --
    migration-era realism, and the thing S6 later flips to `True` as
    subsystems upgrade.
    """

    name: str
    class_weights: tuple[float, float, float, float]
    relative_volume: float
    legacy_fraction: float


_TENANT_PROFILES: tuple[TenantProfile, ...] = (
    # Weights are a documented modelling choice describing what each kind of
    # tenant's traffic looks like -- they are not security constants and set no
    # floors (Hard Rule 4 governs the floor table in env/masking.py, which is
    # what actually maps a class to a minimum tier). The aggregate mix these
    # produce is checked against SMARTKEYNET_BUILD_SPEC.md §3.2's
    # `class_mix: {public: 0.35, internal: 0.35, confidential: 0.20, secret: 0.10}`
    # by tests/test_request_generator.py::test_aggregate_class_mix_matches_spec.
    TenantProfile("hospital", (0.05, 0.15, 0.35, 0.45), relative_volume=0.9, legacy_fraction=0.10),
    TenantProfile("fintech", (0.10, 0.30, 0.50, 0.10), relative_volume=1.0, legacy_fraction=0.02),
    TenantProfile("logistics", (0.30, 0.48, 0.20, 0.02), relative_volume=1.1, legacy_fraction=0.08),
    TenantProfile("telemetry", (0.50, 0.49, 0.01, 0.00), relative_volume=1.3, legacy_fraction=0.05),
    TenantProfile(
        "legacy_scada", (0.40, 0.60, 0.00, 0.00), relative_volume=0.6, legacy_fraction=0.60
    ),
)
"""Realised rate-weighted class mix at `n_nodes=50`, averaged over 8
seeds: **0.389 / 0.312 / 0.216 / 0.084**, against the spec's target of
0.35 / 0.35 / 0.20 / 0.10 (L1 error 0.11). The hybrid-mandatory
fraction it implies -- classes S2+S3 -- is **0.300**, matching
SMARTKEYNET_BUILD_SPEC.md §11.2's "0.20 + 0.10 = 0.30" exactly, which
is the number the scarcity arithmetic in `configs/default.yaml`
depends on.

These weights and volumes were chosen by fitting the realised mix to
that target while holding tenant *character* fixed. An unconstrained
fit over `relative_volume` reaches L1 error 0.047, but only by pushing
telemetry to 0.2x volume and logistics to 3.0x -- which inverts the
premise that telemetry is the high-volume, low-sensitivity tenant, and
would make S4 (telemetry floods the API) incoherent. Matching a
default config's class mix is not worth trading away the realism the
scenarios are built on, so the residual 0.11 is accepted and recorded
here rather than optimised away."""

_HYBRID_MANDATORY_MIN_CLASS: SensitivityClass = SensitivityClass.S2
"""Classes at or above this are emitted with `hybrid_mandatory=True`.

Mirrors `env/masking.py`'s policy table read down its **highest-posture
column**: S2 and S3 both floor at `SERVE_HYBRID` under
`ThreatPosture.HIGH`, so those are exactly the classes that can ever
become hybrid-mandatory. Deriving the flag from the class this way
(rather than the independent 20% coin flip `random_request_generator`
uses) is what makes the graph source's demand internally consistent
with the floors the env will actually compute, and it reproduces
SMARTKEYNET_BUILD_SPEC.md §11.2's "hybrid-mandatory fraction =
0.20 + 0.10 = 0.30".

This is a *mirror*, not an import: `env/request_generator.py` must not
depend on the policy table (the generator runs before any posture is
known). If the floor table changes, this constant and its test move
with it.
"""

_LEGACY_MAX_CLASS: SensitivityClass = SensitivityClass.S0
"""A `pqc_capable=False` edge may not carry a class above this.

SMARTKEYNET_BUILD_SPEC.md §S3 requires the generator to assert this:
"an edge with `pqc_capable: false` must not carry a class whose floor
can exceed T0 under any posture. Otherwise you create an unservable
request (nothing legal), and the masking layer would have to choose
between HR2 and liveness."

KNOWN TENSION, deliberately left alone this session and flagged for a
follow-up: `env/masking.py`'s `compute_mask` does not currently consult
`pqc_capable` at all, so no request is unservable today no matter what
class a legacy edge carries. The spec's rule 2 (interoperability
masking: `if not pqc_capable: mask[SERVE_PQC] = mask[SERVE_HYBRID] =
False`) is not implemented. This generator enforces the invariant
anyway so that turning interoperability masking on later cannot
retroactively produce unservable requests -- but note that even S0
floors at `SERVE_PQC` under `ThreatPosture.HIGH` in the current
placeholder table, so the spec's "floor can exceed T0 under any
posture" reading is strictly unsatisfiable against that table. Closing
that properly needs a policy-table decision (an explicit
legacy-endpoint exemption row), not a generator change.
"""

_MMPP_ON_RATE: float = 0.02
_MMPP_OFF_RATE: float = 0.10
_MMPP_ON_MULTIPLIER: float = 6.0
"""Markov-modulated Poisson (MMPP-2) burst parameters, from
SMARTKEYNET_BUILD_SPEC.md §3.2's `burstiness:` block.

Spec §S3: "an on/off burst state, which produces the peak-to-mean
behaviour real KMS traffic has and a plain Poisson process does not."

**Chains are per-TENANT, not per-edge** (this deviates from the
spec's literal "per edge e" and the deviation is deliberate; see
`measure_fano_factor` for how burstiness is measured and why). Two
reasons, one empirical and one about realism:

  - Empirical: with independent per-edge chains the bursts average
    out. Measured on 40,000 steps, per-edge chains gave a 25-step
    binned Fano factor of 2.35 at 10 edges but only 1.32 at 55 edges
    -- burstiness *fell* as the graph grew, which is backwards. Aggre-
    gating n independent modulators drives the Fano factor toward 1 as
    n grows; the graph cannot be scaled toward the ~50 nodes PLAN.md
    §4 asks for without destroying the very property the MMPP is
    there to provide.
  - Realism: real KMS load spikes are tenant-level events -- a shift
    change, a batch window, an incident -- that light up all of one
    tenant's flows together, not one flow in isolation. Correlated
    tenant bursts are also what make SMARTKEYNET_BUILD_SPEC.md §7.1
    fix A's "save keys for the hospital burst" a real skill rather
    than noise the pool buffer absorbs.
"""

_FANO_BIN_STEPS: int = 25
"""Bin width (steps) at which burstiness is reported.

Per-step arrival counts cannot express burstiness in this environment:
`env/environment.py` decides at most one request per tick, so the
arrival process is calibrated to ~1 request/step and per-step counts
are almost always 0 or 1 -- a series whose Fano factor is pinned near
1.0 by construction, whatever the modulating process does. Measured
per-step Fano is 1.12 for MMPP against 0.99 for plain Poisson: the
right ordering, but no headroom.

Binning at 25 steps measures dispersion at the timescale that actually
matters for the pool, which refills from empty in ~116 steps (see
`configs/default.yaml`'s scarcity calibration). At this bin width the
same two processes separate cleanly. The scenario table should report
the bin width alongside the number -- an unqualified Fano factor is
meaningless here.
"""


def _mmpp_stationary_multiplier() -> float:
    """Long-run mean rate multiplier of the MMPP-2 chain.

    `P(on) = on_rate / (on_rate + off_rate)`; the expected multiplier
    is then `P(on)*on_multiplier + P(off)*1`. Dividing configured edge
    rates by this keeps the graph source's *mean* arrival rate equal to
    its nominal target, so switching sources changes burstiness without
    silently changing load (which would confound every comparison
    against `random_request_generator`).
    """
    p_on = _MMPP_ON_RATE / (_MMPP_ON_RATE + _MMPP_OFF_RATE)
    return p_on * _MMPP_ON_MULTIPLIER + (1.0 - p_on) * 1.0


def build_tenant_graph(
    n_nodes: int = 10,
    seed: int | None = None,
    total_arrival_rate: float = _ARRIVAL_RATE_PER_STEP,
) -> nx.Graph[int]:
    """Build the synthetic tenant/service graph (PLAN.md "Datasets &
    Provenance" -> "Tenant graph" row: NetworkX synthetic, documented
    generator).

    Nodes are service endpoints, each owned by one tenant
    (`node.tenant`, `node.service`). Edges are flows, carrying the
    three attributes the rest of the system reads:

      - `sensitivity_class` -- drawn from the owning tenant's
        `class_weights` (tenant-conditioned, see `TenantProfile`).
      - `traffic_rate` -- mean arrivals per step for this flow, scaled
        so the whole graph's long-run rate equals `total_arrival_rate`
        once MMPP burstiness is averaged out.
      - `pqc_capable` -- `False` on legacy flows; S6 flips these to
        `True` as subsystems upgrade.

    Generation procedure (documented, seeded, reproducible):
      1. Distribute `n_nodes` service nodes round-robin across the
         tenant profiles, so every tenant is represented even at
         `n_nodes=10` (PLAN.md §10 step 4 starts small; §4's
         architecture diagram scales to ~50).
      2. Connect each tenant's nodes into a ring (or a single edge for
         a 2-node tenant), then add one cross-tenant edge per tenant to
         a node of the next tenant, so the graph is connected and flows
         are not perfectly partitioned by tenant.
      3. Assign each edge to the tenant of its lower-indexed endpoint,
         draw its class from that tenant's weights, and mark it legacy
         with probability `legacy_fraction`.
      4. Force any legacy edge's class down to `_LEGACY_MAX_CLASS`
         (see that constant for the invariant and its known tension).
      5. Scale all `traffic_rate` values so the graph's expected
         arrivals per step equal `total_arrival_rate`.

    Raises `ValueError` if `n_nodes` is too small to give every tenant
    profile at least two nodes.
    """
    n_tenants = len(_TENANT_PROFILES)
    if n_nodes < 2 * n_tenants:
        raise ValueError(
            f"n_nodes must be at least {2 * n_tenants} to give each of the "
            f"{n_tenants} tenant profiles two nodes, got {n_nodes}"
        )

    rng = np.random.default_rng(seed)
    graph: nx.Graph[int] = nx.Graph()

    # 1. nodes, round-robin across tenants
    nodes_by_tenant: dict[str, list[int]] = {profile.name: [] for profile in _TENANT_PROFILES}
    for node_index in range(n_nodes):
        profile = _TENANT_PROFILES[node_index % n_tenants]
        service = _SERVICES[(node_index // n_tenants) % len(_SERVICES)]
        graph.add_node(node_index, tenant=profile.name, service=f"{service}-{node_index}")
        nodes_by_tenant[profile.name].append(node_index)

    # 2. intra-tenant ring + one cross-tenant edge per tenant
    edges: list[tuple[int, int]] = []
    for profile_index, profile in enumerate(_TENANT_PROFILES):
        own_nodes = nodes_by_tenant[profile.name]
        if len(own_nodes) == 2:
            edges.append((own_nodes[0], own_nodes[1]))
        else:
            for position, node in enumerate(own_nodes):
                edges.append((node, own_nodes[(position + 1) % len(own_nodes)]))

        next_profile = _TENANT_PROFILES[(profile_index + 1) % n_tenants]
        edges.append((own_nodes[0], nodes_by_tenant[next_profile.name][0]))

    # 3/4. edge attributes, grouped by owning tenant so classes can be
    #      assigned by stratified allocation rather than i.i.d. draws
    edges_by_owner: dict[str, list[tuple[int, int]]] = {
        profile.name: [] for profile in _TENANT_PROFILES
    }
    for source, target in edges:
        if graph.has_edge(source, target) or source == target:
            continue
        owner_name = graph.nodes[min(source, target)]["tenant"]
        graph.add_edge(source, target, tenant=owner_name)
        edges_by_owner[owner_name].append((source, target))

    for profile in _TENANT_PROFILES:
        owned_edges = edges_by_owner[profile.name]
        classes = _stratified_classes(len(owned_edges), profile.class_weights, rng)
        for (source, target), sensitivity_class in zip(owned_edges, classes, strict=True):
            pqc_capable = bool(rng.random() >= profile.legacy_fraction)
            if not pqc_capable:
                sensitivity_class = min(sensitivity_class, int(_LEGACY_MAX_CLASS))

            attrs = graph.edges[source, target]
            attrs["sensitivity_class"] = int(sensitivity_class)
            attrs["pqc_capable"] = pqc_capable
            # Unnormalised rate; step 5 rescales the whole graph at once.
            attrs["traffic_rate"] = float(profile.relative_volume) * float(rng.uniform(0.5, 1.5))

    # 5. scale to the target mean arrival rate, undoing MMPP's mean inflation
    raw_total = sum(attrs["traffic_rate"] for _, _, attrs in graph.edges(data=True))
    if raw_total <= 0.0:
        raise ValueError("generated graph has zero total traffic_rate")
    scale = total_arrival_rate / (raw_total * _mmpp_stationary_multiplier())
    for _, _, attrs in graph.edges(data=True):
        attrs["traffic_rate"] *= scale

    _assert_graph_is_consistent(graph)
    return graph


def _stratified_classes(
    n_edges: int, class_weights: tuple[float, float, float, float], rng: np.random.Generator
) -> list[int]:
    """Allocate `n_edges` sensitivity classes matching `class_weights`
    as closely as an integer allocation can, then shuffle.

    Drawing each edge's class i.i.d. from the weights is the obvious
    implementation and it is too noisy to be usable here: a flow's
    class is fixed for the whole episode, so with only a handful of
    edges per tenant the *realised* class mix is a single small sample
    that can miss entire classes. Measured at `n_nodes=10`, i.i.d.
    draws produced an aggregate mix of 0.71/0.00/0.10/0.19 against a
    0.35/0.35/0.20/0.10 target -- the internal class was absent from
    the graph entirely, which silently deletes a whole tier of demand
    from every scenario built on that seed.

    Largest-remainder allocation removes that variance without
    removing the tenant heterogeneity that matters: each tenant still
    gets *its own* skew (hospital toward S2/S3, telemetry toward
    S0/S1), it is just realised exactly instead of approximately. The
    shuffle keeps class uncorrelated with position in the ring, so
    which flow carries which class still varies by seed.
    """
    if n_edges <= 0:
        return []

    exact_counts = [weight * n_edges for weight in class_weights]
    counts = [int(value) for value in exact_counts]
    remainders = [value - int(value) for value in exact_counts]

    # hand out the leftover slots to the largest fractional remainders
    shortfall = n_edges - sum(counts)
    for class_index in sorted(range(len(counts)), key=lambda i: remainders[i], reverse=True)[
        :shortfall
    ]:
        counts[class_index] += 1

    classes: list[int] = []
    for class_index, count in enumerate(counts):
        classes.extend([class_index] * count)
    rng.shuffle(classes)
    return classes


def _assert_graph_is_consistent(graph: nx.Graph[int]) -> None:
    """Generation-time invariants (SMARTKEYNET_BUILD_SPEC.md §S3's
    "Consistency assertion at generation time"). Raises `ValueError`
    rather than `assert` so the check survives `python -O`."""
    for source, target, attrs in graph.edges(data=True):
        if not attrs["pqc_capable"] and attrs["sensitivity_class"] > int(_LEGACY_MAX_CLASS):
            raise ValueError(
                f"legacy edge ({source}, {target}) carries sensitivity class "
                f"{attrs['sensitivity_class']}, above the permitted "
                f"{int(_LEGACY_MAX_CLASS)} -- see _LEGACY_MAX_CLASS"
            )
        if attrs["traffic_rate"] < 0.0:
            raise ValueError(f"edge ({source}, {target}) has negative traffic_rate")


class RequestGenerator:
    """Samples a request stream from a tenant graph (PLAN.md §4
    architecture diagram: "request stream <- sampled from graph").

    Each edge is an independent MMPP-2 source: it walks an on/off
    Markov chain and, each step, emits `Poisson(lambda_e(t))` requests
    where `lambda_e(t) = traffic_rate * (on_multiplier if on else 1)`.
    See `_MMPP_ON_RATE` for why bursty arrivals rather than plain
    Poisson.

    Hard Rule 3: this class shapes *which* requests arrive and nothing
    else. It is interchangeable with `random_request_generator` behind
    the same `Request` contract, which is what the HR3 substitution
    test exercises -- the agent cannot tell which source it is running
    against, because nothing graph-derived reaches the state vector.

    `tenant_flood`, if given, is the S4 mechanism (PLAN.md §5 S4:
    "One low-sensitivity tenant floods API"): during
    `[start_step, end_step)` every edge owned by `tenant` has its rate
    multiplied by `rate_multiplier`. Unlike the 2026-08-10
    `load_spike` diagnostic in `random_request_generator`, this
    genuinely targets one tenant, which is what makes "protecting
    critical tenants' pool share" a testable question.
    """

    def __init__(
        self,
        graph: nx.Graph[int],
        seed: int | None = None,
        tenant_flood: TenantFlood | None = None,
    ) -> None:
        self._graph = graph
        self._seed = seed
        self._tenant_flood = tenant_flood
        self._edges = [(source, target, attrs) for source, target, attrs in graph.edges(data=True)]
        if not self._edges:
            raise ValueError("cannot generate requests from a graph with no edges")

        # one MMPP chain per tenant, shared by all of that tenant's flows
        self._tenants = sorted({str(attrs["tenant"]) for _, _, attrs in self._edges})
        self._tenant_index = {name: index for index, name in enumerate(self._tenants)}

        self._rng = np.random.default_rng(seed)
        self._tenant_is_on = np.zeros(len(self._tenants), dtype=bool)
        self._request_index = 0
        self.reset()

    def reset(self) -> None:
        """Rewind the request stream for a new episode."""
        self._rng = np.random.default_rng(self._seed)
        self._tenant_is_on = np.zeros(len(self._tenants), dtype=bool)
        self._request_index = 0

    def _advance_mmpp_states(self) -> None:
        """One transition of every tenant's on/off chain.

        Per-tenant rather than per-edge -- see `_MMPP_ON_RATE` for why
        this deviates from the spec's literal per-edge wording.
        """
        draws = self._rng.random(len(self._tenants))
        for index in range(len(self._tenants)):
            if self._tenant_is_on[index]:
                if draws[index] < _MMPP_OFF_RATE:
                    self._tenant_is_on[index] = False
            else:
                if draws[index] < _MMPP_ON_RATE:
                    self._tenant_is_on[index] = True

    def _flood_multiplier(self, tenant: str, step: int) -> float:
        flood = self._tenant_flood
        if flood is None or tenant != flood.tenant:
            return 1.0
        if not flood.start_step <= step < flood.end_step:
            return 1.0
        return flood.rate_multiplier

    def step(self, step: int) -> list[Request]:
        """Return the requests that arrive at this step (possibly empty)."""
        self._advance_mmpp_states()

        arrivals: list[Request] = []
        for source, target, attrs in self._edges:
            rate = float(attrs["traffic_rate"])
            if self._tenant_is_on[self._tenant_index[str(attrs["tenant"])]]:
                rate *= _MMPP_ON_MULTIPLIER
            rate *= self._flood_multiplier(attrs["tenant"], step)

            for _ in range(int(self._rng.poisson(rate))):
                self._request_index += 1
                sensitivity_class = int(attrs["sensitivity_class"])
                arrivals.append(
                    Request(
                        request_id=f"graph-{self._request_index}",
                        step=step,
                        tenant=str(attrs["tenant"]),
                        service=f"{source}-{target}",
                        sensitivity_class=sensitivity_class,
                        pqc_capable=bool(attrs["pqc_capable"]),
                        hybrid_mandatory=sensitivity_class >= int(_HYBRID_MANDATORY_MIN_CLASS),
                    )
                )

        return arrivals

    def as_stream(self) -> Iterator[Request]:
        """Adapt this generator to the flat `Iterator[Request]` shape
        `env/environment.py` already consumes from
        `random_request_generator`.

        Walks an internal step counter forward and yields each step's
        batch in order, so the environment's pending-request deque sees
        an identical interface either way -- this is the seam that
        makes the Hard Rule 3 substitution test a config change rather
        than a code change.
        """
        step = 0
        while True:
            yield from self.step(step)
            step += 1


def measure_fano_factor(per_step_counts: list[int], bin_steps: int = _FANO_BIN_STEPS) -> float:
    """Fano factor (variance / mean) of an arrival-count series,
    measured over `bin_steps`-wide bins.

    SMARTKEYNET_BUILD_SPEC.md §S3: "Report the Fano factor ... in the
    scenario table -- it is one number that tells a reviewer your
    traffic is bursty rather than smooth." A plain Poisson process
    gives ~1.0 at any bin width; correlated MMPP bursts push it above
    1.0 once the bin is wide enough to contain a burst.

    `bin_steps=1` reproduces the textbook per-step definition, which
    in this environment is pinned near 1.0 for every process -- see
    `_FANO_BIN_STEPS` for why, and always report the bin width
    alongside the value.
    """
    if bin_steps < 1:
        raise ValueError(f"bin_steps must be >= 1, got {bin_steps}")

    counts = np.asarray(per_step_counts, dtype=float)
    n_whole_bins = len(counts) // bin_steps
    if n_whole_bins == 0:
        raise ValueError(
            f"need at least {bin_steps} steps to measure a Fano factor at "
            f"bin_steps={bin_steps}, got {len(counts)}"
        )
    binned = counts[: n_whole_bins * bin_steps].reshape(n_whole_bins, bin_steps).sum(axis=1)

    mean = float(binned.mean())
    if mean <= 0.0:
        return 0.0
    return float(binned.var() / mean)


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
            multiplier = (
                load_spike["spike_rate_multiplier"]
                if in_spike
                else load_spike["low_rate_multiplier"]
            )
            rate = _ARRIVAL_RATE_PER_STEP * multiplier
        n_arrivals = int(rng.poisson(rate))
        for _ in range(n_arrivals):
            request_index += 1
            sensitivity_class = int(rng.integers(0, n_sensitivity_classes))
            pqc_capable = bool(rng.random() < _PQC_CAPABLE_PROB)
            hybrid_mandatory = bool(rng.random() < _HYBRID_MANDATORY_PROB)

            # Consistency clamp (2026-08-15). These three fields were
            # drawn independently, which can produce a request that is
            # both hybrid-mandatory and unable to negotiate hybrid --
            # contradictory, and now (since `compute_mask` consults
            # both fields) unservable: it would sit in the deferral
            # queue forever. A legacy endpoint is therefore clamped to
            # the same invariant the graph source enforces
            # structurally: low class, never hybrid-mandatory.
            if not pqc_capable:
                sensitivity_class = min(sensitivity_class, int(_LEGACY_MAX_CLASS))
                hybrid_mandatory = False

            yield Request(
                request_id=f"synthetic-{request_index}",
                step=step,
                tenant=str(rng.choice(_TENANTS)),
                service=str(rng.choice(_SERVICES)),
                sensitivity_class=sensitivity_class,
                pqc_capable=pqc_capable,
                hybrid_mandatory=hybrid_mandatory,
            )
        step += 1
