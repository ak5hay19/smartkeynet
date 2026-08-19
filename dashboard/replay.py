"""
dashboard/replay.py

Rebuilds replayable dashboard state from a written `events.jsonl.gz` and
**nothing else**.

---------------------------------------------------------------------
Why this module exists
---------------------------------------------------------------------
SMARTKEYNET_BUILD_SPEC.md §S13 is explicit about the dashboard: it "reads
`events.jsonl.gz` -- never reaches into env internals. This decoupling means
the dashboard can replay any run, works offline in the viva, and cannot slow
training down."

Until 2026-08-19 `dashboard/app.py` did the opposite. It constructed a live
`SmartKeyNetEnv`, stepped it, and read `env._current_request`,
`env._sessions`, `env._policy_table._ratcheted_posture`, `env._regret_log`
and `env._deferral_queue` directly. Three consequences, in increasing order
of how much they matter:

  1. It could not replay a recorded run at all -- every view required
     re-simulating, so a figure could never be regenerated from the run that
     produced it.
  2. It coupled the demo to five private attributes. Any of the renames done
     in this project's refactors would have broken the viva demo silently.
  3. It meant the event-log schema in §4.4 was never exercised, so
     `metrics/event_log.py` sat written, tested, and dead.

This module consumes only the public log. It imports nothing from `env/`,
which `tests/test_api_and_dashboard.py` asserts by AST scan -- the same
enforcement pattern used for Hard Rule 1, because "don't reach into
internals" is exactly as easy to violate by accident.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from metrics.event_log import read_events

_TIER_NAMES: dict[int, str] = {0: "CLASSICAL", 1: "PQC", 2: "HYBRID", 3: "REUSE", 4: "REKEY"}
"""`Action` index -> display name. Duplicated as a literal rather than
imported from `env.contracts` on purpose: importing the env's enums is the
first step back to importing the env."""


@dataclass
class ReplayFrame:
    """One step of dashboard state, reconstructed from the log."""

    step: int
    pool_fill: float
    pool_keys: int
    skr: float
    qber: float
    threat_score: float
    posture: int
    floor: int
    tenant: str
    served_tier: str
    latency_ms: float
    regret_events_total: int
    queue_depth: int
    overflow_keys_total: int


@dataclass
class ReplayEpisode:
    """A full episode's frames plus the counters the panels summarise."""

    label: str
    frames: list[ReplayFrame] = field(default_factory=list)

    @property
    def pool_curve(self) -> list[float]:
        return [frame.pool_fill for frame in self.frames]

    @property
    def regret_curve(self) -> list[int]:
        return [frame.regret_events_total for frame in self.frames]

    @property
    def overflow_curve(self) -> list[int]:
        return [frame.overflow_keys_total for frame in self.frames]

    @property
    def tier_histogram(self) -> dict[str, int]:
        counts = {"CLASSICAL": 0, "PQC": 0, "HYBRID": 0, "REUSE": 0, "REKEY": 0}
        for frame in self.frames:
            if frame.served_tier in counts:
                counts[frame.served_tier] += 1
        return counts

    @property
    def floor_curve(self) -> list[int]:
        return [frame.floor for frame in self.frames]


def frames_from_events(events: Iterable[dict[str, Any]], label: str = "replay") -> ReplayEpisode:
    """Fold an event stream into one frame per `serve`.

    A `serve` is the natural frame boundary: it is exactly one agent decision,
    which is what the scrubber steps through. Pool physics arrive on
    `pool_refill` events that interleave with serves, so the most recent
    refill is carried forward -- the log is ordered by `t`, so "most recent"
    is well defined without sorting.
    """
    episode = ReplayEpisode(label=label)

    pool_keys = 0
    pool_capacity_keys = 1
    skr = 0.0
    qber = 0.0
    overflow_total = 0

    for event in events:
        event_type = event.get("type")

        if event_type == "pool_refill":
            pool_keys = int(event.get("pool_keys_after", 0))
            pool_capacity_keys = max(1, int(event.get("pool_capacity_keys", 1)))
            skr = float(event.get("skr_kbps", 0.0))
            qber = float(event.get("qber", 0.0))
            overflow_total += int(event.get("overflow_keys", 0))
            continue

        if event_type == "serve":
            episode.frames.append(
                ReplayFrame(
                    step=int(event["t"]),
                    pool_fill=pool_keys / pool_capacity_keys,
                    pool_keys=pool_keys,
                    skr=skr,
                    qber=qber,
                    threat_score=float(event.get("threat_score", 0.0)),
                    posture=int(event.get("posture", 0)),
                    floor=int(event.get("floor", -1)),
                    tenant=str(event.get("tenant", "?")),
                    served_tier=_TIER_NAMES.get(int(event.get("tier_served", -1)), "NONE"),
                    latency_ms=float(event.get("latency_ms", 0.0)),
                    regret_events_total=int(event.get("regret_events_total", 0)),
                    queue_depth=int(event.get("queue_depth", 0)),
                    overflow_keys_total=overflow_total,
                )
            )

    return episode


def load_episode(path: str | Path, label: str | None = None) -> ReplayEpisode:
    """Read a written `events.jsonl.gz` and rebuild the episode.

    This is the only entry point the dashboard needs, and it takes a file
    path -- so any run recorded at any time, on any machine, replays.
    """
    resolved = Path(path)
    return frames_from_events(read_events(resolved), label=label or resolved.stem)
