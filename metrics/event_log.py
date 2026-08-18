"""
metrics/event_log.py

The step-level event log (SMARTKEYNET_BUILD_SPEC.md §3.3 and §4.4).

Why this exists rather than the dashboard reaching into the environment:
§S13 requires the dashboard to read `events.jsonl.gz` and "never reach into
env internals". That decoupling buys three things -- the dashboard can replay
any past run, it works offline in the viva, and it cannot slow training down.
It also means the event schema is a contract, so a plotting change can never
silently depend on an environment attribute that later gets renamed.

One JSON object per line, gzipped. Every event carries `t`, `type` and
`episode`; the payload keys per type are fixed by §4.4.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

EVENT_TYPES: frozenset[str] = frozenset(
    {
        "serve",
        "defer_onset",  # this IS the regret_event
        "defer_step",
        "defer_resolved",
        "pool_refill",
        "pool_exhausted",
        "forced_rekey",
        "posture_change",
        "floor_change",
        "attribution",
    }
)
"""The closed set from §4.4. Writing an unlisted type raises rather than
silently producing a log the dashboard cannot interpret."""

REQUIRED_KEYS: frozenset[str] = frozenset({"t", "type", "episode"})


@dataclass
class EventLog:
    """Accumulates events and writes them as gzipped JSONL."""

    episode: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, event_type: str, step: int, **payload: Any) -> None:
        if event_type not in EVENT_TYPES:
            raise ValueError(
                f"unknown event type {event_type!r} -- the schema in spec §4.4 is closed. "
                f"Known: {sorted(EVENT_TYPES)}"
            )
        self.events.append({"t": int(step), "type": event_type, "episode": self.episode, **payload})

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(destination, "wt", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event) + "\n")
        return destination

    def __len__(self) -> int:
        return len(self.events)


def read_events(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream events back. Used by the dashboard, which must not import the
    environment at all."""
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_episode_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append one `episodes.jsonl` row (§3.3).

    The dashboard, the plots and the report table all read this and nothing
    else, so the key set is fixed and validated on write -- a missing key is
    a broken plot three weeks later, and this turns it into an immediate
    error.
    """
    required = {
        "policy",
        "scenario",
        "seed",
        "episode",
        "return",
        "regret_events",
        "pool_exhaustion_events",
        "floor_violations",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"episodes.jsonl row is missing required keys: {sorted(missing)}")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def read_episode_rows(path: str | Path) -> list[dict[str, Any]]:
    destination = Path(path)
    if not destination.exists():
        return []
    with open(destination, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
