"""Tests for `metrics.event_log` -- the §3.3/§4.4 logging contract."""

from __future__ import annotations

import pytest

from metrics.event_log import (
    EVENT_TYPES,
    EventLog,
    read_episode_rows,
    read_events,
    write_episode_row,
)


def test_round_trips_through_gzipped_jsonl(tmp_path):
    log = EventLog(episode=3)
    log.emit("serve", step=1, request_id="r1", tier_served=2)
    log.emit("defer_onset", step=2, request_id="r2", keys_required=1)
    path = log.write(tmp_path / "events.jsonl.gz")

    recovered = list(read_events(path))
    assert len(recovered) == 2
    assert recovered[0]["type"] == "serve"
    assert recovered[0]["episode"] == 3
    assert recovered[1]["request_id"] == "r2"


def test_every_event_carries_the_required_keys(tmp_path):
    log = EventLog(episode=0)
    for event_type in sorted(EVENT_TYPES):
        log.emit(event_type, step=1)
    for event in log.events:
        assert {"t", "type", "episode"} <= set(event)


def test_unknown_event_type_is_rejected():
    """The schema is closed -- an unlisted type would produce a log the
    dashboard cannot interpret."""
    with pytest.raises(ValueError, match="unknown event type"):
        EventLog().emit("something_invented", step=0)


def test_episode_row_requires_the_fixed_key_set(tmp_path):
    with pytest.raises(ValueError, match="missing required keys"):
        write_episode_row(tmp_path / "episodes.jsonl", {"policy": "dqn"})


def test_episode_rows_append_and_read_back(tmp_path):
    path = tmp_path / "episodes.jsonl"
    for episode in range(3):
        write_episode_row(
            path,
            {
                "policy": "dqn",
                "scenario": "s3",
                "seed": 0,
                "episode": episode,
                "return": -100.0 * episode,
                "regret_events": episode,
                "pool_exhaustion_events": episode,
                "floor_violations": 0,
            },
        )
    rows = read_episode_rows(path)
    assert len(rows) == 3
    assert [r["episode"] for r in rows] == [0, 1, 2]
    assert all(r["floor_violations"] == 0 for r in rows)


def test_missing_episodes_file_reads_as_empty(tmp_path):
    assert read_episode_rows(tmp_path / "nope.jsonl") == []
