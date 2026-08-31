"""Tests for `dashboard.app`, the minimal local server presentation
layer over the six already-rendered panels in `dashboard/samples/`.

Route logic is tested against `resolve_route` directly (pure function,
no socket) for speed and hermeticity, plus one real end-to-end HTTP
round trip on an OS-assigned ephemeral port (`build_server(port=0)`),
started and torn down within a single test, to prove the handler
actually wires `resolve_route` up to a real socket correctly.
"""

from __future__ import annotations

import threading
import urllib.request

from dashboard.app import PANELS, build_server, render_index_html, resolve_route

_ALL_REAL_FILENAMES = [pf.filename for panel in PANELS for pf in panel.files]


def test_panels_registry_has_six_panels() -> None:
    assert len(PANELS) == 6


def test_index_route_returns_200_html() -> None:
    status, content_type, body = resolve_route("/")
    assert status == 200
    assert content_type.startswith("text/html")
    assert b"SmartKeyNet" in body


def test_index_route_references_every_real_panel_filename() -> None:
    status, _content_type, body = resolve_route("/")
    assert status == 200
    text = body.decode("utf-8")
    assert len(_ALL_REAL_FILENAMES) == 10  # 3 Living System + 3 Explain Decision + 4 single-file panels
    for filename in _ALL_REAL_FILENAMES:
        assert f"/samples/{filename}" in text, f"index page is missing a link to {filename}"


def test_render_index_html_matches_resolve_route_body() -> None:
    # render_index_html is the pure function resolve_route delegates to;
    # they must not silently diverge.
    _status, _content_type, body = resolve_route("/")
    assert body.decode("utf-8") == render_index_html()


def test_every_real_panel_file_route_returns_200() -> None:
    for filename in _ALL_REAL_FILENAMES:
        status, content_type, body = resolve_route(f"/samples/{filename}")
        assert status == 200, f"{filename} did not resolve to 200"
        assert content_type.startswith("text/html")
        assert len(body) > 0


def test_nonexistent_sample_path_returns_404() -> None:
    status, content_type, body = resolve_route("/samples/does_not_exist.html")
    assert status == 404
    assert content_type.startswith("text/plain")
    assert body == b"Not Found"


def test_unknown_top_level_path_returns_404() -> None:
    status, _content_type, _body = resolve_route("/nope")
    assert status == 404


def test_path_traversal_outside_whitelist_returns_404() -> None:
    # /samples/ only ever serves the exact registered filenames above --
    # confirm an attempted traversal past the whitelist 404s rather than
    # reading an arbitrary file off disk.
    status, _content_type, _body = resolve_route("/samples/../app.py")
    assert status == 404


def test_server_end_to_end_over_real_http() -> None:
    server = build_server(port=0)  # OS-assigned ephemeral port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]

        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=5) as resp:
            assert resp.status == 200
            assert b"SmartKeyNet" in resp.read()

        first_panel_file = PANELS[0].files[0].filename
        with urllib.request.urlopen(f"http://{host}:{port}/samples/{first_panel_file}", timeout=5) as resp:
            assert resp.status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
