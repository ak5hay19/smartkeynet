"""Tests for `dashboard.app`, the minimal local server presentation
layer over the already-rendered panels in `dashboard/samples/`.

Route logic is tested against `resolve_route` directly (pure function,
no socket) for speed and hermeticity, plus one real end-to-end HTTP
round trip on an OS-assigned ephemeral port (`build_server(port=0)`),
started and torn down within a single test, to prove the handler
actually wires `resolve_route` up to a real socket correctly.
"""

from __future__ import annotations

import threading
import urllib.request

from dashboard.app import PANELS, build_server, render_index_html, render_index_v2_html, resolve_route

_ALL_REAL_FILENAMES = [pf.filename for panel in PANELS for pf in panel.files]

# Strings that only ever appear in the mockup's own fabricated
# placeholder data/framing -- if any of these show up in a real,
# data-driven page, that page has drifted onto mockup content instead
# of the real PANELS registry.
_MOCKUP_ONLY_MARKERS = (
    "mockup · sample data",
    "illustrative placeholders",
    "capture_2026-08-14.pcapng",
)


def test_panels_registry_has_nine_panels() -> None:
    # Was 7; Living System gained two trained-DQN-driven cards (S1 + S2)
    # this session, alongside the existing StaticThresholdPolicy-driven ones.
    assert len(PANELS) == 9


def test_index_route_returns_200_html() -> None:
    status, content_type, body = resolve_route("/")
    assert status == 200
    assert content_type.startswith("text/html")
    assert b"SmartKeyNet" in body


def test_index_route_references_every_real_panel_filename() -> None:
    # "/" now serves the v2 shell (default since the 2026-09-14 routing
    # flip); it must still reference every real panel file, same as the
    # classic grid always did.
    status, _content_type, body = resolve_route("/")
    assert status == 200
    text = body.decode("utf-8")
    assert len(_ALL_REAL_FILENAMES) == 19  # 3 Living System S1 + 3 S2 + 3 S1-DQN + 3 S2-DQN + 3 Explain Decision + 4 single-file panels
    for filename in _ALL_REAL_FILENAMES:
        assert filename in text, f"default (/) page is missing a reference to {filename}"


def test_root_route_serves_the_v2_shell_by_default() -> None:
    # The 2026-09-14 routing flip: "/" is now an alias for "/v2", not
    # the classic card grid. A parity audit (see PROGRESS.md) confirmed
    # no panel/description/honesty-framing/legend/nav path present on
    # the old grid was missing from the tabbed shell before this flip.
    status_root, _ct_root, body_root = resolve_route("/")
    status_v2, _ct_v2, body_v2 = resolve_route("/v2")
    assert status_root == status_v2 == 200
    assert body_root == body_v2 == render_index_v2_html().encode("utf-8")


def test_index_html_alias_follows_the_default_not_the_classic_grid() -> None:
    # "/index.html" means "whatever's current default" (the v2 shell),
    # not "the classic grid" -- that lives at the explicit /classic path.
    status, _content_type, body = resolve_route("/index.html")
    assert status == 200
    assert body == render_index_v2_html().encode("utf-8")


def test_classic_route_serves_the_old_card_grid_unchanged() -> None:
    # The old card-grid index is kept, byte-for-byte, at an explicit
    # fallback path -- a presentation-critical safety net, not deleted
    # just because /v2 became the default.
    status, content_type, body = resolve_route("/classic")
    assert status == 200
    assert content_type.startswith("text/html")
    assert body.decode("utf-8") == render_index_html()


def test_classic_html_alias_returns_same_body_as_classic() -> None:
    status_a, _ct_a, body_a = resolve_route("/classic")
    status_b, _ct_b, body_b = resolve_route("/classic.html")
    assert status_a == status_b == 200
    assert body_a == body_b


def test_classic_route_references_every_real_panel_filename() -> None:
    status, _content_type, body = resolve_route("/classic")
    assert status == 200
    text = body.decode("utf-8")
    for filename in _ALL_REAL_FILENAMES:
        assert f"/samples/{filename}" in text, f"classic page is missing a link to {filename}"


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


def test_v2_route_returns_200_html() -> None:
    status, content_type, body = resolve_route("/v2")
    assert status == 200
    assert content_type.startswith("text/html")
    assert b"SmartKeyNet" in body


def test_v2_html_route_alias_returns_same_body_as_v2() -> None:
    status_a, _ct_a, body_a = resolve_route("/v2")
    status_b, _ct_b, body_b = resolve_route("/v2.html")
    assert status_a == status_b == 200
    assert body_a == body_b


def test_render_index_v2_html_matches_resolve_route_body() -> None:
    # Same contract as render_index_html/resolve_route("/"): the pure
    # function and the route must not silently diverge.
    _status, _content_type, body = resolve_route("/v2")
    assert body.decode("utf-8") == render_index_v2_html()


def test_v2_shell_references_every_real_panel_filename() -> None:
    # This is the test that would fail if the shell accidentally
    # pointed at mockup data instead of a real dashboard/samples/*.html
    # file: every filename in the real PANELS registry must appear
    # (as a JSON-embedded iframe src target) in the v2 page.
    body = render_index_v2_html()
    for filename in _ALL_REAL_FILENAMES:
        assert filename in body, f"v2 shell is missing a reference to real panel file {filename}"


def test_v2_shell_does_not_reference_the_mockup_file_or_its_fabricated_data() -> None:
    body = render_index_v2_html()
    assert "smartkeynet_dashboard_mockup_v2.html" not in body
    for marker in _MOCKUP_ONLY_MARKERS:
        assert marker not in body, f"v2 shell leaked mockup-only content: {marker!r}"


def test_v2_shell_groups_living_system_panels_into_one_tab() -> None:
    # 9 real panels (4 Living System + Explain Decision + Dose-Response
    # + S3 Comparison Table + Budgeting Brain + Migration Wave) collapse
    # to 6 tabs -- Living System's 4 variants share one tab, switched
    # via a client-side variant selector, everything else gets its own.
    from dashboard.app import _v2_panel_data

    tabs = _v2_panel_data()
    assert len(tabs) == 6
    living_system_tabs = [t for t in tabs if t["tabLabel"] == "Living System"]
    assert len(living_system_tabs) == 1
    assert len(living_system_tabs[0]["variants"]) == 4


def test_v2_shell_tab_data_covers_every_real_panel_exactly_once() -> None:
    from dashboard.app import _v2_panel_data

    tabs = _v2_panel_data()
    seen_filenames = [f["filename"] for t in tabs for v in t["variants"] for f in v["files"]]
    assert sorted(seen_filenames) == sorted(_ALL_REAL_FILENAMES)
    assert len(seen_filenames) == len(set(seen_filenames))  # no file listed twice


def test_classic_index_links_back_to_the_default_v2_shell() -> None:
    status, _content_type, body = resolve_route("/classic")
    assert status == 200
    assert b'href="/"' in body


def test_v2_shell_links_to_classic_index() -> None:
    status, _content_type, body = resolve_route("/v2")
    assert status == 200
    assert b'href="/classic"' in body


def test_v2_shell_shows_a_file_count_badge_per_variant() -> None:
    # Parity finding (2026-09-14 audit): the classic grid shows a
    # "N file(s)" badge per card; the v2 shell must carry equivalent
    # information, not silently drop it. It's computed client-side per
    # active variant, so the check is against the JS scaffold + data.
    body = render_index_v2_html()
    assert "fileCountBadge" in body
    assert "variant.files.length" in body


def test_server_end_to_end_over_real_http() -> None:
    server = build_server(port=0)  # OS-assigned ephemeral port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]

        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=5) as resp:
            assert resp.status == 200
            root_body = resp.read()
            assert b"SmartKeyNet" in root_body

        first_panel_file = PANELS[0].files[0].filename
        with urllib.request.urlopen(f"http://{host}:{port}/samples/{first_panel_file}", timeout=5) as resp:
            assert resp.status == 200

        with urllib.request.urlopen(f"http://{host}:{port}/v2", timeout=5) as resp:
            assert resp.status == 200
            v2_body = resp.read()
            assert b"SmartKeyNet" in v2_body
            assert first_panel_file.encode("utf-8") in v2_body
            assert v2_body == root_body  # "/" is an alias for "/v2", the default

        with urllib.request.urlopen(f"http://{host}:{port}/classic", timeout=5) as resp:
            assert resp.status == 200
            classic_body = resp.read()
            assert b"SmartKeyNet" in classic_body
            assert f"/samples/{first_panel_file}".encode("utf-8") in classic_body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
