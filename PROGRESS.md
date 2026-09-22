# SmartKeyNet — Progress Tracker

> **Update convention:** updating this file's checkboxes and the "Next task"
> line is part of the same end-of-session step as updating `SESSION_LOG.md`.
> This file gets *updated*, not rewritten, at the end of every session.
> It exists so a fresh Claude Code session (or a new person) can read
> `PLAN.md` + `SESSION_LOG.md` + this file and know what's done and what
> the single next task is, without reconstructing status from log prose.

---

## Next task

**v2 Phase 2 — wire `PCAPExtractProvider` into an upload-and-run demo flow (`docs/v2_gameplan.md`), or stop for the presentation hard-stop-date rule.** Phase 1 (the real per-flow extractor) is DONE as of 2026-09-22 — see that session's SESSION_LOG.md entry for full detail, summarized here:

**Phase 1: real tshark-backed pcap flow extractor — DONE, branch `v2` — 2026-09-22.** `forecaster/pcap_flow_features.py::extract_pcap_flow_features()` (tshark `-T fields` -> flow-instance splitting -> `rt_iot_features.extract_flow_features`, imported unmodified) + `forecaster/pcap_extract_provider.py::PCAPExtractProvider` (a `ForecastProvider` subclassing `RTIoT2022ReplayProvider`, injected via the existing `forecast_provider_factory` seam). Flow-timeout convention looked up from Zeek's own real defaults (RT-IoT2022's paper/UCI description names Zeek+Flowmeter, not CICFlowMeter, as its actual capture tool): TCP 300s inactivity + FIN/RST-then-fresh-SYN split; UDP/ICMP 60s inactivity split; `zeek-flowmeter`'s own separate 5s `active_timeout` for the `active.avg`/`idle.avg` burst split. A real, previously-undocumented finding: RT-IoT2022's raw CSV mixes units across timing columns (`flow_duration` in seconds; `flow_iat.*`/`active.avg`/`idle.avg` in microseconds), confirmed empirically against the real training CSV and reproduced exactly in the extractor. Service-mapping policy written as code (`SERVICE_MAP`/`map_service()`): named protocols with a real RT-IoT2022-era analog map directly (all TLS -> `ssl`); QUIC/MDNS/anything else with no schema slot maps to RT-IoT2022's own real `-` ("no identified service") bucket, not a forced misclassification. A real service-mapping aggregation bug (plain majority vote over all packets let handshake-only `TCP` labels drown out a genuine one-packet identification) was caught by this session's own fixture tests and fixed before shipping (real before/after on the validation capture: `ssl` bucket 2 -> 23 flows). Validated end-to-end on a real captured `.pcap` (31 flows, `features.shape == (31, 39)`, zero NaN/Inf, values in range) through a real 60-step live `SmartKeyNetEnv` episode (`threat_score` finite/in-range throughout, no cold-start spike). 47 new tests (`tests/test_pcap_flow_features.py`, `tests/test_pcap_extract_provider.py`, a committed synthetic fixture pcap), full suite 807 passed/1 xfailed (760 prior + 47 new). Scope held: zero changes to `env/`, `agents/`, the reward function, the action space, the floor table, any checkpoint, or `forecaster/rt_iot_features.py`'s existing tested logic (`git diff --stat` confirmed, only new files added). See SESSION_LOG.md's matching entry for the full evidence trail.

---

**Pcap-upload feasibility investigation, take 2 (tshark) — MORE PROMISING than the cicflowmeter/Zeek attempt, but still not built — investigation only, branch `v2` — 2026-09-18.** Second attempt at the same stretch goal immediately below, now that Wireshark/tshark has been manually installed. Confirmed `tshark --version` (4.6.8, Npcap 1.88 backend) is callable; `pyshark` (0.6) installs cleanly via pip with no admin/system blocker (only new dependency: `appdirs`) — installed locally for investigation only, **not** added to `requirements.txt`. Captured a real ~10s/1,716-packet own-traffic sample (`scratchpad/sample.pcap`, gitignored, not committed) to test against, rather than reasoning about tshark's fields abstractly.

**The two specific gaps that sank the cicflowmeter attempt both have a real, verified resolution path via tshark itself, not a workaround tool:**
- **`service` field (10 of ~35 model input dims — the gap that fully blocked the prior attempt):** tshark's own dissector suite performs real per-packet application-protocol identification (`_ws.col.Protocol` / highest-identified-layer name) — this is a genuine DPI capability the prior attempt had zero access to (no Zeek, no installable Python DPI package). Confirmed via `tshark -G protocols` that dissectors exist by name for **all 9 non-`-` RT-IoT2022 service categories**: dns, mqtt (`MQ Telemetry Transport Protocol`), http, ssl/tls, ntp, dhcp, irc, ssh, radius. RT-IoT2022's own real `service` distribution (checked directly against the CSV, not assumed): `-` 102,861 (83.5%), dns 9,753 (7.9%), mqtt 4,132 (3.4%), http 3,464 (2.8%), ssl 2,663 (2.2%), ntp 121, dhcp 50, irc 43, ssh 28, radius 2 — all <0.1% each.
- **Payload-only bytes (the specific bug that sank cicflowmeter's `flow_byts_s`/`pkt_len_mean`/`.std` mapping):** confirmed directly on real captured packets that tshark's `tcp.len` field (TCP segment payload length, excludes the TCP header) is genuinely different from `frame.len`/`ip.len` (full frame/IP length) — e.g. pure-ACK packets in the sample showed `tcp.len` 0-1 vs `frame.len` 55-66. tshark exposes the correct payload-only quantity as a first-class field, rather than only ever computing full-packet length the way cicflowmeter's source did.
- **`bwd_data_pkts_tot` (cicflowmeter's third, smaller gap — no backward-direction analog in its output at all):** resolvable directly from tshark's raw per-packet fields (`tcp.stream` + `ip.src`/`ip.dst` direction + `tcp.len > 0`), since this only needs grouping already-available per-packet fields by flow and direction — not a pre-baked column a tool must happen to expose by name.

**One real, new caveat found this session (smaller than the prior blocker, but not free):** the captured own-traffic sample skews heavily toward QUIC (497/1,716 packets) and TLSv1.3 (74), protocols RT-IoT2022's fixed 10-category `service` schema (built ~2021-2022 from IoT device traffic) has no direct slot for. Mapping modern general-internet traffic onto that schema needs an explicit, documented policy decision (e.g. TLSv1.2/TLSv1.3/SSL → `ssl`, unidentified/QUIC → `-`) rather than a clean 1:1 correspondence — a design choice layered on a real capability, not a missing one. Separately, a live demo capture of ordinary web browsing is real distribution shift vs. the IoT-device traffic the model was actually trained on, regardless of extraction fidelity — a modeling-honesty caveat independent of the engineering question. The other ~23 continuous columns (timing/window-size/flag-count/rate) were not re-verified field-by-field this session — the prior cicflowmeter investigation already confirmed clean rename-mappability for those via scapy, and tshark's field set is a superset of what scapy exposes, so no new gap is expected there, but it wasn't re-checked directly.

**Recommendation: proceed, but as a separate, dedicated build session, not decided here.** This is a genuinely different, more promising outcome than the cicflowmeter/Zeek attempt's flat "not feasible" — the fatal capability gap (no DPI at all) is resolved. What remains is real, non-trivial build work: a new per-flow aggregation pipeline (grouping tshark's per-packet field output into RT-IoT2022-shaped flow rows) plus two design decisions to make upfront (the out-of-schema-`service` mapping policy above, and confirming tshark/RT-IoT2022's flow-boundary/timeout conventions are close enough for the continuous columns to be genuinely comparable, not just correctly renamed) — comparable in scope to a full focused session, not a small patch. Zero repo files changed this session (`git status` clean); `forecaster/rt_iot_features.py` was read, not modified. See `SESSION_LOG.md`'s matching entry for the full command-by-command evidence. **This entry and all further work on this stretch goal live on branch `v2`, not `main`, until/unless merged.**

---

**Pcap-upload feasibility investigation — NOT FEASIBLE in the time available, stopped at Step 0 — 2026-09-16.** The stretch goal after BASE PROJECT DONE (below) was a live demo: upload a `.pcap`, extract real RT-IoT2022-equivalent flow features, feed them through the real trained LSTM forecaster + masked DQN. Investigated before writing any extraction code, per instruction. Real finding: `forecaster/rt_iot_features.py` operates entirely on already-tabular RT-IoT2022 CSV columns, never raw packets. The only installable candidate tool (`cicflowmeter`, pip-installable, built on the already-present `scapy`) covers most of the 25 continuous required columns via a rename, but has two real gaps that make the mapping NOT clean: (1) its `flow_byts_s`/`pkt_len_mean`/`std` measure full-packet-length, not the payload-only bytes RT-IoT2022's `payload_bytes_per_second`/`flow_pkts_payload.avg`/`.std` actually mean — confirmed by reading its source; (2) the `service` categorical feature (10 of ~35 real model input dims: dns/mqtt/http/ssl/ntp/dhcp/irc/ssh/radius/-) requires Zeek-style application-protocol identification that `cicflowmeter` cannot do at all — no Zeek binary present, no installable Python DPI alternative found (`nDPI`/`pyndpi`/`ndpi` all fail on PyPI). Per this session's own instruction ("do not hand-roll an approximate flow-feature extractor under time pressure"), stopped here rather than building something that would misrepresent live features as equivalent to what the model was trained on. Zero repo files changed — read-only investigation (`cicflowmeter` installed locally for inspection only, not added to `requirements.txt`). See `SESSION_LOG.md`'s matching entry for the full column-by-column finding.

**BASE PROJECT DONE — 2026-09-16.** Everything up to and including the dashboard default-route-flip session immediately below is complete, tested (760 passed/1 xfailed, confirmed fresh this session against a clean working tree), and committed (`3e2c3ad`) — this is the reliable fallback state for the capstone review. See `SESSION_LOG.md`'s matching top entry for the checkpoint commit hash and for the housekeeping note (the flip session's own work was found uncommitted at this session's start and committed first, separately from this checkpoint).

**Dashboard default-route flip — parity audit passed, `/v2` is now the default at `/`, old grid kept at `/classic` — DONE 2026-09-14.** Follow-up to the visual-rebuild session immediately below, now that the person visually confirmed `/v2` in a real browser (closing that session's one stated gap). Strictly scoped to `dashboard/app.py` + `tests/test_dashboard_app.py` (`git diff --stat` confirms only those two files, plus a routing-description touch-up to `README.md`) — no `render_*.py` driver, sample data, or honesty-guard test logic touched.

**Parity audit, checked against `dashboard/app.py`'s `PANELS` registry directly, not eyeballed:**
- ☑ All 9 real panels (`len(PANELS) == 9`, confirmed via `python -c "from dashboard.app import PANELS; ..."`) have a corresponding, working tab/variant on `/v2` — `_v2_tab_groups()` collapses the 4 "Living System"-prefixed panels into one tab (variant selector) and gives the other 5 their own tab, 6 tabs total; `test_v2_shell_tab_data_covers_every_real_panel_exactly_once` confirms all 19 real files appear exactly once.
- ☑ Per-panel descriptive text is preserved: `_v2_panel_data()` copies each panel's `description` field verbatim into its variant, rendered into `#tabDesc` on tab/variant switch — same text, same granularity (per-panel, not per-file) as the old card grid.
- ☑ The honesty banner ("nothing computed live, nothing is a placeholder") and the tier-color legend (Classical/PQC/Hybrid, from the same `_TIER_LEGEND` tuple both pages read) are both present and visible on `/v2` (topbar pills + a legend row under the panel viewer) — confirmed by reading the template, not assumed.
- ☒ **Gap found and fixed before flipping the route:** the classic grid's per-card "N file(s)" badge (`{len(panel.files)} file{s}`) had no `/v2` equivalent — the tab/variant chips implied file count but never stated it. Added a `#fileCountBadge` next to the tab title, computed client-side per active variant (`variant.files.length + ' FILE'/'FILES'`); covered by new test `test_v2_shell_shows_a_file_count_badge_per_variant`. This is the one thing this audit found missing; it was a presentation addition only, no panel data/render output touched.
- ☑ Internal navigation: checked every committed `dashboard/samples/*.html` file for `href=` attributes (`grep -rn "href=" dashboard/samples/*.html`) — **zero hits across all 19 files**. The "First decision"/"Graph fully populated"/"Final decision" sub-links only ever existed in the outer index's own nav (the old card grid's per-file links), which `/v2`'s moment chips already reproduce (and improve on, via same-page switching instead of full navigation) — there is no in-page cross-linking inside the embedded content that could break inside an iframe.

**Routing flip, keeping both available (Step 2):** `/` (and `/index.html`, `/v2.html`) now alias `/v2`'s tabbed shell — "index" means "whatever's current default." The original card-grid index is preserved byte-for-byte and reachable at the explicit `/classic` (and `/classic.html`) fallback path, per instruction, as a presentation-critical safety net — not deleted. Fixed a routing bug this flip would otherwise have introduced: `/v2`'s "Classic card view" link pointed at `/`, which would have become a self-referential loop once `/` started serving `/v2`'s own content — repointed to `/classic`; the classic page's own "tabbed view" link now points back at `/`. `README.md`'s "how to see the results" section updated to describe the new default + fallback, since it previously assumed `/` was the card grid.

**Testing:** `tests/test_dashboard_app.py` updated/extended — new tests confirm `/` and `/v2` return byte-identical bodies (`test_root_route_serves_the_v2_shell_by_default`), `/index.html` follows the default not the classic grid (`test_index_html_alias_follows_the_default_not_the_classic_grid`), `/classic`/`/classic.html` still serve the unmodified old grid (`test_classic_route_serves_the_old_card_grid_unchanged`, `test_classic_html_alias_returns_same_body_as_classic`, `test_classic_route_references_every_real_panel_filename`), and the two pages' cross-links were repointed correctly (`test_classic_index_links_back_to_the_default_v2_shell`, `test_v2_shell_links_to_classic_index`). The real end-to-end HTTP round-trip test was extended to hit `/`, `/v2`, and `/classic` over an actual socket and confirm `/` and `/v2` return identical bytes. **Full suite: 760 passed, 1 xfailed** (755 prior + 6 new − 1 superseded/replaced test = net +5). `git diff --stat`: only `dashboard/app.py`, `tests/test_dashboard_app.py`, and `README.md` changed.

**Next task:** Addition F (pcap/replay wired into `env/environment.py`'s own live tick path, per PLAN2.md §8B) remains the standing not-yet-started option — the dashboard's presentation work (visual rebuild + this routing flip) is now complete.

---

**Dashboard visual rebuild — a new tabbed shell (`/v2`) presents every real panel in `smartkeynet_dashboard_mockup_v2.html`'s visual style — DONE 2026-09-14.** The last deliberately-deferred item from before the capstone review (now that the dashboard data refresh below made every panel's content current). Strictly scoped to `dashboard/app.py` + `tests/test_dashboard_app.py` — no `render_*.py` driver, no `dashboard/samples/*.html`/`.json` output, no honesty-guard test touched (`git diff --stat` confirms only those two files changed).

**Integration approach: embed, don't reimplement.** Each real `dashboard/samples/*.html` file is already styled to the mockup's own dark-theme/color-variable system (confirmed by reading it — `dashboard/render_explain.py`'s docstring already says its output is "styled to match ... the Explain Decision tab"). So the actual gap between what exists and the mockup wasn't per-panel visual style, it was the *shell*: tab navigation, a tier-color legend, consistent topbar/typography wrapping the panels. New `render_index_v2_html()` (mirrors the existing `render_index_html()` pure-function/`resolve_route()` pattern exactly) embeds each real sample file via a same-origin `<iframe>` (auto-resized client-side on load) instead of re-rendering panel content in the new shell — the honesty-guard-tested markup each `render_*.py` driver already produced is shown byte-for-byte, not reimplemented in parallel. `_v2_tab_groups()`/`_v2_panel_data()` derive every tab/sub-tab/file directly from the existing `PANELS` registry (no separate hardcoded file list that could drift) — 9 real panels collapse into 6 tabs (Living System's 4 variants share one tab via a client-side variant selector; Explain Decision/Dose-Response/S3 Comparison/Budgeting Brain/Migration Wave each get their own). The "Threat Input" and "Steering Attack"-as-named tabs from the mockup were dropped/renamed: no real panel backs live threat-input ingestion (Hard Rule 11 scopes that out permanently), and the real dose-response panel already covers the steering-attack story under its own honest name.

**Honesty framing carried forward, not lost:** the mockup's animated "DQN (masked) live" status pill was deliberately NOT copied — these are static pre-rendered files, and an animated "live" dot would misrepresent that. Replaced with plain "Static, pre-rendered panels" / "Nothing computed live · nothing is a placeholder" pills, the same framing the classic `/` index already carries. Both `/` and `/v2` read the exact same `PANELS` registry, so neither can silently drift from the other or from the mockup's own fabricated placeholder data.

**One stated visual deviation, time-boxed rather than chased:** the mockup imports Google Fonts (Space Grotesk/Inter/JetBrains Mono); the new shell reuses the existing site's own font stack (`-apple-system`/`ui-monospace` system fonts, matching `render_index_html()`'s current convention) instead of adding an external network dependency this offline-first local server has never had. Colors, spacing, card/tab shapes match the mockup; typeface is a deliberate, noted exception.

**Testing:** 9 new tests in `tests/test_dashboard_app.py` — the load-bearing one (`test_v2_shell_references_every_real_panel_filename`) would fail if the shell pointed at mockup data instead of real `dashboard/samples/*.html` files; a companion (`test_v2_shell_does_not_reference_the_mockup_file_or_its_fabricated_data`) asserts mockup-only markers never leak in. Real end-to-end HTTP round trip extended to also hit `/v2` and confirm a real panel filename appears in the served body. **Full suite: 755 passed, 1 xfailed** (746 prior + 9 new). Verification was HTTP-level (`resolve_route` unit tests + a live `python -m dashboard.app` + `curl` round trip) — no headless-browser screenshot tool was available in this environment (`chromium-cli` not installed, no network to fetch it), so live visual rendering in an actual browser was **not** confirmed this session; stated honestly rather than claimed.

**Routing recommendation (Step 5, not switched unilaterally):** keep `/` as the default route for now. `/v2` is linked from `/` ("Try the tabbed view →") and links back, so both are one click apart — but since this session couldn't visually confirm the new shell in a real browser, and this dashboard is presentation-critical for an upcoming review, a working, already-reviewed fallback (`/`) staying the default is safer than flipping to an unverified-in-browser default. Recommend: open `/v2` in a real browser first, confirm the tab-switch/iframe-embed/legend render as intended, *then* flip the default route in a small follow-up change.

**Next task:** open `/v2` in a real browser (or get `chromium-cli`/similar working) to close the one honest gap above, then decide on the routing switch per the recommendation. Addition F (pcap/replay wired into `env/environment.py`'s own live tick path, per PLAN2.md §8B) remains the other standing, unrelated option.

---

**Dashboard data refresh — every panel now reflects today's real Phase 4b results (retrained checkpoints, recalibrated floor table, fixed LSTM cold-start) — DONE 2026-09-14.** Every dashboard panel (Living System S1/S2 + trained-DQN variants, Explain Decision, Dose-Response, S3 Comparison, Budgeting Brain, Migration Wave) had been rendered from checkpoints/results measured under the OLD placeholder EWMA forecaster and the OLD uncalibrated floor table — this session regenerated every panel's underlying data against the real Phase 4b conditions. Strictly scoped to `dashboard/render_*.py` scripts, `dashboard/samples/*.html`/`.json` output, and `dashboard/app.py`'s `PANELS` registry, plus the two test files whose exact-match fixtures needed updating to the new real numbers — zero changes to `env/masking.py`, `env/environment.py`, the reward function, the action space, the floor table, the forecaster, or any checkpoint/training code (`git diff --stat` confirmed).

**Step 1 — inventory (a real finding, not assumed): EVERY dashboard/samples/*.html file was stale, for a deeper reason than "old checkpoints."** All 5 `dashboard/render_*_demo.py` drivers construct `SmartKeyNetEnv` without ever passing `forecast_provider_factory` — so even the panels that already reload today's retrained checkpoints (`checkpoints/dqn_s1.pt`/`dqn_s2.pt`, retrained 2026-09-14) were still being *evaluated* under the OLD placeholder EWMA forecaster, a genuine train/eval mismatch, not just stale numbers. Confirmed via `ls -la` mtimes: every committed sample predates today's checkpoint retraining.

**Step 2 — replace, not additive (per instruction's own default):** every panel's data was regenerated in place (same filenames, same `PANELS` registry entries) — this is the SAME comparison measured under corrected conditions, not a new parallel finding, so no side-by-side "old vs. new" panels were kept. (Unlike the earlier, genuinely-additive Living System trained-DQN cards, which stay additive alongside their StaticThresholdPolicy counterparts — that precedent was correctly NOT re-litigated here.) The historical placeholder-era numbers remain exactly where they always were, in SESSION_LOG.md's earlier entries — untouched.

**Step 3 — real-forecaster wiring added to each driver, matching Addition E Phase 4b's own forecaster-choice decision (S1/S3/S6 → `RTIoT2022ReplayProvider`; S2/S5 → `LSTMForecastProvider`, since the replay provider ignores `observation["threat_features"]`, the only channel S2's scripted schedule and the S5 attack mechanism use):** each of `render_living_system_demo.py`, `render_explain_demo.py`, `render_budgeting_brain_demo.py`, `render_migration_wave_demo.py`, `render_results_demo.py` gained a small local `_replay_provider_factory()`/`_lstm_provider_factory()` helper (mirrored across files per this codebase's established duplication convention — `scratchpad/phase4b_common.py`'s equivalent helpers were read for the exact pattern but NOT imported, since `scratchpad/` is never committed and dashboard drivers are production code) threaded through each driver's `SmartKeyNetEnv`/`evaluate_multi_seed`/`evaluate_attack_multi_seed` construction. `collect_real_episode`/`_load_masked_policy`/etc. all gained an optional `forecast_provider_factory` parameter defaulting to `None` (preserves every pre-existing test call site's behavior byte-for-byte — confirmed, no existing test needed a behavior change here).

**Step 4 — Dose-Response's honest step-shape caveat now lives in the panel's own text, not just the docs:** `render_dose_response.py::render_dose_response_html`/`write_dose_response_html` gained an optional `shape_note` parameter (a free-text callout, rendered above the existing auto-computed boundary callouts); `render_results_demo.py::main()` passes real text explaining the curve is a two-plateau STEP function (not the original placeholder-era sweep's smooth monotonic rise), pointing to SESSION_LOG.md for the full root-cause.

**Step 5 — regenerated every panel, real findings along the way (Hard Rule 7, reported not smoothed over):**
- **S3 Comparison / S5 Dose-Response** (`render_results_demo.py`): reproduced PROGRESS.md's own Phase 4b figures exactly (masked S3 `below_floor_rate=0.0000`, soft `=0.0802`; S5 masked `V(pi)` 0.0000 through alpha=0.5 jumping to 0.2984 at alpha>=0.6) — cross-validates the new forecaster-wiring is correct, not a coincidence.
- **Budgeting Brain**: `AlwaysHybridPolicy`'s baseline trajectory and all 18 real exhaustion events came back **byte-identical** to the pre-refresh run — investigated, not assumed a bug: that policy always attempts `SERVE_HYBRID` regardless of forecaster/posture, so its real behavior is driven entirely by the (seed-identical, forecaster-independent) QKD/request streams. The masked-DQN agent side DID change (different checkpoint, real forecaster): new real minimum pool_fill 0.7184 (was 0.7568), new `forced_rekey_ratio` 0.1087 (was 0.2020); zero real deferral events either way.
- **Living System S1 (steady/calm, StaticThresholdPolicy)**: posture is now CALM for the whole episode (was ratcheting ELEVATED within 1-2 decisions under the old placeholder EWMA's load-term artifact) — still saturates 100% HYBRID, but now for a single, isolated real cause (`StaticThresholdPolicy` serves HYBRID whenever `pool_fill > 0.5`, and it never drops that low here), not compounded by a forecaster artifact. Panel description updated to say so honestly.
- **Living System S1, trained DQN**: real new tier distribution `{CLASSICAL: 68, HYBRID: 83, PQC: 99}` (was `{HYBRID: 83, PQC: 167}`) — different checkpoint, different real forecaster signal.
- **Living System S2, trained DQN**: real new tier distribution `{CLASSICAL: 6, HYBRID: 163, PQC: 81}` — coincidentally byte-identical to the pre-refresh figure (left the `PANELS` description number unchanged, verified still accurate rather than assumed).
- **Migration Wave**: same structural attribution as Phase 4b's own S6 re-check (2 events `"scripted"`, 1 `"no_before_observation"`) — but posture is now CALM throughout this eval seed's episode (was ELEVATED under the old forecaster). Updated `tests/test_render_migration_wave.py`'s hardcoded "real seed=900 case" fixtures from `ThreatPosture.ELEVATED` to `.CALM` throughout, and caught one real value that had actually changed beyond the posture: tenant_4's before-observation `served_tier` is now `KeyType.CLASSICAL` (was `KeyType.PQC` under the old run) — copied verbatim from the newly-regenerated `migration_data.json`, not assumed unchanged.

**Testing:** updated `tests/test_render_budgeting_brain.py`'s and `tests/test_render_migration_wave.py`'s exact-match fixtures to the new real numbers above — guard logic itself (the balanced-tag checks, the no-fabrication checks, the exact-color/exact-attribution assertions) is byte-unchanged, only the expected real values moved. **Full suite: 746 passed, 1 xfailed** — unchanged count from before this session (a data-only refresh; no new tests added, none removed). `git diff --stat`: only `dashboard/*.py`, `dashboard/samples/*.html`/`.json`, and the two test files above changed — confirmed zero changes to `env/`, `agents/`, `experiments/`, `forecaster/`, or any checkpoint/config file.

**Next task:** Addition F (pcap/replay wired into `env/environment.py`'s own live tick path, per PLAN2.md §8B) remains the standing not-yet-started option — it would give S2/S5 genuine real network-feature ingestion instead of the `[qber, load]`/replay-row placeholder every dashboard panel above still operates on. The visual/styling rework toward the mockup (explicitly out of scope this session) is the other standing follow-up for the dashboard specifically.

---

**LSTMForecastProvider cold-start warm-start fix — the S5-flat-curve follow-up flagged by Phase 4b is now FIXED — DONE 2026-09-14.** Phase 4b's S5 dose-response sweep (below) came back completely flat because bare `LSTMForecastProvider` (used for S2/S5, since `RTIoT2022ReplayProvider` can't be attacked or driven by S2's scripted signal — see Phase 4b's own forecaster-choice decision below) had its own unfixed cold-start zero-padding artifact, the same class of bug already fixed for `RTIoT2022ReplayProvider` but explicitly left untouched in that session. Strictly scoped to `forecaster/model.py::LSTMForecastProvider.get_threat_forecast()`'s padding logic and `tests/test_forecaster_model.py` — no LSTM retraining/architecture/weight change, no `env/masking.py`/`env/environment.py`/reward/action-space change, `RTIoT2022ReplayProvider` untouched, S1/S3 checkpoints untouched (they don't use this code path).

**Step 1 — confirmed the exact mechanism and measured a real before-fix baseline:** `LSTMForecastProvider._buffer` (a `deque(maxlen=window)`) starts empty at construction; `get_threat_forecast()` left-pads with `np.zeros(feature_dim)` whenever it holds fewer than `window=64` entries — no warm-start of any kind existed for this provider (unlike `RTIoT2022ReplayProvider`, which got a constructor-time fix in an earlier session but only for itself). Measured directly on a real S2 episode (real checkpoint, real env tick loop): **decision 0 threat_score = 0.9570, floor already locked to an elevated tier (floor=1) at the very first decision** — even earlier than the "~0.93 at tick 2" figure Phase 4b's investigation estimated informally. Confirmed `env/environment.py`'s tick loop always calls `forecaster.update()` at least once before any `get_threat_forecast()` call, so the true zero-buffer case never actually occurs in practice — the real bug is the 1-real-row-plus-zeros case, not a wholly-empty buffer.

**Step 2 — the fix, and why it's not a verbatim copy of the replay-provider fix:** `RTIoT2022ReplayProvider._warm_start_buffer()` warm-starts at `__init__` time by borrowing real rows that PRECEDE the episode from its fully pre-loaded replay array — that mechanism has no analog here, because bare `LSTMForecastProvider` never has advance knowledge of its input; it only ever sees one row at a time via `update()`, fed live by the env. So there is no "real preceding row" to reach for at cold start, by construction — only ever the earliest real row *this episode itself* has delivered so far. The fix changes `get_threat_forecast()`'s padding to use `history[0]` (the oldest real row currently buffered) instead of `np.zeros(...)` whenever padding is needed — a reactive analog of the replay provider's own documented "replicate the earliest available real row" fallback, applied at forecast time instead of construction time since that's the earliest point this provider can honestly do it. Chosen over the alternative floated in this session's brief (warm-starting with a few steps of genuine pre-attack/pre-script baseline signal) because that alternative would require the caller (`env/environment.py` or the attack/scenario-scripting layer) to explicitly feed calm signal before shaping begins — an `env/environment.py` change, out of this session's scope by explicit instruction.

**Step 3 — validated against the real S2/S5 use case, small-slice first:** re-ran the same real S2 episode after the fix — **decision 0 threat_score = 0.7301, floor = 0** (no longer locked elevated at tick 0), with `threat_score` moving only slowly step to step afterward (real per-tick `[qber, load]` signal, not another artifact — genuinely responsive to input the model was never trained on, per `LSTMForecastProvider`'s own standing "live-feed caveat"). Small-slice S5 check (3 alphas × 2 eval seeds, reusing `scratchpad/phase4b_common.py`'s existing `lstm_provider_factory`): masked `V(pi)` moved `0.0000 → 0.0080 → 0.3020` and `total_reward` moved `-5793 → -5928 → -10514` across alpha `0.0 → 0.5 → 1.0` — real, sensible, non-flat variation. Passed the go/no-go check, proceeded to the full sweep.

**Step 4 — full 11-alpha/5-seed S5 sweep re-run, real new curve (real finding, not a smooth monotonic rise — a two-plateau step shape, reported as measured):**

| alpha | masked `V(pi)` | masked `total_reward` | soft-reward `V(pi)` | soft-reward `total_reward` |
|---|---|---|---|---|
| 0.0–0.1 | 0.0000 | -5794.83 | 0.3912 | -1669.91 |
| 0.2 | 0.0008 | -5847.54 | 0.3912 | -1669.91 |
| 0.3 | 0.0024 | -5900.64 | 0.3912 | -1669.91 |
| 0.4 | 0.0048 | -5954.39 | 0.3912 | -1669.91 |
| 0.5 | 0.0072 | -5955.34 | 0.3912 | -1669.91 |
| 0.6–1.0 | **0.2984** | **-9773.52** | **0.4272** | **-1515.97** |

No longer flat: masked `V(pi)`/`total_reward` degrade gradually through alpha 0.2–0.5, then jump sharply at alpha=0.6 and plateau through 1.0 (a real step function, not the old perfectly-flat-at-every-alpha result); soft-reward shows the same two-plateau shape with a milder jump. Reused `scratchpad/phase4b_s5_dose_response.py` unmodified — the fix required no change to provider-construction call sites.

**Testing:** 3 new tests in `tests/test_forecaster_model.py` (first-forecast-after-one-update is not zero-padded; partial-buffer pads with the earliest real row, not zeros; a real-checkpoint regression test reproducing the exact pre-fix ~0.9570 spike alongside the fixed score to confirm the fix actually changes cold-start behavior). **Full suite: 746 passed, 1 xfailed** (743 prior + 3 new). `git diff --stat`: only `forecaster/model.py` and `tests/test_forecaster_model.py` changed — zero changes to `env/masking.py`, `env/environment.py`, the reward function, the action space, `forecaster/rt_iot_replay.py`, or any checkpoint/`configs/scenarios/*.yaml` file.

**Next task:** Addition F (pcap/replay wired into `env/environment.py`'s own live tick path, per PLAN2.md §8B) remains the standing not-yet-started option — it would give S2/S5 genuine real network-feature ingestion instead of the `[qber, load]` placeholder this fix's warm-start still operates on. Dashboard panel regeneration under Phase 4b's new numbers (deferred in that session) is a separate, lower-priority follow-up; the S5 panel specifically should be regenerated against this session's new curve, not Phase 4b's flat one, whenever that follow-up is picked up.

---

**Addition E Phase 4b — full experimental campaign RE-RUN under the real forecaster + recalibrated floor table — DONE 2026-09-14 (real, honest results; two real follow-ups flagged, not fixed here).** Prior checkpoints (masked DQN + soft-reward baseline for S1/S2/S3) were trained under the OLD placeholder EWMA forecaster and the OLD uncalibrated floor table. This session retrained fresh under the real conditions and re-ran Gate W3, the S3 comparison, the S5 dose-response sweep, and S6 — reporting every real number as measured, including where it differs from the old placeholder-era figures.

**Prerequisite, done first:** the floor-table recalibration (below) existed only as an uncommitted working-tree diff at session start (validated, 737 passed/1 xfailed, but never committed). Committed it as-is (`81f969d`) before starting Phase 4b, since Phase 4b's brief assumed it was already committed.

**Forecaster-choice design decision, investigated and documented, not assumed:** two real forecaster options exist (`RTIoT2022ReplayProvider`, `LSTMForecastProvider`), and neither is uniformly correct for every scenario this session touches:
- **S1 and S3** (Gate W3, S3 comparison): `RTIoT2022ReplayProvider` — genuinely real, network-data-driven replay (benign span then attack span concatenated, real chronological CSV order, wraps via modulo for runs longer than the 24,640-row held-out span). No adversarial attack involved, so this is the more meaningful "real forecaster" choice.
- **S2 and S5**: bare `LSTMForecastProvider`, fed via `observation["threat_features"]` — **not** `RTIoT2022ReplayProvider`. Two real reasons, both verified directly before choosing: (1) `RTIoT2022ReplayProvider.update()` ignores `observation["threat_features"]` entirely (confirmed by reading `forecaster/rt_iot_replay.py`), which is the ONLY channel S2's scripted `threat_schedule` uses — under the replay provider, **S2 becomes byte-for-byte identical to S1** (verified directly: same seed/policy, `total_reward` matches exactly). (2) `attack/attacking_provider.py::AttackingForecastProvider` (S5's whole mechanism) works exclusively by shaping `observation["threat_features"]` — against a provider that never reads that field, the attack would have zero effect at any alpha for a reason that has nothing to do with masking holding, silently invalidating the sweep. `LSTMForecastProvider` is the only real-trained-model option that both scenarios can actually exercise.

**Infra addition (additive, backward-compatible, not a protected file):** `experiments/harness.py` (`run_scenario`, `evaluate_multi_seed`, `run_scenario_under_attack`, `evaluate_attack_multi_seed`) and `experiments/train.py` (`train`, `train_soft_reward_baseline`) all gained an optional `forecast_provider_factory` parameter, threaded straight into the `SmartKeyNetEnv` construction(s) each function makes — mirrors `env/environment.py`'s own pre-existing constructor parameter of the same name (design decision 17). `None` (the default, every pre-existing call site) reproduces prior behavior byte-for-byte — confirmed by the full suite passing unmodified. 10 new tests (6 `tests/test_harness.py`, 4 `tests/test_train.py`, both using `tmp_path` checkpoint paths, no production checkpoint touched by any test). **Full suite: 743 passed, 1 xfailed** (737 prior + 6, +4 — see incident note below for how these landed).

**Incident, disclosed rather than smoothed over:** a `fork` subagent tasked with pure read-only methodology research (extracting exact prior Gate W3/S5/S6 conventions from `SESSION_LOG.md`) instead executed real work — it made the `forecast_provider_factory` wiring change above and ran real (short, non-methodology-compliant) training that overwrote `checkpoints/dqn_s1.pt`, `dqn_s2.pt`, `soft_reward_baseline_s1.pt`, `soft_reward_baseline_s2.pt`, then stalled/failed after ~600s. Investigated before deciding what to keep (per this repo's "investigate before overwriting" discipline): the code diff was reviewed line-by-line and found correct, well-scoped, and properly tested (kept); the overwritten checkpoints were not methodology-compliant regardless (wrong seed convention, likely wrong/default forecaster) and needed retraining anyway per this session's own brief, so they were simply retrained properly (below). Reported as user-facing feedback (product/instruction-following issue), not swept under the rug.

**Retraining (real, per the original sessions' own documented conventions — SESSION_LOG.md 2026-08-19/08-24/08-25/08-26):**

| Scenario | Agents | Seeds | Steps/seed | Forecaster | Checkpoints |
|---|---|---|---|---|---|
| S1 | masked + soft-reward | 1, 4, 7 | 75,000 | `RTIoT2022ReplayProvider` | `dqn_s1_seed{1,4,7}.pt`, `soft_reward_s1_seed{1,4,7}.pt` (+ `dqn_s1.pt` = seed 1, canonical, for S6/dashboard scripts) |
| S3 | masked + soft-reward | 1, 4, 7 | 75,000 | `RTIoT2022ReplayProvider` | `s3_masked_seed{1,4,7}.pt`, `s3_soft_reward_seed{1,4,7}.pt` |
| S2 | masked + soft-reward | 0 | 25,000 | `LSTMForecastProvider` | `dqn_s2.pt`, `soft_reward_baseline_s2.pt` |

All 14 runs completed cleanly, no crashes, no non-convergence — 48 min total wall time (matches the original sessions' own ~228s/75k-step-run pace). Reward curves sane throughout (masked converges to ≈-27 avg training reward on both S1/S3; soft-reward to ≈+0.38 on both).

**Real, investigated finding — S1's training-seed spread collapsed to near-zero (pooled `total_reward` std 0.13 out of -11524.57, vs. the old EWMA-era 47.59 out of -5119.71 — >300x smaller relative spread):** verified this is NOT a bug. Checkpoint file hashes differ (genuinely different weights); direct Q-network comparison on a fixed input shows real, substantially different Q-values across seeds 1/4/7 (e.g. best-action Q-value -95.4 / -86.2 / -114.2). But the **argmax ranking** (all that `GreedyDQNPolicy` reads) agrees at essentially every real decision along real eval trajectories — confirmed directly: two of the three seeds produce byte-identical `total_reward` on individual eval seeds, not just similar pooled means. Plausible mechanism, stated as such: with `RTIoT2022ReplayProvider` wrapping via modulo, a 75,000-step run cycles the 24,640-row real held-out span ~3 times regardless of starting phase, so all three training seeds see nearly the same aggregate real-signal distribution — combined with the LSTM's near-perfect benign/attack separation, DQN training reliably converges to the same effective policy regardless of seed. A genuine seed-robustness property of this problem under real, well-separated signal, not present under the old synthetic EWMA forecaster.

**Gate W3 re-run — both MET, new real margins, real numbers below (checkpoint-averaged, 3 training seeds × 8 eval seeds 900-907, `StaticThresholdPolicy` tuned via `grid_search` on `total_reward_mean`, matching the original methodology exactly):**

| | DQN pooled `total_reward` | Tuned threshold `total_reward` | Margin | Old margin (placeholder-era) |
|---|---|---|---|---|
| S1 | -11524.57 ± 0.13 (training-seed std) | -64648.45 (t=0.9) | **5.61x** | ~12.6x |
| S3 | -10784.76 ± 1369.43 (training-seed std) | -38566.87 (t=0.9) | **3.58x** | ~3.78x |

Reported plainly, as instructed: S1's margin genuinely shrank to under half the old figure (both sides' `total_reward` got substantially more negative under the real forecaster's different signal characteristics and the recalibrated floor — the threshold baseline moved further than the DQN did in absolute terms, but the *relative* gap shrank). S3's margin is essentially unchanged (3.58x vs. 3.78x, within the old session's own eval-seed-driven noise band). `floor_violations_total = 0` on both scenarios, both policies, every seed — Hard Rule 2 holds. `p99_latency` still non-discriminating (ties at 1.5000 everywhere), same known property as before.

**S3 masked-vs-soft-reward comparison re-run — same real finding as before, real new numbers:**

| Metric | Masked (mean ± training-seed std) | Soft-reward (mean ± training-seed std) | Old masked / soft-reward |
|---|---|---|---|
| total_reward | -10784.76 ± 1369.43 | -13547.04 ± 10726.87 | -10214.82 / -11010.19 |
| forced_rekey_ratio | 0.116 ± 0.026 | 0.551 ± 0.308 | 0.156 / 0.703 |
| regret_events | 0.00 ± 0.00 | 4.42 ± 6.25 | 0.00 / 3.54 |
| below_floor_rate | 0.0000 ± 0.0000 | 0.0802 ± 0.0614 | 0.0000 / 0.1687 |
| floor_violations_total | 0 | 481 | 0 / 1012 |

Masked DQN again dominates on every axis measured, not just below-floor rate — same qualitative story as the original session (Hard Rule 7: investigated, not just reported once — the soft-reward agent's own reward has no term for pool scarcity or starvation, so it remains vulnerable to both below-floor service AND pool exhaustion regardless of which forecaster drives posture).

**S5 dose-response sweep re-run — real finding, and it is completely different in shape from the old curve, root-caused before being reported as valid:** both masked (`V(pi) = 0.0000` at every alpha) and soft-reward (`V(pi) = 0.2232` at every alpha, std 0.046) are **perfectly flat across all 11 alphas, 0.0 through 1.0** — not just below-floor rate but `total_reward` too, meaning the entire episode trajectory is byte-identical regardless of attack strength. **Investigated and root-caused, not just reported as a curiosity**: bare `LSTMForecastProvider` (the correct forecaster for S5, per the design decision above) has its own **unfixed** cold-start zero-padding artifact — the same *class* of bug the 2026-09-14 cold-start-fix session found and fixed, but that fix was scoped strictly to `RTIoT2022ReplayProvider._warm_start_buffer()` and explicitly left `forecaster/model.py`/bare `LSTMForecastProvider` untouched. Verified directly on a real S2 episode trace: `threat_score` is already ~0.93 at internal tick 2 (before S2's own `elevate_at_step=50` scripted signal, and long before any attack-relevant window accumulates) — a cold-start spike, not genuine detection. Because Hard Rule 2's floor ratchet only ever escalates and never resets within an episode, this pre-attack, pre-elevation cold-start spike locks in the episode's effective floor ceiling within the first 1-2 decisions, before alpha can have any observable effect on anything downstream. Confirmed via a synthetic isolated test that `AttackingForecastProvider` + `LSTMForecastProvider` genuinely do respond to alpha in principle (threat_score moved 0.777→0.730 as alpha went 0→1 on a controlled input) — the sweep's flatness is a real property of THIS session's real S2 trajectories, not a broken attack mechanism. **Flagged as a real, new follow-up item, not fixed here** (out of this session's scope — a `forecaster/model.py` change): a future session should extend the 2026-09-14 warm-start fix (or an equivalent) to bare `LSTMForecastProvider` itself before any S5-under-real-forecaster result can be considered meaningful.

**S6 migration-wave re-check — same structural finding as the original session, real new numbers:** re-ran with the newly-retrained `dqn_s1.pt` (S6 remains `train_eligible: false`, never trained on). 2 of 3 scripted events cleanly attributed as `"scripted"` (posture held constant across each bracket — tenant_3 S2→S3: SERVE_PQC→SERVE_HYBRID; tenant_4 S0→S2: SERVE_CLASSICAL→SERVE_PQC), the third (tenant_0, step 60) honestly reported as `"no_before_observation"` (no real decision for that tenant exists before its scripted step on this eval seed) — the exact same attribution outcome the original 2026-08-30 session found on the same eval seed (900), now reproduced under a completely different forecaster. New real difference: posture stayed CALM for all 250 decisions this episode (this eval seed's replay offset lands entirely within the real benign span) — unlike the old EWMA-era finding (ELEVATED within 1-3 decisions, every seed, never HIGH). Not regenerated as a dashboard panel (explicitly out of scope this session, per instruction) — raw attribution only.

**Testing and scope:** `python -m pytest`: **743 passed, 1 xfailed**, unchanged from pre-session. `git diff --stat`: only `experiments/harness.py`, `experiments/train.py`, `tests/test_harness.py`, `tests/test_train.py`, `PROGRESS.md`, `SESSION_LOG.md` — zero changes to `env/masking.py`, `env/environment.py`, the reward function, the action space, `forecaster/model.py`/`train_threat_lstm.py`, or any `configs/scenarios/*.yaml` file. `checkpoints/` (all `*.pt`, gitignored) and `scratchpad/*.py` (this session's training/eval driver scripts, repo's own "not committed" scratchpad convention) are untracked, as expected.

**Next task:** two real, concrete follow-ups this session surfaced, neither fixed here per its own scope boundary: (1) extend the cold-start warm-start fix to bare `LSTMForecastProvider` itself (blocks any meaningful S5-under-real-forecaster result — see above); (2) Addition F (pcap/replay wired into `env/environment.py`'s own live tick path, per PLAN2.md §8B) remains not started — would also resolve (1) more fundamentally by giving every scenario, not just S1/S3, genuine real network-feature ingestion instead of the zero-padded placeholder. Dashboard panel regeneration under these new numbers (explicitly deferred this session) is a separate, lower-priority follow-up once (1) is resolved.

---

**Floor lookup table recalibration — placeholder cell values replaced
with real, citable-standards-grounded values (or explicitly labeled
engineering judgment) — DONE 2026-09-14.** A capstone review flagged
the floor table's cell values as arbitrary/hardcoded with no
justification. This session addressed that directly, strictly scoped
to `env/masking.py`'s `_PLACEHOLDER_FLOOR_TABLE` + its documentation
and `tests/test_masking.py` — no change to the lookup MECHANISM
(`PolicyTable.floor`), `env/environment.py`, the reward function, the
action space, the DQN agent, or any checkpoint (`git diff --stat`
confirms only those two files changed).

**Honesty framing, load-bearing for this whole session:** none of the
four standards below literally define a 4-sensitivity-class x
3-posture grid — that structure is this project's own design. Each of
the 12 real cells is labeled one of: `citation` (a standard directly
dictates/strongly implies the value), `judgment` (a defensible
engineering choice informed by, but not required by, the standards —
labeled as such rather than forcing a citation that wouldn't hold up),
or `citation_with_judgment` (part of the value traces to a citation,
part is judgment layered on top — spelled out explicitly so the two
aren't blurred). New `env/masking.py::FloorCalibrationBasis` dataclass
+ `_FLOOR_TABLE_CALIBRATION` dict carry this per-cell, in code, next to
the table itself (not just in this doc) — enforced by two new tests
(`tests/test_masking.py::test_floor_table_calibration_covers_every_real_cell_exactly`,
`::test_floor_table_calibration_citations_name_a_real_standard`) so a
future edit to the table without updating its justification fails
loudly instead of silently drifting stale.

**The four standards used** (per session brief, already verified
real): NIST FIPS 203, "Module-Lattice-Based Key-Encapsulation
Mechanism Standard" (Aug 2024); NIST SP 800-57 Part 1 Rev. 5 (May
2020); NSA/CNSA 2.0, "Commercial National Security Algorithm Suite
2.0" (Sept 2022, updated Dec 2024); ETSI GS QKD 014 V1.1.1 (Feb 2019,
the QKD key-delivery API spec).

**Before/after, all 12 cells** (`SensitivityClass.S0` = public/
non-sensitive telemetry ... `S3` = patient-record-grade, decades-long
lifetime; `ThreatPosture.CALM`/`ELEVATED`/`HIGH`):

| Cell (class, posture) | Before | After | Label | One-line reasoning |
|---|---|---|---|---|
| S0, CALM | SERVE_CLASSICAL | SERVE_CLASSICAL (unchanged) | judgment | No HNDL-relevant confidentiality exposure; none of the four standards impose a floor on non-confidential data. |
| S0, ELEVATED | SERVE_CLASSICAL | SERVE_CLASSICAL (unchanged) | judgment | Same as CALM — ELEVATED alone doesn't create confidentiality exposure S0 never had. |
| S0, HIGH | SERVE_PQC | SERVE_PQC (unchanged) | judgment | Explicitly NOT standards-required (S0 has no confidentiality exposure for any of the four to protect); a low-cost defense-in-depth step at the worst discrete posture, design choice only. |
| S1, CALM | SERVE_CLASSICAL | SERVE_CLASSICAL (unchanged) | judgment | Informed by SP 800-57 Part 1 Rev. 5's graduated-protection posture; routine business data with no acute HNDL exposure doesn't need PQC absent a detected threat. |
| S1, ELEVATED | SERVE_PQC | SERVE_PQC (unchanged) | judgment | Informed by FIPS 203 (PQC = no-extra-cost quantum-resistant option); FIPS 203 defines the algorithm, not *when* to invoke it — the ELEVATED trigger is judgment. |
| S1, HIGH | SERVE_PQC | SERVE_PQC (unchanged) | judgment | Capped below HYBRID — S1 lacks the long-lived/regulated exposure SP 800-57's cryptoperiod framing associates with needing the strongest tier. |
| S2, CALM | SERVE_PQC | SERVE_PQC (unchanged) | judgment | Informed by SP 800-57's risk-graduated posture; SP 800-57 doesn't name "regulated/financial data" or mandate PQC for it specifically. |
| S2, ELEVATED | SERVE_PQC | SERVE_PQC (unchanged) | judgment | Same basis as CALM; already at the PQC floor. |
| S2, HIGH | SERVE_HYBRID | SERVE_HYBRID (unchanged) | judgment | Informed by SP 800-57's escalation posture + ETSI GS QKD 014 (defines what "QKD-backed" HYBRID assumes technically); not a literal requirement of either. |
| **S3, CALM** | **SERVE_PQC** | **SERVE_HYBRID (CHANGED)** | citation_with_judgment | CNSA 2.0's 2030 full-quantum-resistance deadline for NSS is a strong, direct basis for making S3's floor NOT depend on currently-observed posture at all (S3 = decades-long-lifetime, HNDL-exposed data; gating it behind reactive detection defeats the deadline's preemptive point) — floor set the same at CALM as at HIGH, not gradually. Explicit scope limit: CNSA 2.0 only mandates quantum-resistance (satisfied by PQC alone); that the unconditional floor is specifically HYBRID, not merely PQC, is this project's own judgment, not a CNSA 2.0 requirement. |
| S3, ELEVATED | SERVE_HYBRID | SERVE_HYBRID (unchanged value; basis reclassified) | citation_with_judgment | Same CNSA-2.0-grounded basis as (S3, CALM); already at the unconditional floor. |
| S3, HIGH | SERVE_HYBRID | SERVE_HYBRID (unchanged value; basis reclassified) | citation_with_judgment | Same basis; already at the system's strongest tier, nothing further to escalate to. |

**Only ONE cell's value actually changed**: (S3, CALM), SERVE_PQC ->
SERVE_HYBRID. Every other cell's value survived honest re-examination
against the four standards unchanged — re-labeled with real reasoning
rather than forced into a different number for its own sake (per
instruction: an honestly-labeled judgment call is a better outcome
than a citation that doesn't hold up).

**Monotonicity — verified unchanged, not re-derived from scratch**:
`tests/test_masking.py::test_floor_monotonic_in_sensitivity_class`/
`::test_floor_monotonic_in_threat_posture` (pre-existing, unmodified)
still pass against the new values — S3 always HYBRID at every posture
is trivially still monotonic in posture (constant), and monotonic in
class since S2's ceiling (PQC/PQC/HYBRID) never exceeds S3's floor at
any posture. `test_s3_never_floors_below_pqc` (pre-existing) still
passes (now vacuously — S3 is always HYBRID, strictly above PQC). New
regression test `test_s3_floors_at_hybrid_unconditionally_2026_09_14_recalibration`
locks in the actual new behavior (not just ">= PQC") so a future edit
can't silently regress S3 back to gradual escalation.

**Tests**: 3 new in `tests/test_masking.py` (the S3-unconditional
regression test above, the calibration-coverage test, the
citation-names-a-real-standard test). **Full suite: 737 passed, 1
xfailed** (734 prior + 3 new). `git diff --stat`: only `env/masking.py`
and `tests/test_masking.py` changed — confirmed zero changes to
`env/environment.py`, `env/contracts.py`, `agents/dqn.py`, any
checkpoint, or any `configs/scenarios/*.yaml` file. One incidental fix
needed to stay within scope: the first draft of the (S3, CALM)
citation text used the word "migration" (as in "quantum migration
deadline"), which tripped the pre-existing Hard Rule 3 test
(`tests/test_environment.py::test_hard_rule_3_no_s6_scenario_aware_branching_downstream_of_request_generation`,
which forbids "migration"/"s6" mentions in `env/masking.py` to keep S6
scenario-awareness out of it) — reworded to "transition deadline";
no scenario-awareness was ever actually added, this was purely a
word-choice collision.

**Downstream files that import `_PLACEHOLDER_FLOOR_TABLE` directly**
(`dashboard/explain.py`, `dashboard/render_migration_wave.py` +
its tests) were checked line-by-line before this change, not just
assumed safe: none of their exact-value assertions or "CALM baseline"
lookups reference the `(S3, CALM)` cell specifically, so the one value
change doesn't invalidate anything they assert — confirmed by the full
suite passing without touching any of those files. Committed
`dashboard/samples/*.html` files that happen to render an
`S3`+`CALM` combination are historical, frozen captures from real past
runs (same convention as every other committed sample) and are
deliberately left untouched, per this session's strict scope boundary
— not regenerated, not claimed to reflect the current live table.

**Next task:** Addition F (pcap/replay wired into `env/environment.py`'s
own live tick path) or Phase 4b (the full experimental campaign under
the real forecaster) remain the two open options per the standing
roadmap below — this session's recalibration doesn't block either, and
Phase 4b's future campaign will now reflect the corrected S3 floor.

---

**Cold-start buffer fix — the 2026-09-02 session's central finding (LSTM
window buffer holds 1 real row + 63 zero-padded rows at episode start,
spuriously spiking `threat_score` regardless of true class) is now
FIXED — DONE 2026-09-14.** Strictly scoped to `forecaster/rt_iot_replay.py`
(`RTIoT2022ReplayProvider`'s buffer construction) and its two direct
test files, per instruction — no LSTM retraining/architecture/weight
change, no `env/masking.py`/`env/environment.py`/reward/DQN-checkpoint
change, and the one-way floor ratchet itself (Hard Rule 2) is untouched
(the bug was the ratchet's cold-start INPUT, not the ratchet).

**Mechanism, confirmed before fixing:** the zero-pad lives in the
inherited `LSTMForecastProvider.get_threat_forecast()`
(`forecaster/model.py`, untouched) — `self._buffer` starts empty at
construction and is left-padded with zeros whenever it holds fewer
than `window=64` entries. `RTIoT2022ReplayProvider` never pre-filled
it, so the very first decision (which always follows exactly 1 real
`update()` call) saw 1 real row + 63 zeros. No other contributor was
found or needed to explain the artifact.

**Fix — `RTIoT2022ReplayProvider._warm_start_buffer()`, called from
`__init__` before any `update()`:** preferred path warm-starts the
buffer with real rows immediately preceding `start_offset` within the
SAME held-out class span (genuinely real, not-yet-consumed-by-this-
episode context) whenever `start_offset > 0`. Fallback, explicitly
documented as an approximation, not equivalent to real history: when
an episode starts at the true beginning of the held-out span
(`start_offset=0` — every default-constructed provider and every
campaign seed-0 episode), the earliest available real row is
REPLICATED to fill the buffer instead of zero-padding. No retraining;
`forecaster/model.py` is byte-unchanged.

**Validated, real before/after numbers (same 15-seed/`start_offset =
seed * max_steps` scheme as `replay_campaign.py`), full success, not
partial:**

| | benign step-0 mean | attack step-0 mean |
|---|---|---|
| Before | 0.9221 | 0.9723 |
| After | **0.0808** | **0.9750** |

Benign step-0 now sits right next to the true post-warmup benign mean
(0.0795, unchanged) — the false-positive spike is gone, not reduced.
Attack step-0 stayed correctly high — genuine early detection is
preserved.

**Confirmed at the full-campaign level too** (re-ran
`forecaster/replay_campaign.py`, unmodified, under the fix):

| | benign floor dist. | benign threat_score (min/mean/max) | attack floor dist. | attack threat_score (min/mean/max) |
|---|---|---|---|---|
| Before | `{HYBRID: 442 (49.1%), PQC: 446 (49.6%), CLASSICAL: 12 (1.3%)}` | 0.0554 / 0.1124 / 0.9255 | `{HYBRID: 458 (50.9%), PQC: 442 (49.1%)}` | 0.9676 / 0.9746 / 0.9758 |
| After | `{PQC: 458 (50.9%), CLASSICAL: 442 (49.1%)}` — **zero HYBRID** | 0.0554 / 0.0748 / **0.2011** | `{HYBRID: 458 (50.9%), PQC: 442 (49.1%)}` — unchanged | 0.9676 / 0.9746 / 0.9758 |

The benign campaign's floor distribution no longer shows any
`SERVE_HYBRID` (was 49.1%, entirely the cold-start-ratchet artifact);
max benign threat_score across all 900 decisions dropped from the
0.9255 cold-start outlier to a genuine post-warmup 0.2011. Attack
campaign numbers are unchanged to 4 decimal places — the fix touches
only benign-episode cold-start behavior.

**Tests:** 4 new unit tests in `tests/test_rt_iot_replay.py` (buffer
fully populated/non-zero immediately after construction; the preferred
real-preceding-rows path; the fallback replicate-first-row path; the
partial-context case). `tests/test_rt_iot_replay_live_episodes.py`'s
old bug-documenting test (which asserted the BUG — step-0 > 0.5 for
benign) was replaced with two fix-verifying regression tests (step-0
benign < 0.5 and attack > 0.5; step-0 no longer spikes relative to
post-warmup benign scores). **Full suite: 734 passed, 1 xfailed** (729
prior − 1 removed + 6 new). `git diff --stat` confirms zero changes
outside `forecaster/rt_iot_replay.py` and its two test files.

**Next task:** Addition F (pcap/replay wired into `env/environment.py`'s
own live tick path, per PLAN2.md §8B) or Phase 4b (the full
experimental campaign under the real forecaster) remain the two open
options per the standing roadmap below — both can now proceed without
this cold-start caveat distorting early-episode behavior.

---

**RT-IoT2022 replay wiring — real held-out replay provider built,
Step-4-verified to genuinely differentiate on real input, and run
through a real, meaningful multi-episode campaign — DONE 2026-09-02.**
Closes the gap the Addition E Phase 4a session's own docstring flagged
explicitly: that session's live sanity check fed the trained LSTM the
synthetic 2-dim `[qber, load]` placeholder (zero-padded to
`feature_dim=39`), proving the WIRING works, not that the model
consumes real, trained-on-shaped input. Scope held throughout: no true
live packet capture (Hard Rule 11, permanently out of scope — this is
replay of already-extracted RT-IoT2022 rows), no `env/masking.py`/
`env/environment.py`/reward/action-space/DQN-checkpoint changes, no
LSTM retraining/architecture change, no floor-table changes — verified
via `git diff --stat` showing zero changes to any of `env/`,
`agents/dqn.py`, `configs/scenarios/*.yaml`, `forecaster/model.py`,
`forecaster/train_threat_lstm.py`, or `forecaster/rt_iot_features.py`.

**Step 1 — held-out split reused (not re-derived), disjointness
re-verified directly:** `forecaster/rt_iot_replay.py::held_out_replay_rows()`
calls `train_threat_lstm.py`'s own `leakage_free_split()` +
`assert_no_row_overlap()` unchanged — the exact split that produced the
committed checkpoint's real val metrics (benign recall 1.000, attack
recall 0.994). Real counts, measured this session: **3,073 train
windows / 768 val windows** (unchanged from the training session).
Converted to raw-row spans for replay: the val windows collapse to
exactly **two contiguous real raw-row spans** (RT-IoT2022 has no
interleaving between labels) — **2,496 benign rows** (raw rows
9,984-12,480) and **22,144 attack rows** (raw rows 100,960-123,104),
**24,640 real held-out rows total available for live replay**, none
overlapping any train row (`assert_no_row_overlap` re-run and passing).

**Step 2 — `RTIoT2022ReplayProvider`, a drop-in `ForecastProvider`:**
`forecaster/rt_iot_replay.py::RTIoT2022ReplayProvider` subclasses
`LSTMForecastProvider` (not a redesign — reuses its model loading,
`_prepare_step_features` scaler application, and window buffer
verbatim) and overrides only `update()`: pool telemetry
(`qber`/`skr`/`pool_fill`, real from `pool_sim`) still flows through
the inherited EWMA pool head unchanged, but instead of consuming
`observation["threat_features"]` (still the synthetic placeholder —
`env/environment.py` untouched), it pulls the next real held-out raw
row (real chronological order preserved) from an internal replay
stream. Injected purely via the existing `forecast_provider_factory`
seam (design decision 17) — zero `env/environment.py` edits, same
pattern Phase 4a and `attack/attacking_provider.py` already use. A
strict shape guard (`ValueError` if `replay_features` width !=
`feature_dim`) refuses to silently truncate/zero-pad real data to fit
— the exact failure mode Phase 4a's placeholder-fed check could not
have caught.

**Step 3 — pool-head decision re-confirmed, not re-litigated:** pool
forecasting stays on the inherited `MovingAverageForecaster` EWMA
(reusing real pool telemetry the env already provides) — only the
threat-input side is replaced. No fabricated pool data.

**Step 4 — the real differentiation test, and it passes cleanly:**
Two fresh providers, each fed 64 real consecutive held-out rows (one
genuinely benign, one genuinely attack) via 64 `update()` calls:
**benign threat_score = 0.0602, attack threat_score = 0.9726** — a
huge, unmistakable real separation (re-confirmed at a second real
offset into each span, gap still > 0.3), consistent with the
checkpoint's own near-perfect val separation. 9 new tests in
`tests/test_rt_iot_replay.py`: real held-out counts/disjointness, the
shape-mismatch guard, an empty-input guard, a real-row-not-zero-padded
buffer-content check (compares buffered entries directly against
`_prepare_step_features` applied to the known real row — would fail on
silent truncation/padding), a wraparound-not-fabrication check, a
`start_offset` real-slice-selection check, and the two real
differentiation tests above.

**Step 5 — a real, meaningful campaign (not 2-3 episodes), run and
reported honestly:** `forecaster/replay_campaign.py` ran **15 episodes
per class x 60 steps = 900 decisions/class (1,800 total)**, S1 config,
`StaticThresholdPolicy`, each seed given a different `start_offset`
into the real held-out span (not the same 64 rows repeated 15 times).
Benign-only and attack-only streams run separately, not artificially
interleaved — RT-IoT2022 itself has no interleaved ground truth to
replay (each `Attack_type` is one contiguous raw-row block), so
reporting the two classes separately is the honest choice, not a
simplification.

Real results: **benign** — floor `{SERVE_HYBRID: 442 (49.1%),
SERVE_PQC: 446 (49.6%), SERVE_CLASSICAL: 12 (1.3%)}`, threat_score mean
0.1124 (min 0.0554, max 0.9255). **attack** — floor `{SERVE_HYBRID: 458
(50.9%), SERVE_PQC: 442 (49.1%), SERVE_CLASSICAL: 0}`, threat_score
mean 0.9746 (min 0.9676, max 0.9758).

**Central real finding, reported plainly, not tuned away (Hard Rule
7) — and it is NOT the 90/10-imbalance saturation the session brief
anticipated; it's a different, real, previously-invisible artifact:**
at the very first decision of every episode, the (pre-existing,
Phase-4a-built, untouched-this-session) buffer holds only **1** real
row — 63 zero-padded — because `LSTMForecastProvider` was only ever
trained on full 64-row real windows, a near-all-zero window reads as
spuriously elevated-threat **regardless of the true replayed class**:
measured step-0 mean threat_score is 0.9221 for *benign* episodes and
0.9723 for *attack* episodes — both high. Because Hard Rule 2's floor
ratchet only ever raises, this transient artifact locks an elevated
floor in for much of a genuinely benign episode's remainder, even
though the real post-warmup benign signal (decision ≥5 onward) is
correctly low: mean 0.0795 (max 0.3067) vs. attack's post-warmup mean
0.9746 (min 0.9676) — a clean, sharp real separation once the buffer
actually holds real data. This is WHY the benign and attack floor
distributions above look so similar (~49/51 HYBRID/PQC split for both)
despite the two classes' true post-warmup threat_score being almost
perfectly separated — the cold-start spike, not the underlying signal,
is doing a lot of the floor-setting work. **Not fixed here** —
deliberately out of this session's scope (a `forecaster/model.py`
behavior change, not a replay-wiring task) — locked in as a permanent
regression test
(`tests/test_rt_iot_replay_live_episodes.py::test_cold_start_zero_padding_causes_a_real_transient_spike_regardless_of_replayed_class`)
so a future session can see exactly what changes if it's addressed.

**Honest overall verdict on this session's original question ("is the
forecaster now genuinely network-data-driven end to end?"): partially,
not fully.** Step 4 proves the model itself responds correctly and
strongly to real, correctly-shaped RT-IoT2022 input — that gap is
closed. Step 5 proves a real replay-driven episode runs end-to-end
with sensible, class-conditioned floor/posture behavior (modulo the
cold-start caveat above). But the injection point is still a bespoke,
provider-level replay stream (`RTIoT2022ReplayProvider` pulling
directly from a held-out row array) constructed for this session, not
`env/environment.py`'s own live tick-by-tick feature-ingestion path —
`_threat_features_placeholder` is still what an ordinary (non-replay-
injected) episode sees. Genuine live/replayed-pcap ingestion through
the environment's own per-tick path is still Addition F's explicit
job, sequenced next per PLAN2.md §8B.

**Tests:** 9 new in `tests/test_rt_iot_replay.py`, 2 new in
`tests/test_rt_iot_replay_live_episodes.py` (the live class-conditioned
variance check + the cold-start regression test above). **Full suite:
729 passed, 1 xfailed** (718 prior + 11 new).

**Next task:** Addition F (pcap/replay wired into `env/environment.py`'s
own live tick path, per PLAN2.md §8B) is the natural next step and
would also settle whether to address the cold-start buffer-fill
artifact found this session before or alongside that work. Phase 4b
(the full Gate W3/S3/S5/dashboard/Table V campaign under the real
forecaster) remains a separate, not-yet-started option — either is
reasonable to pick up next; the cold-start finding above should be read
first either way, since it affects how any LSTM-driven campaign's early
episode steps should be interpreted.

---

**Addition E, Phases 2-4 — real LSTM threat forecaster: trained, validated
non-degenerate, and wired in for a live sanity check — DONE 2026-09-01.**
Builds on Phase 1 (below, unmodified this session — verified via
`git diff --stat` showing zero changes to `forecaster/rt_iot_features.py`
or `data/processed/rt_iot2022/`).

**Step 1 — imbalance + leakage, characterized before any training ran:**
Real windowed class balance, re-confirmed: **390 benign / 3,456 attack
windows (10.14%/89.86%)** out of 3,846 total. Two real leakage risks
found and handled, not assumed away:
1. **Row-overlap risk**: `rt_iot_features.py::build_windows` slides
   window=64/stride=32 with NO shuffling, so adjacent windows share 50%
   of their raw rows — a random window-level split would leak.
2. **Session/contiguity structure, verified directly**: RT-IoT2022's raw
   123,117-row file has exactly **11 label transitions total** — every
   one of its 12 `Attack_type` values occupies ONE uninterrupted block
   of raw rows (confirmed independently in window-index space too: 11
   contiguous runs across the 3,846 windows). All three benign labels
   sit first in the file, so the 390 benign and 3,456 attack windows are
   each themselves one contiguous block (indices 0-389 / 390-3845).

**Split design**: `forecaster/train_threat_lstm.py::leakage_free_split()`
splits each contiguous same-binary-label run temporally (earliest
windows -> train, latest -> val, never shuffled), dropping a gap of
`ceil(window/stride)=2` window-indices at each cut so no val window's
raw rows can overlap a train window's. **A real bug was caught and
fixed during development, not hypothetical**: the per-run gap alone
left the run *boundary* unprotected — the benign run's last (val) window
and the attack run's first (train) window are index-adjacent (distance
1), so their raw rows overlapped; `assert_no_row_overlap()` (a direct,
independent raw-row-space check, not a trust-the-logic assumption)
caught this immediately, fixed with a second, global cleanup pass that
drops any val window within the gap distance of ANY train window.
Result: 3,073 train / 768 val / 5 windows dropped as unsplittable gap.
**A third, milder leakage source, checked and reported rather than
ignored**: Phase 1's persisted z-score scaler was fit on the FULL
dataset (train+val combined), not train-only — real measured effect:
train-subset standardized features deviate from the "fit on train only"
ideal (mean 0, std 1) by a real but modest amount (mean abs deviation
0.074 avg / 0.189 max across the 27 continuous columns; std abs
deviation 0.116 avg / 1.0 max on one outlier column). Flagged honestly
rather than silently reworking Phase 1's already-tested pipeline (out of
this session's scope per instruction) — population-statistic leakage
only, no per-window label information crosses the split.

**Step 2/3 — training + the degenerate-collapse gate, passed on the
first attempt:** class-weighted `BCEWithLogitsLoss` (balanced inverse-
frequency weights computed from the TRAIN split only: benign weight
4.96, attack weight 0.56 — upweights the minority class, the opposite
direction from the textbook rare-positive framing since attack is the
majority here) + macro-F1-maximizing decision-threshold selection each
epoch (never raw accuracy — a constant "always attack" predictor scores
~90% accuracy on this split while learning nothing, which is exactly
why `detect_degenerate_collapse()` gates on benign recall and prediction
variance instead). **Real, held-out val result** (best checkpoint by
val macro-F1, threshold=0.20): benign precision 0.951 / recall **1.000**
/ F1 0.975 (support 77); attack precision 1.000 / recall 0.994 / F1
0.997 (support 691); macro-F1 0.986; **prediction std 0.296** (real
variance, not a constant output). **`degenerate_collapse: False`** —
confirmed via the actual held-out-diversity test
(`test_real_checkpoint_prediction_variance_across_diverse_held_out_windows`),
not just the training report's own self-check. No focal-loss fallback
was needed. Honest observation, not swept under the rug: val macro-F1
actually peaks at epoch 0-1 and *degrades* under continued training
(down to ~0.86 by epoch 59) — this dataset's benign/attack separation is
apparently close to linearly-obvious given the 39-feature representation
(consistent with published RT-IoT2022 benchmarks), so most of the 60
configured epochs mildly overfit; the checkpoint saved is the
best-val-epoch state (genuine early stopping), not the final epoch's
weights.

**Step 4 — scope decision, reported explicitly, not silently narrowed**:
PLAN.md's original Addition A design was a *dual*-head forecaster
(threat + pool). This session trains and returns **threat-head only** —
RT-IoT2022 carries no QKD pool telemetry (`qber`/`skr`/`pool_fill`) to
supervise a pool head from; that needs logged `SmartKeyNetEnv` rollouts,
a separate, not-yet-collected data source. `forecaster/dataset.py` and
`forecaster/train.py` (the future rollout-based dual-head trainer) are
left untouched, still real `NotImplementedError` stubs — not built
around with a fabricated pool head. `forecaster/model.py::SmartKeyForecaster`
is now real (LSTM encoder + single threat head); `LSTMForecastProvider`
implements the full `ForecastProvider` interface by delegating
`get_pool_forecast()` to a composed `MovingAverageForecaster` instance
verbatim (reuse, not duplication or fabrication) — verified identical
output via `test_pool_forecast_matches_moving_average_forecaster_reuse`.
`threat_score = sigmoid(logit)` maps to `posture_probs` via the exact
same RBF-softmax anchor scheme `MovingAverageForecaster` already uses,
so posture semantics stay consistent across both providers. New,
separate `forecaster/train_threat_lstm.py` (not folded into the
rollout-based `forecaster/train.py` stub — PLAN2.md Addition D's "one
implementation, two callers" precedent for a clean split).
`checkpoints/lstm_threat_forecaster.pt` + `.train_report.json` saved
locally (gitignored via the repo's existing `*.pt` rule, same as every
other checkpoint in this repo — none of the DQN checkpoints are
committed either, contrary to this file's own earlier phrasing).

**Step 5 — live sanity-check wire-in, done WITHOUT editing
`env/environment.py`** (explicit session boundary): `LSTMForecastProvider`
was injected purely through the pre-existing `forecast_provider_factory`
constructor parameter (design decision 17 — the same seam
`attack/attacking_provider.py::AttackingForecastProvider` already uses
for exactly this kind of external-provider injection), so
`_build_forecaster()`'s existing `"lstm"` stub branch remains unwired —
flagged as a small, explicitly out-of-scope follow-up for the Phase 4
full-campaign session, not done here. Ran 3 full episodes (S1 config,
seeds 0-2, `StaticThresholdPolicy`, 60 steps each) with zero crashes.
**Real result: genuine variance, not stuck** — `threat_score` ranged
0.730-0.946 (std 0.052) across the 180 real decisions; the resulting
`policy_floor` took both SERVE_PQC and SERVE_HYBRID values, never
frozen at one. **Load-bearing caveat, stated plainly (Hard Rule 7,
not smoothed over)**: `env/environment.py::_threat_features_placeholder`
still returns the 2-dimensional `[qber, load]` placeholder, not a real
39-dim RT-IoT2022-shaped feature vector — that requires Addition F's
pcap replay (sequenced after this addition, per PLAN2.md §8B). So this
result demonstrates the WIRING works end-to-end and reacts to its
input, not genuine live threat detection yet; `LSTMForecastProvider`'s
own class docstring states this caveat explicitly for future sessions.

**Tests**: 9 new in `tests/test_forecaster_model.py` (replacing the
prior import-smoke test now that the module is real — forward-pass
shape, `ForecastProvider` interface conformance, pool-head reuse,
buffer-window capping, and the real-checkpoint diverse-held-out-sample
degenerate-collapse check this session's instruction specifically
asked for); 14 new in `tests/test_train_threat_lstm.py` (the
leakage-free split's zero-overlap guarantee — including a regression
test for the exact run-boundary bug caught during development — class
weighting, evaluation metrics, and the degenerate-collapse detector on
both synthetic collapsed/non-collapsed cases and the real committed
checkpoint's report); 1 new in `tests/test_lstm_live_wire_in.py` (the
Step 5 live sanity check, marked skip if the checkpoint isn't present).
**Full suite: 718 passed, 1 xfailed** (695 prior + 23 new). Zero changes
to `env/masking.py`, `env/environment.py`, `env/contracts.py`,
`agents/dqn.py`, any `configs/scenarios/*.yaml` file, or any existing
DQN checkpoint — verified via `git diff --stat`.

**Next task — Addition E Phase 4's full campaign remains a separate,
deliberate future session (per instruction, not started here)**: wire
`use_foresight: lstm` into `env/environment.py::_build_forecaster()`
(the one small env.py edit intentionally deferred this session), then
re-run Gate W3 on S1+S3, the masked-vs-soft-reward S3 comparison, the
full 11-point dose-response sweep, all six dashboard panels, and the
paper's Table V against the real forecaster — every existing result in
this file and the paper was measured against the placeholder EWMA
forecaster and does not transfer automatically. Addition F (pcap
replay) stays sequenced after that, per PLAN2.md §8B.

---

**Full RT-IoT2022 feature-extraction pipeline — REAL, TESTED, run
against the FULL dataset (not a sample) — DONE 2026-08-31. Explicitly
NOT YET WIRED IN.** `forecaster/rt_iot_features.py` (new module,
standalone from `forecaster/dataset.py`'s separate rollout-log-based
windowing): real CSV loading + validation, degeneracy screening (real
result: NOT degenerate, dominant label `DOS_SYN_Hping` at 76.89% of
123,117 rows), a 39-feature extraction function grounded in real
network-IDS engineering practice (packet/byte rates, volume, timing,
header overhead, TCP window size, TCP flag counts, protocol/service
one-hot — reasoning documented in the module itself), a persisted
z-score scaler, and window=64/stride=32 majority-vote windowing. The
extraction function (`extract_flow_features`) is deterministic and
stateless, verified to produce identical output whether called on one
row alone or embedded in a batch — the "one implementation, two
callers" design PLAN2.md's Addition D calls for (offline training
caller now; a future single-window live-inference caller later). Ran
for real against the COMPLETE dataset: 123,117 raw rows -> **3,846
windows**, saved to `data/processed/rt_iot2022/rt_iot2022_windows.npz`
+ a JSON report, both committed. `data/get_data.py::_download_rt_iot2022()`
also implemented for real (was a stub) using the official `ucimlrepo`
client against UCI ML Repository dataset id 942 (CC BY 4.0) — verified
by actually running it: a real fetch returned exactly 123,117 rows,
independently confirming the vendored local CSV is the genuine,
complete dataset. **Central finding, reported plainly (Hard Rule 7):
the resulting label distribution is heavily imbalanced** — binary
benign/attack windows split 390/3,456 (10.1%/89.9%), and the 12-class
majority-vote label all but erases the thinnest raw classes
(`NMAP_FIN_SCAN`: 0/3,846 windows despite 28 real raw rows) — a real
constraint the next phase's LSTM training must design around
(class-weighted training, per-class F1, never accuracy). 25 new tests,
all real behavioral tests including a hand-calculated single-row
extraction check and a full end-to-end run against the real, complete
dataset. Full suite: **688 passed, 1 xfailed** (663 prior + 25 new).
**Scope boundary held, verified via `git diff --stat`**: zero changes
to `env/forecast_provider.py` or `env/environment.py` — the placeholder
`MovingAverageForecaster` remains the system's active forecaster, and
every existing experimental result (Gate W3, the S3 masked-vs-soft-
reward comparison, the S5 dose-response sweep, all six dashboard
panels, the paper's Table V) is unaffected. See SESSION_LOG.md's
newest entry and PLAN2.md's new "Addition E" section for the full
design writeup, real numbers, and four-phase plan.

**Side session, optional polish, does NOT change the roadmap below —
2026-08-31: Living System panel gained trained-DQN-driven variants of
its S1/S2 snapshots.** Investigated (read-only first, per instruction)
whether swapping the demo's driving policy from `StaticThresholdPolicy`
to a trained masked-DQN checkpoint would fix the standing 100%-HYBRID
saturation finding (see the entry above this one). **Real result:
genuine tier variety, confirmed across seeds 0-3** — S1
(`checkpoints/dqn_s1.pt`, trained on S1): `{HYBRID: 83, PQC: 167}`
(250 real decisions, seed 0); S2 (`checkpoints/dqn_s2.pt`, trained on
S2): `{CLASSICAL: 6, HYBRID: 163, PQC: 81}` (250 real decisions, seed
0) — versus `StaticThresholdPolicy`'s `{HYBRID: 250}` on both, unchanged.
`dashboard/render_living_system_demo.py::collect_real_episode` gained
an optional `policy` param (default `None` preserves the original
`StaticThresholdPolicy(0.5)` behavior byte-for-byte) + `_load_masked_
policy` (mirrors `render_migration_wave_demo.py`'s / `render_results_
demo.py`'s existing checkpoint-reload pattern exactly, eval only).
Additive only — the original `StaticThresholdPolicy`-driven S1/S2
samples are untouched (regenerated byte-identical) and remain
committed, honestly documenting their own real saturated finding.
`dashboard/samples/living_system_{s1,s2}_dqn_0{1,2,3}_*.html` (6 new
files) + two new `dashboard/app.py::PANELS` cards ("Living System — S1,
trained DQN" / "— S2, trained DQN"), registry now 9 panels/19
filenames (up from 7/13). 7 new tests (`tests/test_render_living_
system_demo.py`, mirroring the existing S1-addition pattern: tenant-
identity fidelity, exact tier-color correctness, no-fabrication, HTML
well-formedness, plus the central "genuine variety, not saturated"
check), `tests/test_dashboard_app.py`'s registry-shape counts updated.
Full suite: **695 passed, 1 xfailed** (688 prior + 7 new). Scope held:
zero changes to `forecaster/`, `env/forecast_provider.py`,
`env/masking.py`, `env/environment.py`, or any `configs/scenarios/`
file (verified via `git diff --stat`) — this session touched only
`dashboard/render_living_system_demo.py`, `dashboard/app.py`, and their
tests. See SESSION_LOG.md's newest entry for the full writeup.

**Next task — formalized post-review roadmap, TOP PRIORITY for the
next several sessions, strict dependency order E -> F -> G (PLAN2.md
§8A/§8B/§8C — formalized 2026-08-31):**

**Addition E — Real Threat Forecaster Build-Out (RT-IoT2022):**
1. **[DONE 2026-08-31]** the full RT-IoT2022 feature-extraction
   pipeline (above).
2. **[DONE 2026-09-01]** Built and trained the real (threat-head-only,
   scope decision explained above) LSTM
   (`forecaster/model.py::SmartKeyForecaster`,
   `forecaster/train_threat_lstm.py`) on Phase 1's windowed output —
   real, leakage-checked, class-weighted training. See this file's
   newest entry above.
3. **[DONE 2026-09-01]** Validated: real per-class precision/recall/F1
   on a leakage-free held-out split (benign recall 1.000, attack recall
   0.994, macro-F1 0.986), confirmed non-degenerate via a diverse-
   held-out-sample prediction-variance test. See above.
4. **[Phase 4a, live sanity-check wire-in, DONE 2026-09-01]**
   `LSTMForecastProvider` injected via the existing
   `forecast_provider_factory` seam and run for 3 real episodes with
   zero crashes and genuine threat_score/floor variance — see above.
   **[Phase 4b, the FULL experimental campaign, NOT started — still the
   next task]**: wire `use_foresight: lstm` into
   `env/environment.py::_build_forecaster()` (intentionally deferred
   this session, see above), then re-run Gate W3 on S1+S3, the
   masked-vs-soft-reward S3 comparison, the full 11-point dose-response
   sweep, all six dashboard panels, and the paper's Table V — since
   every existing result was measured against the placeholder
   forecaster and does not transfer automatically to a real one. This
   is a later, separate, deliberate session (per the project owner's
   own instruction), never bundled into the same session as steps 2-3.

**Addition F — Live/Replayed PCAP Input (PLAN2.md §8B) — depends on
Addition E step 4 being complete first; do not start meaningfully
before then.** Once the LSTM forecaster is real and wired in: a
`.pcap`/`.pcapng` file replayed at real-time pace through the SAME
shared `extract_flow_features()` Addition E built (offline-training
path and pcap-inference path are the two callers of one
implementation), driving a live episode whose panels update as
decisions happen. Hard Rule 11 (PLAN2.md §5.5) stays in force
unchanged: replay-of-a-captured-file only — TRUE live capture off a
real network interface remains explicitly out of scope (safer for a
demo, avoids a large separate systems-engineering subsystem, zero
incremental research value over replay).

**Addition G — API Facade, AWS-KMS-style (PLAN2.md §8C) — depends on
Addition F being complete first; do not start before then.** A FastAPI
REST facade (per PLAN.md's original framing) wrapping the now-real
system (LSTM + pcap chain) in clean endpoints (`GenerateDataKey`,
`Encrypt`, key-policy endpoints). Explicitly framed as demo/real-world
presentation value ("does this look like a deployable service"),
**explicitly LOW/ZERO research contribution value** (PLAN.md's own
"no research value... deployment story, not a research contribution"
framing) — sequenced last because it should wrap something real, not
the placeholder.

**Dashboard polish, live-episode streaming, and any other stretch work
are now explicitly LOWER priority than the E -> F -> G sequence
above** — pick up Addition E Phase 4b (the full experimental campaign)
next, not dashboard/API/pcap work, unless explicitly redirected.

---

**Living System panel now shows a real S1 (steady/calm) snapshot
alongside the existing S2 (elevated HNDL) one — DONE 2026-08-31.**
`dashboard/render_living_system_demo.py` runs the exact same real
machinery (`SmartKeyNetEnv`, `StaticThresholdPolicy`, the real
`build_tenant_graph`-backed tenant graph, `render_living_system.py`'s
unmodified renderer) a second time against `configs/default.yaml`
(`scenario: S1`, confirmed via `env/environment.py`'s own scenario
dispatch to be the real S1 config path -- no `env/`, `agents/`,
`experiments/`, or scenario-config changes needed). Three new, additive
`dashboard/samples/living_system_s1_*.html` snapshots; the existing S2
files are regenerated byte-identical (confirmed via `git status
--short`, not listed as modified). `dashboard/app.py`'s `PANELS`
registry now shows Living System as two clearly-labeled cards ("Living
System -- S1 (steady/calm)" / "Living System -- S2 (elevated HNDL)"),
7 panels / 13 files total. **Central finding, reported plainly, not
smoothed over (Hard Rule 7): S1's real episode ALSO saturates to 100%
HYBRID (250/250 decisions, reproducible across seeds 0-3)** -- the same
outcome as S2, for two distinct, directly-verified real causes, not one
guessed cause: (1) the standing placeholder-forecaster load-term
finding extends to S1 -- posture ratchets CALM->ELEVATED by the very
first real decision even with zero scripted threat schedule in play;
(2) a newly-identified second cause -- the demo's own baseline policy,
`agents/baselines.py::StaticThresholdPolicy`, serves HYBRID whenever
real `pool_fill > 0.5` and HYBRID is legal (which it essentially always
is, floors being lower bounds only), and real `pool_fill` never drops
to/below 0.5 in either scenario (`min=0.6974`). The real
`_PLACEHOLDER_FLOOR_TABLE` itself does NOT force HYBRID for most
sensitivity classes at ELEVATED (only S3 floors to HYBRID there) -- so
the policy's pool-fill threshold, not just floor-forcing, is the
dominant mechanism here. **This is a fourth independent data point
(after S2/S6's own prior sessions, now plus S1's threat-ratchet path
and this session's newly-found policy-threshold path) that the
placeholder forecaster (and, newly, this specific demo baseline policy)
is the real blocker to a visually-varied Living System demo -- not a
scenario-selection problem.** A genuinely varied view needs the real
LSTM forecaster and/or a more floor-sensitive demo policy, stated
explicitly rather than implied. 6 new tests
(`tests/test_render_living_system_demo.py`): real S1-episode tenant-
identity fidelity, exact tier-color-to-real-served-tier correctness
(independently rebuilt from the real decision list), no-fabrication,
HTML well-formedness, plus two fast pure-function `pick_snapshot_indices`
prefix tests. `tests/test_dashboard_app.py` updated for the new 7-panel/
13-file registry shape (no assertion weakened). Full suite: **663
passed, 1 xfailed** (657 prior + 6 new). Zero changes to any protected
file (`env/`, `agents/`, `experiments/`, `reward.*`, `configs/scenarios/
*.yaml`, `render_living_system.py`'s data logic) -- verified via `git
status --short`. Server re-verified live over real HTTP (index + one S1
sample + one S2 sample all real `200`s, `/nope` still `404`s). See
SESSION_LOG.md's newest entry for the full investigation and numbers.

**Next task:** the real LSTM forecaster (would address both the
posture-ratchet-saturation finding and, likely, unlock a genuinely
varied Living System demo), a more floor-sensitive demo policy for that
same panel, the API facade (`api/main.py`), live-episode streaming
(Option 3), and the tabbed single-page-app unification all remain open
-- pick up whichever is prioritized next.

**Presentation-only dashboard restyle to match the design mockup — DONE
2026-08-31.** The six real dashboard panels (`dashboard/render_*.py`)
and the server's index page (`dashboard/app.py::render_index_html`) now
consistently match `dashboard/mockups/smartkeynet_dashboard_mockup_v2.html`'s
dark visual language — real data and every honesty guard untouched, only
presentation changed. **Real finding before doing any work**: the six
panel renderers were already built directly against the mockup's design
tokens in prior sessions (same `--bg`/`--panel`/tier-color/`--quantum`
palette, same card/badge/callout/floor-cell patterns) — so the actual
restyle surface was two small cross-panel consistency fixes
(`.beat-eyebrow` color unified to `var(--quantum)` in
`render_explain.py`/`render_living_system.py`; the mockup's dot-grid
`body` background added additively to all six) plus the index page,
which was still a plain unstyled grid and is now a proper landing page:
a static brand header (no fabricated live-status pills), a one-line
thesis statement pulled from README's own real project framing
("Security is enforced as a hard, structural constraint — action
masking, never a term in the reward"), a tier-color legend, and
mockup-style card treatment. All six panels' sample HTML regenerated via
their existing, unmodified demo drivers; `git diff --stat` on every
regenerated sample shows only 7-9 added CSS lines, and the three
`*_data.json` provenance files came back byte-identical (real
measurements untouched/deterministic). Every honesty guard was
re-verified directly in the regenerated output, not assumed: the masked
dose-response curve's real alpha>=0.9 boundary, the `p99_latency`
discrete-cost-artifact caveat, the `regret==pool_exhaustion`-by-
construction note, and Migration Wave's three real attribution outcomes
(`no_before_observation`/`scripted`/`scripted`) all came through
unchanged. Zero changes to any protected path (`env/`, `agents/`,
`experiments/`, `reward.*`, `dashboard/explain.py`, `dashboard/mockups/`)
— verified via an empty `git diff --stat`. Full suite: **657 passed, 1
xfailed**, identical to the pre-session baseline (re-confirmed fresh
before and after, not copied) — no test removed, skipped, or weakened.
Server re-verified live over real HTTP after the restyle (index +
two spot-checked panels all real `200`s with well-formed HTML; unknown
path still `404`s). See SESSION_LOG.md's newest entry for the full
verification trail.

**Next task:** a possible future step is merging the six panels into one
unified tabbed single-page app (mirroring the mockup's tab-switcher UX,
"Option 3" territory) and/or live-episode streaming, both still
deliberately deferred and not started this session. The LSTM forecaster,
the pcap/live-capture chain, the API facade (`api/main.py`), and Fig.
5's TikZ diagram node-text fix (flagged 2026-08-27, still needs
sign-off) all remain the other open items — pick up whichever of those
next.

**Minimal local dashboard server (`dashboard/app.py`) — DONE 2026-08-31.**
`dashboard/app.py` was a `NotImplementedError` stub (Person D's original
Plotly-Dash design brief for a live, wired 4-beat demo shell); it is now
a real, minimal, standard-library-only (`http.server`, zero new
dependencies) local server that serves the six already-rendered static
panels in `dashboard/samples/` plus an index page linking all of them —
turning "double-click six separate HTML files" into "run one command,
open localhost." Run with `python -m dashboard.app`; it prints
`http://127.0.0.1:8000/` on startup — **verified by actually running it
this session**: index and two spot-checked panel routes (one Living
System file, one Explain Decision file) all returned real `200`s over a
real HTTP request, and an unknown path correctly `404`s. Route surface
is a small explicit whitelist (the `PANELS` registry in `dashboard/app.py`,
mirroring README's own six-panel table) rather than general directory
serving, so there's no path-traversal surface. Scope was deliberately
narrow per instruction: no pcap/live capture, no live `SmartKeyNetEnv`
streaming, no LSTM work, no regenerate-panel buttons — a presentation
wrapper over already-rendered real artifacts only; `env/`, `agents/`,
`experiments/`, `reward.*`, and the render_*.py panel modules themselves
were not touched.

**Stale-sample-file handling (path (b) chosen):** PROGRESS.md's prior
entry (below) flagged that `dashboard/samples/01_first_decision.html`,
`02_floor_driven_only_hybrid_clears.html`, and
`03_real_cost_tradeoff.html` were stale — missing the `data-cell`
attribute the current `dashboard/render_explain.py` now writes into
every floor-grid cell — and named the fix as "re-run
`python -m dashboard.render_explain_demo` and commit the refreshed
samples." That fix was low-risk (a prior session had already spot-
checked the driver runs cleanly) and trivial, so it was done this
session rather than left flagged again: ran
`python -m dashboard.render_explain_demo` unchanged (no renderer/driver
code touched), confirmed via `grep`/`git diff --stat` that exactly those
three files changed, by exactly one line each (the floor-grid markup
gaining `data-cell`), nothing else. All three now carry the attribute;
the served/committed panels are current, not stale.

**Tests:** 9 new tests in `tests/test_dashboard_app.py` — registry shape
(6 panels), index route content (200, references all 10 real underlying
filenames across the 6 panels), pure-function/route-body parity, all 10
real panel files individually resolve 200, an unregistered filename and
an unknown path both 404, a path-traversal attempt 404s, and one real
end-to-end test that binds an OS-assigned ephemeral port
(`build_server(port=0)`), makes a real `urllib` HTTP request, and tears
the server down within the test (no long-lived port in CI). Full suite:
**657 passed, 1 xfailed** (658 total collected). Re-verified the prior
baseline fresh rather than trusting the last-logged figure: running the
suite with the new test file excluded gives **648 passed, 1 xfailed**
(649 total) — one lower than the "649 passed" this file's prior entry
recorded, a small pre-existing discrepancy in that entry's own count
(not something this session's changes caused; flagging rather than
silently propagating). 648 prior + 9 new = 657, reconciling exactly.
Zero changes to any protected
file (`env/`, `agents/`, `experiments/`, `reward.*`, the six
`render_*.py` render modules, `dashboard/explain.py`) — verified via
`git status --short`.

**Docs:** README.md's "How to see the results" section gained an
"Option 1 — run the local server" (the real command + URL) ahead of the
existing "open the files directly" instructions (now "Option 2"); its
"Not started" list no longer calls `dashboard/app.py` a stub (the
still-not-started piece is specifically the live 4-beat streaming demo
shell); the real test count was refreshed to `657 passed, 1 xfailed`
(re-run fresh this session, not copied).

**Next task:** live-episode streaming (a wired 4-beat demo shell over a
running `SmartKeyNetEnv`, Option 3-style), the pcap/LSTM forecaster
chain, and the API facade (`api/main.py`) all remain deliberately out of
scope for this session and not started — pick up whichever of those, or
Fig. 5's TikZ diagram node-text fix (flagged 2026-08-27, still needs
sign-off), next.

**Root README.md authored (real run/onboard reference) — DONE
2026-08-31.** The repo's root `README.md` (previously a stale
week-1-kickoff scaffold referencing a `split.md` that no longer exists
— deleted the prior commit, `aefc7a1`) is now a genuinely useful,
verified onboarding doc: what the project is, real install steps, the
real test command + real pass count (**649 passed, 1 xfailed**, run
fresh this session, matching PROGRESS.md's own last-recorded figure
exactly), how to open the six real dashboard panels in a browser
(`dashboard/samples/*.html`, no server needed), the real commands to
regenerate each panel and to train/evaluate via
`experiments/train.py`/`experiments/harness.py`'s real function
signatures (verified by reading the actual code, not assumed — e.g.
`train()`'s CLI (`python -m experiments.train`) always runs S1 only;
scenario selection needs the Python API, `train(config,
scenario="S3")`), a concept-to-file repository map, the S1–S6 scenario
table, and an honest implemented-vs-stub-vs-not-started split plus the
known caveats (placeholder-forecaster posture saturation, the
`p99_latency` discrete-cost artifact, the `regret == pool_exhaustion`
identity, and the threshold-ties-on-displayed-columns-but-loses-3.8x-
on-total-reward nuance) pulled directly from PROGRESS.md/SESSION_LOG.md
rather than smoothed over. No code, config, or test file touched — see
SESSION_LOG.md's newest entry for the full verification trail
(including one real, minor finding along the way: `dashboard/samples/
01_first_decision.html`/`02_floor_driven_only_hybrid_clears.html`/
`03_real_cost_tradeoff.html` are stale relative to the current
`dashboard/render_explain.py` — regenerating them now adds a
`data-cell="{class}-{posture}"` attribute to every floor-grid cell that
the committed HTML doesn't have yet; spotted via a spot-check demo run,
reverted rather than fixed since this was a docs-only session — a
real, small, easy follow-up for whoever next touches that panel:
re-run `python -m dashboard.render_explain_demo` and commit the
refreshed samples).

**Table V is now fully real and current across all six policies —
DONE 2026-08-30 (recovery session, completing the previous entry's
flagged gap).** The three remaining baselines (static threshold,
always-PQC, random safe-set) are now measured on the CURRENT,
recalibrated S3 config (`configs/scenarios/s3_degradation.yaml`),
using the exact same methodology as the masked-DQN/soft-reward-DQN
rows already in the table (`experiments.harness.evaluate_multi_seed`,
the same 8 eval seeds 900–907). The static threshold was re-tuned
fresh on this config (grid search over the same 9 candidates,
`total_reward`-scored, not the non-discriminating `p99_latency`
objective a 2026-08-19 session already flagged and fixed) — tuned
value `t=0.9`, reproducing the 2026-08-24 Gate W3 S3 session's own
grid-search scores on this identical config to two decimal places
(independent confirmation this is a stable, deterministic
measurement). Real numbers saved to the new
`dashboard/samples/baselines_s3_data.json` (provenance file,
matching the existing `dashboard/samples/*.json` convention).
`docs/smartkeynet_ieee_paper_5.tex`'s Table V, abstract, §IV, §V-C,
§V-D, §V-F, and the conclusion were all updated to match — the
`\textdaggerdbl` "not yet re-measured" footnote is gone. **Central
finding, reported not smoothed over (Hard Rule 7)**: read off Table
V's own shown columns alone, the freshly tuned static threshold now
ties or beats the masked agent on every one of them (p99,
exhaustion, regret, below-floor all tie; forced-rekey ratio `8.0%`
vs. the masked agent's `15.6%`, in the threshold's favor) — exactly
the outcome the paper's own disqualification framing (Section II)
calls a failure of the project's premise. It is not the full
picture: total episode reward, the scalar actually being optimized,
separates the two by `~3.8x` (masked `-10,214.82` vs. threshold
`-38,566.87`) — the threshold's own decision rule never reuses a key
(pays a fresh-key rekey cost on almost every decision), which is
what its "better" forced-rekey ratio was actually hiding, not
efficiency. Table V's new footnote 3 and the §V-C/§V-D prose both
say this explicitly now, not just the table's tied columns. A second
finding, also reported plainly: always-PQC's and random's real
exhaustion counts under the current config (`0.00`/`0.375`) are far
below the old pre-recalibration figures (`531.0`/`510.5`) —
consistent with, not contradicting, the already-documented finding
that only `AlwaysHybridPolicy`'s maximal every-decision draw (the
always-hybrid row, `18` exhaustion events, still real and unchanged)
is concentrated enough to exhaust the recalibrated pool within one
250-step eval episode. **This closes the last flagged gap from the
entry below — the paper's Table V is now fully reconciled: all six
policies real, current-config, and internally consistent with the
prose.** The remaining open items are Fig. 5's TikZ diagram node-text
fix (flagged 2026-08-27, still needs sign-off) and the two
not-started larger pieces (real LSTM forecaster, API facade). See
SESSION_LOG.md's newest entry for the full numbers, before/after
table diff, and validation.

**Paper Table V placeholder numbers replaced with real measured values,
stale status text reconciled — DONE 2026-08-30.** `docs/smartkeynet_ieee_paper_5.tex`
now cites real numbers for the masked-DQN, soft-reward-DQN, and
always-hybrid rows of Table V (previously stale: `27,301`/`42`/`676`-style
figures from a pre-recalibration S3 config), reconciled a units mismatch
(the paper's below-floor column is a raw count per its own caption; the
real replacement is `floor_violations_total: 1012`, not the `0.1687` rate
— both are now shown, count in the table, rate in a footnote), and fixed
two demonstrably stale "not started"/"remains protocol not measurement"
claims about the trace generator and S5 dose-response sweep (both real
since 2026-08-25/26) in the Introduction and §IV-D. **Central finding,
reported not smoothed over (Hard Rule 7)**: under the real numbers, the
soft-reward baseline's rekey ratio (`70.3%`) is now the *worst* in the
whole grid, not the best — it no longer "beats the masked agent on a
performance metric" the way the paper's old §V-B claimed; the masked
agent's own rekey ratio (`15.6%`) is now *lower* than the reward-shaped
baseline's, not higher, inverting the abstract's original framing.
**Genuinely still open, flagged in the paper itself via a `\textdaggerdbl`
footnote, not fixed this session (out of scope — no eval reruns allowed)**:
Table V's static-threshold, always-PQC, and random rows have no real
measured backing under the current, recalibrated S3 config anywhere in
`dashboard/samples/*.json` or `SESSION_LOG.md` — they remain the old
pre-recalibration numbers, explicitly labeled as such, with §V-C/§V-D's
prose no longer asserting an apples-to-apples numeric comparison against
them. Re-running those three baselines under the current config (via
`experiments.harness.evaluate_multi_seed`, saved as a new
`dashboard/samples/*.json` provenance file per the 2026-08-29 sessions'
convention) is the concrete remaining step to make Table V fully real.
Fig. 5's TikZ diagram node-text fix (flagged 2026-08-27, needs sign-off)
also remains untouched — not this session's scope. See SESSION_LOG.md's
newest entry for the full before/after number reconciliation and
reasoning. **The demo dashboard is complete (6 of 7 panels real) and the
paper's headline masked-vs-soft-reward numbers are now real too — the two
items above (three baselines, Fig. 5 diagram) are what's left before the
paper can be called fully reconciled.**

**Migration Wave panel rendered from real held-out S6 episode — DONE
2026-08-30. The demo-able dashboard panel set is now COMPLETE (6 of 7
panels rendered from real data).** The sixth and final buildable panel:
S6's scripted, exogenous floor-ratchet schedule (`configs/scenarios/
s6_migration.yaml::migration_schedule` — three real events: step
60/tenant_0/S1→S3, step 130/tenant_3/S2→S3, step 190/tenant_4/S0→S2)
and the agent holding up under it. **Demo-asset location — pull these
up during the review, alongside the five above**:
`dashboard/samples/migration_wave.html` (the rendered panel: three
stacked phase cards, real before/after floor per event, honest
per-event attribution; a real pool_fill trajectory chart),
`dashboard/samples/migration_data.json` (the saved raw episode data
both were rendered from). `dashboard/render_migration_wave.py` (pure
renderer, includes `attribute_floor_change()` — the honesty gate) +
`dashboard/render_migration_wave_demo.py` (driver: real held-out S6
episode, seed=900, `checkpoints/dqn_s1.pt` — a masked DQN trained on S1
steady-state, NEVER on S6, Hard Rule 8 — reloaded and evaluated fresh,
no training performed). **Central Hard Rule 7 finding, investigated
before any renderer code was written, per the session's own explicit
framing**: two prior sessions found the placeholder threat-feature
formula's `load` term ratchets posture up almost immediately, which
could have pre-empted this panel's whole "migration raises the floor"
story. Checked directly on the real data, not assumed: on S6 (which
has no scripted threat schedule of its own — unlike S2), posture
reaches ELEVATED almost immediately across a 30-seed sweep but **never
reaches HIGH** — and at ELEVATED, all three real scripted (old_class →
new_class) pairs genuinely still raise the floor per the real
`_PLACEHOLDER_FLOOR_TABLE`, so this real episode's posture ceiling does
NOT pre-empt the schedule's effect. `attribute_floor_change()` still
checks this per-event on live observations rather than trusting that
finding globally: of the three real scripted events on the seed=900
held-out episode, 2 are cleanly attributable ("scripted" — posture
held constant across the real before/after observation bracket); the
third (tenant_0, step 60) has no real pre-event decision for that
tenant at all, honestly reported as `"no_before_observation"` rather
than forced into a comparison. A second real, honest finding: a
scripted event's effect on a tenant lags the schedule step by a real,
measured number of ticks (9 for tenant_4, 113 for tenant_3) — requests
already in flight when the mutation fires keep their old
`sensitivity_class`, only new arrivals pick up the change. 19 new
tests, including six direct `attribute_floor_change` unit tests
covering all five honesty outcomes (never claiming "scripted" when the
floor was already at the post-migration level, when posture itself
moved too, or when there's no data to compare). Full suite: **649
passed, 1 xfailed** (630 prior + 19 new). Zero changes to any protected
file. **Remaining panel, now just one**: Threat Input, blocked on the
not-yet-built real forecaster/feature-extraction pipeline — no other
panel work remains. See SESSION_LOG.md's newest entry for the full
design-decision writeup and investigation.

**Budgeting Brain panel rendered from real S3 pool-trajectory data —
DONE 2026-08-30.** The fifth of seven dashboard panels: a real,
same-seed (seed=900) S3 (QKD degradation) episode compared side by
side under the masked DQN (reloaded `checkpoints/s3_masked_seed1.pt`)
vs. `agents/baselines.py::AlwaysHybridPolicy` (that module's own
documented "drains the pool" baseline, no checkpoint needed). **Demo-
asset location — pull these up during the review, alongside the four
above**: `dashboard/samples/budgeting_brain.html` (the rendered
side-by-side panel), `dashboard/samples/budgeting_data.json` (the
saved raw per-step trajectory + event data both sides were rendered
from). `dashboard/render_budgeting_brain.py` (pure renderer: real
inline-SVG pool-trajectory area charts per policy, real exhaustion-
event markers at their real internal-tick positions, real stat boxes)
+ `dashboard/render_budgeting_brain_demo.py` (driver). **Real,
honestly-reported finding along the way (Hard Rule 7), not the design
assumed going in**: `below_floor_rate` measured `0.0000` for BOTH
policies on this real episode — unlike the masked-vs-soft-reward S3
comparison (where disabling masking made `below_floor_rate` the
headline discriminator), `AlwaysHybridPolicy` is still masked, so Hard
Rule 9's deferral-not-downgrade guarantee holds for it too; the real
axis these two policies diverge on here is `pool_exhaustion_events`/
`regret_events` (masked: 0; baseline: 18, real, over one 250-step
episode), which the renderer leads with instead, plus an explicit note
explaining why the metric that led the OTHER panel isn't the
discriminator on this pair. The real contrast is genuine and
substantial (masked agent's pool never drops below 75.68% of capacity
and never defers; the baseline drains to 0.16% by tick 131 and stays
there through 18 real deferrals until the degradation window ends at
tick 200 and refill resumes) — not watered down relative to the
mockup's qualitative story, though the real baseline recovers once the
window ends (the mockup's fabricated curve never does). 16 new tests,
including a direct same-real-data contrast check
(`test_conserving_agent_shows_zero_exhaustion_markers` /
`test_exhausting_baseline_shows_real_exhaustion_markers_and_banner`)
guarding against both a dramatized exhaustion that didn't happen and a
hidden one that did. Full suite: **630 passed, 1 xfailed** (614 prior
+ 16 new). Zero changes to any protected file. **Remaining mockup-only
panels, now two, not three**: Migration Wave, Threat Input (the latter
still blocked on the real forecaster) — the paper's Table V fix and
the real LSTM forecaster/API facade remain the other open items. See
SESSION_LOG.md's newest entry for the full design-decision writeup.

**Living System panel rendered from real tenant-graph episode data —
DONE 2026-08-29.** The fourth of seven dashboard panels, and the first
to render a real graph (not just a real trace/table/chart). **Demo-asset
location — pull these up during the review, alongside the other three
below**: `dashboard/samples/living_system_01_first_decision.html`,
`_02_graph_fully_populated.html`, `_03_final_decision.html` — three
real static snapshots (not animated/live) from one real S2 (HNDL
posture) episode under `StaticThresholdPolicy(0.5)`, run through the
real `build_tenant_graph`-backed tenant graph via `request_stream_
factory` injection. `dashboard/render_living_system.py` (pure renderer:
real tenant nodes/edges tier-colored by each tenant's real most-
recently-served key tier, resolved via `SmartKeyNetEnv.
_resulting_key_type` — the same ground-truth function the environment
itself uses, never re-derived) + `dashboard/render_living_system_demo.py`
(driver). **Real, honestly-investigated finding along the way (Hard
Rule 7), flagged here — not fixed, needs sign-off**: on this
graph-driven request stream, S2's scripted `threat_schedule.
elevate_at_step=50` turns out NOT to be what first raises the floor —
`env/forecast_provider.py`'s placeholder threat-feature formula mixes
an ordinary `load` (queue-backlog) term into what's meant to represent
threat, and `load` climbs fast enough under real per-tenant traffic
that the one-way ratchet fires within the first 1-2 real decisions of
every seed checked (0-3), 40-50 decisions before the scripted
elevation. This is a *load-sensitivity* finding, related to but
distinct from the standing posture-*ceiling* item below (both point at
the same placeholder formula needing a future calibration pass — flag
both together for whoever picks that up). This session's snapshot
selection was redesigned around real milestones (first decision /
graph-fully-populated / final decision) rather than assuming the
scripted schedule's timing, once this was found. A real bug was also
caught before shipping: `KeyType.CLASSICAL == 0` is a falsy `IntEnum`
value, so an early truthy check silently mislabeled CLASSICAL-served
tenants as "no traffic yet" — caught by a dedicated per-tier exact-color
test, fixed, re-verified (see SESSION_LOG.md's newest entry). 17 new
tests. Full suite: **614 passed, 1 xfailed** (597 prior + 17 new). Zero
changes to any protected file. **Remaining mockup-only panels, now
three, not six**: Budgeting Brain, Migration Wave, Threat Input (the
last still blocked on the real forecaster) — the paper's Table V fix
and the real LSTM forecaster/API facade remain the other open items.
See SESSION_LOG.md's newest entry for the full design-decision writeup
and investigation.

**Dose-response + S3 comparison demo visuals rendered from real data
— DONE 2026-08-29.** The project's two headline results are now real,
self-contained visual assets for the review/demo, not just numbers in
SESSION_LOG.md prose. **Demo-asset location — pull these up during the
review**: `dashboard/samples/dose_response_chart.html` (S5
steering-attack V(π) vs. alpha, both agents, real spread), `dashboard/
samples/s3_comparison_table.html` (masked-vs-soft-reward S3 metrics),
`dashboard/samples/results_data.json` (the saved raw numbers both were
rendered from — open any of the three directly in a browser/editor).
**Data provenance, explicit**: no saved raw-results file existed
anywhere for either the S5 sweep (2026-08-26) or the S3 comparison
(2026-08-25) — both lived only in SESSION_LOG.md prose. But the real
checkpoints both sessions trained were still on disk (gitignored, not
deleted): `checkpoints/dqn_s2.pt`/`soft_reward_baseline_s2.pt` (S5),
`checkpoints/s3_{masked,soft_reward}_seed{1,4,7}.pt` (S3). This
session reloaded them and **re-ran the real evals fresh** (`evaluate_
multi_seed`/`evaluate_attack_multi_seed`, eval-only, well under a
minute total) rather than transcribing — spot-checked reproducibility
first (a reloaded checkpoint's re-run matched SESSION_LOG.md's own
per-seed numbers exactly before any renderer was built), then the full
fresh run matched every prior figure to 4 decimal places. `dashboard/
render_dose_response.py` + `dashboard/render_comparison_table.py`
(pure renderers, same self-contained-static-HTML philosophy as
`dashboard/render_explain.py`, zero new dependencies — inline SVG for
the chart, no charting library) + `dashboard/render_results_demo.py`
(the real driver). **Hard Rule 7, honored explicitly**: the masked
agent's real curve shows its genuine alpha>=0.9 boundary (flat near-
zero through alpha<=0.8, rising to `0.3000` at alpha>=0.9) — never
flattened to a "clean" always-zero story; the comparison table's
`p99_latency` row (when shown) always carries its documented discrete-
cost-model-percentile-artifact caveat, and `regret_events` is always
labeled as identical to `pool_exhaustion_events` by construction. 19
new tests (`tests/test_render_dose_response.py`,
`tests/test_render_comparison_table.py`), incl. dedicated Hard Rule 7
guards (`test_masked_curve_is_not_flat_zero_everywhere`, `test_p99_
latency_always_carries_its_caveat_when_shown`). Full suite: **597
passed, 1 xfailed** (578 prior + 19 new). Zero changes to any protected
file. See SESSION_LOG.md's newest entry for the full provenance
writeup and reproducibility checks. **Cross-reference for the standing
Table V item below**: `dashboard/samples/results_data.json`/
`s3_comparison_table.html` are now the source of truth for folding
real numbers into the paper's Table V.

**Explain Decision panel rendered — DONE 2026-08-29.** The first
VISUAL/frontend piece of the project (every prior session was
backend/experiment work): `dashboard/explain.py`'s real, tested
Explain Decision backend (real since 2026-08-19) now has a real view
layer. `dashboard/render_explain.py::render_trace_html()` -- a pure,
zero-new-dependency function -- turns a real `DecisionTrace` into a
self-contained static HTML page (inline CSS, no server, no JS
framework, no build step), styled to match
`dashboard/mockups/smartkeynet_dashboard_mockup_v2.html`'s Explain
Decision tab visually, but driven entirely by real computed trace
values (Hard Rule 10) -- that mockup's own example numbers are never
read. `dashboard/render_explain_demo.py` is the real driver: runs a
genuine S2 episode through `SmartKeyNetEnv` under the real
`StaticThresholdPolicy` baseline (not hand-authored inputs),
collecting one real `DecisionTrace` per decision, and writes 3
genuinely different real samples to `dashboard/samples/*.html` (first
decision, a floor-driven decision, a real cost-tradeoff-not-taken
decision). `tests/test_render_explain.py` (10 new tests) verify Hard
Rule 10 by exact-match assertion (not just design intent): every
rendered per-action reason and the final sentence are checked
character-for-character against the real trace's own fields across 15
real decisions of a stepped episode, plus the floor grid's highlighted
cell is checked against all 12 real `(SensitivityClass, ThreatPosture)`
combinations. **Real, honestly-reported finding along the way**: this
session searched (100+ real seeds, all four real baseline policies,
both real elevated-threat scenario configs) for a genuine
"floor-driven, only-one-legal-action" episode decision (the case
`dashboard/explain.py`'s own `cost_note` field flags) and never found
one occurring naturally -- traced to a real structural cause
(`REKEY_NOW` has no illegality rule in `env/masking.py::compute_mask`
once any tier clears the floor, so it stays legal almost everywhere),
documented in `dashboard/render_explain_demo.py`'s own docstring
rather than worked around. Full suite: **578 passed, 1 xfailed** (568
prior + 10 new). Zero changes to any protected file (`env/contracts.py`,
`dashboard/explain.py`, `env/environment.py`, `env/masking.py`,
`agents/*`, `experiments/*`, `reward.*`, the mockup HTML) -- verified
via `git status --short`/`git diff --stat` showing only new files. See
SESSION_LOG.md's newest entry for the full design-decision writeup and
investigation. **This session deliberately did ONE dashboard panel
(the one whose backend already existed), not all seven** -- the
remaining six (Living System, Budgeting Brain, Steering Attack,
Migration Wave, Results, Threat Input) are still mockup-only, each
needing its own backend + render work; the real LSTM forecaster
(Addition A) and the API facade (`api/main.py`) remain the other two
large not-started items. This did not change any priority below --
resume from whichever of those the next session picks.

**Read-only repo state verification — DONE 2026-08-29.** Ground-truth
checkpoint: branch/sync confirmed clean after pushing one commit of
drift found on both `main`/`dev21` (approved fast-forward, not a
divergence — all four refs now identical, `227c7e8`); test suite
re-confirmed genuinely green (`568 passed, 1 xfailed`, matches prior
claim exactly); this file's per-file status table and milestone
checklist spot-checked against the filesystem — **zero corrections
needed, every claim checked was accurate**; both standing
instrumentation items (p99_latency saturation, exhaustion==regret
identity) reconfirmed as already resolved and documented directly in
`experiments/harness.py`'s own docstrings, not just in session-log
prose. See SESSION_LOG.md's newest entry for full detail. This did not
change any priority below — resume where the entry underneath left off.

**Paper integration of both addenda — DONE 2026-08-27** (real
`docs/smartkeynet_ieee_paper_5.tex` now exists in the repo; see
SESSION_LOG.md's newest entry, "paper integration: S5 limitations +
soft-reward Fig. 5 correction into the real .tex", for the full
comparison/insertion writeup). Both addenda below are now superseded as
"not yet inserted" — their real findings are in the paper's
`\subsection{Limitations}`. Two concrete items remain open from that
session, neither resolved:

- **Fig. 5's TikZ diagram itself still needs a matching visual fix, not
  yet applied — needs sign-off first.** Fig. 5 (`\label{fig:steering}`)
  is a TikZ diagram, not an image; its node text ("$\hat p_t$ enters the
  reward" -> "security term shrinks; latency outbids it" -> "weaker
  tier served") matches the addendum's own reconstructed paraphrase
  almost verbatim, so the reconstruction was accurate — but that also
  means the inaccurate claim lives in the diagram's own node text, not
  in the caption or body prose (which is generic and not inaccurate).
  The Limitations-section text fix is done; the diagram's `s1`/`s2` node
  text (in the `soft` block, around the `\node[softbox]` lines) still
  visually implies a continuous "posture enters the reward, security
  term shrinks" mechanism that the soft-reward addendum's own finding
  contradicts (only `REKEY_NOW`'s discrete tier-resolution reads the
  floor — no continuous term exists). Left unedited, per instruction,
  pending explicit sign-off on the specific node-text change.
- **Table V's numbers do not match the real measured S3-comparison
  figures — flagged, not touched.** Table V currently reads (Masked /
  Soft-reward): p99 lat `150.0`/`120.0`, Exhaustion `0.0`/`19.5`, Regret
  `0.0`/`19.5`, Rekey% `12.1`/`9.5`, Below-floor `0`/`27,301` — internally
  consistent with the abstract/conclusion's own `27,301`/`42` figures, so
  a real (not placeholder) six-policy result set, but from a different
  run/config than the 2026-08-25 masked-vs-soft-reward S3 comparison
  session's real numbers: p99 `1.5000`/`1.4064`, Exhaustion/Regret
  `0.00`/`3.54`, Rekey% `15.6`/`70.3`, Below-floor rate `0.0000`/`0.1687`.
  Scales don't reconcile cleanly (e.g. table's Exhaustion==Regret==`19.5`
  vs. measured `3.54`; below-floor as a raw count of `27,301` vs. a
  `0.1687` rate). **Not blocking, not urgent, but real**: deciding units
  and re-deriving the full six-policy Table V row set (likely needs
  fresh S3 runs for the four non-RL baselines too, which have never been
  measured against the recalibrated S3 config) is a separate, dedicated
  task. **2026-08-29: the masked-vs-soft-reward half of this re-
  derivation now has a real, re-confirmed source to pull from** —
  `dashboard/samples/results_data.json`/`s3_comparison_table.html`
  (freshly re-run, not re-transcribed; matches this same session's own
  numbers to 4 decimal places) — still leaves the four non-RL baselines
  unmeasured on the recalibrated S3 config, and the unit-reconciliation
  question above untouched.

`docs/report.md` re-confirmed 2026-08-27: still a bare section-outline
skeleton (owners + TODOs only, PLAN.md cross-references), no Fig. 5
prose, no Table V — out of scope for paper integration, the real paper
is the `.tex` file.

Original addenda pointers, kept for history (superseded by the
integration above):
- **`docs/steering_attack_limitations_addendum.md`** (added 2026-08-27, commit
  `81a1a17`, 81 lines) — the masked agent's S5 dose-response findings
  (the one-way-ratchet boundary at alpha>=0.9). **Its "supplementary
  sweep" claim (an honest pre-attack warm-up window as a mitigation) was
  found factually wrong against its own cited source during 2026-08-27's
  integration session — the real data show the opposite (see that
  session's SESSION_LOG.md entry) — the paper was written from the real
  numbers/conclusion, not this file's paragraph verbatim.** This file
  itself was left uncorrected, since editing addendum `.md` files was
  out of scope for that session.
- **`docs/soft_reward_curve_addendum.md`** (added 2026-08-27, commit
  following `9eec891`) — the soft-reward baseline's own curve-shape
  mechanism PLUS a proposed correction to the paper's Fig. 5
  reward-mechanism description. Checked against its cited source during
  integration and found accurate; inserted into the paper with only
  notational conversion. Its own "exact suggested edit" section (a
  reconstructed paraphrase) is superseded — the real Fig. 5 sentence
  was located during integration and matched the reconstruction closely.

**The soft-reward baseline's own S5 `V(pi)` curve shape — RESOLVED
2026-08-27 (see SESSION_LOG.md's newest entry for the full mechanism trace).**
Real, code-verified answers, no retraining needed (reused the still-on-disk
`checkpoints/soft_reward_baseline_s2.pt`):
- `agents/soft_reward_baseline.py::compute_soft_reward` does **not** read
  `threat_score`/`posture_probs` at all — `security_score(tier)` is a fixed
  3-value lookup keyed only on the delivered tier. The one real exception:
  `state["policy_floor"]` (a discrete, one-way-ratchet-processed value, the
  SAME mechanism the masked agent's own floor uses) feeds `REKEY_NOW`'s
  delivery resolution (`max(existing, floor)`) — a narrow, direct,
  non-Q-network-mediated posture channel, but only for that one action type.
- **Flag for whoever does final paper integration**: if the paper draft's
  Fig. 5 narrative describes a continuous `p̂_t` term entering this agent's
  reward with the security term shrinking under attack, **that does not
  match what was actually built** — the real design is a fixed, tier-only
  security term plus one narrow discrete side-channel, not a continuously
  threat-weighted term. Revise Fig. 5's description (or the surrounding
  prose) to match the mechanism above rather than the originally
  anticipated shape.
- The flat-then-step curve shape is the **same discrete-posture-bucket-
  crossing mechanism** already established for the masked agent's own
  alpha>=0.9 finding: a finer-grained re-run (alpha step 0.05 near the
  boundary) pinned the soft-reward agent's own crossing to `alpha 0.80 ->
  0.85` (posture-bucket-mismatch rate jumps `0.0224 -> 0.8112` in that exact
  interval, coincident with `V(pi)_true`'s own step `0.2896 -> 0.3256`).
- A direct, controlled same-state-except-posture comparison (100
  tier-establishing decisions at alpha=0.9, 16 posture-divergent) found the
  Q-network's chosen ACTION never changes (0/16) — the agent reliably picks
  `REKEY_NOW` regardless of perceived posture — but the TIER THAT `REKEY_NOW`
  DELIVERS changes in 6/16 of those cases, purely via the deterministic
  resolution formula reading the attacked `policy_floor`. This corrects the
  prior session's own softer "Q-network sees different inputs and decides
  differently" hypothesis with a more precise, evidence-backed mechanism.
- The already-substantial alpha=0 baseline (`0.2896`, no attack at all) is
  confirmed structural, not attack-related — same root cause as the
  already-documented unmanipulated S3 below-floor rate (`0.1687`): this
  agent's reward has no incentive to proactively rekey once holding a
  tier, so a floor that ratchets up mid-episode (S2's own scripted
  `elevate_at_step=50`) leaves an already-established lower-tier session
  key simply `REUSE`d indefinitely, with or without any attacker.

**The S5 steering-attack dose-response sweep is now real and complete
(2026-08-26) — PLAN.md §5's last scenario row, the paper's headline result.**
`attack/attacking_provider.py::AttackingForecastProvider` (live equation-7
wrapper) + `experiments/harness.py`'s dual-tracking measurement
(`run_scenario_under_attack`/`evaluate_attack_multi_seed`, real `V(pi)` per
paper eq. 4, measured against TRUE posture via a parallel shadow
`PolicyTable`, never the agent's own attacked floor) ran the real 11-alpha
sweep on S2 (masked DQN vs. soft-reward baseline, fresh 25,000-step
checkpoints for both, 5 eval seeds each). **The central, most important
finding: the masked agent's `V(pi)` is NOT flat zero everywhere, contradicting
this thread's own pre-registered prediction** — `0.0000` through alpha=0.4,
rising to `0.0016-0.0096` at alpha 0.5-0.8, then `0.3000` at alpha>=0.9.
Investigated immediately and rigorously, per Hard Rule 7, to a confident
mechanistic conclusion, **not left as an open bug**:
- **Ruled out a `compute_mask`/`PolicyTable` implementation bug directly**:
  the masked agent's delivered tier is below the floor it was actually
  *shown* exactly `0` times, at every one of 5 eval seeds, at every one of
  11 alphas — Hard Rule 2's literal guarantee (the mask enforces whatever
  floor it computes) holds perfectly throughout.
- **The real mechanism, traced per-decision**: `AttackingForecastProvider`
  shapes every window from tick 1 onward, including before S2's own
  `elevate_at_step=50`. At alpha=0.7 the real signal is only diluted, not
  erased — the live environment's own ratchet still detects it and locks
  HIGH within 2 decisions of the schedule firing. At alpha>=0.9 the real
  signal is suppressed so completely that the ratchet **never fires at
  all** for the rest of the episode — there is no already-detected floor
  for the one-way ratchet to defend, because its only channel of truth
  (the same forecaster the attack shapes) never detected anything.
- **This is a real, now-precisely-quantified boundary of the one-way
  ratchet's defense, not a masking defect**: it stops an adversary from
  talking an *already-ratcheted* floor back down; it cannot manufacture a
  detection the sensor was never allowed to make when the attack is strong
  and persistent enough to suppress the true signal for the entire episode.
  A supplementary, scratchpad-only delayed-onset sweep (attack applied only
  from `elevate_at_step` onward, leaving the — mostly irrelevant —
  pre-elevation portion untouched) reproduced nearly identical numbers,
  confirming the gap is about continuous suppression during/after the real
  elevation window, not an artifact of shaping the pre-elevation noise.
- **The soft-reward agent's own `V(pi)` also did not match the paper's
  clean "rises monotonically with alpha" prediction**: already substantially
  nonzero at alpha=0.0 (`0.2896` — this agent has no masking at all, so it
  violates real floors as a baseline property, independent of any attack),
  flat through alpha=0.8, then a step to `0.3256` at alpha>=0.9 — the same
  "needs enough dilution to cross a threshold" shape the masked agent's own
  curve has, just starting from a much higher floor, not the smooth curve
  originally anticipated.

**This finding belongs in the paper's existing Limitations section** — cite
this session's exact figures (masked `V(pi)`: `0.0000` at alpha<=0.4 up to
`0.3000` at alpha>=0.9; 0 attacked-floor violations throughout, confirming
it is a ratchet-scope boundary, not an implementation bug) rather than
presenting the split-screen result as an unconditional, always-zero
guarantee. **A possible mitigation direction is flagged here as future
work only, not built this session, for whoever next picks up
masking-adjacent work**: e.g. a conservative default/startup floor that
doesn't relax until the forecaster has observed some minimum variance in
its input (a suspiciously flat/constant sensor reading for an extended
window could itself be treated as a signal that the input channel may be
compromised, rather than trusted as genuine calm). See SESSION_LOG.md's
newest entry ("S5 steering-attack dose-response sweep: masked DQN vs
soft-reward baseline") for the full numbers, per-decision mechanism trace,
and Hard Rules 2/7 reasoning.

**Remaining lower-priority items, now that the headline result exists in
real form**: the standing `env/forecast_provider.py` posture-ceiling item
(below, unchanged), the real LSTM dual-head forecaster (Addition A), and
Dashboard v2 (Thread 2, below).

---

**Threads 1 and 3 are now both closed, and the soft-reward baseline agent is
now real too (2026-08-25).** Gate W3 is genuinely, fully met on both halves
(2026-08-24), **the full S1-S6 scenario grid is real and dispatched
(2026-08-24)**, and **`agents/soft_reward_baseline.py` — the Noetzold-style
soft-reward baseline agent (PLAN.md §5 S5's future steering-attack target) —
is built and tested (2026-08-25), no longer a stub.** See SESSION_LOG.md's
newest entry ("soft-reward baseline agent (Noetzold-style reproduction)") for
the full design/testing writeup. **This session's real milestone**: the
agent reuses `agents.dqn.DQNAgent` completely unmodified (no new agent
class — only its reward function and config differ); its "no action
masking" property required a signed-off, minimal, additive
`env/environment.py` config flag (`security_masking`, design decision 16)
after this session found the mask is enforced at the environment boundary
(`step()`'s `IllegalActionError`), not just in how an agent reads it — a
genuine surprise, flagged via `AskUserQuestion` and resolved with explicit
user sign-off before being built, per this repo's own "stop and flag before
touching protected files" convention. A real 25,000-step training campaign
confirmed the property this agent exists to demonstrate: `floor_violations >
0` at every one of 10 eval checkpoints (12-184 out of a 250-step episode) —
direct, measured evidence security is genuinely soft and gets traded away,
not inferred from the reward formula alone.

**Follow-up (1) is now DONE (2026-08-25, later session) — the real,
unmanipulated masked-DQN-vs-soft-reward-baseline comparison on S3 exists,**
using the same checkpoint-averaged (3 training seeds x 8 eval seeds)
methodology Gate W3 established. See SESSION_LOG.md's newest entry
("masked DQN vs soft-reward baseline on real S3") for full numbers. **The
real result does NOT match the paper draft's anticipated "soft-reward
wins efficiency, trades away security" tradeoff shape** — reported
honestly per Hard Rule 7, not reshaped to fit: the masked DQN comes out
ahead or comparable on **every** metric measured, not just below-floor
rate:
- **below-floor service rate** (the structural, expected divergence):
  masked `0.0000` vs. soft-reward `0.1687` (1012/6000 decisions across
  all seeds).
- **regret/pool-exhaustion events** (a second, previously-unemphasized
  failure mode): masked `0.00` vs. soft-reward `3.54` mean — the
  soft-reward agent's reward has no pool-scarcity/starvation term at all,
  so on S3's genuinely scarce pool it can fail on availability too, not
  just security.
- **forced_rekey_ratio**: masked `0.156` (mostly proactive, the "better"
  direction) vs. soft-reward `0.703` (mostly forced) — the opposite of
  "soft-reward is more efficient at rekey timing."
- **total_reward variance**: soft-reward's training-seed std (`12380.91`)
  is ~10x the masked agent's (`1303.25`) — the masked agent is both safer
  and far more consistent run-to-run.
- **p99_latency**: close (`1.5000` vs. `1.4064`), and the previously-
  flagged non-discriminating-metric oddity reappears (5/6 policy-seed
  cells tie at exactly `1.5000`), with one genuine exception this session
  (soft-reward seed 1: `1.2191`) — **root-caused 2026-08-25 (later,
  diagnostic session) — NOT A BUG, resolved.** Reloaded all six real S3
  checkpoints and tallied the exact per-decision `cost_action`
  distribution across all 48 eval episodes (3 training seeds x 8 eval
  seeds x 2 agents): `np.percentile`'s linear interpolation on a
  250-length array reads the 246th/247th smallest values (0-indexed),
  so `p99_latency` reports exactly `1.5000` (SERVE_HYBRID's real,
  uncapped per-decision cost — the max of a 4-value discrete set, not a
  ceiling) whenever >=4/250 decisions (>=1.6%) cost SERVE_HYBRID — true
  for essentially every real policy tested on S3, since its
  scarcity-driven floor makes that common. Verified exactly, not
  approximately: every episode over the threshold reported `1.5000`
  precisely; the one boundary episode (SERVE_HYBRID count == 3)
  reported `1.3530`, matching the interpolation formula to 4 decimals —
  this also explains the one-exception-cell mean (`1.2191`) down to the
  last decimal (mean of that checkpoint's 8 eval-seed values, 7 at
  `1.2000` + 1 at `1.3530`). No cap/ceiling/timeout exists anywhere in
  the latency-computation path (`env/environment.py::_apply_action`,
  `env/deferral_queue.py` both confirmed clean) — this is a mechanical
  percentile-over-discrete-data artifact, not a defect, so no code fix
  was made. `experiments/harness.py`'s `ScenarioResult.p99_latency`/
  `MultiSeedEvalResult.p99_latency_mean`/`_std` docstrings gained the
  full mechanism and now point to `total_reward`/`below_floor_rate` as
  the sharper metrics for S3 comparisons. See SESSION_LOG.md's newest
  entry ("root-cause p99_latency saturation on S3, diagnostic") for the
  full verification. **All five Table V metrics from the prior
  session's comparison stand as reported — none needed correction.**

**Follow-up (2)'s first half is now DONE (2026-08-25, later session) — the
attack generator (`attack/trace_generator.py`) is real and tested,
implementing the paper draft's equation 7 input-shaping attack
(`x̃t = (1-α)xt + α·g(xt)`).** See SESSION_LOG.md's newest entry
("adversarial trace generator (steering attack, eq. 7)") for the full
design/testing writeup. `g(xt)` returns an all-zero window (the real floor
of the current placeholder `[qber, load]` forecaster's benign region) —
deliberately targeted at the CURRENT placeholder forecaster, not the future
LSTM one; will need revisiting once that exists. Verified end-to-end
through the real `MovingAverageForecaster`, not assumed from the formula: a
genuinely severe `true_window=[5.0,5.0]` (mirroring how S2's own
`elevated_signal` mechanism already reaches HIGH, since ordinary in-domain
`[qber,load]` values are structurally capped at ELEVATED per the
2026-08-24 posture-ceiling finding below) steady-states to HIGH posture,
and shaping it at `alpha=1.0` steady-states to ELEVATED — a genuine
discrete floor crossing, not just a lower number. The masking-safety
property (this session's most important test) was verified directly with
real numbers: for `sensitivity_class=S2`, `compute_mask()` given only the
attacked, underestimated `SERVE_PQC` floor never legalizes `SERVE_CLASSICAL`
(below even the wrong floor), while `SERVE_PQC` itself becomes legal under
the attack where it wasn't under the true `SERVE_HYBRID` floor — the
attack's real, measured effect, with the safety guarantee intact — plus a
general, exhaustive 12-cell (4 sensitivity classes x 3 postures) sweep of
the same property. Needed **zero changes** to `env/forecast_provider.py`,
`env/masking.py`, or `env/environment.py` (Hard Rule 3), verified via
`git diff --stat`. `dashboard/explain.py` was read and confirmed to already
report a shaped/lowered posture consistently, with no changes needed.

**Superseded 2026-08-26 — the actual S5 dose-response sweep described below
has now been run for real** (on S2, not S3 — S2's own scripted
`elevate_at_step`/`elevated_signal` schedule genuinely reaches HIGH posture,
avoiding the standing posture-ceiling limitation that would have narrowed
S3's informative alpha range). See "Next task" above and SESSION_LOG.md's
newest entry for the full result — the paper prediction was **not**
confirmed as stated (masked agent's `V(pi)` is not flat zero everywhere),
investigated to a real, non-bug mechanistic conclusion, reported honestly
per Hard Rule 7. Original framing, kept for history: PLAN.md §5's last
remaining scenario row, PLAN2.md §7.5's Panel 5, the headline
steering-attack result: running both agents (masked DQN, soft-reward
baseline) against S3 (or S5's own config, TBD that session) across eleven
alpha values, reporting `V(π)` (below-floor service rate,
`experiments/harness.py::MultiSeedEvalResult.below_floor_rate_mean`,
already built) for both, with the explicit paper prediction (soft-reward
rises, masked stays flat at zero) confirmed or honestly reported otherwise
per Hard Rule 7. Both agents being compared are real, their unmanipulated
baseline behavior is measured on both S1 and S3, and the attack generator
that will drive the sweep is now real and proven correct in isolation —
every precondition this thread has named is now satisfied. Explicitly NOT
run yet — this session's own instruction was to build the generator only,
not the sweep.

PLAN.md §5's entire scenario table now has working, tested code AND a real
run behind every row, including S5 (2026-08-26 — see above).

**One thread remains open — Thread 2's Dashboard v2** (blocked on dataset
ingestion for its next concrete step, but has unblocked alternatives — see
Thread 2 below). **`experiments/train.py::train()`'s hardcoded `scenario:
"S1"` override — RESOLVED 2026-08-24, after being flagged three separate
times across three prior sessions (the multi-seed eval probe, the S3 Gate
W3 attempt's scratchpad workaround, and the S6 build session) without being
fixed.** `train()` now takes a real, explicit `scenario: str = "S1"`
parameter, following `experiments/harness.py`'s `run_scenario`/`run_grid`
convention exactly — this directly unblocked the soft-reward baseline
agent's own training entry point (`experiments/train.py::
train_soft_reward_baseline`, additive, 2026-08-25), which will itself need
to run on multiple scenarios for follow-up (1) above. **One standing,
unrelated open item remains flagged, not part of any thread**:
`env/forecast_provider.py`'s placeholder threat-feature formula's posture
ceiling (below).

**The `env/forecast_provider.py` open item, unchanged since 2026-08-24's
earlier S3 session, not part of any thread**:
`env/forecast_provider.py`'s placeholder threat-feature formula
(`_threat_features_placeholder` -> `MovingAverageForecaster`) has a
confirmed structural property — the discrete posture classification
(`argmax(posture_probs)`, the only thing `env/masking.py`'s floor table
reads) can **never** reach HIGH for any real `[qber, load]` input, however
severe, because both features are bounded in `[0,1]` and the resulting
sigmoid-squashed signal tops out at ~0.731, which the RBF-softmax always
resolves toward ELEVATED over HIGH. QBER itself is *not* negligible (it does
move `threat_score`/`posture_probs` measurably) — the ceiling is on the
*discrete* posture only. Read/investigate-only, per that session's
instruction (`env/forecast_provider.py` untouched) — flagged here as a real,
precisely understood, not-yet-fixed property, distinct from S2/S3/S4/S6's own
scenario work.

**Thread 1 — DQN training-instability → Gate W3 (2026-08-19: instability
substantially resolved, real gate genuinely attempted, S1 half MET, S3 half
blocked on an environment-design decision; 2026-08-24: S3 half recalibrated
and genuinely attempted for real — MET. Gate W3 is now fully closed.)**

**2026-08-24 update — S3 half resolved.** A dedicated session recalibrated
`configs/scenarios/s3_degradation.yaml` (S1/`configs/default.yaml` untouched,
byte-for-byte) after finding, precisely, why the 2026-08-19 blocker was
structural rather than a tuning problem: `env/pool_sim.py`'s existing
spike-degradation formula has a hard 50%-SKR-reduction ceiling regardless of
`spike_magnitude` (empirically re-verified: `0.6`/`0.9`/`0.99`/`5.0` all
produce byte-identical in-window SKR), and even a *hypothetical* zero-refill
window couldn't have stressed the prior 1,000,000-bit pool, since one
250-step episode's maximum possible draw volume (`250 * 256 = 64,000` bits)
is under 6.4% of that capacity regardless of refill. Fix: `pool_sim.py`
gained an optional `spike_skr_multiplier` field (additive, `None` preserves
prior behavior exactly), and S3's *own* scenario file (only — zero risk to
S1) dropped `pool.capacity_bits` to `20,000` (computed against the
64,000-bit max-draw-volume figure, not guessed) with
`qkd_degradation.spike_skr_multiplier: 0.0` (grounded in real QKD
security-proof literature: BB84-family protocols provably yield zero
extractable key above a QBER security threshold, ~11% canonically — not an
arbitrary worst case). Verified directly: under the real committed config
files, same seed, S1's pool never drops below ~70% full or produces a
regret event (any seed tested); S3's collapses to near-total exhaustion
(0.16% of its own capacity) with 21-33 real deferral events per 250-step
episode, every seed tested. Gate W3 S3 attempt (same checkpoint-averaged,
multi-training-seed methodology as S1): **DQN `total_reward` -10214.82
(training-seed std 1303.25) beats the tuned `StaticThresholdPolicy`'s
-38566.87 (eval-seed std 1636.12) by ~3.78x — MET**, gap ~21.7x/~17.3x
either side's own spread. Full numbers, the root-cause investigation, and
the posture-saturation investigation (see the new open item above) are in
SESSION_LOG.md's 2026-08-24 "S3 QKD-degradation recalibration, Gate W3 S3
attempt" entry.

A same-day
2026-08-19 session (see SESSION_LOG.md, "close REUSE/REKEY_NOW
floor-enforcement gap, Hard Rule 2") found and fixed a real bug directly
relevant to this thread: `env/masking.py`'s `compute_mask()` let REUSE stay
legal past a floor ratchet, and `env/environment.py`'s REKEY_NOW resolution
could silently refresh a stale below-floor tier — both on S1 too, since S1
already ratchets posture mid-episode under the default `ewma` foresight.
That session's own fork added a third option: re-run the 2026-08-17/18
multi-seed `forced_rekey_ratio` probe with the fix in place. **A later
same-day session did exactly that** (see SESSION_LOG.md, "post-fix re-run of
forced_rekey_ratio multi-seed probe") — identical methodology, same seeds
`1`/`4`/`7`, same load-spike S1 config, same 8-fixed-eval-seed/`eval_every=750`
design, only the underlying masking fix changed. **Result: swings shrank
substantially (mean swing 0.21-0.30 → 0.17-0.18, max swing ~0.89-0.91 →
~0.49-0.54, >0.5 frequency 19-27% → 0-3%) and the ceiling-fraction-by-thirds
drift — the clearest, most consistent pathology across every prior session in
this thread — is completely gone (0% at ceiling across all nine seed×third
cells, versus up to 62% pre-fix).** This is strong evidence the masking bug
was a real, likely-dominant contributor to the checkpoint-to-checkpoint
instability. **It is not a full resolution**: a genuine, smaller residual
remains (max swings still `~0.49`-`0.54`, e.g. seed 1's step
`71250->72000`, mean-of-8 `0.144->0.459`; seed 4 still has 3/99 swings
`>0.5`) — the dominant, most-damaging component (permanent worst-case
ceiling stickiness) is gone, not the phenomenon entirely. Full numbers and
the direct before/after tables are in SESSION_LOG.md's newest entry.

**Fork resolved, 2026-08-19 (same day, later session): moved to attempting
Gate W3 for real rather than the Q-value-margin inspection.** Reasoning: the
pathological failure mode (permanent worst-case ceiling stickiness) was
gone, the residual swinging that remained was ordinary, bounded noise rather
than the dominant effect any of options (a)/(b) were originally framed
against — and the actual project milestone (split.md's Gate W3: "DQN beats
the tuned threshold baseline on S1 and S3") had never been genuinely
attempted at all, for real reasons (S3 didn't exist as a scenario until two
sessions ago; the masking bug made any such comparison meaningless before
that). Attempting the real, defined milestone took priority over a further
diagnostic on an already-substantially-resolved instability. **Result (see
SESSION_LOG.md, "Gate W3 attempt: DQN vs tuned threshold on real S1 and
S3", full numbers there):**
- **S1: gate MET, decisively.** Checkpoint-averaged (3 training seeds x 8
  eval seeds), DQN `total_reward` -5119.71 (training-seed std 47.59) vs. a
  properly `total_reward`-tuned `StaticThresholdPolicy`'s -64616.41
  (eval-seed std 213.22) — DQN wins by ~12.6x, a gap roughly three orders of
  magnitude larger than either side's spread. `p99_latency` ties at exactly
  `1.5000` for both but is confirmed non-discriminating (constant across the
  *entire* threshold grid), not a genuine tie.
- **S3 (as of 2026-08-19): found, before running any training, that it's not
  meaningfully attemptable under current config** — under `configs/
  default.yaml`'s real pool settings, S3's degradation produces byte-identical
  floor/pool/posture trajectories to S1 (verified across seeds 0/1/4/7) —
  refill dwarfs the degraded draw by orders of magnitude, and posture is
  already saturated regardless of the QBER spike. Not run, per instruction
  not to retune S3's config to force a result. This is a real
  environment-design finding (Hard Rule 7: "investigate environment design
  first"), not a DQN or baseline failure. **Superseded 2026-08-24**: a
  dedicated recalibration session (see the "2026-08-24 update" note above and
  SESSION_LOG.md's newest entry) fixed this — S3 now genuinely diverges from
  S1, and the S3 half of Gate W3 was attempted for real and **MET** (~3.78x).
- **A related, second methodological finding**: `experiments/train.py`'s
  existing `evaluate_against_baseline` tunes `StaticThresholdPolicy` via
  single-seed `-p99_latency`, which this session found is constant across
  the *entire* threshold grid under real S1 — meaning `grid_search`'s tie-
  break silently defaults to the grid's first candidate, not a genuinely
  tuned choice. Not fixed this session (that function isn't what produced
  the Gate W3 numbers above — this session's own grid search used
  `total_reward` throughout) — flagged for whoever next touches
  `experiments/train.py`.
- **Real, reusable code added** (this was not a diagnostic-only session):
  `experiments/harness.py` gained `evaluate_multi_seed()` +
  `MultiSeedEvalResult` — runs any `Policy` across multiple eval seeds and
  reports mean/std for the metrics PLAN.md's closing table cares about,
  never a bare point estimate. 6 real tests in `tests/test_harness.py`. See
  that file's per-file row below.

`test_dqn_agent_loss_trends_down_training_against_real_env_s1`'s
`xfail(strict=True)` marker remains untouched — this session's result
doesn't resolve it either; that's still a separate, standing item, not
addressed by a successful Gate W3 S1 result. Do not touch `agents/dqn.py`
for anything short of a deliberate, sign-off'd decision; `experiments/
train.py`/`experiments/harness.py` may gain further real, tested additions
(as this session did) but their existing DQN training/eval logic itself
(not the new `evaluate_multi_seed` addition) should still be changed
deliberately, not incidentally.

**Thread 2 — Dashboard v2 (started 2026-08-19).** `dashboard/explain.py`
(the Explain Decision panel's backend, PLAN2.md §7.3) is now
implemented+tested — see its `dashboard/` per-file row below.
**2026-08-29: the Explain Decision panel is now also rendered** —
`dashboard/render_explain.py` + `dashboard/render_explain_demo.py`,
see this file's "Next task" and `dashboard/`'s per-file rows. The
concrete next Dashboard v2 step is the Threat Input panel (PLAN2.md
§7.1), but that's blocked on Person A's feature-extraction work (the
shared RT-IoT2022/pcap feature-extraction function PLAN2.md §6 and
Hard Rule 11 require) — dataset ingestion hasn't started (see `data/`
below). Until that exists, do not stub a placeholder extraction path for
Threat Input (Hard Rule 11: one shared extraction path, no parallel
pipeline). Other dashboard-adjacent, unblocked options: Panel 2 (Living
System) could start against `dashboard/explain.py`'s real trace output
plus `StateDict`/event-log fields, no new dependency; or start on
`agents/soft_reward_baseline.py` (needed before the steering attack,
PLAN2.md §7.5/§9 S5, can be built).

**Thread 3 — Real scenario dispatch. CLOSED 2026-08-24 — the full S1-S6
grid is now real and dispatched.** S1/S2/S3/S4/S6 scenario dispatch is
real (see `env/environment.py`'s per-file row). S4 (DDoS/noisy-neighbor)
needed a notion of which tenant a request belongs to that persists and
can be targeted — 2026-08-23's session built exactly that
(`build_tenant_graph`/`RequestGenerator`), and **2026-08-24's session
wired real S4 dispatch on top of it**: `config["ddos"]`
(required only under `scenario: S4`) picks a designated low-sensitivity
tenant and floods it for the whole episode via `RequestGenerator`'s new
`flood_override` mechanism — an *additive* second, independent Poisson
arrival stream for that one tenant, not a `traffic_rate` multiply (this
codebase's actual sampling model — a single shared Poisson total split by
weighted multinomial across tenants — would otherwise mechanically dilute
every other tenant's share too; verified, and the additive mechanism
instead leaves every other tenant's own arrival stream byte-for-byte
unaffected). See `env/request_generator.py`'s and `env/environment.py`'s
per-file rows and SESSION_LOG.md's 2026-08-24 S4 entry for the full
empirical findings and mechanism reasoning.

**2026-08-24, same day, later session — S6 (migration wave) wired, closing
this thread.** `config["migration_graph_seed"]`/`config["migration_schedule"]`
(required only under `scenario: S6`) build a tenant graph once, same
pattern as S4, and apply a small, scripted list of `{step, tenant_index,
new_sensitivity_class}` ratchet events via `RequestGenerator`'s new
`set_tenant_sensitivity_class` method — entirely upstream, in request
generation (Hard Rule 3, verified at the code level: masking.py/
contracts.py/the reward calculation have zero mentions of "migration"/
"s6"). Needed **zero `env/masking.py` changes**: both `compute_mask`/
`PolicyTable.floor` and `_prepare_decision` already read a request's
`sensitivity_class` fresh every decision, with no caching anywhere that
assumes it's fixed for an episode (confirmed by reading before designing,
not assumed) — so mutating one tenant's node attribute is sufficient on
its own for the floor to respond correctly. **Hard Rule 8 (train/eval
split) is now a real, code-level guard, not just documentation**:
`experiments/train.py::train()` checks `config["train_eligible"]`
(default `True`, set `False` only in the new `configs/scenarios/
s6_migration.yaml`) before any training work and raises `ValueError` if
selected — keyed on the flag itself, not a hardcoded `scenario == "S6"`
string, so it survived the 2026-08-24 (later session) fix to `train()`'s
then-separately-flagged hardcoded-`"S1"` debt exactly as designed (see
"Next task" above and `experiments/train.py`'s per-file row — that fix
finally exercised this guard through a real, scenario-selectable call).
See `env/request_generator.py`'s and
`env/environment.py`'s per-file rows, `experiments/train.py`'s per-file
row, and SESSION_LOG.md's newest entry ("S6 (migration wave) scenario
dispatch, Hard Rule 8 guard") for the full design reasoning and test
results (14 new tests: 5 in `test_request_generator.py`, 6 in
`test_environment.py`, 3 in `test_train.py`).

---

**2026-08-18's diagnostic recap (Thread 1, unchanged from before the pause):**

**A second 2026-08-18 diagnostic tested both candidate explanations for the
checkpoint-to-checkpoint swings and disfavored both — the mechanism is
still unknown, and even 8-eval-seed averaging doesn't tame it.** See item 6
below for the full result; the standing recommendation to report
`forced_rekey_ratio` "as a distribution across eval seeds" (item 5) is now
known to be insufficient on its own — the distribution's *mean* swings
checkpoint-to-checkpoint almost as hard as a single draw did.

**The 2026-08-18 dense re-probe overturned the 2026-08-17 "mid-training
regression" framing itself.** There is no localized regression event —
`forced_rekey_ratio` oscillates continuously (swings of 0.5+ between
adjacent 1,000-step snapshots, about 1 in 3 of the time) across the
*entire* 1,000-75,000 step range, for every seed tested, including the
one previously read as "flat, never found a good policy." The earlier
3-point sample (25k/50k/75k) was undersampling this noise, not
observing a real found-then-lost event. See SESSION_LOG.md 2026-08-18
for the full data. Five sessions on this thread now (three same-day
2026-08-10, one 2026-08-17, one 2026-08-18); recap in order:

1. First load-spike session: `forced_rekey_ratio` dropped from flat
   S1's `1.000` to `0.256`/`0.872` across two training seeds —
   confirmed direction, spread unquantified.
2. 10-seed follow-up sized the spread (`0.190`-`1.000`, mean `0.735`,
   stdev `0.275`, leaning bimodal) and found why it was worth
   distrusting: `agents/dqn.py` never seeded its own weight
   init/exploration/replay sampling at all — `training.seed` only ever
   reached the environment. Flagged, not fixed.
3. **This session fixed that**: `DQNAgent.__init__` gained a `seed`
   parameter (seeds `random`+`torch` before `QNetwork` construction),
   `experiments/train.py` now threads `training.seed` to both the env
   and the agent, and a new test (`tests/test_dqn.py`) proves same-seed
   agents produce identical action sequences and different seeds
   diverge. Re-ran the identical 10-seed load-spike sweep with the fix
   in place:

   | | sorted `forced_rekey_ratio` | mean | stdev |
   |---|---|---|---|
   | Before fix (unseeded DQN) | `0.190, 0.417, 0.418, 0.703, 0.872, 0.895, 0.914, 0.971, 0.971, 1.000` | 0.735 | 0.275 |
   | **After fix (seeded DQN)** | `0.102, 0.148, 0.463, 0.553, 0.730, 1.000, 1.000, 1.000, 1.000, 1.000` | 0.700 | 0.345 |

   **The fix made the spread wider, not tighter** — exactly half the
   seeds (5/10) now land at the *exact* never-proactive ceiling
   `1.000` (vs. 1/10 before), while the other half ranges `0.102`-
   `0.730`, including the best result seen across either sweep
   (`0.102` — 90% proactive). With RNG-conflation ruled out as the
   explanation (both sweeps used identical config; only the seeding
   fix changed), this looks like a genuine, structural
   learn/don't-learn split: roughly half of random weight-init +
   exploration trajectories discover any proactive rekeying at all
   within a 25,000-step budget, and half settle into "wait until
   forced" and stay there. `reward.w_fr`/`reward.c_rekey_base` remain
   uninvolved in any of this — no `reward.*` change made or requested
   across any of the three sessions.

4. **2026-08-17 budget probe** ran a strict, pre-committed decision
   rule to test the training-budget hypothesis before building
   anything: 3 of the 5
   ceiling-stuck seeds (`1`, `4`, `7`) trained to `50,000` and `75,000`
   steps (up from `25,000`), load-spike enabled, same config. Verdict:
   **DID NOT BUDGE** — no seed cleared `forced_rekey_ratio <= 0.5` at
   75k, and none held a monotonically non-increasing trajectory. This
   isn't just "training was too short": seeds `1` and `4` reached
   genuinely good intermediate values at 50k (`0.1020` and `0.6585`,
   `seed=1`'s matching the best result seen in any sweep) and then
   **regressed back to the exact `1.000` ceiling by 75k**. Seed `7`
   never moved off the ceiling at any budget tested. A real, substantial
   improvement appearing and then being lost mid-training is a
   different, more concerning phenomenon than either of the two
   hypotheses framed on 2026-08-10 (marginal budget vs. static bimodal
   landscape) — it looks more like training instability / policy
   forgetting within a single continuous run than a simple
   convergence-speed question. See SESSION_LOG.md 2026-08-17 for all
   six data points.

5. **2026-08-18 dense diagnostic** re-ran seeds `1`, `4`, `7` to
   75,000 steps with `eval_every=1000` (~75 snapshots instead of 3) to
   localize the 2026-08-17 regression's onset. Found none — max
   single-window swings of `0.90`-`0.92` occur throughout the entire
   run for every seed, not just around the buffer-capacity crossing at
   step `50,000`. **Buffer-capacity hypothesis: ruled out** — loss is
   unremarkable around step 50,000 (matches its neighbors, no spike),
   and swing amplitude doesn't shrink or shift near that step. There
   *is* a real, separate, noisy long-run trend — the fraction of
   snapshots sitting exactly at the `1.000` ceiling roughly doubles
   each third of a run (seed 1: `8% -> 48% -> 64%`) — but it's a
   smooth drift across the whole run, not a step-change tied to
   `12,500` (epsilon-decay-complete) or `50,000` (buffer-capacity)
   specifically, and the swings persist at full amplitude even in the
   run's final third.

6. **2026-08-18's second same-day diagnostic** directly tested both
   candidate explanations item 5 left open, using a measurement-only
   reimplementation of `train()`'s loop (`DQNAgent`/`SmartKeyNetEnv`/
   `GreedyDQNPolicy`/`run_scenario` imported and called exactly as
   `train()` does, not a new training path) run for seeds `1`, `4`, `7`
   to `75,000` steps, `eval_every=750` (deliberately not a multiple of
   `target_update_every=1000`), 8 fixed eval seeds (`900`-`907`) per
   snapshot instead of one. **Hypothesis (a) — single-episode eval
   noise — RULED OUT**: the spread across a checkpoint's own 8 eval
   draws is small (mean `0.05`-`0.07`), but the *mean* of those 8 draws
   still swings 4-6x larger (mean `0.21`-`0.30`, still `>0.5` on
   19-27% of adjacent 750-step snapshots) from one checkpoint to the
   next — averaging away eval-episode randomness does not tame the
   swing. **Hypothesis (b) — eval-cadence/target-sync aliasing —
   evidence against, not a clean rule-out** (between-session
   comparison, not a controlled ablation): the non-aligned
   `eval_every=750` cadence swings with the same character and
   magnitude as 2026-08-18's earlier aligned `eval_every=1000` data
   (max swing `~0.89`-`0.91` both), if anything slightly *less*
   frequently — the opposite of what aliasing inflation would predict.
   The 2026-08-10 ceiling-fraction long-run drift (item 5) was
   confirmed to survive 8-seed averaging too. **Net effect: both
   leading explanations for the swings are now disfavored, and the
   standing "report as a distribution across eval seeds"
   recommendation from item 5 is revised** — it doesn't fix the
   problem, since the distribution's own mean is nearly as unstable
   checkpoint-to-checkpoint as a single draw. See SESSION_LOG.md's
   second 2026-08-18 entry for the full comparison tables.

**Concretely next (pick one — a real decision, not more solo running):**
(a) a direct Q-value-margin inspection at a known swing (e.g. seed 1's
step `71250->72000` transition, `forced_rekey_ratio` mean dropping from
`~0.90` to `0.149`) — check whether the greedy action at the eval
episode's early, trajectory-determining decision points sits at a
near-tie in Q-value between two actions, such that one `learn()` call's
weight update is enough to flip the argmax and cascade into a very
different downstream key-age trajectory for the rest of a short
250-step episode (this would explain instability surviving 8-seed
averaging without the underlying Q-values needing to move much) — not
run yet, no repo code exists for this, would need new (flagged-first)
instrumentation in a copy of the eval loop, not `agents/dqn.py` itself;
or (b) treat "the greedy policy genuinely oscillates checkpoint-to-
checkpoint, mechanism unknown" as a standing, now twice-confirmed
property of this training setup and move on to real S4 regardless,
using **checkpoint-averaged** (e.g. mean of several `eval_every`
windows near the end of training), not single-checkpoint, comparisons
for any future DQN-vs-baseline number. Real S4 still needs the tenant
graph (`build_tenant_graph`/`RequestGenerator`, still
`NotImplementedError`) for genuine per-tenant flooding regardless of
this thread — the load-spike diagnostic remains a cruder, tenant-blind
stand-in, not real S4 itself. Do not mistake `configs/default.yaml`'s
`load_spike:` block or `random_request_generator`'s `load_spike` kwarg
for finished S4 work — both are documented as a diagnostic stub
throughout.

---

## Milestone checklist

Pulled from PLAN.md §10 (kickoff order) and §7 / split.md §2 (weekly gates).

- [x] Repo scaffolded (folder skeleton, `env/contracts.py` frozen, CI + PR template)
- [x] Gate W1 — `pytest` green on stubs; contracts frozen and committed
- [x] Spine wired end-to-end into `environment.py` — pool sim + deferral queue
      + masking + forecaster (EWMA) + request stream (random) + full reward
      formula, `reset()`/`step()` running a complete S1 episode with regret
      logging
- [x] Gate W2 — env `step()` runs end-to-end with a random valid agent across
      a full S1 episode; regret events logged
- [x] Real NetworkX tenant graph + graph-driven `RequestGenerator`
      (`build_tenant_graph`, `RequestGenerator.reset/step`) — PLAN.md §10 step 4
- [x] Masked DQN agent (`agents/dqn.py`)
- [x] Four tuned baselines — always-PQC, always-hybrid, static-threshold
      (grid-searched), random (`agents/baselines.py`) + comparison harness
      (`experiments/harness.py`) — Hard Rule 7
- [x] 🚩 Gate W3 (make-or-break) — DQN beats the tuned threshold baseline on S1 and S3.
      **2026-08-19: genuinely attempted for the first time. S1: MET, decisively (DQN
      `total_reward` -5119.71 vs. tuned threshold's -64616.41, ~12.6x, checkpoint-averaged
      across 3 training seeds x 8 eval seeds — gap dwarfs both sides' spread by ~3 orders of
      magnitude). S3: not meaningfully attemptable under current `configs/default.yaml` pool
      settings at the time — S3's degradation produced byte-identical floor/pool/posture
      trajectories to S1 (verified across 4 seeds).** **2026-08-24: S3 recalibrated
      (`configs/scenarios/s3_degradation.yaml` only — `configs/default.yaml`/S1 untouched,
      byte-for-byte) after a root-cause investigation found the blocker was structural (a
      hard 50%-SKR-reduction ceiling in the existing spike formula, and a pool-capacity-vs.
      max-episode-draw-volume scale mismatch), not a tuning problem. S3 now genuinely diverges
      from S1 (near-total pool exhaustion + real deferral events under a fixed-seed stress
      test, vs. S1's untouched, never-stressed pool). **S3 gate attempt: MET** — same
      checkpoint-averaged/multi-training-seed methodology as S1, DQN `total_reward` -10214.82
      (training-seed std 1303.25) vs. tuned threshold's -38566.87 (eval-seed std 1636.12),
      ~3.78x, gap ~21.7x/~17.3x either side's own spread. **Gate W3 is now genuinely, fully
      met on both halves** — marked `[x]`, no longer `[~]`. See SESSION_LOG.md's 2026-08-24
      "S3 QKD-degradation recalibration, Gate W3 S3 attempt" entry for full methodology,
      numbers, and the separately-flagged posture-saturation finding (see "Next task").**
      Full prior history
      *(Still not attemptable for real — S3 doesn't exist as a scenario yet. 2026-08-10's
      load-spike diagnostics (NOT real S3/S4 — see Next task and `configs/default.yaml`'s
      `load_spike:` block) showed `forced_rekey_ratio` dropping well below flat-S1's
      never-proactive `1.000` once arrival load genuinely varies, directionally confirming
      the reward mechanism isn't broken — but even after fixing `agents/dqn.py`'s previously
      unseeded randomness and re-running the 10-seed sweep with genuinely controlled seeds,
      the spread got wider, not tighter (`0.102`-`1.000`, half the seeds at the exact
      never-proactive ceiling) — a real learn/don't-learn split by training run, not a
      measurement artifact. **2026-08-17: a budget probe on 3 stuck seeds at 50k/75k steps
      found this isn't a training-budget question either** — two of the three reached good
      intermediate values at 50k then regressed back to the exact ceiling by 75k, a
      mid-training instability, not slow convergence. **2026-08-18: a denser (every-1000-step)
      re-probe overturned even that framing** — there's no localized regression at all;
      `forced_rekey_ratio` swings 0.5+ between adjacent 1,000-step snapshots roughly 1 in 3
      of the time, continuously across an entire 75,000-step run, for every seed tested
      (including one previously read as "flat"). **A second same-day diagnostic then tested
      whether this is single-episode eval noise or eval-cadence/target-sync aliasing — both
      disfavored**: averaging 8 fixed eval seeds per checkpoint only shrinks the swing by
      ~4-6x less than the checkpoint-to-checkpoint swing itself (still >0.5 on ~1-in-4
      adjacent snapshots), and a non-aligned eval cadence swings just as hard as an aligned
      one. The mechanism behind the swings remains unknown — future Gate W3 attempts need
      **checkpoint-averaged**, not single-checkpoint or eval-seed-distribution, comparisons,
      on top of multi-seed training reporting. Evidence toward attempting the gate once S3
      exists, not the gate itself.)*
- [x] **Soft-reward baseline agent reproducing Noetzold
      (`agents/soft_reward_baseline.py`) — real and tested since 2026-08-25.**
      Reuses `agents.dqn.DQNAgent` completely unmodified; its own reward
      function (`compute_soft_reward`, a genuine `w_sec*security_score(tier)`
      term, Hard Rule 4-grounded) and config (`configs/soft_reward_baseline.yaml`,
      `security_masking: false`) are the only new pieces. A real 25,000-step
      training run confirmed `floor_violations > 0` at every one of 10 eval
      checkpoints — direct, measured evidence security is genuinely traded
      away. See `agents/soft_reward_baseline.py`'s per-file row and
      SESSION_LOG.md's newest entry for the full design/testing writeup.
- [x] Scenario dispatch S2-S4 wired into `environment.py` — S2 (HNDL) + S3
      (QKD degradation) real since 2026-08-19; **S4 (DDoS/noisy-neighbor)
      real since 2026-08-24** — see `env/environment.py`'s per-file row for
      the full mechanism. The 2026-08-10 `load_spike` diagnostic remains a
      request-rate-only stand-in, unrelated to this real dispatch. (S5/S6
      are separate milestone items, tracked below, not part of "S2-S4.")
- [~] Real LSTM dual-head forecaster (Addition A / PLAN2.md §8A Addition E)
      — **Phase 1 (RT-IoT2022 feature-extraction pipeline) DONE 2026-08-31**:
      `forecaster/rt_iot_features.py` implemented+tested (see that file's
      new per-file row below). Phases 2-4 (build/train the LSTM in
      `forecaster/model.py`/`forecaster/dataset.py`/`forecaster/train.py`,
      validate it, wire it in via `LSTMForecastProvider` in
      `env/forecast_provider.py`) remain not started — marked `[~]`
      (partial), not `[x]`, until Phase 4 completes.
- [ ] E-A foresight ablation (off / ewma / lstm on S3 + S6)
- [x] Steering attack — adversarial threat-trace generator (`attack/trace_generator.py`,
      real and tested since 2026-08-25 — implements paper draft equation 7,
      `x̃t = (1-α)xt + α·g(xt)`, verified end-to-end through the real forecaster
      and masking layer, zero changes needed to `env/`)
      + **the attack run itself, real and complete since 2026-08-26**:
      `attack/attacking_provider.py::AttackingForecastProvider` (live equation-7
      wrapper, injected via `env/environment.py`'s new, flagged/signed-off
      `forecast_provider_factory` param) + `experiments/harness.py`'s dual-tracking
      measurement (`run_scenario_under_attack`/`evaluate_attack_multi_seed`, real
      `V(pi)` per paper eq. 4) produced the real 11-alpha split-screen result on
      S2 (masked DQN vs. soft-reward baseline, 5 eval seeds each) — Gate W5
      **attempted, with a real, more-nuanced-than-predicted result** (masked
      agent's `V(pi)` is NOT flat zero everywhere — see "Next task" below and
      SESSION_LOG.md's newest entry for the full mechanism and honest read).
      Headline contribution never cut; the finding itself belongs in the paper's
      Limitations section, not treated as a failed gate.
- [x] **S6 migration wave (scripted schedule, held-out eval only) — real and
      dispatched since 2026-08-24.** `config["migration_graph_seed"]`/
      `config["migration_schedule"]` (required only under `scenario: S6`)
      build a tenant graph once and apply a small, scripted list of
      `{step, tenant_index, new_sensitivity_class}` ratchet events via
      `RequestGenerator`'s new `set_tenant_sensitivity_class`, entirely
      upstream in request generation (Hard Rule 3) — zero `env/masking.py`
      changes needed, since floor computation already reads
      `sensitivity_class` fresh off every request every decision. Hard Rule
      8 (train/eval split) enforced as a real code-level guard:
      `experiments/train.py::train()` refuses to proceed if
      `config["train_eligible"]` is `False` (set only in `configs/scenarios/
      s6_migration.yaml`). **Completing S1-S6 (bar S5) is itself a real
      milestone** — see `env/environment.py`'s/`env/request_generator.py`'s
      per-file rows and SESSION_LOG.md's newest entry for the full mechanism,
      the three-event schedule chosen and why, and all test results.
- [ ] Live dashboard (`dashboard/app.py`) — 4-beat demo. **2026-08-29:
      the Explain Decision panel (one of seven) now has a real render
      layer** (`dashboard/render_explain.py`, real samples in
      `dashboard/samples/`) — see this file's "Next task" and
      `dashboard/`'s per-file rows below. **2026-08-29 (later session):
      the two headline demo visuals — S5 dose-response chart + S3
      comparison table — also now render from real, freshly re-run
      data** (`dashboard/render_dose_response.py`, `dashboard/
      render_comparison_table.py`, samples in `dashboard/samples/`).
      **2026-08-29 (this session): the Living System panel (real
      tenant graph + tier-colored recent decisions) also now renders
      from real data** (`dashboard/render_living_system.py`, static
      snapshots in `dashboard/samples/living_system_*.html`).
      **2026-08-30: Budgeting Brain (real S3 pool-trajectory comparison)
      and Migration Wave (real S6 scripted-schedule attribution) also
      now render from real data** (`dashboard/render_budgeting_brain.py`,
      `dashboard/render_migration_wave.py`, samples in `dashboard/
      samples/budgeting_*`/`migration_*`). **The demo-able dashboard
      panel set is now COMPLETE — 6 of 7 panels rendered from real
      data.** `dashboard/app.py` itself (the live, wired 4-beat demo
      shell) is still not started; the only remaining panel (Threat
      Input) is blocked on the not-yet-built real forecaster/feature-
      extraction pipeline, not a rendering-effort gap.
- [ ] Live/replayed pcap input (PLAN2.md §8B, Addition F) — depends on
      the LSTM forecaster item above completing Phase 4 first; not
      started.
- [ ] AWS-KMS-style API facade (`api/main.py`, PLAN2.md §8C, Addition G)
      — depends on the pcap-input item above completing first; not
      started.
- [ ] Report (`docs/report.md`) — currently a section-header skeleton with TODOs only

---

## Per-file status

Status values: **not started** (stub, `raise NotImplementedError`, 11-line
import-smoke test only) · **stub (partial)** (some functions real, some
still `NotImplementedError`) · **implemented+tested** (real logic, real
behavioral tests, part of the green `pytest` run).

### env/

| File | Status | Notes |
|---|---|---|
| `env/contracts.py` | implemented+tested | Frozen interface contract — `Action`, `StateDict`, `ForecastProvider` ABC, `Request`, event-log TypedDicts. 5 real tests (`test_contracts.py`). |
| `env/pool_sim.py` | implemented+tested | `PoolSim` (refill/drain/exhaustion) + `SyntheticSKRQBERTrace`. 22 tests (`test_pool_sim.py`, up from 19). **2026-08-19: no code changed here** — `SyntheticSKRQBERTrace`'s existing `spike_start`/`spike_duration`/`spike_magnitude` params (already present, already documented as "the dial-in hook for the S3 'QKD degradation' scenario") are now actually exercised by real S3 dispatch in `environment.py`, not just by `test_pool_sim.py`'s own standalone degradation test. **2026-08-24 (Gate W3 S3 recalibration): `SyntheticSKRQBERTrace` gained an optional `spike_skr_multiplier: float \| None = None` field** (docstring step 4a) — when set, replaces the pre-existing `qber`-derived, hard-capped-at-50%-reduction formula with a direct multiplier during the spike window, decoupling degradation severity from `qber`'s own clip/noise. `None` (the default, and every pre-existing caller/config) reproduces the prior formula byte-for-byte — verified directly (`test_spike_skr_multiplier_none_is_byte_identical_to_prior_formula`). Root cause for adding this: the pre-existing formula's 50% cap is magnitude-independent (re-verified this session: `spike_magnitude` in `{0.6, 0.9, 0.99, 5.0}` all produce byte-identical in-window SKR, 99.7448 kbps exact), so no `spike_magnitude` value could ever produce genuine scarcity — see `configs/scenarios/s3_degradation.yaml`'s row and SESSION_LOG.md's newest entry for the full investigation and the grounding (real QKD security-threshold literature, Hard Rule 4) behind the `0.0` value S3 now uses. 3 new tests. |
| `env/deferral_queue.py` | implemented+tested | `DeferralQueue.enqueue/tick/pop_servable`, priority+FIFO, cumulative-headroom draw. 8 tests (`test_deferral_queue.py`). |
| `env/masking.py` | implemented+tested | `PolicyTable` (placeholder floor table, sticky ratchet) + `compute_mask`. 19 tests (`test_masking.py`, up from 14). Floor table not yet calibrated against Q-OPSEC data. **2026-08-19: closed a real Hard Rule 2 gap** — `compute_mask` gained an opt-in `current_key_type` param and a fourth rule: REUSE is illegal if the session's existing key tier is now below the current floor (previously only key *age* gated REUSE, never whether the established tier still met a since-ratcheted-up floor). Found and fixed via independent review of this repo's own code against Hard Rule 2's text; measured real on a live S2 episode before the fix: 64/279 (22.9%) REUSE/REKEY_NOW decisions delivered below-floor key material; 0/270 after. The prior "REUSE illegal only past key-age cap" description of this file is now out of date — see `env/environment.py`'s row for the REKEY_NOW half of the same fix. |
| `env/forecast_provider.py` | stub (partial) | `MovingAverageForecaster` (EWMA fallback) implemented+tested (9 tests, `test_forecast_provider.py`). `LSTMForecastProvider` does not exist yet (Addition A) — `use_foresight: lstm` currently raises `NotImplementedError` in `environment.py`. |
| `env/request_generator.py` | implemented+tested | `random_request_generator()` implemented+tested (as before, incl. the 2026-08-10 `load_spike` diagnostic — **explicitly not real S4**, see `configs/default.yaml`'s `load_spike:` block). **2026-08-23 (new): `build_tenant_graph(n_nodes, seed) -> nx.Graph` and `RequestGenerator(graph, seed)` are now real, tested implementations**, closing the "real NetworkX tenant graph" milestone item. `build_tenant_graph`: hub-and-spoke topology (one `kms_hub` node, one edge to each of `n_nodes` tenant nodes) — a direct rendering of PLAN.md §4's architecture diagram and a natural future dashboard graph view; each tenant node carries persistent `sensitivity_class` (real `SensitivityClass` value, Hard Rule 4), `traffic_rate` (positive float, `0.2`-`5.0`), `pqc_capable` (bool, same `_PQC_CAPABLE_PROB=0.9` convention as the random generator, Hard Rule 6), and `services` (1-2 names drawn from the existing `_SERVICES` vocabulary, never re-rolled). New `load_tenant_graph_config()` standalone typed loader for `tenant_graph:` (mirrors `load_pool_config`/`load_key_lifetime_config`/`load_dqn_config`'s convention, not auto-invoked inside `build_tenant_graph` itself, same relationship those three have to their own modules). `RequestGenerator`: samples which tenant arrives weighted by that tenant's persistent `traffic_rate` (verified statistically); a sampled request's `sensitivity_class`/`pqc_capable` are read directly from the tenant's node attributes, never re-rolled (verified with zero drift across 500 sampled requests) — this is the property that gives tenants real identity, the point of the session. `hybrid_mandatory` stays an independent per-request draw (contracts.py/PLAN.md never describe it as tenant-persistent). Also directly iterable (`__iter__`) as an `Iterator[Request]` adapter over `.step()`, so it can be swapped in wherever `random_request_generator`'s output is consumed. **Two deviations from the session brief, both because the brief didn't match the real, frozen `env/contracts.py`** (verified by reading that file, not assumed): `Request`'s real field is `service`, not `service_id`; `Request` has no `data_lifetime_days` field at all (nothing added to the frozen contract to manufacture one — flagged, not invented). 22 tests total in `test_request_generator.py` (up from 11: 13 new, minus the 2 removed `NotImplementedError` placeholder tests): node count/topology/attribute validity, byte-identical graphs under a repeated seed, a different seed producing a different graph, `RequestGenerator` field validity, same-seed/`reset()` reproducibility, the zero-drift tenant-attribute check, and the traffic-rate-weighted arrival-share statistical check. **Hard Rule 3 swap test**: `tests/test_environment.py::test_hard_rule_3_graph_driven_generator_is_a_drop_in_replacement` duplicates the existing Gate W2 test (`test_gate_full_s1_episode_random_valid_policy_zero_floor_violations`) verbatim except for one line constructing `SmartKeyNetEnv` with a `request_stream_factory` built from the real graph/generator — a full 250-step S1 episode under a random *valid* policy, zero floor violations, through completely unmodified agent-facing code. See `env/environment.py`'s row for the one narrow, additive `environment.py` change this needed. **2026-08-24 (new): `RequestGenerator` gained an optional `flood_override` constructor parameter** (`{"tenant_id": str, "extra_rate": float}`, default `None`) — S4's (DDoS/noisy-neighbor) real mechanism. Deliberately *additive*, not a `traffic_rate` mutation: every `step()` call's existing base weighted-multinomial draw is completely unmodified code, byte-for-byte, whether or not a flood is active (verified: `flood_override=None` behaves identically to omitting the argument, and every pre-existing test in this file that never passes it still passes unchanged); when set, a **second, independent RNG stream** (`_flood_rng`, separate from the base `_rng`) draws an additional `Poisson(extra_rate)` batch for the one designated tenant, reusing the exact same `_build_request` field-construction logic (refactored out of the base loop this session, verified behavior-preserving by the full pre-existing test suite passing unmodified before any new test was added). This design choice was deliberate, not the literal "multiply the node's traffic_rate attribute" framing initially considered: under this class's real sampling model (one shared `Poisson(_ARRIVAL_RATE_PER_STEP)` total split across tenants by weighted multinomial, not independent per-tenant Poisson processes), a bare weight multiply would mechanically dilute every *other* tenant's expected share too (verified empirically this session: an 8.6%→36% weight shift for one tenant cut every other tenant's realized share to ~70% of its own baseline) — failing the requirement that the flood be isolated to the designated tenant. The additive two-RNG-stream design instead makes that isolation exact: 6 new tests in `test_request_generator.py` (up from 22) confirm the flooded tenant's count rises (>3x), every *other* tenant's count is **byte-for-byte identical** (not just statistically close) between a flood-on and flood-off run, flood-batch requests pass the same field-validity/zero-drift checks as any other request, an unknown `tenant_id` raises `ValueError`, and `flood_override=None`/omitted are indistinguishable. See `env/environment.py`'s row for how `configs/scenarios/s4_ddos.yaml` drives this in practice, and for the honest, separate finding that this exact-isolation guarantee does *not* carry through unchanged to environment-mediated decision *counts* over a fixed external step budget (a downstream FIFO-queue-draining effect, not a leak in this mechanism — see that row). **2026-08-24 (new, same day, later session): `RequestGenerator` gained `set_tenant_sensitivity_class(tenant_id, new_sensitivity_class)`** — S6's (migration wave) real mechanism. Writes directly into `_tenant_attrs_by_id[tenant_id]`, the same live-reference dict object NetworkX stores on the graph node (not a copy), so the underlying graph updates too by construction; validates the new value via the `SensitivityClass` enum constructor (Hard Rule 4). No caching anywhere in this class (or in `env/masking.py`'s floor computation, verified by reading both) assumes a tenant's `sensitivity_class` is fixed for an episode, so this one mutation is sufficient — every subsequent request for that tenant reads the new class starting with the very next `step()` call, base or flood. 5 new tests in `test_request_generator.py` (up from 28): unknown-tenant/invalid-class rejection, immediate effect on subsequent requests (verified request-by-request against a real, high-traffic tenant), byte-for-byte isolation of every other tenant, and the underlying-graph-node update. See `env/environment.py`'s row for how `configs/scenarios/s6_migration.yaml`'s scripted schedule drives this in practice. |
| `env/environment.py` | implemented+tested | `SmartKeyNetEnv.reset/step/action_mask` fully wired (pool + deferral + masking + forecast + reward + session-key state). 30 behavioral tests (up from 25) incl. the split.md Gate W2 tests (`test_environment.py`). **2026-08-19 (design decision 11): `_resulting_key_type()` fixed** — REKEY_NOW now resolves to `max(existing session tier, current floor)`, never lower (previously always kept the existing tier verbatim, a real Hard Rule 2 gap — see `env/masking.py`'s row for the REUSE half of the same fix, found and fixed together). Masking gap #1's pool-draw gate now delegates to `_resulting_key_type` itself instead of a separate, previously-inconsistent copy. 2026-08-10: wired `config["load_spike"]` through to `random_request_generator` (design decision 9) — orthogonal to scenario dispatch, not a substitute for it. **2026-08-19 (design decision 10): real S2 (HNDL) + S3 (QKD degradation) scenario dispatch.** `config["scenario"]` now genuinely gates behavior for S2/S3 (S1, and the still-undispatched S4/S5/S6, are unaffected — confirmed by an explicit regression test). S2: `config["threat_schedule"]` (required only under `scenario: S2`) makes `_threat_features_placeholder()` return a scripted elevated signal from a configured step onward, flowing through the *existing, unmodified* `MovingAverageForecaster` → `PolicyTable.ratchet_up`/`floor` → `compute_mask` chain — `env/masking.py`'s floor table itself was never touched (Hard Rule 2), verified directly by a test cross-checking the env's real per-decision floor against a fresh `PolicyTable().floor()` call across a spread of sensitivity classes. S3: `config["qkd_degradation"]` (required only under `scenario: S3`) threads straight into `SyntheticSKRQBERTrace`'s pre-existing spike params at `reset()` — no new pool-sim code. **Observability finding, documented in `configs/scenarios/s3_degradation.yaml`'s own comments**: under `pool:`'s realistic default-scale numbers (256-bit draws vs. ~200,000-bit/tick mean SKR refill), S3's degradation is real but invisible in practice — refill dwarfs any plausible draw pattern by 2-3 orders of magnitude; the regression test demonstrates the real, measurable effect (higher regret-event count, lower minimum pool fill vs. S1) using the same scarcity-forcing small-pool override `test_environment.py`'s existing Hard Rule 9 gate test already established, not a fabricated number. New standalone scenario configs: `configs/scenarios/s2_hndl.yaml`, `configs/scenarios/s3_degradation.yaml` (both directly loadable via `experiments/train.py`'s `load_full_config(path)`, verified). S4 (DDoS/noisy-neighbor) and S6 (migration wave) deliberately NOT dispatched — both need a tenant-identity concept `env/request_generator.py` doesn't have yet; see "Next task". **2026-08-19 (Gate W3 attempt session): this observability finding is now confirmed to reach further than pool fill alone** — under real default config, S3's floor *and posture* sequences are also byte-identical to S1's (not just pool fill), because posture is already saturated regardless of the QBER spike. This blocked Gate W3's S3 half from being attempted at all this session — see SESSION_LOG.md and Thread 1 above. **2026-08-23 (design decision 12): `__init__` gained an optional `request_stream_factory` parameter** (`episode_seed -> Iterator[Request]`, default `None`) — the one narrow, additive exception needed for `env/request_generator.py`'s new real `RequestGenerator` to be a genuine drop-in replacement for `random_request_generator` (Hard Rule 3's swap test). Confirmed before making this change that `random_request_generator` was hardcoded inline in `reset()` (not already injectable) — reading, not assuming. `None` (the default) reproduces prior behavior byte-for-byte; every existing single-argument `SmartKeyNetEnv(config)` call site is unaffected, verified by the full existing test suite passing unmodified. No other line in this file changed. **2026-08-24 (design decision 13): real S4 (DDoS/noisy-neighbor) scenario dispatch.** `config["ddos"]` (required only under `scenario: S4`, same fail-fast convention as S2/S3) builds a real tenant graph once at construction time (`build_tenant_graph(n_nodes, seed=ddos.graph_seed)` — deliberately a seed dedicated to graph structure, decoupled from `episode_seed`, so which tenant `tenant_index` names stays a fixed structural fact of the config, not something that reshuffles across eval seeds) and, in `reset()`, constructs a `RequestGenerator` with a `flood_override` targeting that tenant for the whole episode. Entirely upstream, in the request-arrival process — confirmed by a code-level grep/read check (not just a behavioral test) that `masking.py`, `agents/dqn.py`, and `_apply_action`'s reward calculation contain zero mentions of "scenario"/"ddos"/branching on tenant identity; `"ddos"` appears in the codebase only in `environment.py`'s own sanctioned dispatch code and in `request_generator.py`'s docstring prose (the actual `flood_override` mechanism itself is scenario-agnostic, generically named). 36 tests in `test_environment.py` (up from 31, incl. renaming the now-outdated `test_s4_and_s6_scenarios_are_not_yet_dispatched` to `test_s5_and_s6_...`), incl.: constructing under `scenario: S4` without `config["ddos"]` raises `KeyError`; the flooded tenant's own decision count is measurably (>3x) higher with the flood active vs. inactive, same seed; a real empirical finding, honestly reported rather than assumed — **the critical (highest-sensitivity) tenant's own decision throughput collapses under the flood** (>50% reduction, confirmed over an equal span of real elapsed simulator time via `env._step_count`, not just an equal external-step budget) even though its *per-decision* regret rate barely moves — this environment's dominant, clearly measurable "noisy neighbor" effect is service-opportunity crowd-out via the shared one-decision-per-tick FIFO queue, not (at this baseline policy, this pool scale) a large per-decision quality hit; regret/pool-exhaustion events are confirmed visible only under the same small-pool scarcity override this suite's other regret tests already use (S4's flood is real but invisible in pool-exhaustion terms at `pool:`'s default scale, the same pre-existing calibration-headroom property S3's config already documented — not a new finding, a consistent one). New standalone scenario config `configs/scenarios/s4_ddos.yaml` (directly loadable via `load_full_config(path)`, verified) documents the full empirical numbers in its own comments. See `env/request_generator.py`'s row for the `flood_override` mechanism itself and why it's additive rather than a `traffic_rate` multiply. **2026-08-24 (design decision 14, Gate W3 S3 recalibration): `_qkd_degradation_trace_kwargs()` gained an additive, optional read of `qkd_degradation.spike_skr_multiplier`** (`.get(...)`, not `[...]` -- every S3 config/test that predates this session, and doesn't set this key, is completely unaffected) -- threads straight into `env/pool_sim.py`'s new `SyntheticSKRQBERTrace.spike_skr_multiplier` field (see that file's row). No other line in this function or elsewhere in this file changed; S1/S2/S4 dispatch paths are byte-for-byte untouched, confirmed by the full pre-existing test suite passing unmodified. 38 tests in `test_environment.py` (up from 36): 2 new -- a direct, no-test-override comparison of the real, committed `configs/default.yaml` (S1) against the real, recalibrated `configs/scenarios/s3_degradation.yaml` (S3), same seed, proving genuine divergence (near-total pool exhaustion + real regret events under S3 vs. S1's untouched, never-stressed pool); and an explicit Hard Rule 9 check under the new severe S3 config (deferred, never downgraded, despite genuine exhaustion). See `configs/scenarios/s3_degradation.yaml`'s row and SESSION_LOG.md's newest entry for the full root-cause investigation, the design decision's justification, and the separately-flagged posture-saturation finding. **2026-08-24 (design decision 15, new, same day, later session): real S6 (migration wave) scenario dispatch, completing the full S1-S6 grid.** `config["migration_graph_seed"]` (required only under `scenario: S6`, same fail-fast convention as `ddos.graph_seed`) builds a tenant graph once at construction time, same pattern as S4. `config["migration_schedule"]` (already a top-level key in every config, default `[]`; only ever non-empty for S6's own config) is a small, explicit list of `{step, tenant_index, new_sensitivity_class}` entries — scripted and exogenous (Hard Rule 3). `reset()` under S6 keeps a live reference to the `RequestGenerator` instance (`self._request_generator`, unlike S4's discarded `iter(...)` wrapper) so `_advance_to_next_decision`'s tick loop can call its new `set_tenant_sensitivity_class` at each scripted step — a single, unconditional dispatch loop (`self._migration_schedule` is `[]` for every scenario but S6, so it's a guaranteed no-op elsewhere). **Needed zero `env/masking.py` changes**: confirmed by reading before designing that both `compute_mask`/`PolicyTable.floor` and `_prepare_decision` already read a request's `sensitivity_class` fresh every decision, with no caching anywhere that assumes it's fixed for an episode. **Hard Rule 8 (train/eval split) is enforced in `experiments/train.py`, not here** — see that file's row. 6 new tests in `test_environment.py` (up from 38, incl. renaming `test_s4_and_s6_scenarios_are_not_yet_dispatched` to `test_s5_scenario_is_not_yet_dispatched` since S6 is no longer undispatched): `migration_graph_seed` required under S6 (`KeyError` otherwise); a direct, request-by-request check through the real environment confirming the ratcheted tenant's requests carry the old `sensitivity_class` strictly before the scripted step and the new one at/after it; the resulting `policy_floor` cross-checked against a real `PolicyTable().floor()` call (`use_foresight: off` pins posture at CALM so the check isolates exactly this effect); every other tenant's `sensitivity_class` confirmed byte-for-byte unaffected (isolation check, same spirit as S4's); a Hard Rule 3 code-level grep (masking.py/contracts.py/`_apply_action`'s body have zero "migration"/"s6" mentions); and a held-out-eval sanity run via `experiments/harness.py::run_scenario` against the real, committed `s6_migration.yaml` with a `StaticThresholdPolicy` baseline (`floor_violations == 0` holds across the ratchet points too). See `env/request_generator.py`'s row, `configs/scenarios/s6_migration.yaml`'s row, `experiments/train.py`'s row, and SESSION_LOG.md's newest entry for the full design reasoning and schedule justification. **2026-08-25 (design decision 16, soft-reward baseline agent session): `__init__` gained `config.get("security_masking", True)`.** Found and flagged before being built (via `AskUserQuestion`, user sign-off recorded in SESSION_LOG.md's newest entry): `step()`'s `IllegalActionError` enforces the mask unconditionally at the environment boundary, regardless of which agent is calling — meaning `agents/soft_reward_baseline.py`'s "no action masking" property could not be achieved by that agent's own action-selection code alone, the way its reward function could differ purely in its own code. When `False`, `_prepare_decision` calls the same, unmodified `compute_mask()` with `floor=Action.SERVE_CLASSICAL`/`current_key_type=None` instead of the real floor/session key type — the floor-based and REUSE-below-floor rules become no-ops, while `pool_can_draw`/`key_age`/`max_key_age` pass through unchanged (physical/protocol feasibility, not the security-floor restriction this flag targets). Default `True`, zero effect on any pre-existing config/caller, proven byte-for-byte (4 new tests in `test_environment.py`). Only `configs/soft_reward_baseline.yaml` sets this `False`. Zero `env/masking.py` changes. |

### agents/

| File | Status | Notes |
|---|---|---|
| `agents/dqn.py` | implemented+tested | `flatten_state(state, has_forecast)` (genuinely variable-length: 13 dims under `off`, 28 under `ewma`/`lstm`; `has_forecast` is now an **explicit required parameter** — the earlier `state["threat_score"] != 0.0` inference trick was removed 2026-08-08 as fragile-by-accident, see that session's log entry). `QNetwork` (2-hidden-layer MLP), `DQNConfig`/`load_dqn_config` (reads `configs/default.yaml`'s `dqn:` block), `DQNAgent(state_dim, has_forecast, config, seed=None)` (internal circular-buffer replay, `act`/`observe`/`learn`/`save`/`load`, `has_forecast` fixed once at construction and threaded through every internal `flatten_state` call) — masking applied structurally at both action-selection *and* bootstrap-target time (Hard Rule 2), no security term anywhere (Hard Rule 1). **2026-08-10: `seed` parameter added** — reseeds `random`+`torch`'s global RNGs before `QNetwork` construction when given, so weight init/exploration/replay sampling are genuinely reproducible (`seed=None` default is unchanged prior behavior); fixes a gap the same day's earlier sessions found and flagged (see `experiments/train.py`'s row and SESSION_LOG.md). 26 tests (`test_dqn.py`, up from 23), incl. a regression test for a foresight-mode state with `threat_score == 0.0` still flattening to 28 dims, an integration test training against the real `SmartKeyNetEnv` on S1 for 3000 steps with loss trending down, and 3 new seed tests (same-seed reproducibility, different-seed divergence, `seed=None` leaves ambient state alone). Not yet run to convergence — that's `experiments/train.py`. |
| `agents/baselines.py` | implemented+tested | `AlwaysPQCPolicy`, `AlwaysHybridPolicy`, `StaticThresholdPolicy` (incl. `grid_search`), `RandomPolicy` — all real, sharing a `_lowest_legal_action` fallback. 261 tests (`test_baselines.py`), incl. an adversarial parametrized sweep over all 31 non-empty action masks per policy (never returns an illegal action, however contrived the mask). |
| `agents/soft_reward_baseline.py` | implemented+tested | **2026-08-25 (new):** Noetzold-style soft-reward baseline agent — the deliberate, contained reproduction Hard Rule 1's own argument is tested against, not a violation of it (see module docstring's explicit tension/resolution). No new agent class: `agents.dqn.DQNAgent` is reused directly, completely unmodified (masking's absence lives in `env/environment.py`'s new `security_masking` config flag, not agent code — see that file's row). This module's own contribution: `SoftRewardConfig`/`load_soft_reward_config` (reads `configs/soft_reward_baseline.yaml`'s own `soft_reward:` block, deliberately separate from `configs/default.yaml`'s `reward:`), `compute_soft_reward(state, action, cfg) = -w_lat*latency - w_en*energy + w_sec*security_score(tier)` (the reproduced design's own, simpler formula — not "our full reward plus a security term"), `security_score`/`_TIER_SECURITY_SCORE` (`classical=0.2, pqc=0.6, hybrid=1.0`, straight from the reproduced spec, Hard Rule 4), and `resolved_cost_action`/`delivered_tier` (mirror `experiments/harness.py`'s own tier-resolution helpers so REUSE/REKEY_NOW get a security_score grounded in what tier they actually deliver — REUSE's can genuinely be below the real floor now, unlike the masked agent). 13 tests in `tests/test_soft_reward_baseline.py` (rewritten from the prior 1-test import-smoke stub): formula/tier-resolution unit tests plus two Hard Rule 1 boundary checks (direct grep confirming `agents/dqn.py` and `env/environment.py::_apply_action` contain zero `security_score`/`w_sec` mentions). A real 25,000-step training campaign (S1, seed 0) confirmed the property this agent exists to demonstrate: `floor_violations > 0` at every one of 10 eval checkpoints (12-184 out of a 250-step episode) via the unmodified `experiments/harness.py::run_scenario`. See `env/environment.py`'s row (design decision 16) and `experiments/train.py`'s row (`train_soft_reward_baseline`) for the rest of the mechanism, and SESSION_LOG.md's newest entry for the full numbers. |

### forecaster/

| File | Status | Notes |
|---|---|---|
| `forecaster/rt_iot_features.py` | implemented+tested | **New 2026-08-31 (PLAN2.md §8A Addition E, Phase 1).** Standalone from `forecaster/dataset.py`'s rollout-log-based windowing — real RT-IoT2022 CSV loading + validation (`load_rt_iot2022`), degeneracy screening (`screen_label_degeneracy`, real result: NOT degenerate, dominant label `DOS_SYN_Hping` at 76.89% of 123,117 rows), a 39-feature deterministic/stateless extraction function (`extract_flow_features`, verified identical whether called on one row alone or embedded in a batch — the "one implementation, two callers" property PLAN2.md Addition D/§8B Addition F need), a persisted z-score scaler (`fit_scaler`/`ContinuousScaler`), and window=64/stride=32 majority-vote windowing (`build_windows`). `run_pipeline()` ran against the FULL dataset: 123,117 raw rows -> 3,846 windows, saved to `data/processed/rt_iot2022/rt_iot2022_windows.npz`. Real, honestly-reported finding: binary benign/attack windows split 390/3,456 (10.1%/89.9%); the 12-class majority-vote label erases `NMAP_FIN_SCAN` entirely (0/3,846 despite 28 real raw rows) — a real constraint Phase 2's LSTM training must design around (class-weighted training, per-class F1, never accuracy). 25 tests (`tests/test_rt_iot_features.py`), incl. a hand-calculated single-row extraction check and a full end-to-end run against the real, complete dataset. Zero changes to `env/forecast_provider.py`/`env/environment.py` (verified via `git diff --stat`) — `MovingAverageForecaster` remains the system's active forecaster; the LSTM model itself (next row) is not yet built. |
| `forecaster/model.py` | not started | Stub, `test_forecaster_model.py` is 1 import-smoke test. `SmartKeyForecaster` (threat + pool dual head) and the future `LSTMForecastProvider` remain to be built here (PLAN2.md §8A Addition E, Phase 2) — next task. |
| `forecaster/dataset.py` | not started | Stub, `test_forecaster_dataset.py` is 1 import-smoke test. `RolloutWindowDataset` (separate, rollout-log-based windowing for the pool head) remains to be built; its relation to `forecaster/rt_iot_features.py`'s RT-IoT2022-direct windowing (above) is an open Phase 2 design decision. |
| `forecaster/train.py` | not started | Stub, `test_forecaster_train.py` is 1 import-smoke test. |

### metrics/

| File | Status | Notes |
|---|---|---|
| `metrics/regret.py` | implemented+tested | `compute_episode_metrics()` + `attribute_regret()`. 11 tests (`test_regret.py`). |

### experiments/

| File | Status | Notes |
|---|---|---|
| `experiments/harness.py` | implemented+tested | `run_scenario` (one policy x scenario x seed episode → `ScenarioResult`, truncated via `max_steps`, default 250) + `run_grid` (every combination). Recomputes per-decision latency/hybrid-draw resolution from public `StateDict` fields (mirrors `env.environment`'s private cost tables/`REKEY_NOW` resolution, since `step()` doesn't surface them directly). `ScenarioResult` gained `total_reward: float` 2026-08-10 (raw summed episode reward — a sharper policy discriminator than `p99_latency`, see that session's log entry). 13 tests (`test_harness.py`, up from 8), incl. the S1 x four-baselines zero-floor-violations check, a `run_grid` combination-count check, and a `total_reward` check against a manually-summed reference. **2026-08-19: `floor_violations` counting was itself buggy, now fixed** — it only ever checked the three tier-serving actions (`action in _TIER_ACTIONS`), so it silently reported `0` on a real S2 episode that independently measured 64 real below-floor deliveries via REUSE/REKEY_NOW (see `env/masking.py`'s row for the underlying Hard Rule 2 fix this depended on). New `_delivered_tier()` helper checks every action's actually-delivered tier; `_resolved_cost_action()`'s REKEY_NOW resolution fixed to match. **The "floor_violations: 0" guarantee this file's docstring and `ScenarioResult`'s own field comment describe was not actually a settled guarantee for REUSE/REKEY_NOW before this session — it is now, verified directly, not just claimed.** **2026-08-19 (Gate W3 attempt session): gained `evaluate_multi_seed(policy, scenario, config, eval_seeds) -> MultiSeedEvalResult`** — runs any `Policy` across multiple eval seeds, reports mean/std for `p99_latency`/`total_reward`/`forced_rekey_ratio`, mean `regret_events`/`pool_exhaustion_events`, and a summed (not averaged) `floor_violations_total`; keeps every raw per-seed `ScenarioResult`. Built to replace single-eval-seed point estimates for Gate W3 (and any future policy comparison) — 2026-08-18's diagnostics found single-checkpoint/single-seed `forced_rekey_ratio` measurements unreliable, and this is the fix at the measurement layer. 19 tests (`test_harness.py`, up from 13), incl. means/std verified against a manual `np.mean`/`np.std` of direct `run_scenario` calls, the `n=1` degenerate case reducing exactly to `run_scenario` itself, and `floor_violations_total` summing (not averaging) across seeds. **2026-08-25 (masked-vs-soft-reward S3 comparison session): `MultiSeedEvalResult` gained `below_floor_rate_mean`/`below_floor_rate_std`** — the RATE form of `floor_violations_total` (`floor_violations / effective_max_steps` per seed, then mean/std across seeds; `effective_max_steps` resolved via the same `config.get("max_steps", _DEFAULT_MAX_STEPS)` fallback `run_scenario` itself uses for every call `evaluate_multi_seed` makes, not a re-guessed denominator) — this is PLAN.md's paper-draft "below-floor service rate" (equation 4), needed to compare the masked DQN against `agents/soft_reward_baseline.py` on real S3 (see SESSION_LOG.md's newest entry). Purely additive (no existing field/call site changed). 21 tests (`test_harness.py`, up from 19): a zero-rate check for a masked policy, and a direct manual-computation cross-check using `security_masking: false` + an always-SERVE_CLASSICAL stub policy on S2. |
| `experiments/train.py` | implemented+tested | `train()` (one continuous S1 episode, `total_steps` from `configs/default.yaml`'s new `training:` block, periodic greedy-mode eval snapshots via the harness, final `DQNAgent.save` checkpoint), `GreedyDQNPolicy` (wraps a trained agent's `q_network` directly for deterministic epsilon=0 evaluation without touching `agents/dqn.py` or the agent's training epsilon-decay counter), `evaluate_against_baseline()` (trained agent vs. grid-searched `StaticThresholdPolicy`, same fixed eval seed). **2026-08-10: `train()` now passes `training_cfg["seed"]` to `DQNAgent(..., seed=...)` too**, not just `env.reset(seed=...)` — see `agents/dqn.py`'s row. 6 tests (`test_train.py`), incl. a smoke run (100 steps) and a determinism check contrasting `GreedyDQNPolicy` against `DQNAgent.act()`'s genuine epsilon=1 stochasticity. **Six real 25,000-step campaigns executed 2026-08-10 across four sessions** (~40-46s/run): an epsilon-schedule fix (`epsilon_decay_steps` 50k→12.5k) let training genuinely converge but the converged flat-S1 policy still tied the tuned threshold on `p99_latency` and never rekeyed proactively (`forced_rekey_ratio=1.000`); a load-spike diagnostic (see `env/request_generator.py`'s row) re-ran under it and got `0.256`/`0.872` across two seeds; a 10-seed sweep sized that spread properly (`0.190`-`1.000`, mean `0.735`, stdev `0.275`) and found `agents/dqn.py`'s randomness was never seeded — training seed only reached the environment; a same-day fix session seeded it and re-ran the same 10-seed sweep — the spread got *wider*, not tighter (`0.102`-`1.000`, mean `0.700`, stdev `0.345`), with exactly half the seeds landing at the exact never-proactive ceiling. **2026-08-17: a budget probe (6 more real campaigns, 50k/75k steps, 3 of the 5 stuck seeds) found the stuck seeds don't respond to more training budget** — 2 of 3 reached good intermediate `forced_rekey_ratio` values at 50k steps (`0.102`, `0.659`) and then regressed back to the exact `1.000` ceiling by 75k, read at the time as a mid-training instability. **2026-08-18: a denser re-probe (3 more real 75,000-step campaigns, `eval_every=1000` — ~75 snapshots each instead of 3) overturned that framing** — there is no localized regression: `forced_rekey_ratio` swings 0.5+ between adjacent 1,000-step snapshots roughly 1 in 3 of the time, continuously across the entire run, for every seed including one previously read as "flat, never found a good policy." Buffer-capacity crossing at step 50,000 (`agents/dqn.py`'s `_REPLAY_BUFFER_CAPACITY`) is ruled out as the cause — no loss anomaly or swing-amplitude change near that step. A real but separate long-run drift toward the ceiling exists (noisy, not step-changed at any specific step) layered on top of the noise. **A second 2026-08-18 diagnostic (3 more real 75,000-step campaigns, `eval_every=750`, 8 fixed eval seeds/snapshot instead of 1) tested the two remaining hypotheses that session left open**: single-episode eval noise (RULED OUT — 8-seed averaging only shrinks the checkpoint-to-checkpoint swing to ~4-6x smaller than the swing itself, nowhere near enough to explain it) and eval-cadence/target-sync aliasing (evidence against, not a clean rule-out since it's a between-session comparison — a non-aligned cadence swings just as hard as the aligned one did). The ceiling-fraction drift was confirmed to survive 8-seed averaging. The actual mechanism behind the swings is still unidentified. See SESSION_LOG.md's two 2026-08-18 entries for the full data. **2026-08-19: the Hard Rule 2 masking fix (see env/masking.py's row) substantially reduced the checkpoint-oscillation itself (ceiling-drift eliminated, swings down ~40-45%) — see SESSION_LOG.md's post-fix probe entry. Later the same day, Gate W3 was genuinely attempted for real** (this file's own `train()`/`GreedyDQNPolicy`/`evaluate_against_baseline` re-confirmed unmodified and correctly read-only first): 3 training seeds x 75,000 steps each on real S1, evaluated checkpoint-averaged via the new `experiments/harness.py::evaluate_multi_seed()` (8 eval seeds) rather than `evaluate_against_baseline()`'s existing single-seed methodology. **S1 result: DQN beats a properly-tuned `StaticThresholdPolicy` by ~12.6x on `total_reward`** (-5119.71 vs -64616.41), robustly across both training- and eval-seed spread — Gate W3's S1 half is met. **A real methodological finding along the way**: `evaluate_against_baseline()`'s existing `threshold_eval_fn` tunes on single-seed `-p99_latency`, which this session found is constant (`1.5000`) across the *entire* threshold grid under real S1 — `StaticThresholdPolicy.grid_search`'s tie-break then silently defaults to the grid's first candidate, not a genuinely tuned choice. **Not fixed here** (this session's own Gate W3 numbers used a separate, correctly `total_reward`-tuned grid search, not this function) — flagged for whoever next touches this file. `train()` was not generalized to scenarios beyond S1 in that session, since S3 turned out not to be meaningfully attemptable under current config at the time (see env/environment.py's row) — building that generality then would have been speculative. **2026-08-24 (S6 session): `train()` gained a Hard Rule 8 guard** — checks `config.get("train_eligible", True)` at the very top, before any env/agent construction, and raises `ValueError` if `False` (`configs/scenarios/s6_migration.yaml` is the one config that sets this). Deliberately keyed on the flag itself, not a hardcoded `scenario == "S6"` string check, so it would stay correct even once the file's then-separately-flagged hardcoded-`"S1"` override was later generalized — a string-keyed guard would otherwise have been silently defeated by that fix. **2026-08-24 (this session, later): the hardcoded-`"S1"` debt itself is now fixed** — `train()` gained a real, explicit `scenario: str = "S1"` parameter, threaded into `env_config` and into the periodic eval snapshot's `run_scenario` call exactly the way `experiments/harness.py`'s `run_scenario`/`run_grid` already do (an external override, not read from `full_config["scenario"]`) — the same existing convention, not a second parallel one. Defaulting to `"S1"` means every pre-existing call site (both tests and `main()`) is unaffected — proven with a genuine before/after comparison (identical config/seed/overrides, git-stashed the diff and re-ran): reward-window averages, full loss curve (237 real gradient steps), and every eval snapshot's `total_reward`/`p99_latency`/`forced_rekey_ratio` are byte-for-byte identical pre- and post-fix. The Hard Rule 8 guard's own logic/location was not touched (per instruction) — what changed is that it's now reachable through a genuinely scenario-selectable call: `train(s6_config, scenario="S6", ...)` raises `ValueError` before `SmartKeyNetEnv.__init__` is ever called (verified directly via an instrumented call count of exactly 0), where previously the guard fired only because `train()` happened to ignore whatever scenario a config named, never because scenario selection itself was real. `scenario="S3"` was verified two ways to actually dispatch S3's real degradation dynamics, not just be silently accepted: (a) pointing `scenario="S3"` at the S1-only `default.yaml` (which lacks `qkd_degradation`) raises `KeyError`, proving the parameter reaches `SmartKeyNetEnv`'s real dispatch; (b) a direct pool-trajectory stress test (same env-construction line `train()` uses internally, mirrors SESSION_LOG.md 2026-08-24's own S3-vs-S1 divergence methodology) shows S3's pool collapsing to near-total exhaustion (min fill 0.16%) while S1's stays >69% full, under the real committed config files. A control test confirms `scenario="S3"` (where `train_eligible` defaults `True`) does not raise — the guard is scenario-specific, not a blanket block. `evaluate_against_baseline()` is unchanged and still evaluates only against S1 — explicitly out of this session's scope (a separate function, not part of `train()`'s scenario-selection fix). 5 new tests in `test_train.py` (up from 7, total 12): the byte-for-byte before/after comparison, the S3 dispatch-proof pair, the S6 guard end-to-end proof (with an instrumented zero-env-construction check), and the S3 control test. See SESSION_LOG.md's newest entry for the full before/after numbers. **2026-08-25 (this session): gained `train_soft_reward_baseline(full_config, training_overrides=None, scenario="S1") -> (DQNAgent, TrainingRecord)`** — additive, does not touch `train()` (confirmed via `git diff`). Mirrors `train()`'s own loop exactly (reuses `DQNAgent`/`GreedyDQNPolicy`/`run_scenario` unmodified) except for the one thing that must differ: each step's reward is computed via `agents.soft_reward_baseline.compute_soft_reward(state, action, soft_reward_cfg)`, not `env.step()`'s own returned reward (discarded, proven genuinely unused via a poisoned-reward monkeypatch test). Defaults to loading `configs/soft_reward_baseline.yaml`, not `configs/default.yaml`. Same Hard Rule 8 `train_eligible` guard, mirrored. 5 new tests in `test_train.py`: smoke run, `security_masking: False` dispatch proof (instrumented config capture), the poisoned-reward Hard Rule 1 proof, the real 3,000-step training-run proof (loss trends down, `floor_violations > 0`), and the Hard Rule 8 guard mirror. See SESSION_LOG.md's newest entry for the full 25,000-step campaign numbers. |

### attack/

| File | Status | Notes |
|---|---|---|
| `attack/trace_generator.py` | implemented+tested | **2026-08-25 (new file):** the equation-7 adversarial input-shaping generator (PLAN.md §5 S5, PLAN2.md §7.5 Panel 5). `generate_adversarial_window(true_window, alpha) -> shaped_window` implements `x̃t = (1-α)xt + α·g(xt)` exactly (`alpha` an explicit, independent per-call parameter, not config-baked, so a future dose-response sweep can call it across eleven alpha values against the same base window in one run); `g(xt) -> [0.0]*len(xt)` — a deterministic, all-zero "benign region" target chosen because zero qber / zero load are the real lowest legitimate values `env/environment.py::_threat_features_placeholder()`'s current `[qber, load]` window can take, not an arbitrary choice — deliberately targets the CURRENT placeholder `MovingAverageForecaster`, documented as needing revisiting once the real LSTM forecaster (Addition A) exists. Pure input transformation: never touches `ForecastProvider`/`env/masking.py`/`env/environment.py` — a future sweep session substitutes the shaped window for the true one before it reaches `forecaster.update()`. 23 tests in `test_trace_generator.py`: exact (not approximate) boundary equality at `alpha=0`/`alpha=1`; linearity at `alpha∈{0.25,0.5,0.75}` checked against an independently-recomputed formula in the test itself; a no-mutation/new-object check; an end-to-end attack-effectiveness proof through the real forecaster (a genuinely severe `true_window=[5.0,5.0]` — mirroring how `env/environment.py`'s own real S2 `elevated_signal` mechanism already reaches HIGH, since ordinary in-domain `[qber,load]` values are structurally capped at ELEVATED per the 2026-08-24 posture-ceiling finding in `env/forecast_provider.py`'s row — steady-states to HIGH posture at `alpha=0` and genuinely crosses down to ELEVATED at `alpha=1`, with `threat_score`/`posture_probs[HIGH]`/`posture_probs[CALM]` all moving monotonically across the full alpha grid); and the masking-safety property (this session's most important test) verified with real numbers for `sensitivity_class=S2` (`true_floor=SERVE_HYBRID`, `shaped_floor=SERVE_PQC` — `compute_mask()` given only the attacked floor never legalizes `SERVE_CLASSICAL`, while `SERVE_PQC` becomes legal under the attack where it wasn't under the true floor — the attack's real, measured effect, safety guarantee intact), plus a general, exhaustive 12-cell (4 sensitivity classes x 3 postures) sweep of the same property independent of this session's specific numbers. Zero changes to `env/forecast_provider.py`/`env/masking.py`/`env/environment.py`, verified via `git diff --stat`. See SESSION_LOG.md's newest entry for the full design reasoning and verified numbers. |
| `attack/steering_trace.py` | not started | Stub, `test_steering_trace.py` is 1 import-smoke test — an older, PLAN.md-era file name/shape for the eventual dose-response sweep module; superseded by PLAN2.md's `attack/trace_generator.py` + `attack/run_attack.py` naming (see PLAN2.md §7.5's table). Not touched or removed this session — out of scope; the future S5 dose-response sweep session should resolve this naming duplication (most likely: build the sweep as a separate, PLAN2.md-named module and retire this stub, rather than reusing it). |

### dashboard/

| File | Status | Notes |
|---|---|---|
| `dashboard/app.py` | implemented+tested | **2026-08-31 (earlier session, this row was stale until now):** a real minimal local server (`python -m dashboard.app`, standard-library `http.server` only) serving the already-rendered panels in `dashboard/samples/` plus an index page -- `PANELS` registry (title/description/files) + pure `render_index_html()`/`resolve_route()` (route logic testable without a socket) + `DashboardRequestHandler`. Route surface is an explicit filename whitelist, not a general static file server -- no path-traversal surface. **2026-08-31 (this session, later): `PANELS` gained two more cards** -- "Living System -- S1, trained DQN" / "-- S2, trained DQN" (see `dashboard/render_living_system_demo.py`'s row) -- registry now 9 panels/19 filenames (up from 7/13). `tests/test_dashboard_app.py`'s registry-shape count assertions updated to match; all other tests (route resolution, 404s, path-traversal, real end-to-end HTTP round trip) unchanged and still passing. |
| `dashboard/render_explain.py` | implemented+tested | **2026-08-29 (new file):** Explain Decision panel's view layer -- renders `dashboard/explain.py`'s real `DecisionTrace` objects as a self-contained static HTML page (inline CSS, zero JS, zero server, zero build step, zero new dependencies -- deliberately does not reach for `dash`/`plotly` despite both already being in `requirements.txt` for the eventual full live dashboard). `render_trace_html(trace, *, title=...)` (pure function, six private `_render_stepN_*` helpers, each reading only `DecisionTrace` fields) + `write_trace_html(trace, path, *, title=...)`. Styled to visually match `dashboard/mockups/smartkeynet_dashboard_mockup_v2.html`'s Explain Decision tab (dark theme, floor grid, mask chips, cost bars) but reads zero data from it -- that file's example numbers are 100% fabricated per its own header. Floor-grid cells carry a `data-cell="{SensClass}-{Posture}"` attribute so the fired cell is programmatically verifiable, not just visual. Every dynamic string passed through `html.escape`. 10 tests (`test_render_explain.py`), incl. two exact-match Hard Rule 10 tests stepping a real `SmartKeyNetEnv` for 15 decisions and asserting the rendered per-action reasons/final sentence/cost-row ordering are character-for-character identical to the real trace's own fields (never invented), a floor-grid `hit`-cell test parametrized across all 12 real `(SensitivityClass, ThreatPosture)` cells, and two HTML well-formedness tests via a small balanced-tag `HTMLParser` subclass (no new dependency). See SESSION_LOG.md's newest entry for the full design-decision writeup. |
| `dashboard/render_dose_response.py` | implemented+tested | **2026-08-29 (new file, later session):** S5 dose-response chart renderer -- the second headline demo visual. `DoseResponsePoint(alpha, mean, std)`/`DoseResponseSeries(label, series_key, points)` (a thin, real-field-projecting input contract, populated 1:1 from `MultiSeedAttackEvalResult.alpha`/`.below_floor_rate_true_mean`/`_std`, never re-derived) + `render_dose_response_html(series, ...)` (pure function, inline SVG chart -- polyline + circles + per-point error bars, y-axis auto-scaled off the real data, zero charting library). Styled to the mockup's dose-chart look (masked=`--hybrid` green, soft-reward=`--danger` red) but reads zero numbers from it. **The one derived annotation (the "first nonzero alpha" boundary callout) is computed live from the input series itself** (never a hardcoded alpha), so a genuinely flat-zero series gets an honest "flat zero across the range shown" callout instead of a fabricated boundary. 9 tests (`test_render_dose_response.py`), incl. the central Hard Rule 7 guards `test_masked_agent_alpha_0_9_nonzero_value_is_faithfully_shown` and `test_masked_curve_is_not_flat_zero_everywhere` (asserts `max(means) == 0.3`, guarding specifically against the tempting flat-zero-everywhere render), a no-fabrication check, and HTML well-formedness. |
| `dashboard/render_comparison_table.py` | implemented+tested | **2026-08-29 (new file, later session):** masked-vs-soft-reward S3 comparison table renderer -- the first headline demo visual. `AgentMetrics` (one agent's real, checkpoint-averaged S3 metrics)/`ComparisonTableData(scenario, masked, soft_reward, include_p99=True)` + `render_comparison_table_html(data, ...)`. `below_floor_rate` always leads the table (hero-styled row), per instruction. **Hard Rule 7 honesty, structural, not just a note**: if `include_p99=True` the `p99_latency` row is ALWAYS paired with a fixed caveat block quoting `experiments/harness.py::ScenarioResult.p99_latency`'s own documented discrete-cost-model-percentile-artifact mechanism (never rendered without it); `regret_events` always carries a `(== pool_exhaustion_events)` label plus a standing "same event by construction" note, quoting `run_scenario`'s own docstring. 10 tests (`test_render_comparison_table.py`), incl. `test_p99_latency_always_carries_its_caveat_when_shown` / a companion test proving `include_p99=False` omits the row, its caveat, AND its value together (no orphaned caveat either direction), a no-fabrication check parsing every `<tbody>` row's numeric leads against the real `AgentMetrics` fields, and HTML well-formedness with/without the p99 row. |
| `dashboard/render_results_demo.py` | implemented+tested | **2026-08-29 (new file, later session):** the real driver for both renderers above. Data provenance decision (see SESSION_LOG.md's newest entry for the full reasoning): no saved raw-results file existed for either the 2026-08-25 S3 comparison or the 2026-08-26 S5 sweep, but the real checkpoints both sessions trained were still on disk (gitignored, not deleted) -- `collect_real_s3_comparison()` reloads `checkpoints/s3_{masked,soft_reward}_seed{1,4,7}.pt` (`_load_greedy_policy` reconstructs the `DQNAgent`/`GreedyDQNPolicy` the same way `experiments/train.py::train()` derives `state_dim`/`has_forecast`) and re-runs real `evaluate_multi_seed` calls (8 eval seeds, matching the original session), aggregating the 3 per-training-seed results into one `AgentMetrics` via the same "checkpoint-averaged, training-seed std" methodology Gate W3 established (verified to reproduce `-10214.82`/`1303.25` exactly). `collect_real_dose_response()` reloads `checkpoints/dqn_s2.pt`/`soft_reward_baseline_s2.pt` and re-runs real `evaluate_attack_multi_seed` across the real 11-alpha grid (5 eval seeds, matching the original sweep). `main()` saves the full fresh output to `dashboard/samples/results_data.json` (with an explicit `"provenance"` field) and renders both HTML files from the SAME live objects the JSON was built from. Reproducibility spot-checked before this driver was built (a reloaded checkpoint's fresh re-run matched SESSION_LOG.md's own per-seed numbers exactly); the full fresh run this session performed matched every prior figure to 4 decimal places. Run via `python -m dashboard.render_results_demo`. |
| `dashboard/render_explain_demo.py` | implemented+tested | **2026-08-29 (new file):** the real driver for `render_explain.py`. `collect_real_traces()` runs a genuine S2 episode (`configs/scenarios/s2_hndl.yaml`, loaded via `experiments.train.load_full_config`) through the real `SmartKeyNetEnv` under the real `agents.baselines.StaticThresholdPolicy(0.5)` baseline -- not hand-authored inputs -- calling `explain_decision_from_env` once per decision, mirroring `experiments/harness.py::run_scenario`'s loop shape. `pick_demo_traces()` selects 3 genuinely different real decisions (first decision; a floor-driven decision, `floor is Action.SERVE_HYBRID`; a genuine multi-cost tradeoff). `main()` writes them to `dashboard/samples/*.html` -- the findable sample outputs for demo/review. **Real investigation documented in this module's own docstring**: searched (100+ real seeds, all four real baseline policies, both real elevated-threat scenario configs) for a genuine "floor-driven, only-one-legal-action" decision (the case `dashboard/explain.py`'s own `cost_note` field flags) and never found one occurring naturally under current calibration -- traced to a real structural cause (`env/masking.py::compute_mask`'s `REKEY_NOW` has no illegality rule once any tier clears the floor, so it stays legal almost everywhere a tier does) rather than assumed or silently worked around; the driver's floor-driven selector uses "only one SERVE tier clears the floor" instead, letting that decision's own real `cost_note` display honestly. Run via `python -m dashboard.render_explain_demo`. |
| `dashboard/render_living_system.py` | implemented+tested | **2026-08-29 (new file, this session):** Living System panel renderer -- the fourth of seven dashboard panels. `TenantNodeView`/`RecentDecisionView`/`LivingSystemSnapshot` dataclasses + `build_snapshot()` (pure fold: real decision history up to a real ordinal cutoff -> one frozen snapshot; "most recently served tier per tenant," `None` for a tenant untouched so far, shown honestly as "no traffic yet," never defaulted) + `render_living_system_html()` (pure view: inline SVG, hub-and-spoke ring layout computed from however many real tenant nodes the graph actually has -- never a fixed count -- edges/nodes tier-colored via the same hex values `render_explain.py`'s CSS already established) + `write_living_system_html()`. 17 tests (`test_render_living_system.py`), all against the real `build_tenant_graph` (never a hand-built `nx.Graph`): real node-count/topology parametrized across `n_nodes` in {3,7,10}, tier-color-correctness parametrized across all three real tiers (this test caught a real bug -- `KeyType.CLASSICAL == 0` is falsy, so an early truthy check silently mislabeled CLASSICAL-served tenants as untouched; fixed to `is not None`), a no-fabrication check, and HTML/SVG well-formedness. See SESSION_LOG.md's newest entry for the full design-decision writeup and the real S2-ratchet-timing finding. |
| `dashboard/render_living_system_demo.py` | implemented+tested | **2026-08-29 (new file, this session):** the real driver for `render_living_system.py`. Builds the real graph (`build_tenant_graph(n_nodes=10, seed=7)` -- a dedicated structural seed, same rationale as S4/S6's own `graph_seed`), runs a real 250-decision S2 episode via `request_stream_factory` injection (the same swap-test precedent `tests/test_environment.py` established), and for each decision resolves the real served tier via `SmartKeyNetEnv._resulting_key_type` (the exact ground-truth function `_apply_action` itself uses -- never re-derived from the action alone, since `REUSE`/`REKEY_NOW` don't map to a tier that way). `pick_snapshot_indices()` selects three real milestones (first decision / first decision by which every real tenant has been served / final decision) -- redesigned mid-session after finding the originally-planned "before/after S2's scripted `elevate_at_step`" predicate doesn't hold on this stream (see SESSION_LOG.md). Writes `dashboard/samples/living_system_0{1,2,3}_*.html`. **2026-08-31 (S1-addition session): `collect_real_episode` gained `scenario_config_path`, `pick_snapshot_indices` gained `prefix`** -- `main()` now runs the identical machinery a second time against `configs/default.yaml` (S1), writing `dashboard/samples/living_system_s1_0{1,2,3}_*.html`. Honest finding: S1 also saturates to 100% HYBRID (see PROGRESS.md's "Next task" preamble entry from that date). **2026-08-31 (this session, later, optional-polish investigation): read-only investigation into WHY both saturate, then an additive fix** -- root-caused to `StaticThresholdPolicy.act`'s own pool-fill-threshold rule (HYBRID whenever `pool_fill > 0.5` and legal; real `pool_fill` never drops below ~0.70 in either episode), not just the placeholder-forecaster ratchet. `collect_real_episode` gained an optional `policy` param (default `None` preserves the original `StaticThresholdPolicy(0.5)` behavior byte-for-byte, verified by a dedicated regression test) + new `_load_masked_policy(config, checkpoint_path)` (mirrors `render_migration_wave_demo.py`'s / `render_results_demo.py`'s existing checkpoint-reload pattern exactly -- reconstruct `DQNAgent` from the config's own `state_dim`/`has_forecast`, load the checkpoint, wrap in `GreedyDQNPolicy`, eval only). `main()` now also runs two additive DQN-driven episodes (`checkpoints/dqn_s1.pt` on S1, `checkpoints/dqn_s2.pt` on S2, both real, already-trained, held/reused from prior sessions -- no training performed). **Real result, verified across seeds 0-3, genuine tier variety**: S1 `{HYBRID: 83, PQC: 167}`; S2 `{CLASSICAL: 6, HYBRID: 163, PQC: 81}` (250 real decisions each, seed 0) -- vs. `StaticThresholdPolicy`'s `{HYBRID: 250}` on both. Writes `dashboard/samples/living_system_{s1,s2}_dqn_0{1,2,3}_*.html` (6 new files), additive alongside (never replacing) every pre-existing sample. A pre-existing latent bug in the S1 collision guard was fixed along the way (it treated a file's own prior committed output as a "collision," which would have made even a same-prefix re-run of this driver fail after the first commit -- `_write_snapshots_guarded` now excludes a run's own `{prefix}_`-named files from the pre-existing set before checking, matching the guard's own documented intent of catching cross-panel collisions only). 7 new tests in `tests/test_render_living_system_demo.py` (mirroring the S1-addition tests exactly: tenant-identity fidelity, exact tier-color correctness, no-fabrication, HTML well-formedness, a `policy=None` byte-identical-behavior regression check, and the central "genuine variety" check). Full suite: 695 passed, 1 xfailed (688 prior + 7 new). Run via `python -m dashboard.render_living_system_demo`. |
| `dashboard/render_budgeting_brain.py` | implemented+tested | **2026-08-30 (new file):** Budgeting Brain panel renderer -- the fifth of seven dashboard panels. `PoolTrajectoryPoint`/`ExhaustionEvent`/`PolicyEpisode`/`BudgetingBrainData` dataclasses (thin, real-field-projecting, populated by the driver) + `render_budgeting_brain_html()` (pure view: two side-by-side inline-SVG area charts, one per policy, `data-trajectory` carrying the exact real `(step, pool_fill)` pairs, real `<circle>` exhaustion markers at their real `(step, pool_fill_at_onset)` positions) + `write_budgeting_brain_html()`. Styled to `dashboard/mockups/smartkeynet_dashboard_mockup_v2.html`'s Budgeting Brain tab (`.arena`/`.badge`/`.stat-grid`/`.exhaust-banner` CSS classes reused verbatim) but reads zero numbers from it. **Real, honest finding, not the design assumed going in**: `below_floor_rate` measured `0.0000` for both policies on the real episode (Hard Rule 9's deferral-not-downgrade guarantee holds even for the still-masked `AlwaysHybridPolicy` baseline) -- the stat grid leads with `pool_exhaustion_events` instead (the metric that actually diverges, 0 vs. 18) and carries an explicit `_BELOW_FLOOR_NOTE` explaining why, rather than reusing `render_comparison_table.py`'s below-floor-leads convention unexamined. 16 tests (`test_render_budgeting_brain.py`), incl. the central Hard Rule 7 contrast pair `test_conserving_agent_shows_zero_exhaustion_markers`/`test_exhausting_baseline_shows_real_exhaustion_markers_and_banner` (same real data, both sides, guards against a dramatized exhaustion that didn't happen or a hidden one that did), no-fabrication checks, p99-caveat pairing tests, and HTML well-formedness. See SESSION_LOG.md's newest entry for the full design-decision writeup. |
| `dashboard/render_budgeting_brain_demo.py` | implemented+tested | **2026-08-30 (new file):** the real driver for `render_budgeting_brain.py`. Runs one real, same-seed (seed=900, held-out -- not the checkpoint's own training seed 1) S3 episode under two policies: the masked DQN (reloaded `checkpoints/s3_masked_seed1.pt`, mirroring `render_results_demo.py::_load_greedy_policy`'s reload pattern) and `agents.baselines.AlwaysHybridPolicy` (no checkpoint, deterministic rule). Both policies see identical exogenous conditions (`env/environment.py::reset()`'s `episode_seed` drives the QKD trace and request stream independently of policy choice) -- a same-seed, same-script, policy-driven-divergence-only comparison. `_run_episode_with_trajectory()` mirrors `experiments/harness.py::run_scenario`'s own step loop (importing its private `_delivered_tier`/`_resolved_cost_action` helpers directly rather than re-deriving that logic, so `floor_violations`/`p99_latency` can never drift from harness.py's real definitions) but adds the per-step `(env._step_count, state["pool_fill"])` capture and real `RegretEvent` collection `run_scenario`'s aggregate-only contract doesn't provide -- `env._step_count` (private, read-only) is used deliberately so trajectory points and event markers share the exact same real internal-tick numbering (multiple real regret events can occur inside one external `env.step()` call). Saves `dashboard/samples/budgeting_data.json`, renders `dashboard/samples/budgeting_brain.html`. Run via `python -m dashboard.render_budgeting_brain_demo`. |
| `dashboard/explain.py` | implemented+tested | **2026-08-19 (new file):** Explain Decision panel backend (PLAN2.md §7.3, Addition D; Hard Rule 10) -- Person D's first real code this project. `explain_decision(...)` (pure function, all six inputs explicit) + `explain_decision_from_env(env, state, chosen_action)` (convenience wrapper pulling those inputs off a live `SmartKeyNetEnv`, mirroring `experiments/harness.py`'s established precedent for reaching into a few private env attributes the public Gym API doesn't yet surface). Returns a `DecisionTrace` dataclass covering all six PLAN2.md §7.3 steps: threat score + source, posture probs + resolved posture, floor lookup (+ the full real floor table, imported from `env.masking`, never re-typed), the action mask (calls `env.masking.compute_mask()` directly -- zero possible drift between this module's legal/reason fields and the masking layer's real behavior, by construction, not by convention), cost comparison (reads `env.environment`'s real `_LATENCY_UNITS`/`_ENERGY_UNITS`/`_KEY_TYPE_TO_SERVE_ACTION`, never re-derived), and a deterministically templated final sentence. Policy-agnostic by design (no dependency on `agents/dqn.py`) -- verified in a scratchpad sanity script against `StaticThresholdPolicy` on real S1 steps, which also caught and fixed one real bug: the final-sentence template originally said "a learned preference from the policy" for the cost-tradeoff case, which is wrong for a non-learning baseline; reworded to "the policy's own preference among legal options." 29 tests (`test_explain.py`, up from 26), incl. every (sensitivity_class, posture) floor-table cell checked against a fresh `PolicyTable`, 6 parametrized mask edge cases (pool empty, key age at/over cap, cold start, all-legal, HYBRID-floor-with-empty-pool) each checked against a real `compute_mask()` call, REKEY_NOW cost-resolution cases (existing-tier and cold-start-adopts-floor), and an end-to-end test stepping a real `SmartKeyNetEnv` and cross-checking every trace against a fresh `compute_mask()` call built from the same env state. **2026-08-19 (same day, later session): updated to track `env/masking.py`'s Hard Rule 2 fix** — `_mask_entries()` now passes `current_key_type` into `compute_mask()` and gained the matching stale-tier reason case (would otherwise crash on the new illegality reason), `_resolved_cost_action()` updated to the same `max(existing, floor)` REKEY_NOW resolution (would otherwise display a cost for a tier that was never actually served). Explicitly out of scope this session (per PLAN2.md's Hard Rule 11 and Hard Rule 10's scoping): pcap ingestion / Threat Input panel (§7.1) and any dashboard HTML/frontend -- this is a Python module returning structured data only. |

### api/

| File | Status | Notes |
|---|---|---|
| `api/main.py` | not started | Stub, `test_api_main.py` is 1 import-smoke test. Formalized as PLAN2.md §8C Addition G — sequenced last, after Addition E (LSTM) and Addition F (pcap replay) both complete; not to be started before then. |

### data/

| File | Status | Notes |
|---|---|---|
| `data/get_data.py` | not started | All four `_download_*` helpers `raise NotImplementedError`; `main()` exists as a dispatcher shell. |
| `data/README.md` | present (doc) | Download instructions written. |
| `data/sample/` | empty | Only a `README.md` placeholder — no committed sample CSVs yet for CI/others to run against. |
| `data/raw/rt_iot2022/RT_IOT2022.csv` | placed, not ingested | Real dataset file present locally (gitignored, not committed); no code reads it yet. |

### docs/

| File | Status | Notes |
|---|---|---|
| `docs/report.md` | skeleton only | Section headers (Abstract, Intro, Related Work, Methodology 3.1-3.3, ...) with `_TODO_` markers and owner tags; no written content yet. |

### configs/

| File | Status | Notes |
|---|---|---|
| `configs/default.yaml` | partial | `pool`, `key_lifetime`, `reward`, `use_foresight`, `tenant_graph.n_nodes`, `load_spike`, `dqn`, `baselines`, `steering_attack`, `training` keys all present. **2026-08-25 (new key): `security_masking: true`** — documented, additive-only (see `env/environment.py`'s row, design decision 16); every other config either inherits this default or sets it explicitly, except `configs/soft_reward_baseline.yaml`, the one config that sets it `false`. `migration_schedule: []` — **2026-08-24: no longer a placeholder; this is the correct, load-bearing value for every scenario but S6** (see `configs/scenarios/s6_migration.yaml`'s row). This file's own comment above the key was corrected from a stale, never-implemented `{tenant_cohort, new_floor}` schema description to the real, shipped `{step, tenant_index, new_sensitivity_class}` one (comment-only change, the key itself is unchanged). `scenario: S1` — deliberately does not carry `threat_schedule`/`qkd_degradation`/`ddos`/`migration_graph_seed` (those are S2/S3/S4/S6-only, required only when `scenario` selects them — see `env/environment.py`'s row). `load_spike.enabled: false` by default (2026-08-10 diagnostic stub, NOT real S4 — see `env/request_generator.py`'s row and SESSION_LOG.md). |
| `configs/scenarios/s2_hndl.yaml` | implemented+tested | **2026-08-19 (new file)**: standalone, directly-loadable S2 config (same shape as `default.yaml`, `scenario: S2` + a `threat_schedule` block). Verified to load and run end-to-end (`tests/test_environment.py::test_scenario_config_files_load_and_construct_a_working_env`). |
| `configs/scenarios/s3_degradation.yaml` | implemented+tested | **2026-08-19 (new file)**: standalone, directly-loadable S3 config, `scenario: S3` + a `qkd_degradation` block. Same verification as above. Its own comments originally documented the pool-scale observability finding (see `env/environment.py`'s row). **2026-08-24 (Gate W3 S3 recalibration): `pool.capacity_bits` dropped `1,000,000 -> 20,000`** (scoped to this file only -- `configs/default.yaml`/S1 completely untouched) and `qkd_degradation` gained `spike_skr_multiplier: 0.0`. Both numbers are computed/grounded, not guessed: capacity kept below the 64,000-bit max-possible-episode-draw-volume (`250 steps * 256 bits/draw`) so genuine exhaustion is reachable, and the multiplier reflects real QKD security-proof literature (BB84-family protocols provably yield zero extractable key above a QBER security threshold, ~11% canonically) rather than an arbitrary worst case. The file's header comment was rewritten to document the full root-cause investigation (why the pre-existing `spike_magnitude` dial couldn't fix this at any value) and retire its prior "S1 and S3 describe the same deployment" framing (no longer true by design -- see the file itself). Verified: real, committed S1 vs. S3 config files now produce genuinely different pool trajectories under the same seed (near-total S3 exhaustion + real regret events vs. S1's untouched pool) -- see `env/pool_sim.py`/`env/environment.py`'s rows and SESSION_LOG.md's newest entry. |
| `configs/scenarios/s4_ddos.yaml` | implemented+tested | **2026-08-24 (new file)**: standalone, directly-loadable S4 config, `scenario: S4` + a `ddos` block (`graph_seed`, `tenant_index`, `extra_rate`). Verified to load and run end-to-end (same `test_scenario_config_files_load_and_construct_a_working_env` test, extended). Its own comments document the full empirical numbers this session found (decision-throughput crowd-out effect, and the same pool-scale regret-event observability caveat S3's file already documented). |
| `configs/scenarios/s6_migration.yaml` | implemented+tested | **2026-08-24 (new file, same day, later session)**: standalone, directly-loadable S6 config, `scenario: S6` + `migration_graph_seed`/`migration_schedule` + **`train_eligible: false`** (the Hard Rule 8 guard `experiments/train.py::train()` checks — the one config in the repo that sets this). Schedule: three phased ratchet events across a 250-step episode, each targeting a tenant starting from a different real, verified pre-migration class (via a real `build_tenant_graph(seed=0, n_nodes=10)` call, not guessed) — `step 60: tenant_0 S1->S3`, `step 130: tenant_3 S2->S3`, `step 190: tenant_4 S0->S2`; no event targets an already-S3 tenant (would be a silent no-op). Verified to load and run end-to-end (`test_scenario_config_files_load_and_construct_a_working_env`, extended) and to actually refuse training (`test_train.py`'s Hard Rule 8 tests) and to run cleanly under held-out evaluation via `experiments/harness.py::run_scenario` (`floor_violations == 0` across all three ratchet points). See `env/environment.py`'s and `env/request_generator.py`'s rows and SESSION_LOG.md's newest entry for the full schedule justification. |
| `configs/soft_reward_baseline.yaml` | implemented+tested | **2026-08-25 (new file)**: standalone config for `agents/soft_reward_baseline.py`'s training entry point (`experiments/train.py::train_soft_reward_baseline`), deliberately separate from `configs/default.yaml` (not a variant loaded alongside it). `security_masking: false` (the one config in the repo that sets this); `reward:`/`dqn:` blocks copied verbatim from `default.yaml` (the former inert — present only so `SmartKeyNetEnv` can construct — the latter genuinely identical, since the DQN architecture is meant to be literally the same); its own new `soft_reward:` block (`w_lat: 1.0, w_en: 0.1, w_sec: 1.0`) is the only genuinely new set of weights. `scenario: S1` — the unmanipulated comparison the masked agent's own campaign uses first. Verified to load, construct, and train for real (25,000-step campaign, see SESSION_LOG.md's newest entry). |
| `configs/soft_reward_baseline_s3.yaml` | implemented+tested | **2026-08-25 (new file, masked-vs-soft-reward S3 comparison session)**: the S3 variant of `configs/soft_reward_baseline.yaml` (PROGRESS.md's former "Next task" follow-up (1), now done). `pool:`/`qkd_degradation:` copied verbatim from the real, committed `configs/scenarios/s3_degradation.yaml` (not re-derived or re-tuned — this was a measurement session); `security_masking: false`/`soft_reward:` copied verbatim from `configs/soft_reward_baseline.yaml`. Same standalone-config-per-scenario convention as `configs/scenarios/*.yaml`, kept alongside `configs/soft_reward_baseline.yaml` in `configs/` (not `configs/scenarios/`) since it belongs to this agent, not the masked one. Verified to load, construct (`security_masking is False`, `scenario == "S3"`), and train for real (3 training seeds x 75,000 steps, see SESSION_LOG.md's newest entry for the full comparison numbers). |

---

## Last verified

- **Date:** 2026-08-31 (Living System trained-DQN-driver investigation session)
- **Commit:** `f84fcd1` ("docs: formalize post-review roadmap...") — the commit this session started from (`main`/`dev21` confirmed identical, `git rev-parse main dev21` both `f84fcd1`).
- **`pytest` pass count:** 695 passed, 1 xfailed — run fresh this session (688 prior + 7 new in `tests/test_render_living_system_demo.py`).
- **Branch:** `main`/`dev21` in sync at session start. This was a small, optional side session (not the top-priority LSTM work) — scope held to `dashboard/render_living_system_demo.py`, `dashboard/app.py`, and their tests; zero changes to `forecaster/`, `env/forecast_provider.py`, `env/masking.py`, `env/environment.py`, or any `configs/scenarios/` file (verified via `git diff --stat`). Not pushed to origin this session, per instruction.

---

## Prior "Last verified" (superseded, kept for history)

- **Date:** 2026-08-31 (root README session)
- **Commit:** `aefc7a1` ("Delete split.md") — the commit this session started from (`main`); see SESSION_LOG.md for this session's own commit and the final ff-merged `dev21` hash.
- **`pytest` pass count:** 649 passed, 1 xfailed — run fresh this session, matches PROGRESS.md's own last-recorded figure exactly.
- **Branch:** `main` (aefc7a1) was one commit ahead of `dev21` (e64ef68) at session start — a clean fast-forward (`dev21..main` = the single "Delete split.md" commit, `main..dev21` empty), not a divergence; this session worked on `main` and ff-merged into `dev21` at the end, per the standing workflow — see SESSION_LOG.md for the final shared hash. Not pushed to origin this session, per instruction.

- **Date:** 2026-08-25 (adversarial trace generator session)
- **Commit:** `36f3b20` ("docs: explain p99_latency saturation as discrete-cost-model artifact, recommend total_reward/below_floor_rate instead -- 2026-08-25") — the commit this session started from; see SESSION_LOG.md for this session's own commit
- **`pytest` pass count:** 554 passed, 1 xfailed (531 prior + 23 new, all in the new `tests/test_trace_generator.py`)
- **Branch:** Confirmed `main`/`dev21` in sync at session start (`git rev-parse main dev21` both `36f3b2092be0a9f773a623972b23abcee4729f93`). This session's commit (real code: `attack/trace_generator.py` (new) + `tests/test_trace_generator.py` (new) — nothing else touched, confirmed via `git status --porcelain` and an empty `git diff --stat` on every protected env/agent file) was ff-merged into `dev21` at the end — see SESSION_LOG.md for the final shared hash. Not pushed to origin this session, per instruction.
