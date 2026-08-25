"""
attack/trace_generator.py

Adversarial input-shaping generator for the steering attack (PLAN.md
Section 5 scenario S5; PLAN2.md Section 7.5 Panel 5 -- the headline
contribution; paper draft equation 7). Owned by Person D (split.md
Section 1).

Implements equation 7 exactly:

    x~t = (1 - alpha) * xt + alpha * g(xt)

where `xt` is the TRUE input window feeding the threat forecaster,
`alpha` in [0, 1] parameterizes attack strength (0 = no attack, 1 =
full replacement by g's output), and `g` maps a window toward the
"benign region of the learned feature space" (paper draft). At
alpha=0, x~t == xt exactly. At alpha=1, x~t == g(xt) exactly.

Interception point: this module produces a shaped INPUT WINDOW only.
It never touches `env/forecast_provider.py`, `env/masking.py`, or
`env/environment.py` -- the shaped window is meant to be substituted
for the true window *before* it reaches `ForecastProvider.update()`,
by whoever orchestrates an episode (a future dose-response sweep
session). The forecaster, masking layer, and environment run
completely unaware an attack occurred, the same way they're unaware
which tenant graph or pcap replay mode produced a request -- this
keeps Hard Rule 3's "the agent must not need to know" spirit intact by
analogy, one layer up the stack.

Design decision -- what g(xt) computes: the real learned feature space
(the LSTM forecaster, PLAN.md Addition A) does not exist yet -- only
`env/forecast_provider.py`'s placeholder EWMA-based
`MovingAverageForecaster`, driven by a `[qber, load]` window built by
`env/environment.py::_threat_features_placeholder`, does. The faithful
interpretation for THIS current placeholder forecaster: g doesn't need
to be learned, it needs to be a well-defined, deterministic function
that, when the CURRENT real forecaster processes its output, reads as
low threat. Concretely: `g(xt)` returns an all-zero window of the same
length as `xt` -- zero qber (no bit-error / no eavesdropping signal)
and zero load (no backlog) are the two lowest legitimate values either
current placeholder feature can take, i.e. the actual floor of this
feature space's benign region, not an arbitrary choice.

This generator targets the CURRENT placeholder forecaster specifically.
If/when the real LSTM forecaster (Addition A) is built, `g` (or an
analogous function against the new learned feature space) will need
revisiting -- expected, not a flaw in this session's work.
"""

from __future__ import annotations

from typing import Sequence


def g(window: Sequence[float]) -> list[float]:
    """Benign-region target for the current placeholder forecaster.

    Deterministic, not learned: an all-zero window of the same length
    as `window` -- the lowest legitimate value both current placeholder
    features (`qber`, `load`, both real, non-negative quantities) can
    take, per this module's docstring design decision.
    """
    return [0.0] * len(window)


def generate_adversarial_window(true_window: Sequence[float], alpha: float) -> list[float]:
    """Shape `true_window` toward `g(true_window)` by strength `alpha`
    (paper draft equation 7): `x~t = (1 - alpha) * xt + alpha * g(xt)`.

    `alpha` is an explicit, independent parameter (not read from any
    config) so a future dose-response sweep can call this repeatedly
    against the same `true_window` across many different alpha values
    in one run.

    `alpha=0.0` returns a window exactly (bit-for-bit) equal to
    `true_window`; `alpha=1.0` returns a window exactly equal to
    `g(true_window)`. Never mutates `true_window` -- always returns a
    new list, safe to call repeatedly against the same base window.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    benign = g(true_window)
    return [(1.0 - alpha) * float(x) + alpha * float(b) for x, b in zip(true_window, benign)]
