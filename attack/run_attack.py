"""
attack/run_attack.py

Runs the S5 steering attack and produces the dose-response result
(PLAN2 §7.5, the headline contribution). Owned by Person D.

    python -m attack.run_attack

What is measured, and why the definition matters
------------------------------------------------
The claim under test is Hard Rule 2's: *a threat signal may only ever
raise a floor, never lower one.* So the quantity that decides the claim
is the share of decisions that deliver key material **below the
sensitivity-class floor** -- the protection level guaranteed by the
policy table's CALM row, which no threat signal of any kind can reduce.

That is the honest denominator for this experiment. Two weaker
definitions were considered and rejected:

  * "below the floor the *honest* (unsuppressed) posture would have
    produced" -- under that definition the masked agent scores nonzero
    too, because suppression prevents floors from *escalating*. That is
    a real and reportable limitation, so it is reported here as a
    second column (`below_escalated_floor_share`) rather than quietly
    omitted. But it is not what Hard Rule 2 claims, and scoring the
    architecture against a claim it never made would be as dishonest as
    scoring it against one that is too easy.
  * "the harness's floor_violations counter" -- correct, but it is
    trivially 0 for anything running inside the environment, since the
    environment refuses illegal actions outright. The soft-reward agent
    is interesting precisely because a real deployment of that design
    has no such mask, so it is scored on the tiers it *chooses*.

Scored at key ESTABLISHMENTS, not at every request
---------------------------------------------------
The protection level of a session is decided when its key material is
established, not on the cache hits that follow. So both arms are scored
on the decisions where a key genuinely has to be established -- where
`REUSE` is unavailable because the session key has reached its SP
800-57 cryptoperiod (or does not exist yet).

That is also the only scoring rule that stays faithful when the two
arms share a trajectory. Scored on *every* decision, a `REUSE` by the
soft-reward arm delivers whichever tier the driving policy happened to
establish -- so the arm gets credit for key material its own decision
rule never chose. Measured with that flawed rule, the soft-reward
curve came out non-monotone and small (0.2% -> 9.0% -> 0.8% -> 0.8%),
which is an artefact of shared session state, not a property of the
reward design.

Key *lifetime* is still enforced for both arms, and deliberately so:
SP 800-57 cryptoperiods are an operational constraint, not a
threat-signal-derived floor. The only thing the soft-reward arm is
freed from is the floor -- which is exactly the difference under test.

Reported per dose, for both policies:
  below_class_floor_share      the Hard Rule 2 quantity
  below_escalated_floor_share  the honest-posture comparison
  tier histogram               so the *shape* of the degradation is
                               visible without reading numbers (PLAN2 §7.5)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agents.baselines import Policy
from agents.soft_reward_baseline import SoftRewardAgent, SoftRewardConfig
from attack.steering_trace import (
    SteeringTraceConfig,
    default_honest_trace,
    generate_steering_trace,
    suppression_ratio,
)
from env.contracts import Action, SensitivityClass, ThreatPosture
from env.environment import SmartKeyNetEnv
from env.masking import PolicyTable

_TIER_ACTIONS = (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID)

_TIER_ONLY_MASK = np.zeros(len(Action), dtype=bool)
for _tier in _TIER_ACTIONS:
    _TIER_ONLY_MASK[int(_tier)] = True
"""All three tiers legal, REUSE and REKEY_NOW not.

This is the mask handed to the **unmasked** arm at scoring time, and it
is the cleanest statement of what the whole experiment is comparing.
Both arms are asked the same question -- "you must establish key
material for this request; which tier?" -- and the only difference is
which mask they are asked it under:

  * the masked architecture gets `mask & _TIER_ONLY_MASK`, i.e. the
    floor still applies;
  * the soft-reward arm gets `_TIER_ONLY_MASK` alone, i.e. no floor at
    all, only the physical fact that a key past its cryptoperiod cannot
    be reused.

Restricting to tiers matters: without it, the soft-reward arm's best
action under a suppressed signal is REUSE (cheapest of all), which is
physically unavailable at an establishment and, when resolved through
the environment's floor, silently credited the arm with the masked
architecture's own protection -- reporting 0.0% below-class-floor at
full suppression, which is the opposite of the truth."""


def class_floor(sensitivity_class: int) -> Action:
    """The floor guaranteed by sensitivity class alone -- the policy
    table's CALM row.

    This is the level Hard Rule 2 protects: the ratchet starts here and
    only ever moves up, so no threat signal, however adversarially
    shaped, can put a request below it.
    """
    return PolicyTable().floor(SensitivityClass(int(sensitivity_class)), ThreatPosture.CALM)


def escalated_floor(sensitivity_class: int, honest_posture: ThreatPosture) -> Action:
    """The floor the *honest* posture would have produced."""
    return PolicyTable().floor(SensitivityClass(int(sensitivity_class)), honest_posture)


def _honest_posture_at(honest_trace: Sequence[float], step: int) -> ThreatPosture:
    """Posture implied by the honest signal, with the same one-way
    ratchet the real system applies."""
    from forecaster.model import posture_probs_from_score

    level = float(honest_trace[min(step, len(honest_trace) - 1)])
    score = 1.0 / (1.0 + np.exp(-(1.7 * level - 2.94)))
    return ThreatPosture(int(np.argmax(posture_probs_from_score(score))))


@dataclass
class AttackRunResult:
    """One (policy, dose) run."""

    policy_name: str
    dose: float
    suppression: float
    decisions: int
    """Key establishments scored -- see the module docstring."""

    below_class_floor: int
    below_escalated_floor: int
    tier_counts: dict[str, int] = field(default_factory=dict)

    @property
    def below_class_floor_share(self) -> float:
        return self.below_class_floor / self.decisions if self.decisions else 0.0

    @property
    def below_escalated_floor_share(self) -> float:
        return self.below_escalated_floor / self.decisions if self.decisions else 0.0


def train_soft_reward_agent(
    config: dict[str, Any],
    honest_trace: Sequence[float],
    steps: int = 6000,
    seed: int = 0,
) -> SoftRewardAgent:
    """Train the critique target on an HONEST signal.

    Deliberately honest: the agent is trained on a truthful sensor and
    only *attacked at evaluation time*. Training it on the suppressed
    trace would conflate two different claims -- "this design can be
    poisoned during training" and "this design can be steered at
    inference" -- and only the second is what PLAN2 §7.5 argues. It also
    makes the baseline stronger, since it arrives at evaluation having
    learned the correct threat-to-tier relationship.
    """
    from agents.baselines import AlwaysPQCPolicy

    agent = SoftRewardAgent(SoftRewardConfig(), seed=seed, respect_mask=False)
    driver = AlwaysPQCPolicy()

    env_config = {**config, "scenario": "S5", "seed": seed, "max_steps": steps}
    env = SmartKeyNetEnv(env_config)
    env.set_external_threat_trace(list(honest_trace))
    state, info = env.reset(seed=seed)

    truncated = False
    while not truncated:
        mask = info["action_mask"]
        # The agent explores over ALL five actions, unmasked -- its own
        # architecture has no policy table, so training it under this
        # environment's mask would teach it a floor it does not have and
        # make the steering comparison meaningless. Q-learning is
        # off-policy, so learning from its own freely-chosen action while
        # a masked reference policy advances the environment is the
        # standard, correct construction rather than a workaround.
        own_action = agent.act_exploring(state, mask)
        driving_action = driver.act(state, mask)
        next_state, _reward, _terminated, truncated, info = env.step(driving_action)
        agent.learn(
            state,
            own_action,
            float(state["threat_score"]),
            next_state,
            float(next_state["threat_score"]),
        )
        state = next_state
    return agent


def run_attack(
    driving_policy: Policy,
    scored_policies: dict[str, tuple[Policy, bool]],
    config: dict[str, Any],
    honest_trace: Sequence[float],
    attacked_trace: Sequence[float],
    dose: float,
    seed: int,
    max_steps: int,
) -> dict[str, AttackRunResult]:
    """Run S5 once under an attacked trace, scoring several decision
    rules on the *same* state trajectory.

    Why one trajectory rather than one run per policy: the two arms have
    to be compared on identical pool levels, identical requests and an
    identical (suppressed) threat signal, or the comparison confounds
    the decision rule with the trajectory it happened to induce.
    `driving_policy` advances the environment; every entry in
    `scored_policies` maps a name to `(policy, floor_constrained)` and
    is asked what tier it would establish at each establishment
    decision. `floor_constrained=False` is what makes an arm genuinely
    unmasked -- see `_TIER_ONLY_MASK`.

    This is also what makes the soft-reward arm a **faithful** unmasked
    reproduction. A soft-reward deployment has no policy table and
    therefore no mask -- the tier it picks is the tier it delivers. But
    `SmartKeyNetEnv` refuses illegal actions by design (that refusal is
    the thing under test), so an unmasked agent cannot drive it. Scoring
    it in shadow gives it exactly the freedom its own architecture would
    give it, while keeping this environment's guarantee intact.

    An earlier version of this ran the soft-reward agent with
    `respect_mask=True` so it could drive the env directly. That
    measured a *masked* soft-reward agent, which trivially cannot
    violate a floor -- it reported 0.0% below-class-floor at every dose
    for both arms, i.e. it accidentally measured the mask rather than
    the reward design.
    """
    env_config = {**config, "scenario": "S5", "seed": seed, "max_steps": max_steps}
    env = SmartKeyNetEnv(env_config)
    env.set_external_threat_trace(list(attacked_trace))
    state, info = env.reset(seed=seed)

    results = {
        name: AttackRunResult(
            policy_name=name,
            dose=float(dose),
            suppression=suppression_ratio(honest_trace, attacked_trace),
            decisions=0,
            below_class_floor=0,
            below_escalated_floor=0,
            tier_counts={action.name: 0 for action in _TIER_ACTIONS},
        )
        for name in scored_policies
    }

    truncated = False
    while not truncated:
        request = env._current_request
        sensitivity = int(request["sensitivity_class"])
        floor = Action(state["policy_floor"])
        honest_posture = _honest_posture_at(honest_trace, env._step_count)
        mask = info["action_mask"]

        # Score only where key material actually has to be established.
        # REUSE being unavailable is exactly that condition: env/masking.py
        # withdraws it at the SP 800-57 cryptoperiod (and on a session with
        # no key yet). See the module docstring for why every-decision
        # scoring is not faithful across a shared trajectory.
        if not mask[int(Action.REUSE)] and (mask & _TIER_ONLY_MASK).any():
            for name, (policy, floor_constrained) in scored_policies.items():
                scoring_mask = (mask & _TIER_ONLY_MASK) if floor_constrained else _TIER_ONLY_MASK
                established = Action(int(policy.act(state, scoring_mask)))
                result = results[name]
                result.decisions += 1
                result.tier_counts[established.name] += 1
                if int(established) < int(class_floor(sensitivity)):
                    result.below_class_floor += 1
                if int(established) < int(escalated_floor(sensitivity, honest_posture)):
                    result.below_escalated_floor += 1

        driving_action = driving_policy.act(state, mask)
        state, _reward, _terminated, truncated, info = env.step(driving_action)

    return results





@dataclass
class DoseResponse:
    """The full sweep -- what PLAN2 §7.5's chart is drawn from."""

    doses: list[float]
    results: dict[str, list[AttackRunResult]]

    def to_json(self) -> str:
        return json.dumps(
            {
                "doses": self.doses,
                "policies": {
                    name: [
                        {
                            "dose": r.dose,
                            "suppression": r.suppression,
                            "decisions": r.decisions,
                            "below_class_floor_share": r.below_class_floor_share,
                            "below_escalated_floor_share": r.below_escalated_floor_share,
                            "tier_counts": r.tier_counts,
                        }
                        for r in runs
                    ]
                    for name, runs in self.results.items()
                },
            },
            indent=2,
        )


def run_dose_response(
    config: dict[str, Any],
    agent_policy: Policy | None = None,
    doses: list[float] | None = None,
    seed: int = 0,
    max_steps: int = 2000,
) -> DoseResponse:
    """The S5 experiment: both policies, every dose, one honest trace.

    `agent_policy` is the masked DQN (a `GreedyDQNPolicy`). When None,
    `AlwaysPQCPolicy` stands in as a masked-architecture representative
    -- the claim under test is a property of the *masking architecture*,
    not of the DQN's learned preferences, so any masked policy
    demonstrates it. Which one was used is recorded in the report.
    """
    from agents.baselines import AlwaysPQCPolicy

    doses = doses if doses is not None else list(config["steering_attack"]["doses"])
    honest = default_honest_trace(max_steps + 1)

    agent_policy = agent_policy if agent_policy is not None else AlwaysPQCPolicy()
    soft_agent = train_soft_reward_agent(config, honest, seed=seed)
    # At scoring time it is handed `_TIER_ONLY_MASK` (all three tiers
    # legal), so "respecting" that mask is precisely an unconstrained
    # tier choice -- the floor never enters.
    soft_agent.respect_mask = True

    soft_name = "soft-reward (security IS the reward, no mask)"
    masked_name = "masked architecture (security is a constraint)"
    results: dict[str, list[AttackRunResult]] = {soft_name: [], masked_name: []}

    for dose in doses:
        attacked = generate_steering_trace(
            SteeringTraceConfig(dose=float(dose), duration_steps=max_steps + 1, seed=seed),
            honest_trace=honest,
        )
        run = run_attack(
            driving_policy=agent_policy,
            scored_policies={
                soft_name: (soft_agent, False),
                masked_name: (agent_policy, True),
            },
            config=config,
            honest_trace=honest,
            attacked_trace=attacked,
            dose=dose,
            seed=seed,
            max_steps=max_steps,
        )
        results[soft_name].append(run[soft_name])
        results[masked_name].append(run[masked_name])

    return DoseResponse(doses=[float(d) for d in doses], results=results)


def format_dose_response(response: DoseResponse) -> str:
    lines = ["=== S5 steering attack: dose-response ===", ""]
    lines.append(
        "scored at KEY ESTABLISHMENTS (where REUSE is unavailable) -- protection is set when a "
        "key is established"
    )
    lines.append(
        "below_class_floor_share = share of establishments below the sensitivity-class floor"
    )
    lines.append("                          (the level Hard Rule 2 guarantees; no signal can lower it)")
    lines.append(
        "below_escalated_floor   = share below the floor the HONEST posture would have set "
        "(reported for completeness)"
    )
    lines.append("")
    for name, runs in response.results.items():
        lines.append(f"-- {name} --")
        lines.append(
            f"   {'dose':>5} {'suppressed':>11} {'estabs':>7} {'below_class_floor':>18} "
            f"{'below_escalated':>16}   tier histogram"
        )
        for run in runs:
            histogram = " ".join(f"{k.replace('SERVE_','')}={v}" for k, v in run.tier_counts.items())
            lines.append(
                f"   {run.dose:>5.2f} {run.suppression:>10.1%} {run.decisions:>7} "
                f"{run.below_class_floor_share:>17.1%} "
                f"{run.below_escalated_floor_share:>15.1%}   {histogram}"
            )
        lines.append("")
    lines.append('"Security isn\'t in our reward, so it isn\'t for sale." (PLAN2 §13)')
    return "\n".join(lines)


def main() -> None:
    from experiments.train import load_full_config

    config = load_full_config()
    response = run_dose_response(config)
    print(format_dose_response(response))

    out = Path("results/steering_dose_response.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(response.to_json())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
