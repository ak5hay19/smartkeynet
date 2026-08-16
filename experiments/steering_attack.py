"""
experiments/steering_attack.py

Scenario S5, the headline experiment (PLAN.md §5; §6 Demo Beat 3).

Runs the same adversarially suppressed threat trace into two agents that
differ in exactly one respect -- where security lives:

  * `SoftRewardAgent` (agents/soft_reward_baseline.py) keeps security in
    its **reward**, reproducing the published designs PLAN.md §2
    critiques.
  * The masked agent keeps security in a **constraint**: the policy
    table sets a floor, `env/masking.py` removes everything below it
    from the action set, and the reward never sees a threat term at all.

Everything else is held fixed: same environment, same seeds, same
arrival stream, same SKR trace, same action mask. The attack enters at
one point only -- the threat features the forecaster observes -- which
is the access a real adversary would plausibly have.

Measured on the **served-tier distribution**, per PLAN.md §6 Beat 3
("the served-tier histogram slides downward"). Both agents act through
the same mask, so neither can literally serve below a floor here; the
question is how far down the ladder each slides *within* what it is
allowed, as the reported threat is quietened.

Run it:

    .venv/bin/python -m experiments.steering_attack
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agents.soft_reward_baseline import (
    GreedySoftRewardPolicy,
    SoftRewardAgent,
    SoftRewardConfig,
    soft_reward,
)
from attack.steering_trace import SuppressionTrace, detectability_score, dose_response_traces
from env.contracts import Action, KeyType
from env.environment import SmartKeyNetEnv
from experiments.train import GreedyDQNPolicy, load_full_config, train

_TIER_OF_KEY_TYPE: dict[KeyType, int] = {
    KeyType.CLASSICAL: 0,
    KeyType.PQC: 1,
    KeyType.HYBRID: 2,
}

_ATTACK_SCENARIO = "S2"
"""The attack is run on top of S2 (elevated threat posture).

S2 is the right substrate because it is the scenario where the threat
signal *matters*: floors ratchet upward in response to it. Running the
attack on S1 -- where the posture is CALM throughout and no floor ever
moves -- would prove nothing, because there would be no protection for
the adversary to try to suppress. The point is to attack a system that
is actively defending itself.
"""


@dataclass
class AttackOutcome:
    """What one agent did under one dose."""

    agent: str
    dose: float
    served_tier_histogram: list[float]
    mean_served_tier: float
    mean_tier_in_window: float
    """Mean served tier measured ONLY inside the attack window. The
    whole-episode mean dilutes the effect with pre-attack steps, so this
    is the number the headline should quote."""

    mean_headroom_in_window: float
    """Mean (established tier - floor) inside the attack window, over
    decisions that actually established key material. THIS IS THE
    HEADLINE NUMBER: it isolates what the agent *chose* from what its
    floor *required*, so it separates the two architectures cleanly.
    Suppression should collapse it for the soft-reward agent (security
    is now worth fewer reward points) and leave it untouched for the
    masked agent (whose reward never saw the threat signal at all)."""

    n_establishing_in_window: int
    mean_floor: float
    max_floor: int
    floor_violations: int
    posture_reversals: int
    """How many times the ratcheted posture moved DOWN. Zero is the
    structural guarantee; anything else is a Hard Rule 2 bug."""

    n_decisions: int

    def summary_row(self) -> str:
        histogram = " ".join(f"{value:.3f}" for value in self.served_tier_histogram)
        return (
            f"{self.agent:14s} dose={self.dose:4.2f}  headroom={self.mean_headroom_in_window:+6.3f}  "
            f"tier={self.mean_tier_in_window:5.3f}  "
            f"[{histogram}]  mean_floor={self.mean_floor:5.3f}  "
            f"viol={self.floor_violations}  reversals={self.posture_reversals}"
        )


def run_attack_episode(
    policy: Any,
    trace: SuppressionTrace,
    config: dict[str, Any],
    n_steps: int,
    seed: int,
    agent_name: str,
    learner: Any | None = None,
) -> AttackOutcome:
    """Drive one agent through one attacked episode.

    `learner`, when supplied, gets a `learn(...)` call per step so the
    soft-reward agent keeps adapting to the manipulated signal *during*
    the attack. That is the realistic threat model: the published
    designs learn online, and it is the online adaptation that makes
    them steerable rather than merely mis-served once.
    """
    env_config = {
        **config,
        "scenario": _ATTACK_SCENARIO,
        "seed": seed,
        "steering_trace": trace,
        "max_steps": n_steps,
    }
    env = SmartKeyNetEnv(env_config)
    state, info = env.reset(seed=seed)

    served_tiers: list[int] = []
    floors: list[int] = []
    in_window_tiers: list[int] = []
    # Voluntary over-provisioning: (established tier - floor), counted
    # ONLY on decisions where the agent actually established key
    # material. This is the agent's revealed preference. Measuring the
    # tier of whatever key happens to be installed instead conflates
    # choice with history, because REUSE carries an old key's tier
    # forward -- an earlier version of this experiment reported the
    # soft-reward agent's tier going *up* under suppression for exactly
    # that reason: suppression made it reuse more, and what it was
    # reusing was high-tier keys bought before the attack began.
    in_window_headroom: list[int] = []
    postures: list[int] = []
    floor_violations = 0
    _ESTABLISHING = {Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID, Action.REKEY_NOW}

    for _ in range(n_steps):
        mask = info["action_mask"]
        floor = int(state["policy_floor"])
        step_index = env._step_count
        postures.append(int(env._policy_table._ratcheted_posture))
        action = policy.act(state, mask)

        # REUSE and REKEY_NOW inherit the session's tier, so what tier
        # actually got served has to be read from session state after
        # the step -- it cannot be inferred from the action alone.
        tenant_service = (env._current_request["tenant"], env._current_request["service"])
        next_state, reward, terminated, truncated, next_info = env.step(action)

        session = env._sessions.get(tenant_service)
        if session is not None and session.key_type is not None:
            tier = _TIER_OF_KEY_TYPE[session.key_type]
            served_tiers.append(tier)
            floors.append(floor)
            if tier < floor:
                floor_violations += 1
            if trace.start_step <= step_index < trace.end_step:
                in_window_tiers.append(tier)
                if action in _ESTABLISHING:
                    in_window_headroom.append(tier - floor)

        if learner is not None:
            learner.learn(state, action, next_state, next_info["action_mask"])

        state, info = next_state, next_info
        if truncated:
            break

    histogram = [0.0, 0.0, 0.0]
    for tier in served_tiers:
        histogram[tier] += 1.0
    total = max(1, len(served_tiers))

    # Did the ratchet ever move DOWN? For the masked agent this must be
    # zero by construction -- PolicyTable has no downward path -- and it
    # is the machine-checked form of "protections can only ratchet up".
    posture_reversals = sum(
        1 for earlier, later in zip(postures, postures[1:]) if later < earlier
    )

    return AttackOutcome(
        agent=agent_name,
        dose=trace.dose,
        served_tier_histogram=[count / total for count in histogram],
        mean_served_tier=float(np.mean(served_tiers)) if served_tiers else 0.0,
        mean_tier_in_window=float(np.mean(in_window_tiers)) if in_window_tiers else 0.0,
        mean_headroom_in_window=float(np.mean(in_window_headroom)) if in_window_headroom else 0.0,
        n_establishing_in_window=len(in_window_headroom),
        mean_floor=float(np.mean(floors)) if floors else 0.0,
        max_floor=int(np.max(floors)) if floors else 0,
        floor_violations=floor_violations,
        posture_reversals=posture_reversals,
        n_decisions=len(served_tiers),
    )


def train_soft_reward_agent(
    config: dict[str, Any], n_steps: int, seed: int
) -> SoftRewardAgent:
    """Pre-train the victim on the *unattacked* signal.

    It has to have learned a sensible threat-responsive policy before
    the attack starts, or the experiment would just be showing that an
    untrained agent behaves randomly. Trained on a dose-0 trace, which
    is the same code path as the attacked runs with the attack turned
    off.
    """
    agent = SoftRewardAgent(config=SoftRewardConfig(), seed=seed)
    control = SuppressionTrace(start_step=0, end_step=n_steps, dose=0.0, ramp_steps=0)

    env_config = {
        **config,
        "scenario": _ATTACK_SCENARIO,
        "seed": seed,
        "steering_trace": control,
        "max_steps": n_steps,
    }
    env = SmartKeyNetEnv(env_config)
    state, info = env.reset(seed=seed)

    for _ in range(n_steps):
        mask = info["action_mask"]
        action = agent.act(state, mask)
        next_state, reward, terminated, truncated, next_info = env.step(action)
        agent.learn(state, action, next_state, next_info["action_mask"])
        state, info = next_state, next_info
        if truncated:
            break

    return agent


def run_steering_attack(
    doses: list[float] | None = None,
    n_steps: int = 3000,
    seeds: list[int] | None = None,
    soft_train_steps: int = 20_000,
    dqn_train_steps: int = 20_000,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full dose-response sweep for both agents."""
    config = config if config is not None else load_full_config()
    config = {**config, "scenario_steps": n_steps}
    doses = doses if doses is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
    seeds = seeds if seeds is not None else [0, 1, 2]

    # The attack starts AFTER S2's first threat window has already
    # fired, which is the only placement that tests the actual claim.
    #
    # S2 elevates the posture over [500, 1100) and again over
    # [1400, 2100). Starting the attack before step 500 would only show
    # that suppressing a signal prevents a floor from ever rising --
    # true, but trivial, and not what the architecture claims. Starting
    # it at 1200 means the ratchet is already engaged at ELEVATED when
    # the adversary arrives, so the question becomes the sharp one:
    # **can suppression walk an established protection back down?**
    # For the masked agent the answer is structurally no (the ratchet
    # has no downward path); for the soft-reward agent the answer is
    # whatever its reward gradient says.
    attack_start = int(config.get("steering_attack", {}).get("start_step", 1200))
    attack_end = n_steps
    if attack_start >= n_steps:
        raise ValueError(
            f"attack window [{attack_start}, {n_steps}) is empty -- run with more steps "
            "(the attack must start after S2's first threat window at step 500-1100)"
        )
    traces = dose_response_traces(attack_start, attack_end, doses)

    print(f"{'=' * 78}\nS5 STEERING ATTACK\n{'=' * 78}")
    print(f"  substrate scenario : {_ATTACK_SCENARIO} (floors actively ratchet with the threat signal)")
    print(f"  attack window      : steps [{attack_start}, {attack_end})")
    print(f"  doses              : {doses}")
    print(f"  seeds              : {seeds}\n")

    # --- the two agents, both pre-trained on the unattacked signal ---
    print("  pre-training the soft-reward victim on the clean signal ...", flush=True)
    soft_agent = train_soft_reward_agent(config, soft_train_steps, seed=0)

    print("  training the masked agent on the clean signal ...", flush=True)
    masked_agent, _record = train(
        full_config=config,
        training_overrides={
            "seed": 0,
            "total_steps": dqn_train_steps,
            "eval_every": max(1, dqn_train_steps),
            "eval_max_steps": 500,
            "checkpoint_path": "checkpoints/steering_masked.pt",
        },
        scenario=_ATTACK_SCENARIO,
    )

    agents = {
        "soft_reward": (GreedySoftRewardPolicy(soft_agent), soft_agent),
        "masked": (GreedyDQNPolicy(masked_agent), None),
    }

    results: dict[str, list[dict[str, Any]]] = {name: [] for name in agents}
    print(f"\n  {'agent':14s} {'':9s} {'mean_tier':>10s}  served-tier histogram   floors")
    for trace in traces:
        for name, (policy, learner) in agents.items():
            per_seed = [
                run_attack_episode(
                    policy, trace, config, n_steps, seed, name, learner=learner
                )
                for seed in seeds
            ]
            averaged = AttackOutcome(
                agent=name,
                dose=trace.dose,
                served_tier_histogram=[
                    float(np.mean([o.served_tier_histogram[t] for o in per_seed])) for t in range(3)
                ],
                mean_served_tier=float(np.mean([o.mean_served_tier for o in per_seed])),
                mean_tier_in_window=float(np.mean([o.mean_tier_in_window for o in per_seed])),
                mean_headroom_in_window=float(np.mean([o.mean_headroom_in_window for o in per_seed])),
                n_establishing_in_window=int(np.sum([o.n_establishing_in_window for o in per_seed])),
                mean_floor=float(np.mean([o.mean_floor for o in per_seed])),
                max_floor=int(np.max([o.max_floor for o in per_seed])),
                floor_violations=int(np.sum([o.floor_violations for o in per_seed])),
                posture_reversals=int(np.sum([o.posture_reversals for o in per_seed])),
                n_decisions=int(np.sum([o.n_decisions for o in per_seed])),
            )
            results[name].append(asdict(averaged))
            print(f"  {averaged.summary_row()}")
        print()

    # --- the headline numbers ---
    report: dict[str, Any] = {
        "scenario": _ATTACK_SCENARIO,
        "attack_window": [attack_start, attack_end],
        "doses": doses,
        "seeds": seeds,
        "n_steps": n_steps,
        "results": results,
        "detectability": {
            str(trace.dose): detectability_score(trace, n_steps) for trace in traces
        },
    }

    print(f"{'=' * 78}\nHEADLINE\n{'=' * 78}")
    for name in agents:
        clean = results[name][0]["mean_headroom_in_window"]
        attacked = results[name][-1]["mean_headroom_in_window"]
        drop = clean - attacked
        report[f"{name}_tier_drop"] = drop
        direction = "DOWN" if drop > 0 else ("up" if drop < 0 else "unchanged")
        print(
            f"  {name:14s} voluntary over-provisioning {clean:+.3f} -> {attacked:+.3f} "
            f"({direction} {abs(drop):.3f}) at full suppression"
        )

    # --- the mechanism, read directly off the two architectures ---
    analytic_tiers = reward_optimal_tier_by_threat(SoftRewardConfig())
    learned_tiers = preferred_tier_by_threat(soft_agent)
    masked_floors = masked_floor_by_threat(config)
    report["soft_reward_optimal_tier_by_threat"] = analytic_tiers
    report["soft_reward_learned_tier_by_threat"] = learned_tiers
    report["masked_floor_by_threat"] = masked_floors

    print("\n  THE MECHANISM -- tier vs reported threat (bin 0 = fully suppressed):")
    print(f"    {'threat':>16s} " + " ".join(f"{b/10:4.1f}" for b in range(10)))
    print(f"    {'soft (analytic)':>16s} " + " ".join(f"{t:4d}" for t in analytic_tiers))
    print(f"    {'soft (learned)':>16s} " + " ".join(f"{t:4d}" for t in learned_tiers))
    print(f"    {'masked (floor)':>16s} " + " ".join(f"{f:4d}" for f in masked_floors))

    slide = analytic_tiers[-1] - analytic_tiers[0]
    report["soft_reward_tier_slide"] = slide
    print(f"\n  The soft reward's optimal tier drops {slide} tiers as the reported threat is")
    print("  driven to zero -- analytically, straight out of the reward formula, so it is a")
    print("  property of the DESIGN and not of any particular training run. The masked")
    print("  architecture has no such table: its reward contains no threat term, so there is")
    print("  no gradient for an adversary to pull on, and its floors move only upward.")

    total_violations = sum(row["floor_violations"] for rows in results.values() for row in rows)
    total_reversals = sum(row["posture_reversals"] for rows in results.values() for row in rows)
    report["total_floor_violations"] = total_violations
    report["total_posture_reversals"] = total_reversals
    print(f"\n  floor violations across every agent, dose and seed : {total_violations}")
    print(f"  posture ratchet reversals (must be 0)              : {total_reversals}")
    print("  Protections can only ratchet upward: PolicyTable has no downward path, so")
    print("  suppression cannot walk back a floor that has already been raised.")

    return report


def reward_optimal_tier_by_threat(config: SoftRewardConfig) -> list[int]:
    """The tier the soft reward makes optimal at each threat level,
    computed analytically from the reward formula itself.

    THIS IS THE DECISIVE EVIDENCE, and it needs no training run at all.
    The critiqued reward is

        r(tier) = w_security * security(tier) * threat - w_cost * cost(tier)

    Every term but `threat` is fixed, so as `threat` falls the security
    term shrinks toward zero and the cost term -- which *increases* with
    tier -- dominates. The argmax therefore walks monotonically down the
    tier ladder. That is not an artifact of a particular training run,
    a seed, or an exploration schedule: it is a property of the reward
    function, and any agent that maximises it inherits the
    vulnerability.

    The masked architecture has no counterpart to this table, because
    its reward contains no threat term at all -- there is no gradient
    for an adversary to pull on. Its floors are a separate, monotone
    non-decreasing function of posture (`masked_floor_by_threat`).
    """
    tier_actions = (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID)
    tiers: list[int] = []
    for threat_bin in range(config.threat_bins):
        threat = threat_bin / config.threat_bins
        best = max(
            tier_actions,
            key=lambda a: soft_reward({}, a, threat, config.w_security, config.w_cost),  # type: ignore[arg-type]
        )
        tiers.append(_TIER_OF_KEY_TYPE[_KEY_TYPE_FOR_ACTION[best]])
    return tiers


def preferred_tier_by_threat(agent: SoftRewardAgent) -> list[int]:
    """The soft-reward agent's preferred tier at each threat level, read
    straight out of its Q-table with every action legal.

    THIS IS THE MECHANISM, stated as directly as it can be stated. The
    agent is tabular precisely so this readout is possible: one row per
    threat bin, and the greedy tier in each row is what the agent would
    serve if the reported threat sat at that level.

    The episode-level metrics above are realistic but confounded --
    `REUSE` carries a key bought at an earlier, higher threat level
    forward, and the floors themselves legitimately move -- so they
    blur the very effect they are meant to show. This readout has no
    such confound: it is the policy itself, as a function of the one
    signal the adversary controls.
    """
    # Restricted to the three tier-establishing actions. REUSE is
    # cheapest overall and wins the unrestricted argmax at every threat
    # level, which tells us nothing about tier preference -- the
    # question here is specifically "when this agent does establish key
    # material, which tier does it pick?"
    tier_actions = (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID)
    tiers: list[int] = []
    for threat_bin in range(agent.config.threat_bins):
        q_row = agent.q_table[threat_bin]
        best = max(tier_actions, key=lambda a: q_row[int(a)])
        tiers.append(_TIER_OF_KEY_TYPE[_KEY_TYPE_FOR_ACTION[best]])
    return tiers


_KEY_TYPE_FOR_ACTION = {
    Action.SERVE_CLASSICAL: KeyType.CLASSICAL,
    Action.SERVE_PQC: KeyType.PQC,
    Action.SERVE_HYBRID: KeyType.HYBRID,
}


def masked_floor_by_threat(config: dict[str, Any]) -> list[int]:
    """The masked architecture's floor at each threat level, for the
    same sweep -- the direct counterpart to the readout above.

    Computed from `PolicyTable` for the highest sensitivity class, with
    a fresh table each time so the ratchet does not carry over between
    points (the ratchet is what makes the *episode* monotone; this
    readout is about the underlying mapping)."""
    from env.contracts import SensitivityClass, ThreatPosture
    from env.masking import PolicyTable

    floors: list[int] = []
    for threat_bin in range(10):
        threat = threat_bin / 10.0
        # posture from the same nearest-anchor rule the forecaster uses
        posture = ThreatPosture(int(min(2, round(threat * 2))))
        floors.append(int(PolicyTable().floor(SensitivityClass.S3, posture)))
    return floors


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the S5 steering attack.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--soft-train-steps", type=int, default=20_000)
    parser.add_argument("--dqn-train-steps", type=int, default=20_000)
    parser.add_argument("--out", type=str, default="results/steering_attack.json")
    args = parser.parse_args()

    report = run_steering_attack(
        n_steps=args.steps,
        seeds=list(range(args.seeds)),
        soft_train_steps=args.soft_train_steps,
        dqn_train_steps=args.dqn_train_steps,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {out_path}")


if __name__ == "__main__":
    main()
