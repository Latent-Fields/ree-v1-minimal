"""
Multi-Step Control Completion Experiment (MECH-057 / EXQ-007)

The existing EXQ-004 (control_completion_requirement.py) failed to differentiate
GATED from UNGATED because GridWorld has only atomic single-step actions — there
is no "action in progress" spanning multiple steps for a completion gate to protect.

MECH-057 claims: the control completion requirement applies specifically to actions
that span multiple sub-steps (macro-actions). The gate must suppress incoming
precision updates during a committed macro-action sequence; admitting updates
mid-sequence corrupts the trajectory.

This experiment provides a richer substrate where macro-actions are sequences of
N sub-steps that should complete as a unit. Three conditions test whether the
completion gate is functionally necessary:

  GATED:    Precision updates (E1/E2 gradient steps) suppressed during macro-action
            execution. Updates occur only at macro-action completion boundaries.
            This is the MECH-057 prediction: gating mid-sequence updates helps.

  UNGATED:  Precision updates admitted at every sub-step regardless of macro-action
            state. Mid-trajectory latent disruption expected.

  NO_MACRO: Baseline — all actions atomic (single step), replicates EXQ-004 setup.
            Updates at every step, no macro-action structure.

Environment: MacroStepGrid
  Standard GridWorld (10×10, 4 hazards) extended with macro-actions:
  - Each macro-action commits to a cardinal direction for MACRO_LENGTH sub-steps.
  - Sub-steps execute sequentially; the action cannot change mid-sequence.
  - GATED: E1/E2/E3 gradient updates deferred until the macro completes.
  - UNGATED: updates at every sub-step (testing mid-sequence disruption).

PASS criteria:
  1. GATED last-Q harm < UNGATED last-Q harm   (gate suppression helps)
  2. GATED last-Q harm < NO_MACRO last-Q harm  (macro-action structure helps)

Both must hold for EXQ-007 PASS (MECH-057 provisional support).

Usage:
    python experiments/multistep_control_completion.py
    python experiments/multistep_control_completion.py --episodes 5 --seeds 7

Claims:
    MECH-057: agentic_extension.control_completion_requirement
    EXQ-007
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent
from ree_core.environment.grid_world import GridWorld
from ree_core.utils.config import REEConfig


DEFAULT_EPISODES = 200
DEFAULT_MAX_STEPS = 100          # macro-action steps per episode
DEFAULT_SEEDS = [7, 42, 99]
DEFAULT_GRID_SIZE = 10
DEFAULT_NUM_HAZARDS = 4
MACRO_LENGTH = 4                  # sub-steps per macro-action
MAX_GRAD_NORM = 1.0
E1_LR = 1e-4
POLICY_LR = 1e-3

CONDITIONS = ["GATED", "UNGATED", "NO_MACRO"]


def make_optimizers(
    agent: REEAgent,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    e1_params = (
        list(agent.e1.parameters())
        + list(agent.latent_stack.parameters())
        + list(agent.obs_encoder.parameters())
    )
    policy_params = list(agent.e3.parameters())
    e1_opt = torch.optim.Adam(e1_params, lr=E1_LR)
    policy_opt = torch.optim.Adam(policy_params, lr=POLICY_LR)
    return e1_opt, policy_opt


def _policy_step(
    policy_opt: torch.optim.Optimizer,
    log_probs: List[torch.Tensor],
    accumulated_harm: float,
) -> None:
    """Apply REINFORCE policy gradient update."""
    if log_probs:
        G = float(-accumulated_harm)
        policy_loss = -(torch.stack(log_probs) * G).mean()
        policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for grp in policy_opt.param_groups for p in grp["params"]],
            MAX_GRAD_NORM,
        )
        policy_opt.step()


def _e1_step(agent: REEAgent, e1_opt: torch.optim.Optimizer) -> float:
    """Apply one E1 world model gradient update. Returns E1 loss value."""
    e1_loss = agent.compute_prediction_loss()
    if e1_loss.requires_grad:
        e1_opt.zero_grad()
        e1_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for grp in e1_opt.param_groups for p in grp["params"]],
            MAX_GRAD_NORM,
        )
        e1_opt.step()
        return e1_loss.item()
    return 0.0


def run_episode_gated(
    agent: REEAgent,
    env: GridWorld,
    e1_opt: torch.optim.Optimizer,
    policy_opt: torch.optim.Optimizer,
    max_macro_steps: int,
    macro_length: int,
) -> Dict[str, Any]:
    """
    GATED condition: select a macro-action, execute MACRO_LENGTH sub-steps
    without gradient updates, then apply one consolidated update at completion.
    """
    agent.reset()
    obs = env.reset()

    total_harm = 0.0
    e1_loss_total = 0.0
    macro_count = 0
    done = False

    for _ in range(max_macro_steps):
        if done:
            break

        obs_tensor = obs.unsqueeze(0) if obs.dim() == 1 else obs

        # Select macro-action at the start of each macro-step
        with torch.no_grad():
            encoded = agent.sense(obs_tensor)
            agent.update_latent(encoded)
            candidates = agent.generate_trajectories(agent._current_latent)

        result = agent.e3.select(candidates)
        macro_log_probs = [result.log_prob] if result.log_prob is not None else []
        action_idx = result.selected_action.argmax(dim=-1).item()

        # Execute sub-steps WITHOUT gradient updates (gated)
        macro_harm = 0.0
        for _sub in range(macro_length):
            if done:
                break
            next_obs, harm, done, _info = env.step(action_idx)
            if harm < 0:
                agent.update_residue(harm)
                macro_harm += abs(harm)
            obs = next_obs

        total_harm += macro_harm

        # Gradient update at macro-action completion boundary (gated)
        _policy_step(policy_opt, macro_log_probs, macro_harm)
        e1_loss_total += _e1_step(agent, e1_opt)
        macro_count += 1

    return {
        "total_harm": total_harm,
        "macro_count": macro_count,
        "mean_e1_loss": e1_loss_total / max(macro_count, 1),
    }


def run_episode_ungated(
    agent: REEAgent,
    env: GridWorld,
    e1_opt: torch.optim.Optimizer,
    policy_opt: torch.optim.Optimizer,
    max_macro_steps: int,
    macro_length: int,
) -> Dict[str, Any]:
    """
    UNGATED condition: same macro-action structure, but gradient updates occur
    at every sub-step — disrupting the committed trajectory mid-execution.
    """
    agent.reset()
    obs = env.reset()

    total_harm = 0.0
    e1_loss_total = 0.0
    macro_count = 0
    sub_count = 0
    done = False

    for _ in range(max_macro_steps):
        if done:
            break

        obs_tensor = obs.unsqueeze(0) if obs.dim() == 1 else obs

        with torch.no_grad():
            encoded = agent.sense(obs_tensor)
            agent.update_latent(encoded)
            candidates = agent.generate_trajectories(agent._current_latent)

        result = agent.e3.select(candidates)
        log_prob = result.log_prob
        action_idx = result.selected_action.argmax(dim=-1).item()

        macro_harm = 0.0
        for sub_idx in range(macro_length):
            if done:
                break
            next_obs, harm, done, _info = env.step(action_idx)
            if harm < 0:
                agent.update_residue(harm)
                macro_harm += abs(harm)
                total_harm += abs(harm)

            # E1 update at EVERY sub-step — this is the ungated mid-sequence disruption.
            # (Policy gradient still applied once per macro to avoid double-backward.)
            e1_loss_total += _e1_step(agent, e1_opt)
            sub_count += 1
            obs = next_obs

        # Policy gradient once per macro-action (log_prob graph used only once)
        _policy_step(policy_opt, [log_prob] if log_prob is not None else [], macro_harm)
        macro_count += 1

    return {
        "total_harm": total_harm,
        "macro_count": macro_count,
        "mean_e1_loss": e1_loss_total / max(sub_count, 1),
    }


def run_episode_no_macro(
    agent: REEAgent,
    env: GridWorld,
    e1_opt: torch.optim.Optimizer,
    policy_opt: torch.optim.Optimizer,
    max_steps: int,
) -> Dict[str, Any]:
    """
    NO_MACRO baseline: atomic single-step actions, update at every step.
    Replicates the EXQ-004 setup.
    """
    agent.reset()
    obs = env.reset()

    total_harm = 0.0
    e1_loss_total = 0.0
    steps = 0
    done = False

    for _ in range(max_steps):
        if done:
            break

        obs_tensor = obs.unsqueeze(0) if obs.dim() == 1 else obs

        with torch.no_grad():
            encoded = agent.sense(obs_tensor)
            agent.update_latent(encoded)
            candidates = agent.generate_trajectories(agent._current_latent)

        result = agent.e3.select(candidates)
        log_prob = result.log_prob
        action_idx = result.selected_action.argmax(dim=-1).item()

        next_obs, harm, done, _info = env.step(action_idx)
        step_harm = 0.0
        if harm < 0:
            agent.update_residue(harm)
            step_harm = abs(harm)
            total_harm += step_harm

        _policy_step(policy_opt, [log_prob] if log_prob is not None else [], step_harm)
        e1_loss_total += _e1_step(agent, e1_opt)
        steps += 1
        obs = next_obs

    return {
        "total_harm": total_harm,
        "steps": steps,
        "mean_e1_loss": e1_loss_total / max(steps, 1),
    }


def run_condition(
    seed: int,
    condition: str,
    num_episodes: int,
    max_steps: int,
    grid_size: int,
    num_hazards: int,
    macro_length: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    env = GridWorld(size=grid_size, num_hazards=num_hazards)
    config = REEConfig.from_dims(env.observation_dim, env.action_dim)
    agent = REEAgent(config=config)
    e1_opt, policy_opt = make_optimizers(agent)

    ep_harms: List[float] = []

    for ep in range(num_episodes):
        if condition == "GATED":
            metrics = run_episode_gated(
                agent, env, e1_opt, policy_opt, max_steps, macro_length
            )
        elif condition == "UNGATED":
            metrics = run_episode_ungated(
                agent, env, e1_opt, policy_opt, max_steps, macro_length
            )
        else:  # NO_MACRO
            metrics = run_episode_no_macro(
                agent, env, e1_opt, policy_opt, max_steps * macro_length
            )

        ep_harms.append(metrics["total_harm"])

        if verbose and (ep + 1) % 50 == 0:
            recent_harm = statistics.mean(ep_harms[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  seed={seed}  cond={condition}  "
                f"harm={recent_harm:.3f}"
            )

    quarter = max(1, num_episodes // 4)
    return {
        "condition": condition,
        "seed": seed,
        "macro_length": macro_length if condition != "NO_MACRO" else 1,
        "first_quarter_harm": round(statistics.mean(ep_harms[:quarter]), 4),
        "last_quarter_harm": round(statistics.mean(ep_harms[-quarter:]), 4),
        "episode_count": num_episodes,
    }


def run_experiment(
    num_episodes: int = DEFAULT_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    seeds: Optional[List[int]] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_hazards: int = DEFAULT_NUM_HAZARDS,
    macro_length: int = MACRO_LENGTH,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if seeds is None:
        seeds = DEFAULT_SEEDS

    run_timestamp = datetime.now(timezone.utc).isoformat()

    if verbose:
        print("[Multi-Step Control Completion — MECH-057 / EXQ-007]")
        print(f"  GridWorld: {grid_size}×{grid_size}, {num_hazards} hazards")
        print(f"  Macro-action length: {macro_length} sub-steps")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions:")
        print(f"    GATED:    Updates only at macro-action completion (N={macro_length} steps)")
        print(f"    UNGATED:  Updates at every sub-step (mid-sequence disruption)")
        print(f"    NO_MACRO: Atomic single-step actions (EXQ-004 baseline)")
        print()
        print("  PASS if: GATED harm < UNGATED AND GATED harm < NO_MACRO")
        print()

    all_results: List[Dict] = []
    for seed in seeds:
        for condition in CONDITIONS:
            if verbose:
                print(f"  Seed {seed}  Condition {condition}")
            result = run_condition(
                seed=seed,
                condition=condition,
                num_episodes=num_episodes,
                max_steps=max_steps,
                grid_size=grid_size,
                num_hazards=num_hazards,
                macro_length=macro_length,
                verbose=verbose,
            )
            all_results.append(result)
            if verbose:
                print(
                    f"    harm {result['first_quarter_harm']:.3f} → "
                    f"{result['last_quarter_harm']:.3f}"
                )
                print()

    def _agg(cond: str, key: str) -> float:
        vals = [r[key] for r in all_results if r["condition"] == cond]
        return round(statistics.mean(vals), 4) if vals else 0.0

    gated_harm = _agg("GATED", "last_quarter_harm")
    ungated_harm = _agg("UNGATED", "last_quarter_harm")
    no_macro_harm = _agg("NO_MACRO", "last_quarter_harm")

    gate_vs_ungated = gated_harm < ungated_harm
    gate_vs_no_macro = gated_harm < no_macro_harm
    verdict = "PASS" if (gate_vs_ungated and gate_vs_no_macro) else "FAIL"

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  GATED    last-Q harm: {gated_harm:.4f}")
        print(f"  UNGATED  last-Q harm: {ungated_harm:.4f}")
        print(f"  NO_MACRO last-Q harm: {no_macro_harm:.4f}")
        print()
        print(f"  Gate vs ungated  (GATED < UNGATED)?   {'YES' if gate_vs_ungated else 'NO'}")
        print(f"  Gate vs no-macro (GATED < NO_MACRO)?  {'YES' if gate_vs_no_macro else 'NO'}")
        print()
        print(f"  MECH-057 / EXQ-007 verdict: {verdict}")
        print()
        if verdict == "PASS":
            print("  Interpretation:")
            print("    Suppressing precision updates during committed macro-action")
            print("    execution reduces harm vs both mid-sequence disruption (UNGATED)")
            print("    and atomic-action baseline (NO_MACRO), confirming that the")
            print("    control completion gate is a functional requirement when")
            print("    actions span multiple sub-steps.")
        else:
            print("  Informative FAIL: loops may not yet be differentiated at")
            print("  ree-v1-minimal scale — useful baseline for future design.")

    result_doc = {
        "experiment": "multistep_control_completion",
        "claim": "MECH-057",
        "exq_id": "EXQ-007",
        "run_timestamp": run_timestamp,
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seeds": seeds,
            "grid_size": grid_size,
            "num_hazards": num_hazards,
            "macro_length": macro_length,
            "e1_lr": E1_LR,
            "policy_lr": POLICY_LR,
        },
        "verdict": verdict,
        "aggregate": {
            "gated_last_quarter_harm": gated_harm,
            "ungated_last_quarter_harm": ungated_harm,
            "no_macro_last_quarter_harm": no_macro_harm,
            "gate_vs_ungated_criterion_met": gate_vs_ungated,
            "gate_vs_no_macro_criterion_met": gate_vs_no_macro,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "multistep_control_completion"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(
        evidence_dir / f"multistep_control_completion_{ts}.json"
    )
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main():
    parser = argparse.ArgumentParser(
        description="MECH-057 / EXQ-007: Multi-step control completion gating"
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--num-hazards", type=int, default=DEFAULT_NUM_HAZARDS)
    parser.add_argument("--macro-length", type=int, default=MACRO_LENGTH)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_experiment(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seeds=args.seeds,
        grid_size=args.grid_size,
        num_hazards=args.num_hazards,
        macro_length=args.macro_length,
        output_path=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
