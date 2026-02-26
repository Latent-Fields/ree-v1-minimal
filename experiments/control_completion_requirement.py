"""
Control Completion Requirement Experiment (MECH-057 / EVB-0042)

Tests whether all four core REE agency loops are required for harm avoidance.
Disabling attribution (harm→residue feedback) or gating (E3 trajectory selection)
should degrade outcomes compared to the full system.

MECH-057 claims: REE-like control completion requires all loops to be active. A system
that predicts well but lacks self-attribution or gating degrades ethically.

The four loops being tested:
  1. E1 prediction: slow world model (always active in all conditions)
  2. E2 rollout: short-horizon trajectory candidates (always active)
  3. E3 gating: trajectory selection based on harm/residue scoring
  4. Harm attribution: harm→residue feedback (agent.update_residue)

Conditions:
  A (FULL):           All loops active — baseline canonical REE.
  B (NO_ATTRIBUTION): Loop 4 disabled — no harm→residue feedback.
                       Agent cannot build a residue map; E3 still scores but field stays empty.
  C (NO_GATING):      Loop 3 disabled — random trajectory selection instead of E3 scoring.
                       E1/E2/attribution still run, but committed action is random.
                       Policy gradient update skipped (REINFORCE on random choice is meaningless).

Key diagnostics:
  1. NO_ATTRIBUTION last-quarter harm > FULL last-quarter harm * 1.10
     Attribution matters — without residue feedback, harm avoidance degrades ≥10%.
  2. NO_GATING last-quarter harm > FULL last-quarter harm * 1.10
     Gating matters — random trajectory selection degrades harm avoidance ≥10%.

Both must hold for MECH-057 PASS.

Note on P3 status: MECH-057 needs conceptual redesign post-JEPA decoupling. If FULL and
ablated conditions show no significant difference (<10% threshold), the result is
informative — the loops are not yet differentiated at ree-v1-minimal scale — rather than
a design failure. This is expected and useful as a baseline for future experiment design.

Usage:
    python experiments/control_completion_requirement.py
    python experiments/control_completion_requirement.py --episodes 5 --seeds 7

Runtime: ~23 min on M2 Air 8GB (3 conditions x 3 seeds x 200 episodes).

Claims:
    MECH-057: agentic_extension.control_completion_requirement
    EVB-0042
"""

import argparse
import json
import random
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
DEFAULT_MAX_STEPS = 100
DEFAULT_SEEDS = [7, 42, 99]
DEFAULT_GRID_SIZE = 10
DEFAULT_NUM_HAZARDS = 4
MAX_GRAD_NORM = 1.0
E1_LR = 1e-4
POLICY_LR = 1e-3

CONDITIONS = ["FULL", "NO_ATTRIBUTION", "NO_GATING"]


def make_optimizers(
    agent: REEAgent,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """Build two optimizers matching REETrainer's parameter grouping."""
    e1_params = (
        list(agent.e1.parameters())
        + list(agent.latent_stack.parameters())
        + list(agent.obs_encoder.parameters())
    )
    policy_params = list(agent.e3.parameters())
    e1_opt = torch.optim.Adam(e1_params, lr=E1_LR)
    policy_opt = torch.optim.Adam(policy_params, lr=POLICY_LR)
    return e1_opt, policy_opt


def run_episode(
    agent: REEAgent,
    env: GridWorld,
    e1_opt: torch.optim.Optimizer,
    policy_opt: torch.optim.Optimizer,
    condition: str,
    max_steps: int,
) -> Dict[str, Any]:
    """
    Run one episode, applying the loop ablation specified by condition.

    FULL:           full REE loop with policy gradient update
    NO_ATTRIBUTION: skip agent.update_residue() entirely; policy gradient still applied
    NO_GATING:      random candidate selection; skip policy gradient (not meaningful)
    """
    agent.reset()
    obs = env.reset()

    log_probs: List[torch.Tensor] = []
    total_harm = 0.0
    ep_residue_costs: List[float] = []
    steps = 0

    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # Encode + latent update (always inside no_grad for inference)
        with torch.no_grad():
            encoded = agent.sense(obs_tensor)
            agent.update_latent(encoded)
            candidates = agent.generate_trajectories(agent._current_latent)

        if condition == "NO_GATING":
            # Random candidate selection — bypass E3 entirely
            rand_idx = torch.randint(len(candidates), (1,)).item()
            selected_traj = candidates[rand_idx]
            action = selected_traj.actions[:, 0, :]  # first action in selected trajectory
            log_prob = None  # no policy gradient for random selection
            # Still compute residue cost for tracking (field may have values)
            try:
                residue_cost = agent.e3.compute_residue_cost(selected_traj).mean().item()
            except Exception:
                residue_cost = 0.0
        else:
            # FULL or NO_ATTRIBUTION: use E3 selection with grad for REINFORCE
            result = agent.e3.select(candidates)
            action = result.selected_action
            log_prob = result.log_prob
            selected_traj = result.selected_trajectory
            try:
                residue_cost = agent.e3.compute_residue_cost(selected_traj).mean().item()
            except Exception:
                residue_cost = 0.0

        ep_residue_costs.append(residue_cost)

        if log_prob is not None:
            log_probs.append(log_prob)

        action_idx = action.argmax(dim=-1).item()
        next_obs, harm, done, _info = env.step(action_idx)

        if harm < 0:
            total_harm += abs(harm)
            if condition != "NO_ATTRIBUTION":
                # FULL and NO_GATING both do attribution; NO_ATTRIBUTION skips it
                agent.update_residue(harm)

        obs = next_obs
        steps += 1
        if done:
            break

    # Policy gradient update (FULL and NO_ATTRIBUTION only; NO_GATING skips)
    policy_loss_val = 0.0
    if log_probs and condition != "NO_GATING":
        G = float(-total_harm)
        policy_loss = -(torch.stack(log_probs) * G).mean()
        policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for grp in policy_opt.param_groups for p in grp["params"]],
            MAX_GRAD_NORM,
        )
        policy_opt.step()
        policy_loss_val = policy_loss.item()

    # E1 update: always active across all conditions
    e1_loss_val = 0.0
    e1_loss = agent.compute_prediction_loss()
    if e1_loss.requires_grad:
        e1_opt.zero_grad()
        e1_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for grp in e1_opt.param_groups for p in grp["params"]],
            MAX_GRAD_NORM,
        )
        e1_opt.step()
        e1_loss_val = e1_loss.item()

    mean_residue_cost = (
        statistics.mean(ep_residue_costs) if ep_residue_costs else 0.0
    )

    return {
        "total_harm": total_harm,
        "steps": steps,
        "mean_trajectory_residue_cost": mean_residue_cost,
        "e1_loss": e1_loss_val,
        "policy_loss": policy_loss_val,
    }


def run_condition(
    seed: int,
    condition: str,
    num_episodes: int,
    max_steps: int,
    grid_size: int,
    num_hazards: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    env = GridWorld(size=grid_size, num_hazards=num_hazards)
    config = REEConfig.from_dims(env.observation_dim, env.action_dim)
    agent = REEAgent(config=config)
    e1_opt, policy_opt = make_optimizers(agent)

    ep_harms: List[float] = []
    ep_e1_losses: List[float] = []
    ep_residue_costs: List[float] = []

    for ep in range(num_episodes):
        metrics = run_episode(agent, env, e1_opt, policy_opt, condition, max_steps)
        ep_harms.append(metrics["total_harm"])
        ep_e1_losses.append(metrics["e1_loss"])
        ep_residue_costs.append(metrics["mean_trajectory_residue_cost"])

        if verbose and (ep + 1) % 50 == 0:
            recent_harm = statistics.mean(ep_harms[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  "
                f"seed={seed}  cond={condition}  "
                f"harm={recent_harm:.3f}"
            )

    quarter = max(1, num_episodes // 4)

    return {
        "condition": condition,
        "seed": seed,
        "first_quarter_harm": round(statistics.mean(ep_harms[:quarter]), 4),
        "last_quarter_harm": round(statistics.mean(ep_harms[-quarter:]), 4),
        "mean_e1_loss": round(statistics.mean(ep_e1_losses), 6),
        "mean_trajectory_residue_cost": round(statistics.mean(ep_residue_costs), 6),
        "episode_count": num_episodes,
    }


def run_experiment(
    num_episodes: int = DEFAULT_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    seeds: Optional[List[int]] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_hazards: int = DEFAULT_NUM_HAZARDS,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if seeds is None:
        seeds = DEFAULT_SEEDS

    run_timestamp = datetime.now(timezone.utc).isoformat()

    if verbose:
        print("[Control Completion Requirement — MECH-057 / EVB-0042]")
        print(f"  GridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions:")
        print("    A (FULL):           All loops active (baseline)")
        print("    B (NO_ATTRIBUTION): Harm→residue feedback disabled")
        print("    C (NO_GATING):      Random trajectory selection, no policy gradient")
        print()
        print("  Diagnostic 1: NO_ATTRIBUTION last-Q harm > FULL * 1.10")
        print("  Diagnostic 2: NO_GATING last-Q harm > FULL * 1.10")
        print()
        est_min = len(seeds) * len(CONDITIONS) * num_episodes * max_steps * 750 / (1000 * 60 * 200)
        print(f"  Est. runtime: ~{est_min:.0f} min on M2 Air 8GB")
        print()

    all_results = []

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
                verbose=verbose,
            )
            all_results.append(result)
            if verbose:
                print(
                    f"    harm {result['first_quarter_harm']:.3f} → "
                    f"{result['last_quarter_harm']:.3f}"
                )
                print()

    full_results = [r for r in all_results if r["condition"] == "FULL"]
    no_attr_results = [r for r in all_results if r["condition"] == "NO_ATTRIBUTION"]
    no_gate_results = [r for r in all_results if r["condition"] == "NO_GATING"]

    def _agg(results: List[Dict], key: str) -> float:
        return round(statistics.mean(r[key] for r in results), 4)

    full_harm_last = _agg(full_results, "last_quarter_harm")
    no_attr_harm_last = _agg(no_attr_results, "last_quarter_harm")
    no_gate_harm_last = _agg(no_gate_results, "last_quarter_harm")

    # PASS criteria (10% degradation threshold)
    threshold = 1.10
    attribution_matters = no_attr_harm_last > full_harm_last * threshold
    gating_matters = no_gate_harm_last > full_harm_last * threshold
    verdict = "PASS" if (attribution_matters and gating_matters) else "FAIL"
    partial = (attribution_matters or gating_matters) and not (attribution_matters and gating_matters)

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  FULL           last-Q harm: {full_harm_last:.3f}  (baseline)")
        print(f"  NO_ATTRIBUTION last-Q harm: {no_attr_harm_last:.3f}  "
              f"({'↑' if no_attr_harm_last > full_harm_last else '↓'} "
              f"{abs(no_attr_harm_last - full_harm_last) / max(full_harm_last, 1e-8) * 100:.1f}%)")
        print(f"  NO_GATING      last-Q harm: {no_gate_harm_last:.3f}  "
              f"({'↑' if no_gate_harm_last > full_harm_last else '↓'} "
              f"{abs(no_gate_harm_last - full_harm_last) / max(full_harm_last, 1e-8) * 100:.1f}%)")
        print()
        print(f"  Attribution matters (>10% degradation)?  {'YES' if attribution_matters else 'NO'}")
        print(f"  Gating matters (>10% degradation)?       {'YES' if gating_matters else 'NO'}")
        print()
        print(f"  MECH-057 verdict: {verdict}")
        if partial:
            print("  (partial — one of two criteria met)")
        if verdict == "FAIL" and not (attribution_matters or gating_matters):
            print()
            print("  Note: MECH-057 is P3 priority and flagged for post-JEPA redesign.")
            print("  A FAIL at ree-v1-minimal scale indicates the loops are not yet")
            print("  differentiated enough to show a measurable ablation effect.")
            print("  This is informative rather than a fundamental design failure.")
        elif verdict == "PASS":
            print()
            print("  Interpretation:")
            print("    Removing either harm attribution or trajectory gating degrades")
            print("    harm outcomes by >10%. Both loops are functionally necessary at")
            print("    ree-v1-minimal scale. MECH-057 control completion requirement confirmed.")
        print()

    result_doc = {
        "experiment": "control_completion_requirement",
        "claim": "MECH-057",
        "evb_id": "EVB-0042",
        "run_timestamp": run_timestamp,
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seeds": seeds,
            "grid_size": grid_size,
            "num_hazards": num_hazards,
            "e1_lr": E1_LR,
            "policy_lr": POLICY_LR,
            "degradation_threshold": threshold,
            "conditions": CONDITIONS,
        },
        "verdict": verdict,
        "partial_support": partial,
        "aggregate": {
            "full_harm_last_quarter": full_harm_last,
            "no_attribution_harm_last_quarter": no_attr_harm_last,
            "no_gating_harm_last_quarter": no_gate_harm_last,
            "degradation_threshold": threshold,
            "attribution_criterion_met": attribution_matters,
            "gating_criterion_met": gating_matters,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "control_completion_requirement"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(
        evidence_dir / f"control_completion_requirement_{ts}.json"
    )
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main():
    parser = argparse.ArgumentParser(
        description="MECH-057: Control Completion Requirement experiment"
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--num-hazards", type=int, default=DEFAULT_NUM_HAZARDS)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_experiment(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seeds=args.seeds,
        grid_size=args.grid_size,
        num_hazards=args.num_hazards,
        output_path=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
