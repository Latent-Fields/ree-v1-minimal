"""
Candidate Count Ablation Experiment (MECH-063 / EVB-0036)

Tests whether the number of E2 trajectory candidates available to E3 selection
is load-bearing for harm avoidance. The E3 selector scores candidates via:
    J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ)
and picks the minimum. With only 1 candidate there is no selection — E3 commits
to whatever E2 generates first. With 32 (standard) it can choose the least harmful
from a meaningful slate. With 128 it has a wider search.

MECH-063 claims: trajectory selection over a diverse candidate set is a functional
requirement, not an optimisation detail. The selection step must see enough candidates
to have a material chance of avoiding the locally harmful one.

Conditions:
  A (SINGLE):    1 candidate — no selection possible, purely E2-driven
  B (STANDARD):  32 candidates — default REE configuration
  C (EXPANDED):  128 candidates — wider search budget

Key diagnostics:
  1. STANDARD last-Q harm < SINGLE last-Q harm
     (selection over multiple candidates reduces harm vs no selection)
  2. EXPANDED last-Q harm <= STANDARD last-Q harm * 1.10
     (128 candidates is at least roughly as good as 32 — no collapse)

Both must hold for MECH-063 PASS.

The primary claim is diagnostic 1: does trajectory *selection* matter?
Diagnostic 2 guards against pathological variance at high candidate counts.

Usage:
    python experiments/candidate_count_ablation.py
    python experiments/candidate_count_ablation.py --episodes 5 --seeds 7

Claims:
    MECH-063: trajectory_selection.candidate_count_load_bearing
    EVB-0036
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
DEFAULT_MAX_STEPS = 100
DEFAULT_SEEDS = [7, 42, 99]
DEFAULT_GRID_SIZE = 10
DEFAULT_NUM_HAZARDS = 4
MAX_GRAD_NORM = 1.0

CONDITIONS: Dict[str, int] = {
    "SINGLE":   1,
    "STANDARD": 32,
    "EXPANDED": 128,
}


def make_optimizers(
    agent: REEAgent,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """Standard two-optimizer setup matching REETrainer."""
    e1_params = (
        list(agent.e1.parameters())
        + list(agent.latent_stack.parameters())
        + list(agent.obs_encoder.parameters())
    )
    policy_params = list(agent.e3.parameters())
    e1_opt = torch.optim.Adam(e1_params, lr=1e-4)
    policy_opt = torch.optim.Adam(policy_params, lr=1e-3)
    return e1_opt, policy_opt


def run_episode(
    agent: REEAgent,
    env: GridWorld,
    e1_opt: torch.optim.Optimizer,
    policy_opt: torch.optim.Optimizer,
    max_steps: int,
    num_candidates: int,
) -> Dict[str, Any]:
    """Run one episode, overriding the candidate count for this condition."""
    agent.reset()
    obs = env.reset()

    log_probs: List[torch.Tensor] = []
    total_harm = 0.0
    steps = 0

    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            encoded = agent.sense(obs_tensor)
            agent.update_latent(encoded)
            # Override num_candidates for this condition
            candidates = agent.generate_trajectories(
                agent._current_latent, num_candidates=num_candidates
            )

        result = agent.e3.select(candidates)
        if result.log_prob is not None:
            log_probs.append(result.log_prob)

        action_idx = result.selected_action.argmax(dim=-1).item()
        next_obs, harm, done, _info = env.step(action_idx)

        if harm < 0:
            agent.update_residue(harm)
            total_harm += abs(harm)

        obs = next_obs
        steps += 1
        if done:
            break

    # Policy gradient update
    policy_loss_val = 0.0
    if log_probs:
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

    # E1 world model update
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

    return {
        "total_harm": total_harm,
        "steps": steps,
        "e1_loss": e1_loss_val,
        "policy_loss": policy_loss_val,
    }


def run_condition(
    seed: int,
    condition: str,
    num_candidates: int,
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

    for ep in range(num_episodes):
        metrics = run_episode(agent, env, e1_opt, policy_opt, max_steps, num_candidates)
        ep_harms.append(metrics["total_harm"])

        if verbose and (ep + 1) % 50 == 0:
            recent_harm = statistics.mean(ep_harms[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  "
                f"seed={seed}  cond={condition}(n={num_candidates})  "
                f"harm={recent_harm:.3f}"
            )

    quarter = max(1, num_episodes // 4)
    return {
        "condition": condition,
        "num_candidates": num_candidates,
        "seed": seed,
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
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if seeds is None:
        seeds = DEFAULT_SEEDS

    run_timestamp = datetime.now(timezone.utc).isoformat()

    if verbose:
        print("[Candidate Count Ablation — MECH-063 / EVB-0036]")
        print(f"  GridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        for cond, n in CONDITIONS.items():
            print(f"    {cond}: n_candidates={n}")
        print()
        print("  Diagnostic 1: STANDARD last-Q harm < SINGLE last-Q harm")
        print("    (selection from multiple candidates is load-bearing)")
        print("  Diagnostic 2: EXPANDED last-Q harm <= STANDARD * 1.10")
        print("    (128 candidates at least as good as 32)")
        print()

    all_results = []

    for seed in seeds:
        for condition, num_candidates in CONDITIONS.items():
            if verbose:
                print(f"  Seed {seed}  Condition {condition} (n={num_candidates})")
            result = run_condition(
                seed=seed,
                condition=condition,
                num_candidates=num_candidates,
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

    def _agg(cond: str, key: str) -> float:
        vals = [r[key] for r in all_results if r["condition"] == cond]
        return round(statistics.mean(vals), 4)

    single_harm = _agg("SINGLE", "last_quarter_harm")
    standard_harm = _agg("STANDARD", "last_quarter_harm")
    expanded_harm = _agg("EXPANDED", "last_quarter_harm")

    selection_ok = standard_harm < single_harm
    stability_ok = expanded_harm <= standard_harm * 1.10
    verdict = "PASS" if (selection_ok and stability_ok) else "FAIL"
    partial = (selection_ok or stability_ok) and not (selection_ok and stability_ok)

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  SINGLE   (n=1)   last-Q harm: {single_harm:.3f}")
        print(f"  STANDARD (n=32)  last-Q harm: {standard_harm:.3f}")
        print(f"  EXPANDED (n=128) last-Q harm: {expanded_harm:.3f}")
        print()
        print(f"  Selection criterion (STANDARD < SINGLE)?  {'YES' if selection_ok else 'NO'}")
        print(f"  Stability criterion (EXPANDED <= STD*1.10)?  {'YES' if stability_ok else 'NO'}")
        print()
        print(f"  MECH-063 verdict: {verdict}")
        if partial:
            print("  (partial — one of two criteria met)")
        print()

    result_doc = {
        "experiment": "candidate_count_ablation",
        "claim": "MECH-063",
        "evb_id": "EVB-0036",
        "run_timestamp": run_timestamp,
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seeds": seeds,
            "grid_size": grid_size,
            "num_hazards": num_hazards,
            "conditions": CONDITIONS,
        },
        "verdict": verdict,
        "partial_support": partial,
        "aggregate": {
            "single_harm_last_quarter": single_harm,
            "standard_harm_last_quarter": standard_harm,
            "expanded_harm_last_quarter": expanded_harm,
            "selection_criterion_met": selection_ok,
            "stability_criterion_met": stability_ok,
            "stability_tolerance_factor": 1.10,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "candidate_count_ablation"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(
        evidence_dir / f"candidate_count_ablation_{ts}.json"
    )
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main():
    parser = argparse.ArgumentParser(
        description="MECH-063: Candidate Count Ablation experiment"
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
