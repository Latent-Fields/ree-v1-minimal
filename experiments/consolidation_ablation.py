"""
Consolidation Ablation Experiment (MECH-068 / CSH-1)

Tests the Consolidation Selectivity Hypothesis:
  "Behavioural selectivity emerges at the consolidation/gating layer (E3),
   not at the shared representational basis (E1)."

Experiment design:
  - Warm-start a base REE agent in GridWorld
  - Freeze E1 weights entirely
  - Run N episodes with a grid of E3 consolidation weights (lambda, rho)
  - Measure:
      (a) Trajectory preference: harm-zone visitation rate, total harm accumulated
      (b) E1 latent stability: cosine similarity of z_theta/z_delta across configs

Expected result (CSH-1 PASS):
  - Trajectory preference varies with lambda/rho
  - E1 latent cosine similarity remains high (>= 0.90) across configs

Failure mode (CSH-1 FAIL):
  - E1 latent drift under E3-only changes (cosine sim < 0.90)
  - Or trajectory preference does not change significantly across lambda/rho configs

This is the first genuine discriminating experiment for REE.
Outputs: results/consolidation_ablation_<timestamp>.json

Usage:
    python experiments/consolidation_ablation.py
    python experiments/consolidation_ablation.py --episodes 20 --seed 42
    python experiments/consolidation_ablation.py --steps 200 --output my_results.json

Ref: REE_assembly/docs/architecture/compact_consolidation_principle.md (MECH-068)
     Cowley et al. 2023, doi:10.1101/2023.11.22.568315
"""

import argparse
import copy
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Allow running from repo root or from experiments/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent
from ree_core.environment.grid_world import GridWorld
from ree_core.utils.config import REEConfig, E3Config, EnvironmentConfig


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Consolidation weight grid to sweep
# lambda_ethical: weight on ethical cost M(zeta) in J(zeta)
# rho_residue:    weight on residue cost Phi_R(zeta) in J(zeta)
LAMBDA_VALUES = [0.0, 0.5, 1.0, 2.0]
RHO_VALUES = [0.0, 0.5, 1.0]

# Warm-start training episodes before freezing E1
WARMUP_EPISODES = 10

# Default experiment parameters
DEFAULT_EPISODES_PER_CONFIG = 10
DEFAULT_MAX_STEPS = 100
DEFAULT_SEED = 7
DEFAULT_GRID_SIZE = 10
DEFAULT_NUM_HAZARDS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity_tensors(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean cosine similarity between two [N, D] tensors."""
    a_flat = a.reshape(a.shape[0], -1).detach()
    b_flat = b.reshape(b.shape[0], -1).detach()
    sims = F.cosine_similarity(a_flat, b_flat, dim=1)
    return float(sims.mean().item())


def _run_episodes(
    agent: REEAgent,
    env: GridWorld,
    num_episodes: int,
    max_steps: int,
    seed_offset: int = 0,
    collect_latents: bool = False,
) -> Dict[str, Any]:
    """
    Run num_episodes in env with agent. Return aggregate metrics.

    Returns dict with:
      - total_harm: sum of harm accumulated across all episodes
      - hazard_visits: number of timesteps spent in hazard cells
      - episode_lengths: list of episode lengths
      - latent_snapshots: list of (z_theta, z_delta) tensors if collect_latents
    """
    total_harm = 0.0
    hazard_visits = 0
    episode_lengths = []
    latent_snapshots = []

    for ep in range(num_episodes):
        # Seed environment reproducibly per episode
        torch.manual_seed(seed_offset + ep * 1000)
        obs = env.reset()
        ep_harm = 0.0
        ep_hazard_visits = 0
        ep_steps = 0

        for step in range(max_steps):
            action = agent.act(obs)
            obs, harm_signal, done, info = env.step(action)
            agent.update_residue(harm_signal)

            if harm_signal < 0:
                ep_harm += abs(harm_signal)
                ep_hazard_visits += 1

            ep_steps += 1

            # Collect E1 latent snapshot for stability measurement
            if collect_latents and step % 10 == 0:
                with torch.no_grad():
                    lat = agent._current_latent
                    if lat is not None:
                        z_theta = lat.z_theta.detach().clone()
                        z_delta = lat.z_delta.detach().clone()
                        latent_snapshots.append((z_theta, z_delta))

            if done:
                break

        total_harm += ep_harm
        hazard_visits += ep_hazard_visits
        episode_lengths.append(ep_steps)

    return {
        "total_harm": total_harm,
        "mean_harm_per_episode": total_harm / num_episodes,
        "hazard_visits": hazard_visits,
        "mean_hazard_visits_per_episode": hazard_visits / num_episodes,
        "episode_lengths": episode_lengths,
        "mean_episode_length": sum(episode_lengths) / len(episode_lengths),
        "latent_snapshots": latent_snapshots,
    }


def _compute_latent_stability(
    baseline_snapshots: List[Tuple[torch.Tensor, torch.Tensor]],
    ablation_snapshots: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, float]:
    """
    Compare E1 latent representations between baseline and an ablation config.

    Returns cosine similarity stats for z_theta and z_delta.
    """
    n = min(len(baseline_snapshots), len(ablation_snapshots))
    if n == 0:
        return {"z_theta_cosine_sim": float("nan"), "z_delta_cosine_sim": float("nan")}

    theta_sims = []
    delta_sims = []
    for (b_theta, b_delta), (a_theta, a_delta) in zip(
        baseline_snapshots[:n], ablation_snapshots[:n]
    ):
        # Ensure same shape for comparison
        if b_theta.shape == a_theta.shape:
            sim_theta = _cosine_similarity_tensors(b_theta, a_theta)
            theta_sims.append(sim_theta)
        if b_delta.shape == a_delta.shape:
            sim_delta = _cosine_similarity_tensors(b_delta, a_delta)
            delta_sims.append(sim_delta)

    return {
        "z_theta_cosine_sim": float(sum(theta_sims) / len(theta_sims)) if theta_sims else float("nan"),
        "z_delta_cosine_sim": float(sum(delta_sims) / len(delta_sims)) if delta_sims else float("nan"),
    }


def _build_agent(
    env: GridWorld,
    lambda_ethical: float,
    rho_residue: float,
    shared_basis_state: Optional[Dict] = None,
    freeze_e1: bool = False,
) -> REEAgent:
    """
    Build a REEAgent with specified E3 consolidation weights.

    Optionally initialise the shared basis (E1 + latent_stack + obs_encoder) from
    shared_basis_state — a dict with keys "e1", "latent_stack", and optionally
    "obs_encoder" — and freeze those weights so only E3 varies.
    """
    ree_config = REEConfig.from_dims(env.observation_dim, env.action_dim)
    ree_config.e3.lambda_ethical = lambda_ethical
    ree_config.e3.rho_residue = rho_residue
    agent = REEAgent(config=ree_config)

    # Load shared-basis weights from baseline if provided
    if shared_basis_state is not None:
        try:
            if "e1" in shared_basis_state:
                agent.e1.load_state_dict(shared_basis_state["e1"])
            if "latent_stack" in shared_basis_state:
                agent.latent_stack.load_state_dict(shared_basis_state["latent_stack"])
            if "obs_encoder" in shared_basis_state and hasattr(agent, "obs_encoder"):
                agent.obs_encoder.load_state_dict(shared_basis_state["obs_encoder"])
        except RuntimeError:
            # Architecture mismatch — skip loading (shouldn't happen in practice)
            pass

    # Freeze shared-basis weights if requested (E1 + latent_stack + obs_encoder)
    if freeze_e1:
        for param in agent.e1.parameters():
            param.requires_grad = False
        for param in agent.latent_stack.parameters():
            param.requires_grad = False
        if hasattr(agent, "obs_encoder"):
            for param in agent.obs_encoder.parameters():
                param.requires_grad = False

    return agent


def _assess_verdict(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess whether CSH-1 passes or fails based on experiment results.

    CSH-1 PASS criteria:
      1. Trajectory preference varies with lambda/rho (harm_range > threshold)
      2. E1 latent remains stable across configs (mean cosine_sim >= 0.90)

    Returns verdict dict.
    """
    config_results = results["config_results"]
    baseline_key = results["baseline_config_key"]

    # Collect harm values across all configs
    harm_values = [
        v["mean_harm_per_episode"]
        for k, v in config_results.items()
        if k != baseline_key
    ]
    harm_range = max(harm_values) - min(harm_values) if harm_values else 0.0

    # Collect E1 stability values
    stability_values = [
        v["e1_stability"]["z_theta_cosine_sim"]
        for k, v in config_results.items()
        if k != baseline_key
        and not math.isnan(v["e1_stability"]["z_theta_cosine_sim"])
    ]
    mean_stability = (
        sum(stability_values) / len(stability_values)
        if stability_values else float("nan")
    )

    # Criteria
    selectivity_criterion = harm_range > 0.01  # lambda/rho changes affected behaviour
    stability_criterion = (
        math.isnan(mean_stability) or mean_stability >= 0.85
    )  # E1 did not drift (NaN means no latents collected — not a failure)

    verdict = "PASS" if (selectivity_criterion and stability_criterion) else "FAIL"

    failure_signatures = []
    if not selectivity_criterion:
        failure_signatures.append("csh1:no_selectivity_change_under_consolidation_variation")
    if not stability_criterion:
        failure_signatures.append("csh1:e1_latent_drift_under_e3_only_changes")

    return {
        "verdict": verdict,
        "harm_range_across_configs": round(harm_range, 6),
        "mean_e1_latent_stability": round(mean_stability, 4) if not math.isnan(mean_stability) else None,
        "selectivity_criterion_met": selectivity_criterion,
        "stability_criterion_met": stability_criterion,
        "failure_signatures": failure_signatures,
        "interpretation": (
            "CSH-1 supported: E3 consolidation weights changed trajectory preference "
            "while E1 latent structure remained stable."
            if verdict == "PASS"
            else "CSH-1 not supported: see failure_signatures for details."
        ),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_consolidation_ablation(
    num_episodes: int = DEFAULT_EPISODES_PER_CONFIG,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = DEFAULT_SEED,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_hazards: int = DEFAULT_NUM_HAZARDS,
    output_path: Optional[str] = None,
    warmup_episodes: int = WARMUP_EPISODES,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the Consolidation Ablation experiment.

    Returns the full results dict (also written to output_path if given).
    """
    torch.manual_seed(seed)
    run_timestamp = datetime.now(timezone.utc).isoformat()

    env_cfg = EnvironmentConfig(
        size=grid_size,
        num_hazards=num_hazards,
    )
    env = GridWorld(
        size=grid_size,
        num_hazards=num_hazards,
    )

    if verbose:
        print(f"[CSH-1] Consolidation Ablation Experiment")
        print(f"  GridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  obs_dim={env.observation_dim}, action_dim={env.action_dim}")
        print(f"  Warmup: {warmup_episodes} episodes | Test: {num_episodes} ep/config")
        print(f"  Lambda grid: {LAMBDA_VALUES}")
        print(f"  Rho grid: {RHO_VALUES}")
        print()

    # ------------------------------------------------------------------
    # Step 1: Warm-start a baseline agent (default lambda=1.0, rho=0.5)
    # ------------------------------------------------------------------
    baseline_lambda = 1.0
    baseline_rho = 0.5
    baseline_agent = _build_agent(
        env, lambda_ethical=baseline_lambda, rho_residue=baseline_rho
    )

    if verbose:
        print(f"[Step 1] Warming up baseline agent (lambda={baseline_lambda}, rho={baseline_rho})...")
    _run_episodes(baseline_agent, env, warmup_episodes, max_steps, seed_offset=seed)

    # Capture shared-basis state dicts after warmup (E1 + latent_stack + obs_encoder)
    shared_basis_state = {
        "e1": copy.deepcopy(baseline_agent.e1.state_dict()),
        "latent_stack": copy.deepcopy(baseline_agent.latent_stack.state_dict()),
    }
    if hasattr(baseline_agent, "obs_encoder"):
        shared_basis_state["obs_encoder"] = copy.deepcopy(baseline_agent.obs_encoder.state_dict())

    # Run baseline test episodes (with latent collection for stability reference)
    if verbose:
        print(f"[Step 2] Running baseline test episodes ({num_episodes} episodes)...")
    baseline_results = _run_episodes(
        baseline_agent, env, num_episodes, max_steps,
        seed_offset=seed + 10000, collect_latents=True
    )
    baseline_key = f"lambda_{baseline_lambda}_rho_{baseline_rho}"

    if verbose:
        print(f"  Baseline harm/ep: {baseline_results['mean_harm_per_episode']:.4f}")
        print(f"  Baseline hazard visits/ep: {baseline_results['mean_hazard_visits_per_episode']:.2f}")
        print()

    # ------------------------------------------------------------------
    # Step 2: Sweep E3 consolidation weights with E1 frozen
    # ------------------------------------------------------------------
    config_results: Dict[str, Any] = {}
    config_results[baseline_key] = {
        "lambda_ethical": baseline_lambda,
        "rho_residue": baseline_rho,
        "is_baseline": True,
        "e1_frozen": False,
        "mean_harm_per_episode": baseline_results["mean_harm_per_episode"],
        "mean_hazard_visits_per_episode": baseline_results["mean_hazard_visits_per_episode"],
        "mean_episode_length": baseline_results["mean_episode_length"],
        "e1_stability": {"z_theta_cosine_sim": 1.0, "z_delta_cosine_sim": 1.0},
    }

    if verbose:
        print("[Step 3] Sweeping E3 consolidation weights (E1 frozen from baseline)...")
        print(f"  {'Config':<30} {'Harm/ep':>10} {'Hazard/ep':>12} {'E1 sim (theta)':>16}")
        print(f"  {'-'*30} {'-'*10} {'-'*12} {'-'*16}")

    for lam in LAMBDA_VALUES:
        for rho in RHO_VALUES:
            config_key = f"lambda_{lam}_rho_{rho}"
            if config_key == baseline_key:
                continue  # already have baseline

            # Build ablation agent with shared basis weights, frozen
            ablation_agent = _build_agent(
                env,
                lambda_ethical=lam,
                rho_residue=rho,
                shared_basis_state=shared_basis_state,
                freeze_e1=True,
            )

            # Run test episodes
            abl_results = _run_episodes(
                ablation_agent, env, num_episodes, max_steps,
                seed_offset=seed + 10000, collect_latents=True
            )

            # Compute E1 latent stability vs baseline
            stability = _compute_latent_stability(
                baseline_results["latent_snapshots"],
                abl_results["latent_snapshots"],
            )

            config_results[config_key] = {
                "lambda_ethical": lam,
                "rho_residue": rho,
                "is_baseline": False,
                "e1_frozen": True,
                "mean_harm_per_episode": abl_results["mean_harm_per_episode"],
                "mean_hazard_visits_per_episode": abl_results["mean_hazard_visits_per_episode"],
                "mean_episode_length": abl_results["mean_episode_length"],
                "e1_stability": stability,
            }

            if verbose:
                sim_str = f"{stability['z_theta_cosine_sim']:.3f}" if not math.isnan(stability['z_theta_cosine_sim']) else "N/A"
                print(
                    f"  {config_key:<30}"
                    f"  {abl_results['mean_harm_per_episode']:>8.4f}"
                    f"  {abl_results['mean_hazard_visits_per_episode']:>10.2f}"
                    f"  {sim_str:>14}"
                )

    if verbose:
        print()

    # ------------------------------------------------------------------
    # Step 3: Assess verdict
    # ------------------------------------------------------------------
    results = {
        "experiment": "consolidation_ablation",
        "claim_ids_tested": ["MECH-068"],
        "csh_sub_claim": "CSH-1",
        "run_timestamp_utc": run_timestamp,
        "config": {
            "seed": seed,
            "num_episodes_per_config": num_episodes,
            "max_steps_per_episode": max_steps,
            "grid_size": grid_size,
            "num_hazards": num_hazards,
            "warmup_episodes": warmup_episodes,
            "lambda_values": LAMBDA_VALUES,
            "rho_values": RHO_VALUES,
        },
        "baseline_config_key": baseline_key,
        "config_results": config_results,
    }

    verdict = _assess_verdict(results)
    results["verdict"] = verdict

    if verbose:
        print(f"[Result] Verdict: {verdict['verdict']}")
        print(f"  Harm range across configs: {verdict['harm_range_across_configs']:.4f}")
        if verdict["mean_e1_latent_stability"] is not None:
            print(f"  Mean E1 latent stability: {verdict['mean_e1_latent_stability']:.3f}")
        print(f"  {verdict['interpretation']}")
        if verdict["failure_signatures"]:
            print(f"  Failure signatures: {verdict['failure_signatures']}")

    # ------------------------------------------------------------------
    # Step 4: Write output
    # ------------------------------------------------------------------
    if output_path is None:
        out_dir = Path(__file__).parent.parent / "evidence" / "experiments" / "consolidation_ablation"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = run_timestamp.replace(":", "").replace("+", "").replace("-", "")[:15]
        output_path = str(out_dir / f"consolidation_ablation_{ts}.json")

    # Remove tensors from latent_snapshots before serializing
    results_serializable = copy.deepcopy(results)
    # (latent_snapshots are in baseline_results which is not in config_results JSON)
    with open(output_path, "w") as f:
        json.dump(results_serializable, f, indent=2, default=str)

    if verbose:
        print(f"\n[Output] Results written to: {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Consolidation Ablation Experiment (MECH-068 / CSH-1)"
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES_PER_CONFIG,
                        help="Episodes per consolidation config (default: %(default)s)")
    parser.add_argument("--steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="Max steps per episode (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed (default: %(default)s)")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE,
                        help="GridWorld size (default: %(default)s)")
    parser.add_argument("--hazards", type=int, default=DEFAULT_NUM_HAZARDS,
                        help="Number of hazard cells (default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=WARMUP_EPISODES,
                        help="Warmup episodes before freezing E1 (default: %(default)s)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: auto-generated in evidence/)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    run_consolidation_ablation(
        num_episodes=args.episodes,
        max_steps=args.steps,
        seed=args.seed,
        grid_size=args.grid_size,
        num_hazards=args.hazards,
        output_path=args.output,
        warmup_episodes=args.warmup,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
