"""
E1/E2 Timescale Ablation Experiment (MECH-058 / EVB-0040)

Tests whether the E1 (slow LSTM deep predictor, lr=1e-4) and E2 (fast short-horizon
rollout generator, lr=1e-3) timescale separation stabilises latent representation
compared to running both at the same (fast) learning rate.

MECH-058 claims: the E1 slow anchor is a functional necessity, not just an
optimisation choice. Separating the timescales keeps the latent basis stable while
allowing fast policy adaptation — the canonical REE "two-speed" architecture.

Conditions:
  A (SEPARATED):      E1 lr=1e-4 (slow anchor), policy/E3 lr=1e-3 (fast adaptor)
  B (SAME-TIMESCALE): E1 lr=1e-3 (same rate as policy — no slow anchor)

Key diagnostics:
  1. SEPARATED mean_latent_stability < SAME-TIMESCALE mean_latent_stability
     Lower stability value = less z_gamma drift across steps = better anchoring.
     (latent_stability: mean std of z_gamma across episode timesteps)
  2. SEPARATED last-quarter harm <= SAME-TIMESCALE * 1.05
     (practical benefit: stable anchor doesn't hurt harm avoidance)

Both must hold for MECH-058 PASS.

Usage:
    python experiments/e1_e2_timescale_ablation.py
    python experiments/e1_e2_timescale_ablation.py --episodes 5 --seeds 7

Claims:
    MECH-058: latent_stack.e1_e2_timescale_separation
    EVB-0040
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

# SEPARATED condition learning rates (canonical REE)
SEPARATED_E1_LR = 1e-4
SEPARATED_POLICY_LR = 1e-3

# SAME-TIMESCALE condition: E1 runs at same rate as policy (ablated)
SAME_TIMESCALE_E1_LR = 1e-3
SAME_TIMESCALE_POLICY_LR = 1e-3


def make_optimizers(
    agent: REEAgent,
    e1_lr: float,
    policy_lr: float,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """
    Build two optimizers matching REETrainer's parameter grouping.
    E1 group: E1 predictor + latent stack + obs encoder (slow anchor in SEPARATED).
    Policy group: E3 selector (fast adaptor).
    """
    e1_params = (
        list(agent.e1.parameters())
        + list(agent.latent_stack.parameters())
        + list(agent.obs_encoder.parameters())
    )
    policy_params = list(agent.e3.parameters())
    e1_opt = torch.optim.Adam(e1_params, lr=e1_lr)
    policy_opt = torch.optim.Adam(policy_params, lr=policy_lr)
    return e1_opt, policy_opt


def run_episode(
    agent: REEAgent,
    env: GridWorld,
    e1_opt: torch.optim.Optimizer,
    policy_opt: torch.optim.Optimizer,
    max_steps: int,
) -> Dict[str, Any]:
    """
    Run one episode with gradient updates.

    Uses low-level pipeline (sense → update_latent → generate_trajectories → select)
    to access z_gamma after each latent update for the stability metric.

    Returns per-episode metrics dict.
    """
    agent.reset()
    obs = env.reset()

    log_probs: List[torch.Tensor] = []
    z_gammas: List[torch.Tensor] = []
    total_harm = 0.0
    steps = 0

    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # Latent update inside no_grad to avoid accumulating graph through encoder
        with torch.no_grad():
            encoded = agent.sense(obs_tensor)
            agent.update_latent(encoded)
            # Capture z_gamma for latent stability metric
            z_gamma_snap = agent._current_latent.z_gamma.detach().clone()
            z_gammas.append(z_gamma_snap)
            candidates = agent.generate_trajectories(agent._current_latent)

        # E3 select outside no_grad so log_prob has grad_fn for REINFORCE
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

    # Latent stability: temporal std of z_gamma across episode steps
    latent_stability = 0.0
    if len(z_gammas) >= 2:
        z_stack = torch.stack(z_gammas).squeeze(1)  # [T, latent_dim]
        std_per_dim = z_stack.std(dim=0)             # [latent_dim], Bessel-corrected
        latent_stability = std_per_dim.mean().item()

    # Policy gradient update (REINFORCE)
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
        "latent_stability": latent_stability,
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

    if condition == "SEPARATED":
        e1_lr, policy_lr = SEPARATED_E1_LR, SEPARATED_POLICY_LR
    else:  # SAME-TIMESCALE
        e1_lr, policy_lr = SAME_TIMESCALE_E1_LR, SAME_TIMESCALE_POLICY_LR

    e1_opt, policy_opt = make_optimizers(agent, e1_lr, policy_lr)

    ep_harms: List[float] = []
    ep_stabilities: List[float] = []
    ep_e1_losses: List[float] = []

    for ep in range(num_episodes):
        metrics = run_episode(agent, env, e1_opt, policy_opt, max_steps)
        ep_harms.append(metrics["total_harm"])
        ep_stabilities.append(metrics["latent_stability"])
        ep_e1_losses.append(metrics["e1_loss"])

        if verbose and (ep + 1) % 50 == 0:
            recent_harm = statistics.mean(ep_harms[-20:])
            recent_stab = statistics.mean(ep_stabilities[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  "
                f"seed={seed}  cond={condition}  "
                f"harm={recent_harm:.3f}  stability={recent_stab:.4f}"
            )

    quarter = max(1, num_episodes // 4)

    return {
        "condition": condition,
        "seed": seed,
        "e1_lr": e1_lr,
        "policy_lr": policy_lr,
        "first_quarter_harm": round(statistics.mean(ep_harms[:quarter]), 4),
        "last_quarter_harm": round(statistics.mean(ep_harms[-quarter:]), 4),
        "mean_latent_stability": round(statistics.mean(ep_stabilities), 6),
        "last_quarter_latent_stability": round(statistics.mean(ep_stabilities[-quarter:]), 6),
        "mean_e1_loss": round(statistics.mean(ep_e1_losses), 6),
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
        print("[E1/E2 Timescale Ablation — MECH-058 / EVB-0040]")
        print(f"  GridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions:")
        print(f"    A (SEPARATED):      E1 lr={SEPARATED_E1_LR}  policy lr={SEPARATED_POLICY_LR}")
        print(f"    B (SAME-TIMESCALE): E1 lr={SAME_TIMESCALE_E1_LR}  policy lr={SAME_TIMESCALE_POLICY_LR}")
        print()
        print("  Diagnostic 1: SEPARATED mean_latent_stability < SAME-TIMESCALE")
        print("    Lower = less z_gamma drift = better slow-anchor effect")
        print("  Diagnostic 2: SEPARATED last-Q harm <= SAME-TIMESCALE * 1.05")
        print()

    all_results = []

    for seed in seeds:
        for condition in ["SEPARATED", "SAME-TIMESCALE"]:
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
                    f"{result['last_quarter_harm']:.3f}  "
                    f"stability={result['mean_latent_stability']:.5f}"
                )
                print()

    separated = [r for r in all_results if r["condition"] == "SEPARATED"]
    same_ts = [r for r in all_results if r["condition"] == "SAME-TIMESCALE"]

    def _agg(results: List[Dict], key: str) -> float:
        return round(statistics.mean(r[key] for r in results), 6)

    sep_stability = _agg(separated, "mean_latent_stability")
    sts_stability = _agg(same_ts, "mean_latent_stability")
    sep_harm_last = round(_agg(separated, "last_quarter_harm"), 4)
    sts_harm_last = round(_agg(same_ts, "last_quarter_harm"), 4)

    # PASS criteria
    stability_ok = sep_stability < sts_stability
    performance_ok = sep_harm_last <= sts_harm_last * 1.05
    verdict = "PASS" if (stability_ok and performance_ok) else "FAIL"
    partial = (stability_ok or performance_ok) and not (stability_ok and performance_ok)

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  SEPARATED      mean latent stability: {sep_stability:.6f}")
        print(f"  SAME-TIMESCALE mean latent stability: {sts_stability:.6f}")
        print(f"  SEPARATED      last-Q harm: {sep_harm_last:.3f}")
        print(f"  SAME-TIMESCALE last-Q harm: {sts_harm_last:.3f}")
        print()
        print(f"  Stability criterion (sep < same)?  {'YES' if stability_ok else 'NO'}")
        print(f"  Performance criterion (harm <=)?   {'YES' if performance_ok else 'NO'}")
        print()
        print(f"  MECH-058 verdict: {verdict}")
        if partial:
            print("  (partial — one of two criteria met)")
        print()
        if verdict == "PASS":
            print("  Interpretation:")
            print("    Slow E1 anchor (lr=1e-4) produces more stable latent representations")
            print("    than fast E1 (lr=1e-3), confirming the E1/E2 timescale separation")
            print("    is a functional necessity, not just an optimisation detail.")

    result_doc = {
        "experiment": "e1_e2_timescale_ablation",
        "claim": "MECH-058",
        "evb_id": "EVB-0040",
        "run_timestamp": run_timestamp,
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seeds": seeds,
            "grid_size": grid_size,
            "num_hazards": num_hazards,
            "separated_e1_lr": SEPARATED_E1_LR,
            "separated_policy_lr": SEPARATED_POLICY_LR,
            "same_timescale_e1_lr": SAME_TIMESCALE_E1_LR,
            "same_timescale_policy_lr": SAME_TIMESCALE_POLICY_LR,
        },
        "verdict": verdict,
        "partial_support": partial,
        "aggregate": {
            "separated_mean_latent_stability": sep_stability,
            "same_timescale_mean_latent_stability": sts_stability,
            "separated_harm_last_quarter": sep_harm_last,
            "same_timescale_harm_last_quarter": sts_harm_last,
            "harm_tolerance_factor": 1.05,
            "stability_criterion_met": stability_ok,
            "performance_criterion_met": performance_ok,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "e1_e2_timescale_ablation"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(
        evidence_dir / f"e1_e2_timescale_ablation_{ts}.json"
    )
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main():
    parser = argparse.ArgumentParser(
        description="MECH-058: E1/E2 Timescale Ablation experiment"
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
