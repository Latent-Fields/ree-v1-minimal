"""
Write-Locus Contamination Ablation (MECH-060 / MECH-067 / EVB-0043)

Tests whether write-locus separation between the pre-commit and post-commit
channels is load-bearing for attribution reliability and residue calibration.

MECH-060 asserts: dual error channels (sim_error pre-commit vs realized_error
post-commit) must route to separate write loci. E1 (world model) should only
be updated on actual observations; the residue field should only accumulate
from post-commit harm. Contaminating either locus degrades attribution
reliability and miscalibrates the residue terrain.

Three conditions:
  FULL:
    Clean write loci — current agent by default.
    E1 trains only on actual observations via compute_prediction_loss().
    Residue accumulates only via update_residue() with post-commit env harm.

  CONTAMINATED_DURABLE:
    Pre-commit E2 harm predictions also distort E1 weights.
    Implemented as: E1 loss is scaled by (1 + CONTAM_E1_WEIGHT * e2_pred_error)
    each episode, where e2_pred_error is the mean |E2_pred - actual_harm|.
    Models: E2's pre-commit speculation leaks into the durable world model's
    training signal, decoupling E1 loss from actual observations.

  CONTAMINATED_RESIDUE:
    E2 harm predictions write residue pre-commit (before env.step).
    Implemented as: agent.update_residue(-e2_pred) called at pre-commit,
    then agent.update_residue(env_harm) called post-commit as normal.
    Models: pre-commit speculation pollutes the residue terrain — harm is
    attributed to latent states the agent has not yet visited.

PASS criteria:
  1. Residue inflation: CONTAMINATED_RESIDUE total residue accumulated across
     all episodes > FULL total residue * RESIDUE_CRITERION_FACTOR (1.1).
     Should be reliably detectable: pre-commit + post-commit double-accumulation.
  2. Harm ordering: FULL last-quarter harm <= max(CONT_DUR, CONT_RES)
     last-quarter harm * HARM_TOLERANCE_FACTOR (1.05).
     Tests that write-locus separation is not worse than contaminated conditions.

Both criteria must hold for PASS. One criterion met = partial FAIL.

Additional diagnostics (computed and stored, not PASS criteria at this scale):
  attr_corr: Pearson(E1_loss_sequence, harm_sequence) per condition.
  Higher |corr| in FULL = E1 loss reliably tracks actual harm (attribution works).
  Expected direction: FULL |corr| > CONTAMINATED |corr|. May not be significant
  at 200-episode scale but is tracked for future calibration.

Note on E2 predictions: trajectory.harm_predictions is Tensor[batch, horizon]
in [0, 1] (Sigmoid output). We take the mean as a scalar pre-commit harm estimate.
A value of 0.2 = E2 predicts moderate harm on the selected trajectory.

Calibration: 3 seeds x 3 conditions x 200 episodes x 100 max_steps.
Estimated runtime: ~22.5 min on Apple M2 Air 8 GB (750 ms/episode-condition).

Usage:
    python experiments/write_locus_contamination.py
    python experiments/write_locus_contamination.py --episodes 5 --seeds 7

Claims:
    MECH-060: commitment.dual_error_channels_pre_post_commit
    MECH-067: agency.write_locus_permission_matrix
    EVB-0043
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
E1_LR = 1e-4
POLICY_LR = 1e-3

# CONTAMINATED_DURABLE: E1 loss scaled by (1 + CONTAM_E1_WEIGHT * pred_error)
# Typical e2_pred_error ~0.1–0.3 → scale factor 1.3–1.9.
CONTAM_E1_WEIGHT = 3.0

# Criterion 1: CONTAMINATED_RESIDUE total residue > FULL total * factor
RESIDUE_CRITERION_FACTOR = 1.1

# Criterion 2: FULL last-Q harm <= max(contaminated) * factor
HARM_TOLERANCE_FACTOR = 1.05


def pearson_corr(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 4:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


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


def get_e2_predicted_harm(trajectory) -> float:
    """
    Extract E2's harm prediction for the selected trajectory.

    trajectory.harm_predictions: Tensor[batch, horizon] in [0,1] (Sigmoid).
    Returns mean over batch and horizon. Returns 0.0 if unavailable.
    """
    if trajectory is None or trajectory.harm_predictions is None:
        return 0.0
    return float(trajectory.harm_predictions.mean().item())


def run_episode(
    agent: REEAgent,
    env: GridWorld,
    e1_opt: torch.optim.Optimizer,
    policy_opt: torch.optim.Optimizer,
    condition: str,
    max_steps: int,
) -> Dict[str, Any]:
    """
    Run one episode under the specified write-locus condition.

    FULL:                  Standard pipeline — post-commit updates only.
    CONTAMINATED_DURABLE:  E1 loss scaled by E2 pred error each episode.
    CONTAMINATED_RESIDUE:  update_residue called pre-commit with E2 prediction,
                           then again post-commit with actual env harm.

    Tracks E1 loss, actual harm, and total residue added per episode.
    """
    agent.reset()
    obs = env.reset()

    log_probs: List[torch.Tensor] = []
    step_actual_harms: List[float] = []
    step_e2_preds: List[float] = []
    committed_count = 0
    total_actual_harm = 0.0
    # Track ALL harm magnitudes passed to update_residue (includes pre-commit in CONT_RES)
    total_residue_added = 0.0
    steps = 0

    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # Encode + latent update inside no_grad (matches commitment_boundary_validation)
        with torch.no_grad():
            encoded = agent.sense(obs_tensor)
            agent.update_latent(encoded)
            candidates = agent.generate_trajectories(agent._current_latent)

        # E3 select outside no_grad so log_prob retains grad_fn
        result = agent.e3.select(candidates)
        if result.log_prob is not None:
            log_probs.append(result.log_prob)
        if result.committed:
            committed_count += 1

        # Pre-commit: read E2's harm prediction for selected trajectory
        e2_pred = get_e2_predicted_harm(result.selected_trajectory)
        step_e2_preds.append(e2_pred)

        # CONTAMINATED_RESIDUE: write residue pre-commit from E2 prediction
        if condition == "CONTAMINATED_RESIDUE" and e2_pred > 0.0:
            # update_residue expects negative harm_signal to trigger accumulation
            agent.update_residue(-e2_pred)
            total_residue_added += e2_pred  # track pre-commit contribution

        # Post-commit: execute in environment (all conditions)
        action_idx = result.selected_action.argmax(dim=-1).item()
        next_obs, harm, done, _info = env.step(action_idx)

        actual_harm = abs(harm) if harm < 0 else 0.0
        step_actual_harms.append(actual_harm)

        # Post-commit residue update (all conditions)
        agent.update_residue(harm)
        if actual_harm > 0.0:
            total_residue_added += actual_harm  # track post-commit contribution

        total_actual_harm += actual_harm
        obs = next_obs
        steps += 1
        if done:
            break

    # --- Policy gradient update ---
    # All conditions: use only realized post-commit harm as the return signal.
    # (Write-locus contamination is in E1 and residue; policy signal stays clean
    # across all three conditions so harm-learning differences reflect E1/residue effects.)
    policy_loss_val = 0.0
    if log_probs:
        G = float(-total_actual_harm)
        policy_loss = -(torch.stack(log_probs) * G).mean()
        policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for grp in policy_opt.param_groups for p in grp["params"]],
            MAX_GRAD_NORM,
        )
        policy_opt.step()
        policy_loss_val = policy_loss.item()

    # --- E1 update ---
    e1_loss_val = 0.0
    e1_loss = agent.compute_prediction_loss()

    if condition == "CONTAMINATED_DURABLE" and step_e2_preds and step_actual_harms:
        # Contamination: E1 gradient is scaled by the E2 prediction error magnitude.
        # Models: pre-commit E2 speculation leaks into E1's update, causing E1 to
        # over-react in episodes where E2 predictions diverge from actual harm.
        # When E2 over-predicts harm (pred_error large), E1 gets an outsized update
        # that decouples E1 loss from the actual observation signal.
        e2_pred_error = statistics.mean(
            abs(p - a) for p, a in zip(step_e2_preds, step_actual_harms)
        )
        e1_loss = e1_loss * (1.0 + CONTAM_E1_WEIGHT * e2_pred_error)

    if e1_loss.requires_grad:
        e1_opt.zero_grad()
        e1_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for grp in e1_opt.param_groups for p in grp["params"]],
            MAX_GRAD_NORM,
        )
        e1_opt.step()
        e1_loss_val = float(e1_loss.item())

    mean_actual = statistics.mean(step_actual_harms) if step_actual_harms else 0.0
    mean_e2_pred = statistics.mean(step_e2_preds) if step_e2_preds else 0.0
    commit_rate = committed_count / steps if steps > 0 else 0.0

    return {
        "total_harm": total_actual_harm,
        "total_residue_added": total_residue_added,
        "steps": steps,
        "mean_actual_harm": mean_actual,
        "mean_e2_predicted_harm": mean_e2_pred,
        "e1_loss": e1_loss_val,
        "policy_loss": policy_loss_val,
        "commit_rate": commit_rate,
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
    ep_residue_totals: List[float] = []

    for ep in range(num_episodes):
        metrics = run_episode(agent, env, e1_opt, policy_opt, condition, max_steps)
        ep_harms.append(metrics["total_harm"])
        ep_e1_losses.append(metrics["e1_loss"])
        ep_residue_totals.append(metrics["total_residue_added"])

        if verbose and (ep + 1) % 50 == 0:
            recent_harm = statistics.mean(ep_harms[-20:])
            recent_residue = statistics.mean(ep_residue_totals[-20:])
            print(
                f"    ep {ep+1:3d}/{num_episodes}  "
                f"seed={seed}  cond={condition}  "
                f"harm={recent_harm:.3f}  residue={recent_residue:.3f}  "
                f"e1_loss={ep_e1_losses[-1]:.4f}"
            )

    quarter = max(1, num_episodes // 4)
    attr_corr = pearson_corr(ep_e1_losses, ep_harms)
    total_residue = sum(ep_residue_totals)

    return {
        "condition": condition,
        "seed": seed,
        "first_quarter_harm": round(statistics.mean(ep_harms[:quarter]), 4),
        "last_quarter_harm": round(statistics.mean(ep_harms[-quarter:]), 4),
        "total_residue_added": round(total_residue, 4),
        "mean_residue_per_episode": round(statistics.mean(ep_residue_totals), 6),
        "attr_corr": round(attr_corr, 4),
        "abs_attr_corr": round(abs(attr_corr), 4),
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
        print("[Write-Locus Contamination Ablation — MECH-060/067 / EVB-0043]")
        print(f"  GridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions:")
        print("    FULL:                 Clean write loci (default agent)")
        print("    CONTAMINATED_DURABLE: E2 pred error scales E1 gradient (pre→durable leak)")
        print("    CONTAMINATED_RESIDUE: E2 pred writes residue pre-commit (pre→residue leak)")
        print()
        print("  Criterion 1: CONTAMINATED_RESIDUE total residue > FULL * 1.1")
        print("  Criterion 2: FULL last-Q harm <= max(CONT_DUR, CONT_RES) last-Q harm * 1.05")
        print()

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        for condition in ["FULL", "CONTAMINATED_DURABLE", "CONTAMINATED_RESIDUE"]:
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
                    f"    harm {result['first_quarter_harm']:.3f}"
                    f"→{result['last_quarter_harm']:.3f}  "
                    f"attr_corr={result['attr_corr']:.3f}  "
                    f"residue={result['total_residue_added']:.3f}"
                )
                print()

    full_results = [r for r in all_results if r["condition"] == "FULL"]
    cont_dur_results = [r for r in all_results if r["condition"] == "CONTAMINATED_DURABLE"]
    cont_res_results = [r for r in all_results if r["condition"] == "CONTAMINATED_RESIDUE"]

    def _agg(results: List[Dict], key: str) -> float:
        return round(statistics.mean(r[key] for r in results), 4)

    full_attr_corr     = _agg(full_results,     "abs_attr_corr")
    cont_dur_attr_corr = _agg(cont_dur_results, "abs_attr_corr")
    cont_res_attr_corr = _agg(cont_res_results, "abs_attr_corr")

    full_residue     = _agg(full_results,     "total_residue_added")
    cont_res_residue = _agg(cont_res_results, "total_residue_added")

    full_harm_last     = _agg(full_results,     "last_quarter_harm")
    cont_dur_harm_last = _agg(cont_dur_results, "last_quarter_harm")
    cont_res_harm_last = _agg(cont_res_results, "last_quarter_harm")

    # --- PASS criteria ---
    # Criterion 1: Pre-commit residue contamination inflates total residue
    residue_criterion = cont_res_residue > full_residue * RESIDUE_CRITERION_FACTOR

    # Criterion 2: Write-locus separation is not worse than contamination
    #   FULL last-Q harm <= max(contaminated) last-Q harm * HARM_TOLERANCE_FACTOR
    max_cont_harm = max(cont_dur_harm_last, cont_res_harm_last)
    harm_criterion = full_harm_last <= max_cont_harm * HARM_TOLERANCE_FACTOR

    verdict = "PASS" if (residue_criterion and harm_criterion) else "FAIL"
    partial = (residue_criterion or harm_criterion) and not (residue_criterion and harm_criterion)

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  FULL           last-Q harm: {full_harm_last:.4f}  |attr_corr|: {full_attr_corr:.4f}  residue: {full_residue:.4f}")
        print(f"  CONT_DURABLE   last-Q harm: {cont_dur_harm_last:.4f}  |attr_corr|: {cont_dur_attr_corr:.4f}")
        print(f"  CONT_RESIDUE   last-Q harm: {cont_res_harm_last:.4f}  |attr_corr|: {cont_res_attr_corr:.4f}  residue: {cont_res_residue:.4f}")
        print()
        print(
            f"  Criterion 1 (residue inflation):  {'YES' if residue_criterion else 'NO'}  "
            f"(cont_res={cont_res_residue:.3f} > full*1.1={full_residue * RESIDUE_CRITERION_FACTOR:.3f}?)"
        )
        print(
            f"  Criterion 2 (harm ordering):      {'YES' if harm_criterion else 'NO'}  "
            f"(full={full_harm_last:.3f} <= max_cont={max_cont_harm:.3f}*1.05={max_cont_harm * HARM_TOLERANCE_FACTOR:.3f}?)"
        )
        print()
        print(f"  MECH-060/067 verdict: {verdict}")
        if partial:
            print("  (partial — one of two criteria met)")
        print()
        if verdict == "PASS":
            print("  Interpretation:")
            print("    Write-locus contamination is detectable: pre-commit residue pollution")
            print("    inflates the residue field, and separation of write loci is at least")
            print("    as good for harm reduction as contaminated conditions.")

    result_doc = {
        "experiment": "write_locus_contamination",
        "claim": "MECH-060",
        "claim_ids_tested": ["MECH-060", "MECH-067"],
        "evb_id": "EVB-0043",
        "run_timestamp": run_timestamp,
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seeds": seeds,
            "grid_size": grid_size,
            "num_hazards": num_hazards,
            "contam_e1_weight": CONTAM_E1_WEIGHT,
            "residue_criterion_factor": RESIDUE_CRITERION_FACTOR,
            "harm_tolerance_factor": HARM_TOLERANCE_FACTOR,
            "e1_lr": E1_LR,
            "policy_lr": POLICY_LR,
        },
        "verdict": verdict,
        "partial_support": partial,
        "aggregate": {
            "full_last_quarter_harm": full_harm_last,
            "contaminated_durable_last_quarter_harm": cont_dur_harm_last,
            "contaminated_residue_last_quarter_harm": cont_res_harm_last,
            "full_total_residue": full_residue,
            "contaminated_residue_total_residue": cont_res_residue,
            "full_abs_attr_corr": full_attr_corr,
            "contaminated_durable_abs_attr_corr": cont_dur_attr_corr,
            "contaminated_residue_abs_attr_corr": cont_res_attr_corr,
            "residue_criterion_factor": RESIDUE_CRITERION_FACTOR,
            "harm_tolerance_factor": HARM_TOLERANCE_FACTOR,
            "residue_criterion_met": residue_criterion,
            "harm_criterion_met": harm_criterion,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "write_locus_contamination"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(
        evidence_dir / f"write_locus_contamination_{ts}.json"
    )
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main():
    parser = argparse.ArgumentParser(
        description="MECH-060/067: Write-Locus Contamination Ablation (EVB-0043)"
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help=f"Episodes per condition (default: {DEFAULT_EPISODES})")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help=f"Max steps per episode (default: {DEFAULT_MAX_STEPS})")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help=f"Random seeds (default: {DEFAULT_SEEDS})")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE,
                        help=f"GridWorld size (default: {DEFAULT_GRID_SIZE})")
    parser.add_argument("--num-hazards", type=int, default=DEFAULT_NUM_HAZARDS,
                        help=f"Number of hazards (default: {DEFAULT_NUM_HAZARDS})")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output path for result JSON")
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
