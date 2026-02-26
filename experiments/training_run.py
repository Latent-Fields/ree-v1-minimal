"""
REE Training Run

Runs the REETrainer training loop and logs per-episode metrics.

This is the second genuine experiment for REE, testing whether the agent
can learn to reduce harm accumulation over time.

Expected result (ARC-015 prerequisite):
  - E1 prediction loss decreases over training (world model improves)
  - Harm per episode decreases over training (agent learns harm avoidance)

This is a discriminating test because:
  - A system where E3 consolidation weights have no effect on behaviour
    would show flat harm curves regardless of training
  - A system where E1 cannot predict state sequences would show flat or
    diverging E1 loss

Usage:
    python experiments/training_run.py
    python experiments/training_run.py --episodes 200 --seed 42
    python experiments/training_run.py --episodes 500 --log-interval 20
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent
from ree_core.environment.grid_world import GridWorld
from ree_core.training.trainer import REETrainer
from ree_core.utils.config import REEConfig


DEFAULT_EPISODES = 200
DEFAULT_MAX_STEPS = 100
DEFAULT_SEED = 7
DEFAULT_GRID_SIZE = 10
DEFAULT_NUM_HAZARDS = 4
DEFAULT_E1_LR = 1e-4
DEFAULT_POLICY_LR = 1e-3
DEFAULT_LOG_INTERVAL = 20


def _rolling_mean(values: List[float], window: int) -> List[float]:
    """Compute rolling mean over a list."""
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start : i + 1]) / (i - start + 1))
    return out


def _assess_training_trend(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Assess whether training showed meaningful improvement.

    Compares first-quarter vs last-quarter mean harm and E1 loss.
    """
    n = len(history)
    if n < 4:
        return {"verdict": "INSUFFICIENT_DATA"}

    quarter = max(1, n // 4)
    first_harm = [h["total_harm"] for h in history[:quarter]]
    last_harm = [h["total_harm"] for h in history[-quarter:]]
    mean_first_harm = sum(first_harm) / len(first_harm)
    mean_last_harm = sum(last_harm) / len(last_harm)
    harm_improved = mean_last_harm < mean_first_harm

    e1_losses = [h["e1_loss"] for h in history if "e1_loss" in h]
    e1_trend = None
    if len(e1_losses) >= 4:
        first_e1 = sum(e1_losses[:quarter]) / quarter
        last_e1 = sum(e1_losses[-quarter:]) / quarter
        e1_trend = "decreasing" if last_e1 < first_e1 else "flat_or_increasing"
        e1_improved = last_e1 < first_e1
    else:
        e1_improved = None

    verdict = "IMPROVED" if harm_improved else "FLAT_OR_DEGRADED"

    return {
        "verdict": verdict,
        "mean_harm_first_quarter": round(mean_first_harm, 4),
        "mean_harm_last_quarter": round(mean_last_harm, 4),
        "harm_reduction": round(mean_first_harm - mean_last_harm, 4),
        "harm_improved": harm_improved,
        "e1_loss_trend": e1_trend,
        "e1_improved": e1_improved,
        "interpretation": (
            "Agent reduced harm accumulation over training."
            if harm_improved
            else "No clear harm reduction — E3 policy gradient may need more episodes or a different configuration."
        ),
    }


def run_training(
    num_episodes: int = DEFAULT_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = DEFAULT_SEED,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_hazards: int = DEFAULT_NUM_HAZARDS,
    e1_lr: float = DEFAULT_E1_LR,
    policy_lr: float = DEFAULT_POLICY_LR,
    output_path: str = None,
    log_interval: int = DEFAULT_LOG_INTERVAL,
    verbose: bool = True,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    run_timestamp = datetime.now(timezone.utc).isoformat()

    env = GridWorld(size=grid_size, num_hazards=num_hazards)

    config = REEConfig.from_dims(env.observation_dim, env.action_dim)
    agent = REEAgent(config=config)

    trainer = REETrainer(
        agent,
        env,
        e1_lr=e1_lr,
        policy_lr=policy_lr,
    )

    if verbose:
        print("[REE Training Run]")
        print(f"  GridWorld: {grid_size}x{grid_size}, {num_hazards} hazards")
        print(f"  obs_dim={env.observation_dim}, action_dim={env.action_dim}")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seed: {seed}")
        print(f"  E1 LR: {e1_lr}  policy LR: {policy_lr}")
        print()

    history = trainer.train(
        num_episodes=num_episodes,
        max_steps=max_steps,
        seed=seed,
        verbose=verbose,
        log_interval=log_interval,
    )

    assessment = _assess_training_trend(history)

    if verbose:
        print()
        print(f"[Assessment] {assessment['verdict']}")
        print(f"  Harm first quarter: {assessment['mean_harm_first_quarter']:.4f}")
        print(f"  Harm last quarter:  {assessment['mean_harm_last_quarter']:.4f}")
        print(f"  Harm reduction:     {assessment['harm_reduction']:.4f}")
        if assessment.get("e1_loss_trend"):
            print(f"  E1 loss trend:      {assessment['e1_loss_trend']}")
        print(f"  {assessment['interpretation']}")

    results = {
        "experiment": "training_run",
        "claim_ids_relevant": ["ARC-015", "MECH-062"],
        "run_timestamp_utc": run_timestamp,
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seed": seed,
            "grid_size": grid_size,
            "num_hazards": num_hazards,
            "e1_lr": e1_lr,
            "policy_lr": policy_lr,
        },
        "assessment": assessment,
        "history": history,
    }

    if output_path is None:
        out_dir = Path(__file__).parent.parent / "evidence" / "experiments" / "training_run"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = run_timestamp.replace(":", "").replace("+", "").replace("-", "")[:15]
        output_path = str(out_dir / f"training_run_{ts}.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"\n[Output] Results written to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="REE Training Run")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--hazards", type=int, default=DEFAULT_NUM_HAZARDS)
    parser.add_argument("--e1-lr", type=float, default=DEFAULT_E1_LR)
    parser.add_argument("--policy-lr", type=float, default=DEFAULT_POLICY_LR)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_training(
        num_episodes=args.episodes,
        max_steps=args.steps,
        seed=args.seed,
        grid_size=args.grid_size,
        num_hazards=args.hazards,
        e1_lr=args.e1_lr,
        policy_lr=args.policy_lr,
        output_path=args.output,
        log_interval=args.log_interval,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
