"""
E1/E2 Associative vs Transition Knowledge Experiment (MECH-058 / EXQ-006)

Probes the core orthogonality claim about E1 and E2:

  E1 = associative engine ("register of what IS"):
    Action-unconditioned LSTM world model. Learns what latent states
    associate with across timescales — e.g., that spatial region R
    consistently produces harm regardless of path taken.

  E2 = transition model ("how things change"):
    Action-conditioned MLP: f(z_t, a_t) → z_{t+1}.
    Learns local dynamics — which action from which latent state
    leads where. Orthogonal to associative harm profile.

The two types of knowledge are ORTHOGONAL, not just timescale-shifted:
  - E1 builds a prior over "what this region IS" (affective/semantic)
  - E2 builds a prior over "how to navigate" (kinematic/transition)

Environment: MultiRoomGrid
  A 14x14 grid divided into four spatial quadrants, each with a distinct
  hazard density (room type). Room type is NOT included in the observation
  directly — the agent receives only position + local 3×3 view + homeostatic.
  E1 must accumulate temporal evidence about which regions are consistently
  dangerous; E2 must learn local movement physics (consistent across rooms).

Four experimental conditions:
  FULL        E1 lr=1e-4 (slow, associative) + E3 lr=1e-3 + E2 trains
  E1_FROZEN   E1 weights frozen (no associative learning); E2+E3 train
  E2_FROZEN   E2 weights frozen (no transition update); E1+E3 train
  SAME_RATE   E1 lr=1e-3 (same as E3 — no timescale separation, existing test)

PASS criteria:
  1. FULL last-Q harm < E1_FROZEN last-Q harm     (E1 associative prior helps)
  2. FULL last-Q harm < E2_FROZEN last-Q harm     (E2 transitions help)
  3. FULL last-Q harm <= SAME_RATE last-Q harm * 1.10  (slow E1 doesn't hurt)

All three must hold for MECH-058 PASS.

Usage:
    python experiments/e1_e2_associative_transition.py
    python experiments/e1_e2_associative_transition.py --episodes 5 --seeds 7

Claims:
    MECH-058: latent_stack.e1_e2_timescale_separation (scope: associative_vs_transition)
    EXQ-006
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


DEFAULT_EPISODES = 200
DEFAULT_MAX_STEPS = 100
DEFAULT_SEEDS = [7, 42, 99]
MAX_GRAD_NORM = 1.0

# Learning rates
FULL_E1_LR = 1e-4          # slow associative anchor
FULL_POLICY_LR = 1e-3
SAME_RATE_E1_LR = 1e-3     # no timescale separation (control)
SAME_RATE_POLICY_LR = 1e-3

# Grid parameters
GRID_SIZE = 14
# Quadrant hazard counts (one per quadrant):
#   Q1 top-left:     high hazard (dangerous)
#   Q2 top-right:    low hazard (mostly safe)
#   Q3 bottom-left:  medium hazard
#   Q4 bottom-right: medium-high hazard (trap zone)
QUADRANT_HAZARDS = [6, 1, 3, 4]  # [Q1, Q2, Q3, Q4]
QUADRANT_RESOURCES = [1, 4, 2, 2]  # compensating resources

CONDITIONS = ["FULL", "E1_FROZEN", "E2_FROZEN", "SAME_RATE"]


# ---------------------------------------------------------------------------
# MultiRoomGrid — inline environment with quadrant-based hazard clustering
# ---------------------------------------------------------------------------

class MultiRoomGrid:
    """
    14×14 grid divided into four spatial quadrants.

    Each quadrant has a fixed hazard density (room type) that is CONSISTENT
    across resets for a given seed configuration. The room type is NOT
    included in the observation — it must be inferred by the agent from
    the latent state trajectory.

    Observation includes:
      - Position one-hot (14×14 = 196 dims)
      - Local 3×3 view flattened (9 cells × 3 channels: empty/hazard/resource = 27 dims)
      - Homeostatic state (health, energy = 2 dims)

    Total observation dim: 196 + 27 + 2 = 225
    Actions: 4 cardinal directions (N, E, S, W)
    """

    ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
    ACTION_DIM = 4
    VIEW_SIZE = 3  # 3×3 local view
    VIEW_CHANNELS = 3  # empty=0, hazard=1, resource=2

    HAZARD_HARM = 0.5
    RESOURCE_BENEFIT = 0.3
    ENERGY_DECAY = 0.01
    INITIAL_HEALTH = 1.0
    INITIAL_ENERGY = 1.0

    def __init__(self, grid_size: int = GRID_SIZE, rng: Optional[np.random.Generator] = None):
        self.size = grid_size
        self.rng = rng if rng is not None else np.random.default_rng(0)

        # Mid-point divides grid into four quadrants (excluding border walls)
        # Inner cells: rows 1..size-2, cols 1..size-2
        mid = grid_size // 2
        self.mid = mid

        # Quadrant definitions: (row_min, row_max, col_min, col_max) — inclusive inner cells
        inner_max = grid_size - 2  # last inner row/col index (1-based)
        self.quadrants = [
            (1, mid - 1, 1, mid - 1),         # Q1: top-left
            (1, mid - 1, mid, inner_max),      # Q2: top-right
            (mid, inner_max, 1, mid - 1),      # Q3: bottom-left
            (mid, inner_max, mid, inner_max),  # Q4: bottom-right
        ]

        # Static hazard/resource positions (set once per seed, not per episode)
        self._hazard_positions: List[Tuple[int, int]] = []
        self._resource_positions: List[Tuple[int, int]] = []
        self._place_static_entities()

        # Episode state
        self.agent_pos = (1, 1)
        self.health = self.INITIAL_HEALTH
        self.energy = self.INITIAL_ENERGY
        self.visited_hazards: set = set()

    def _quadrant_cells(self, q_idx: int) -> List[Tuple[int, int]]:
        r0, r1, c0, c1 = self.quadrants[q_idx]
        return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    def _place_static_entities(self) -> None:
        """Place hazards and resources per quadrant. Called once per seed."""
        self._hazard_positions = []
        self._resource_positions = []
        for q_idx in range(4):
            cells = self._quadrant_cells(q_idx)
            self.rng.shuffle(cells)
            n_hazards = min(QUADRANT_HAZARDS[q_idx], len(cells))
            n_resources = min(QUADRANT_RESOURCES[q_idx], len(cells) - n_hazards)
            self._hazard_positions.extend(cells[:n_hazards])
            self._resource_positions.extend(cells[n_hazards:n_hazards + n_resources])

    @property
    def observation_dim(self) -> int:
        pos_dim = self.size * self.size
        view_dim = self.VIEW_SIZE * self.VIEW_SIZE * self.VIEW_CHANNELS
        homeo_dim = 2
        return pos_dim + view_dim + homeo_dim

    @property
    def action_dim(self) -> int:
        return self.ACTION_DIM

    def reset(self) -> torch.Tensor:
        # Randomise starting position to an inner cell not occupied by hazard/resource
        occupied = set(self._hazard_positions) | set(self._resource_positions)
        candidates = [
            (r, c)
            for r in range(1, self.size - 1)
            for c in range(1, self.size - 1)
            if (r, c) not in occupied
        ]
        if candidates:
            idx = int(self.rng.integers(len(candidates)))
            self.agent_pos = candidates[idx]
        else:
            self.agent_pos = (1, 1)

        self.health = self.INITIAL_HEALTH
        self.energy = self.INITIAL_ENERGY
        self.visited_hazards = set()
        return self._get_obs()

    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        dr, dc = self.ACTIONS[action_idx]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        # Wall check
        if 0 < nr < self.size - 1 and 0 < nc < self.size - 1:
            self.agent_pos = (nr, nc)

        harm = 0.0
        pos = self.agent_pos

        # Hazard effect (once per step at current position)
        if pos in self._hazard_positions:
            harm = self.HAZARD_HARM
            self.health -= harm

        # Resource effect
        if pos in self._resource_positions:
            self.health = min(1.0, self.health + self.RESOURCE_BENEFIT)
            self.energy = min(1.0, self.energy + self.RESOURCE_BENEFIT * 0.5)

        # Energy decay
        self.energy = max(0.0, self.energy - self.ENERGY_DECAY)

        done = self.health <= 0.0 or self.energy <= 0.0
        return self._get_obs(), -harm, done, {}

    def _get_obs(self) -> torch.Tensor:
        # Position one-hot
        r, c = self.agent_pos
        pos_vec = torch.zeros(self.size * self.size)
        pos_vec[r * self.size + c] = 1.0

        # Local 3×3 view (channels: empty, hazard, resource)
        view = torch.zeros(self.VIEW_SIZE * self.VIEW_SIZE * self.VIEW_CHANNELS)
        half = self.VIEW_SIZE // 2
        idx = 0
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                nr, nc = r + dr, c + dc
                ch = 0  # empty by default (walls also = empty channel)
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if (nr, nc) in self._hazard_positions:
                        ch = 1
                    elif (nr, nc) in self._resource_positions:
                        ch = 2
                view[idx * self.VIEW_CHANNELS + ch] = 1.0
                idx += 1

        # Homeostatic state
        homeo = torch.tensor([self.health, self.energy], dtype=torch.float32)

        return torch.cat([pos_vec, view, homeo])


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------

def make_optimizers(
    agent: REEAgent,
    e1_lr: float,
    policy_lr: float,
    freeze_e1: bool = False,
    freeze_e2: bool = False,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """
    Build E1 and policy optimizers. Optionally freeze E1 or E2 parameters.
    """
    if freeze_e1:
        for p in agent.e1.parameters():
            p.requires_grad_(False)
    if freeze_e2:
        for p in agent.e2.parameters():
            p.requires_grad_(False)

    e1_params = [
        p for p in (
            list(agent.e1.parameters())
            + list(agent.latent_stack.parameters())
            + list(agent.obs_encoder.parameters())
        )
        if p.requires_grad
    ]
    policy_params = [p for p in agent.e3.parameters() if p.requires_grad]

    e1_opt = torch.optim.Adam(e1_params, lr=e1_lr) if e1_params else torch.optim.Adam(
        [nn.Parameter(torch.zeros(1))], lr=e1_lr  # dummy to avoid empty optimizer
    )
    policy_opt = torch.optim.Adam(policy_params, lr=policy_lr) if policy_params else torch.optim.Adam(
        [nn.Parameter(torch.zeros(1))], lr=policy_lr
    )
    return e1_opt, policy_opt


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    agent: REEAgent,
    env: MultiRoomGrid,
    e1_opt: torch.optim.Optimizer,
    policy_opt: torch.optim.Optimizer,
    max_steps: int,
    freeze_e1: bool,
    freeze_e2: bool,
) -> Dict[str, Any]:
    agent.reset()
    obs = env.reset()

    log_probs: List[torch.Tensor] = []
    total_harm = 0.0
    steps = 0
    e1_loss_val = 0.0

    for _ in range(max_steps):
        obs_tensor = obs.unsqueeze(0) if obs.dim() == 1 else obs

        with torch.no_grad():
            encoded = agent.sense(obs_tensor)
            agent.update_latent(encoded)
            candidates = agent.generate_trajectories(agent._current_latent)

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
    if log_probs:
        G = float(-total_harm)
        policy_loss = -(torch.stack(log_probs) * G).mean()
        policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for grp in policy_opt.param_groups for p in grp["params"] if p.requires_grad],
            MAX_GRAD_NORM,
        )
        policy_opt.step()

    # E1 world model update (skipped if frozen)
    if not freeze_e1:
        e1_loss = agent.compute_prediction_loss()
        if e1_loss.requires_grad:
            e1_opt.zero_grad()
            e1_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for grp in e1_opt.param_groups for p in grp["params"] if p.requires_grad],
                MAX_GRAD_NORM,
            )
            e1_opt.step()
            e1_loss_val = e1_loss.item()

    return {
        "total_harm": total_harm,
        "steps": steps,
        "e1_loss": e1_loss_val,
    }


# ---------------------------------------------------------------------------
# Condition runner
# ---------------------------------------------------------------------------

def run_condition(
    seed: int,
    condition: str,
    num_episodes: int,
    max_steps: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = MultiRoomGrid(grid_size=GRID_SIZE, rng=rng)

    config = REEConfig.from_dims(env.observation_dim, env.action_dim)
    agent = REEAgent(config=config)

    freeze_e1 = condition == "E1_FROZEN"
    freeze_e2 = condition == "E2_FROZEN"

    if condition == "FULL":
        e1_lr, policy_lr = FULL_E1_LR, FULL_POLICY_LR
    elif condition == "SAME_RATE":
        e1_lr, policy_lr = SAME_RATE_E1_LR, SAME_RATE_POLICY_LR
    elif condition == "E1_FROZEN":
        e1_lr, policy_lr = FULL_E1_LR, FULL_POLICY_LR   # E1 lr unused (frozen)
    else:  # E2_FROZEN
        e1_lr, policy_lr = FULL_E1_LR, FULL_POLICY_LR

    e1_opt, policy_opt = make_optimizers(
        agent, e1_lr, policy_lr, freeze_e1=freeze_e1, freeze_e2=freeze_e2
    )

    ep_harms: List[float] = []
    ep_e1_losses: List[float] = []

    for ep in range(num_episodes):
        metrics = run_episode(
            agent, env, e1_opt, policy_opt, max_steps, freeze_e1, freeze_e2
        )
        ep_harms.append(metrics["total_harm"])
        ep_e1_losses.append(metrics["e1_loss"])

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
        "freeze_e1": freeze_e1,
        "freeze_e2": freeze_e2,
        "e1_lr": e1_lr if not freeze_e1 else 0.0,
        "policy_lr": policy_lr,
        "first_quarter_harm": round(statistics.mean(ep_harms[:quarter]), 4),
        "last_quarter_harm": round(statistics.mean(ep_harms[-quarter:]), 4),
        "mean_e1_loss": round(statistics.mean(ep_e1_losses), 6),
        "episode_count": num_episodes,
    }


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------

def run_experiment(
    num_episodes: int = DEFAULT_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    seeds: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if seeds is None:
        seeds = DEFAULT_SEEDS

    run_timestamp = datetime.now(timezone.utc).isoformat()

    if verbose:
        print("[E1/E2 Associative vs Transition — MECH-058 / EXQ-006]")
        print(f"  MultiRoomGrid {GRID_SIZE}×{GRID_SIZE}, 4 quadrants")
        print(f"  Quadrant hazards: {QUADRANT_HAZARDS}  (Q1=dangerous, Q2=safe, Q3=med, Q4=trap)")
        print(f"  Episodes: {num_episodes}  max_steps: {max_steps}  seeds: {seeds}")
        print()
        print("  Conditions:")
        print(f"    FULL      E1 lr={FULL_E1_LR}  policy lr={FULL_POLICY_LR}")
        print(f"    E1_FROZEN E1 frozen (no associative learning)")
        print(f"    E2_FROZEN E2 frozen (no transition update)")
        print(f"    SAME_RATE E1 lr={SAME_RATE_E1_LR}  policy lr={SAME_RATE_POLICY_LR}")
        print()
        print("  PASS if: FULL harm < E1_FROZEN AND < E2_FROZEN AND <= SAME_RATE*1.10")
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

    full_harm = _agg("FULL", "last_quarter_harm")
    e1f_harm = _agg("E1_FROZEN", "last_quarter_harm")
    e2f_harm = _agg("E2_FROZEN", "last_quarter_harm")
    sr_harm = _agg("SAME_RATE", "last_quarter_harm")

    assoc_ok = full_harm < e1f_harm
    trans_ok = full_harm < e2f_harm
    rate_ok = full_harm <= sr_harm * 1.10
    verdict = "PASS" if (assoc_ok and trans_ok and rate_ok) else "FAIL"

    if verbose:
        print("=" * 60)
        print("[Summary]")
        print(f"  FULL       last-Q harm: {full_harm:.4f}")
        print(f"  E1_FROZEN  last-Q harm: {e1f_harm:.4f}")
        print(f"  E2_FROZEN  last-Q harm: {e2f_harm:.4f}")
        print(f"  SAME_RATE  last-Q harm: {sr_harm:.4f}")
        print()
        print(f"  Associative criterion (FULL < E1_FROZEN)?   {'YES' if assoc_ok else 'NO'}")
        print(f"  Transition criterion  (FULL < E2_FROZEN)?   {'YES' if trans_ok else 'NO'}")
        print(f"  Rate criterion        (FULL <= SR*1.10)?    {'YES' if rate_ok else 'NO'}")
        print()
        print(f"  MECH-058 / EXQ-006 verdict: {verdict}")
        print()
        if verdict == "PASS":
            print("  Interpretation:")
            print("    Both E1 associative knowledge and E2 transition knowledge")
            print("    contribute independently to harm reduction, confirming that")
            print("    E1 and E2 carry orthogonal information types rather than")
            print("    redundant representations at different timescales.")

    result_doc = {
        "experiment": "e1_e2_associative_transition",
        "claim": "MECH-058",
        "exq_id": "EXQ-006",
        "run_timestamp": run_timestamp,
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seeds": seeds,
            "grid_size": GRID_SIZE,
            "quadrant_hazards": QUADRANT_HAZARDS,
            "quadrant_resources": QUADRANT_RESOURCES,
            "full_e1_lr": FULL_E1_LR,
            "full_policy_lr": FULL_POLICY_LR,
            "same_rate_e1_lr": SAME_RATE_E1_LR,
            "same_rate_policy_lr": SAME_RATE_POLICY_LR,
        },
        "verdict": verdict,
        "aggregate": {
            "full_last_quarter_harm": full_harm,
            "e1_frozen_last_quarter_harm": e1f_harm,
            "e2_frozen_last_quarter_harm": e2f_harm,
            "same_rate_last_quarter_harm": sr_harm,
            "associative_criterion_met": assoc_ok,
            "transition_criterion_met": trans_ok,
            "rate_criterion_met": rate_ok,
        },
        "per_run": all_results,
    }

    evidence_dir = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "experiments" / "e1_e2_associative_transition"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    save_path = output_path or str(evidence_dir / f"e1_e2_associative_transition_{ts}.json")
    with open(save_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    if verbose:
        print(f"  Results saved to: {save_path}")

    return result_doc


def main():
    parser = argparse.ArgumentParser(
        description="MECH-058 / EXQ-006: E1 Associative vs E2 Transition orthogonality"
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_experiment(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seeds=args.seeds,
        output_path=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
