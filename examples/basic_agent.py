#!/usr/bin/env python3
"""
Basic REE Agent Example

Demonstrates the core REE agent loop:
1. Create environment and agent
2. Run episodes with the agent
3. Observe residue accumulation from harm
4. Periodic offline integration

This example shows:
- How to initialize the REE agent
- The agent's action selection process
- Residue accumulation when harm occurs
- Basic episode statistics
"""

import torch
from ree_core import REEAgent, GridWorld
from ree_core.agent import run_episode


def main():
    print("=" * 60)
    print("REE-v1 Minimal: Basic Agent Example")
    print("=" * 60)

    # Create environment
    print("\n1. Creating Grid World environment...")
    env = GridWorld(
        size=10,
        num_resources=5,
        num_hazards=3,
        num_other_agents=1,
        seed=42
    )
    print(f"   Environment: {env.size}x{env.size} grid")
    print(f"   Observation dim: {env.observation_dim}")
    print(f"   Action dim: {env.action_dim}")

    # Create REE agent
    print("\n2. Creating REE Agent...")
    agent = REEAgent.from_config(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        latent_dim=64
    )
    print(f"   Latent dim: 64")
    print(f"   Components: LatentStack, E1, E2, E3, ResidueField")

    # Run multiple episodes
    print("\n3. Running episodes...")
    num_episodes = 5

    for episode in range(num_episodes):
        print(f"\n   Episode {episode + 1}/{num_episodes}")

        # Run episode
        stats = run_episode(agent, env, max_steps=200, render=False)

        # Report statistics
        print(f"      Steps: {stats['steps']}")
        print(f"      Total harm: {stats['total_harm']:.3f}")
        print(f"      Total reward: {stats['total_reward']:.3f}")
        print(f"      Final health: {stats['final_health']:.3f}")
        print(f"      Final energy: {stats['final_energy']:.3f}")
        print(f"      Accumulated residue: {stats['total_residue']:.3f}")
        print(f"      Harm events: {stats['num_harm_events']}")

    # Show final residue state
    print("\n4. Final Residue Field Statistics:")
    residue_stats = agent.get_residue_statistics()
    print(f"   Total residue accumulated: {residue_stats['total_residue'].item():.3f}")
    print(f"   Number of harm events: {residue_stats['num_harm_events'].item()}")
    print(f"   Active RBF centers: {residue_stats['active_centers'].item()}")

    # Demonstrate that residue persists across resets
    print("\n5. Demonstrating residue persistence...")
    residue_before = residue_stats['total_residue'].item()
    agent.reset()  # Reset agent for new episode
    residue_after = agent.get_residue_statistics()['total_residue'].item()
    print(f"   Residue before reset: {residue_before:.3f}")
    print(f"   Residue after reset: {residue_after:.3f}")
    print(f"   Residue preserved: {residue_before == residue_after}")

    # Show commitment state
    print("\n6. Agent Commitment State:")
    state = agent.get_state()
    print(f"   Current precision: {state.precision:.3f}")
    print(f"   Committed: {state.is_committed}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
