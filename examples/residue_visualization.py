#!/usr/bin/env python3
"""
Residue Field Visualization Example

Demonstrates the residue field as persistent ethical geometry:
1. Accumulate residue from multiple harm events
2. Visualize the 2D field projection
3. Show how trajectories through residue regions cost more

Requires matplotlib: pip install matplotlib
"""

import torch
import numpy as np

from ree_core import ResidueField
from ree_core.utils.config import ResidueConfig


def main():
    print("=" * 60)
    print("REE-v1 Minimal: Residue Field Visualization")
    print("=" * 60)

    # Create residue field
    print("\n1. Creating residue field...")
    config = ResidueConfig(
        latent_dim=32,
        num_basis_functions=16,
        kernel_bandwidth=1.0,
        accumulation_rate=0.1
    )
    field = ResidueField(config)

    # Simulate harm events at specific locations
    print("\n2. Simulating harm events...")
    harm_locations = [
        torch.zeros(32),                      # Origin
        torch.zeros(32) + 1.0,                # Offset
        torch.randn(32) * 0.5,                # Random near origin
        torch.randn(32) * 0.5 + 2.0,          # Random, offset
    ]

    for i, loc in enumerate(harm_locations):
        field.accumulate(loc.unsqueeze(0), harm_magnitude=1.0)
        stats = field.get_statistics()
        print(f"   Event {i+1}: Total residue = {stats['total_residue'].item():.3f}")

    # Visualize the field
    print("\n3. Generating field visualization...")
    X, Y, values = field.visualize_field(
        z_range=(-3, 3),
        resolution=50,
        slice_dims=(0, 1)
    )

    # Print text visualization
    print("\n   2D projection of residue field (dimensions 0 and 1):")
    print("   (Higher values = more residue = higher ethical cost)")

    # Create ASCII visualization
    values_np = values.numpy()
    max_val = values_np.max()
    min_val = values_np.min()

    if max_val > min_val:
        normalized = (values_np - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(values_np)

    # Downsample for ASCII display
    display_size = 20
    step = values_np.shape[0] // display_size

    chars = " .:-=+*#%@"
    print("\n   " + "-" * (display_size + 2))
    for i in range(0, values_np.shape[0], step):
        row = "   |"
        for j in range(0, values_np.shape[1], step):
            val = normalized[i, j]
            char_idx = min(int(val * (len(chars) - 1)), len(chars) - 1)
            row += chars[char_idx]
        row += "|"
        print(row)
    print("   " + "-" * (display_size + 2))

    # Compare trajectory costs
    print("\n4. Comparing trajectory costs...")

    # Trajectory through high-residue region
    traj_through = torch.zeros(1, 10, 32)  # Near origin (harm location)
    cost_through = field.evaluate_trajectory(traj_through)

    # Trajectory avoiding residue
    traj_avoid = torch.ones(1, 10, 32) * 5  # Far from harm locations
    cost_avoid = field.evaluate_trajectory(traj_avoid)

    print(f"   Trajectory through harm region: cost = {cost_through.item():.3f}")
    print(f"   Trajectory avoiding harm: cost = {cost_avoid.item():.3f}")
    print(f"   Path through harm is {cost_through.item() / max(cost_avoid.item(), 0.001):.1f}x more costly")

    # Demonstrate integration
    print("\n5. Offline integration (contextualization)...")
    total_before = field.get_statistics()['total_residue'].item()
    integration_metrics = field.integrate(num_steps=100)
    total_after = field.get_statistics()['total_residue'].item()

    print(f"   Integration loss: {integration_metrics['integration_loss']:.4f}")
    print(f"   Residue before: {total_before:.3f}")
    print(f"   Residue after: {total_after:.3f}")
    print(f"   Residue preserved: {total_after >= total_before}")

    # Summary
    print("\n" + "=" * 60)
    print("Key observations:")
    print("  - Residue accumulates at harm locations")
    print("  - Residue creates a 'cost landscape' in latent space")
    print("  - Trajectories through harm regions cost more")
    print("  - Integration contextualizes but never removes residue")
    print("=" * 60)


# Optional: Save matplotlib visualization if available
def save_matplotlib_visualization(field, filename="residue_field.png"):
    """Save residue field visualization using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        X, Y, values = field.visualize_field(resolution=100)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        contour = ax.contourf(
            X.numpy(), Y.numpy(), values.numpy(),
            levels=50, cmap=cm.viridis
        )
        fig.colorbar(contour, ax=ax, label="Residue Cost")

        ax.set_xlabel("Latent Dimension 0")
        ax.set_ylabel("Latent Dimension 1")
        ax.set_title("Residue Field φ(z) - Ethical Cost Landscape")

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"\n   Saved visualization to: {filename}")

    except ImportError:
        print("\n   (Install matplotlib for graphical visualization)")


if __name__ == "__main__":
    main()
