"""
REE-v1-Minimal: Reflective-Ethical Engine Reference Implementation

A minimal implementation of the REE architecture demonstrating:
- Multi-timescale latent stack (L-space)
- E1/E2 predictors for world modelling
- E3 trajectory selection with ethical scoring
- Persistent residue field for moral continuity
"""

from ree_core.agent import REEAgent
from ree_core.latent.stack import LatentStack
from ree_core.predictors.e1_deep import E1DeepPredictor
from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.trajectory.e3_selector import E3TrajectorySelector
from ree_core.residue.field import ResidueField
from ree_core.environment.grid_world import GridWorld

__version__ = "0.1.0"

__all__ = [
    "REEAgent",
    "LatentStack",
    "E1DeepPredictor",
    "E2FastPredictor",
    "E3TrajectorySelector",
    "ResidueField",
    "GridWorld",
]
