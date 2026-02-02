"""Predictor modules for REE (E1 and E2)."""

from ree_core.predictors.e1_deep import E1DeepPredictor
from ree_core.predictors.e2_fast import E2FastPredictor

__all__ = ["E1DeepPredictor", "E2FastPredictor"]
