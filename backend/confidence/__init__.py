"""
SatQuery AI Confidence Scoring & Calibration Subsystem.
Config-driven multi-factor confidence combining token probabilities,
cross-tool spatial agreement, and sensor/input quality metrics.
"""
from .scorer import ConfidenceScorer, get_confidence_scorer
from .calibration import compute_ece, evaluate_calibration

__all__ = ["ConfidenceScorer", "get_confidence_scorer", "compute_ece", "evaluate_calibration"]
