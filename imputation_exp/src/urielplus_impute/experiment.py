from __future__ import annotations

from .feature_types import FEATURE_TYPES


DEFAULT_REGIMES = (
    "mcar",
    "empirical_copy",
    *(f"Local_block_{feature_type}" for feature_type in FEATURE_TYPES),
    *(f"Global_block_{feature_type}" for feature_type in FEATURE_TYPES),
)

SIMPLE_MODELS = ("logistic_regression", "decision_tree")
FULL_PREDICTOR_SET = "full"
