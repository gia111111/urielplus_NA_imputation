from __future__ import annotations

DEFAULT_SPLIT_MANIFEST = (
    "imputation_exp/runs/stratified_splits/manifests/split_manifest.csv"
)

DEFAULT_REGIMES = (
    "mcar",
    "resource_copy",
)

SIMPLE_MODELS = ("logistic_regression", "decision_tree")
FULL_PREDICTOR_SET = "full"
