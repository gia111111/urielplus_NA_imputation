from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_types import FEATURE_TYPES


METRIC_COLUMNS = ["rmse", "macro_f1", "accuracy", "p", "r"]
SUMMARY_METRIC_COLUMNS = [
    "rmse",
    "rmse_std",
    "macro_f1",
    "macro_f1_std",
    "accuracy",
    "accuracy_std",
    "p",
    "p_std",
    "r",
    "r_std",
]


def _binary_values(y_true) -> np.ndarray:
    values = np.asarray(y_true, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("y_true contains non-finite values.")
    rounded = np.rint(values).astype(int)
    if not np.allclose(values, rounded) or not np.all(np.isin(rounded, [0, 1])):
        unique = np.unique(values)
        raise ValueError(f"y_true must contain only binary 0/1 values; got {unique[:10]}.")
    return rounded


def _precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: int,
) -> tuple[float, float, float]:
    true_positive = float(np.sum((y_true == label) & (y_pred == label)))
    false_positive = float(np.sum((y_true != label) & (y_pred == label)))
    false_negative = float(np.sum((y_true == label) & (y_pred != label)))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def compute_binary_imputation_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the shared binary-imputation metrics from probability-like scores.

    Scores are clipped to [0, 1] before RMSE and thresholding. Classification
    metrics use a fixed threshold, which is 0.5 throughout the experiment.
    Precision, recall, and F1 use the conventional macro average over class
    labels present in the true or thresholded predictions.
    """
    true = _binary_values(y_true)
    score = np.asarray(y_score, dtype=float).reshape(-1)
    if len(true) != len(score):
        raise ValueError(f"y_true has {len(true)} values but y_score has {len(score)}.")
    if len(true) == 0:
        raise ValueError("Cannot compute metrics for zero scored cells.")
    if not np.all(np.isfinite(score)):
        raise ValueError("y_score contains non-finite values.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1]; got {threshold}.")

    score = np.clip(score, 0.0, 1.0)
    predicted = (score >= threshold).astype(int)
    labels = np.unique(np.concatenate([true, predicted]))
    per_label = [
        _precision_recall_f1(true, predicted, int(label))
        for label in labels
    ]
    return {
        "rmse": float(np.sqrt(np.mean((score - true) ** 2))),
        "macro_f1": float(np.mean([values[2] for values in per_label])),
        "accuracy": float(np.mean(predicted == true)),
        "p": float(np.mean([values[0] for values in per_label])),
        "r": float(np.mean([values[1] for values in per_label])),
    }


def compute_stratified_metrics(
    y_true,
    y_score,
    feature_types_for_cells,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute shared metrics separately for each present S/P/M/INV group."""
    true = _binary_values(y_true)
    score = np.asarray(y_score, dtype=float).reshape(-1)
    feature_types = np.asarray(feature_types_for_cells, dtype=str).reshape(-1)
    if len(true) != len(score) or len(true) != len(feature_types):
        raise ValueError(
            "y_true, y_score, and feature_types_for_cells must have equal lengths; "
            f"got {len(true)}, {len(score)}, and {len(feature_types)}."
        )
    unsupported = sorted(set(feature_types).difference(FEATURE_TYPES))
    if unsupported:
        raise ValueError(
            f"Unsupported feature types in scored cells: {unsupported}; "
            f"expected only {FEATURE_TYPES}."
        )

    rows = []
    present = [target_type for target_type in FEATURE_TYPES if np.any(feature_types == target_type)]
    for target_type in present:
        selected = feature_types == target_type
        rows.append(
            {
                "feature_type": target_type,
                "n": int(selected.sum()),
                **compute_binary_imputation_metrics(
                    true[selected],
                    score[selected],
                    threshold=threshold,
                ),
            }
        )
    result = pd.DataFrame(rows, columns=["feature_type", "n", *METRIC_COLUMNS])
    if set(result["feature_type"]) != set(present):
        raise AssertionError(
            f"Stratified metrics groups {sorted(result['feature_type'].tolist())} "
            f"do not match present scored groups {present}."
        )
    return result


def summarize_metrics_across_seeds(
    seed_metrics: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Average shared metrics across seeds using the standard output schema."""
    missing = set(METRIC_COLUMNS).difference(seed_metrics.columns)
    if missing:
        raise ValueError(f"Seed metrics are missing columns: {sorted(missing)}")
    groups = [column for column in group_columns if column in seed_metrics.columns]
    if not groups:
        raise ValueError("At least one summary grouping column is required.")
    grouped = seed_metrics.groupby(groups, dropna=False)
    summary = grouped[METRIC_COLUMNS].mean().reset_index()
    standard_deviation = grouped[METRIC_COLUMNS].std().reset_index()
    standard_deviation = standard_deviation.rename(
        columns={metric: f"{metric}_std" for metric in METRIC_COLUMNS}
    )
    summary = summary.merge(standard_deviation, on=groups, how="left")
    return summary[groups + SUMMARY_METRIC_COLUMNS]
