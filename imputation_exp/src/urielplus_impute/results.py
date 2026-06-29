from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import METRIC_COLUMNS, SUMMARY_METRIC_COLUMNS, summarize_metrics_across_seeds


RESULT_ID_COLUMNS = ["model", "variant", "regime", "seed", "split"]
SUMMARY_ID_COLUMNS = ["model", "variant", "regime", "split"]
STRATIFIED_SUMMARY_ID_COLUMNS = [
    "model",
    "variant",
    "regime",
    "split",
    "feature_type",
]


def _ordered_seed_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    identifiers = [*RESULT_ID_COLUMNS, "feature_type"]
    leading = [column for column in identifiers if column in frame.columns]
    standard = [column for column in ["n", *METRIC_COLUMNS] if column in frame.columns]
    remaining = [
        column for column in frame.columns if column not in set(leading + standard)
    ]
    return frame[leading + standard + remaining]


def _validate_summary_columns(summary: pd.DataFrame, id_columns: list[str]) -> None:
    present_ids = [column for column in id_columns if column in summary.columns]
    expected = present_ids + SUMMARY_METRIC_COLUMNS
    if list(summary.columns) != expected:
        raise AssertionError(
            f"Summary schema mismatch. Expected {expected}, got {list(summary.columns)}."
        )


def write_result_tables(
    seed_metrics: pd.DataFrame,
    stratified_seed_metrics: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the same raw/overall/feature-type result schema for every runner."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    overall_seed_metrics = seed_metrics.copy()
    overall_seed_metrics["feature_type"] = "ALL"
    overall = summarize_metrics_across_seeds(
        overall_seed_metrics,
        SUMMARY_ID_COLUMNS,
    )
    _validate_summary_columns(overall, SUMMARY_ID_COLUMNS)
    overall_path = output_dir / "overall_metrics_summary.csv"
    overall.to_csv(overall_path, index=False)
    outputs["overall_metrics_summary"] = overall_path

    if stratified_seed_metrics.empty:
        stratified_seed_metrics = pd.DataFrame(
            columns=[*RESULT_ID_COLUMNS, "feature_type", "n", *METRIC_COLUMNS]
        )
        feature_summary = pd.DataFrame(
            columns=[*STRATIFIED_SUMMARY_ID_COLUMNS, *SUMMARY_METRIC_COLUMNS]
        )
    else:
        feature_summary = summarize_metrics_across_seeds(
            stratified_seed_metrics,
            STRATIFIED_SUMMARY_ID_COLUMNS,
        )
        _validate_summary_columns(feature_summary, STRATIFIED_SUMMARY_ID_COLUMNS)

    combined_seed_metrics = pd.concat(
        [overall_seed_metrics, stratified_seed_metrics],
        ignore_index=True,
        sort=False,
    )
    combined_seed_metrics = _ordered_seed_metrics(combined_seed_metrics)
    seed_path = output_dir / "seed_metrics.csv"
    combined_seed_metrics.to_csv(seed_path, index=False)
    outputs["seed_metrics"] = seed_path

    feature_summary_path = output_dir / "feature_type_metrics_summary.csv"
    feature_summary.to_csv(feature_summary_path, index=False)
    outputs["feature_type_metrics_summary"] = feature_summary_path
    return outputs
