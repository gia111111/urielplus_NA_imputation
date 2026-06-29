#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urielplus_impute.data import load_dataset
from urielplus_impute.metrics import (
    compute_binary_imputation_metrics,
    compute_stratified_metrics,
)
from urielplus_impute.results import write_result_tables
from urielplus_impute.softimpute import SoftImpute, softimpute_parameter_grid
from urielplus_impute.split_io import load_split, load_split_manifest_rows


DEFAULT_SPLIT_MANIFEST = "imputation_exp/runs/splits/manifests/split_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SoftImpute baseline on saved URIEL+ splits."
    )
    parser.add_argument("--typological", default="urielplus_analysis/typological_data.csv")
    parser.add_argument("--languages", default="urielplus_analysis/languages.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--regimes", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--index-col", default=None)
    parser.add_argument("--drop-empty-languages", action="store_true")
    parser.add_argument("--keep-special-languages", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=["rmse", "macro_f1"],
        default="rmse",
        help="Validation metric used only to select SoftImpute hyperparameters.",
    )
    parser.add_argument(
        "--softimpute-shrinkage-grid",
        nargs="+",
        type=float,
        default=[0.0, 1.0, 5.0, 10.0],
    )
    parser.add_argument("--softimpute-max-rank", type=int, default=None)
    parser.add_argument("--softimpute-max-iters", type=int, default=400)
    parser.add_argument("--softimpute-tol", type=float, default=1e-4)
    parser.add_argument("--softimpute-verbose", action="store_true")
    return parser.parse_args()


def _selection_score(metrics: dict[str, float], selection_metric: str) -> float:
    return -metrics["rmse"] if selection_metric == "rmse" else metrics["macro_f1"]


def _save_predictions(
    outdir: Path,
    regime: str,
    seed: int,
    split_name: str,
    cells: pd.DataFrame,
    scores: np.ndarray,
) -> None:
    prediction_dir = outdir / "predictions" / regime
    prediction_dir.mkdir(parents=True, exist_ok=True)
    predictions = cells.copy()
    predictions["y_score"] = np.clip(scores, 0.0, 1.0)
    predictions["y_pred"] = (predictions["y_score"] >= 0.5).astype(int)
    predictions.to_csv(
        prediction_dir / f"seed_{seed}_{split_name}.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    metrics_dir = outdir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        args.typological,
        args.languages,
        index_col=args.index_col,
        drop_empty_languages=args.drop_empty_languages,
        filter_special_families=not args.keep_special_languages,
    )
    manifest_path = Path(args.split_manifest)
    split_rows = load_split_manifest_rows(
        manifest_path,
        regimes=args.regimes,
        seeds=args.seeds,
    )
    manifest_dir = outdir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                **row,
                "split_path": str(row["split_path"]),
                "metadata_path": str(row["metadata_path"]),
            }
            for row in split_rows
        ]
    ).to_csv(manifest_dir / "softimpute_split_manifest.csv", index=False)

    seed_metric_rows: list[dict] = []
    stratified_rows: list[pd.DataFrame] = []
    parameter_grid = softimpute_parameter_grid(
        args.softimpute_shrinkage_grid,
        max_rank=args.softimpute_max_rank,
        max_iters=args.softimpute_max_iters,
        tol=args.softimpute_tol,
        verbose=args.softimpute_verbose,
    )

    for row in split_rows:
        loaded = load_split(row, dataset.X, dataset.feature_types)
        print(
            f"[split] regime={loaded.regime} seed={loaded.seed} "
            f"train_visible={int(loaded.masks.train_visible_mask.sum())}"
        )
        best: dict | None = None
        tuning_rows = []
        for params in parameter_grid:
            started = time.perf_counter()
            model = SoftImpute(**params).fit(loaded.train_matrix)
            validation_scores = np.clip(
                model.predict_cells(loaded.val_cells),
                0.0,
                1.0,
            )
            validation_metrics = compute_binary_imputation_metrics(
                loaded.val_cells["true_value"],
                validation_scores,
                threshold=0.5,
            )
            score = _selection_score(validation_metrics, args.selection_metric)
            tuning_rows.append(
                {
                    "regime": loaded.regime,
                    "seed": loaded.seed,
                    "params": json.dumps(params, sort_keys=True),
                    "elapsed_seconds": time.perf_counter() - started,
                    "n_iters": model.n_iters_,
                    "relative_change": model.relative_change_,
                    "effective_rank": model.effective_rank_,
                    **validation_metrics,
                }
            )
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "model": model,
                    "params": params,
                    "validation_scores": validation_scores,
                }
        if best is None:
            raise RuntimeError("SoftImpute parameter grid is empty.")

        tuning_dir = outdir / "tuning" / loaded.regime
        tuning_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(tuning_rows).to_csv(
            tuning_dir / f"seed_{loaded.seed}_grid.csv",
            index=False,
        )
        test_scores = np.clip(
            best["model"].predict_cells(loaded.test_cells),
            0.0,
            1.0,
        )
        for split_name, cells, scores in (
            ("val", loaded.val_cells, best["validation_scores"]),
            ("test", loaded.test_cells, test_scores),
        ):
            metrics = compute_binary_imputation_metrics(
                cells["true_value"],
                scores,
                threshold=0.5,
            )
            base = {
                "model": "softimpute",
                "variant": "softimpute",
                "regime": loaded.regime,
                "seed": loaded.seed,
                "split": split_name,
                "n": int(len(cells)),
                **metrics,
                "threshold": 0.5,
                "selection_metric": args.selection_metric,
                "selected_params": json.dumps(best["params"], sort_keys=True),
                "split_path": str(loaded.split_path),
                "metadata_path": str(loaded.metadata_path),
            }
            seed_metric_rows.append(base)
            stratified = compute_stratified_metrics(
                cells["true_value"],
                scores,
                cells["feature_type"],
                threshold=0.5,
            )
            for key, value in {
                "model": "softimpute",
                "variant": "softimpute",
                "regime": loaded.regime,
                "seed": loaded.seed,
                "split": split_name,
            }.items():
                stratified[key] = value
            stratified_rows.append(stratified)
            _save_predictions(
                outdir,
                loaded.regime,
                loaded.seed,
                split_name,
                cells,
                scores,
            )

    outputs = write_result_tables(
        pd.DataFrame(seed_metric_rows),
        pd.concat(stratified_rows, ignore_index=True),
        metrics_dir,
    )
    for name, path in outputs.items():
        print(f"[wrote] {name}: {path}")


if __name__ == "__main__":
    main()
