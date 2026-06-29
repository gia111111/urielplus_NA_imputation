#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urielplus_impute.data import load_dataset
from urielplus_impute.diagnostics import (
    write_decision_tree_diagnostics,
    write_logistic_diagnostics,
)
from urielplus_impute.experiment import FULL_PREDICTOR_SET, SIMPLE_MODELS
from urielplus_impute.metrics import (
    compute_binary_imputation_metrics,
    compute_stratified_metrics,
)
from urielplus_impute.predictors import PredictorConfig, ProposalPredictorBuilder, select_predictor_columns
from urielplus_impute.results import write_result_tables
from urielplus_impute.simple import SklearnSimpleImputer, grid_for_model
from urielplus_impute.split_io import (
    load_split,
    load_split_manifest_rows,
)
from urielplus_impute.splits import observed_cells_df


DEFAULT_SPLIT_MANIFEST = "imputation_exp/runs/splits/manifests/split_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run proposal simple models for URIEL+ imputation.")
    parser.add_argument("--typological", default="urielplus_analysis/typological_data.csv")
    parser.add_argument("--languages", default="urielplus_analysis/languages.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--split-manifest",
        default=DEFAULT_SPLIT_MANIFEST,
        help=(
            "split_manifest.csv produced by make_splits.py. "
            "The simple models reuse these exact train/val/test files."
        ),
    )
    parser.add_argument("--models", nargs="+", default=list(SIMPLE_MODELS), choices=list(SIMPLE_MODELS))
    parser.add_argument(
        "--variant",
        default=FULL_PREDICTOR_SET,
        choices=[FULL_PREDICTOR_SET],
        help="Predictor set label kept in output tables; the revised proposal uses only `full`.",
    )
    parser.add_argument("--regimes", nargs="+", default=None, help="Optional regime filter applied to --split-manifest rows.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional seed filter applied to --split-manifest rows.")
    parser.add_argument("--index-col", default=None)
    parser.add_argument("--drop-empty-languages", action="store_true")
    parser.add_argument("--keep-special-languages", action="store_true", help="Keep special high-missingness groups instead of filtering them before masking/training.")
    parser.add_argument("--max-train-cells", type=int, default=None)
    parser.add_argument("--selection-metric", default="rmse", choices=["rmse", "macro_f1"])
    parser.add_argument("--k-geo", type=int, default=50)
    parser.add_argument("--geo-backend", choices=["knn", "macroarea"], default="knn")
    parser.add_argument("--top-corr-features", type=int, default=32)
    parser.add_argument("--corr-shrinkage", type=float, default=20.0)
    parser.add_argument("--min-corr-overlap", type=int, default=20)
    parser.add_argument("--skip-logit-diagnostics", action="store_true", help="Skip statsmodels logistic coefficient/VIF/AIC/BIC diagnostics.")
    parser.add_argument("--stats-max-cells", type=int, default=None, help="Optional per-split row cap for statsmodels diagnostic refits.")
    parser.add_argument("--stats-max-iter", type=int, default=100, help="Maximum IRLS iterations for statsmodels logistic diagnostics.")
    parser.add_argument(
        "--tree-plot-max-depth",
        type=int,
        default=0,
        help="Write decision-tree PNG/PDF diagnostics to this depth; 0 disables plots.",
    )
    parser.add_argument(
        "--save-predictor-table-heads",
        action="store_true",
        help="Write the first 5,000 training predictor rows for each regime/seed.",
    )
    return parser.parse_args()


def maybe_sample_train_cells(train_cells: pd.DataFrame, max_train_cells: int | None, seed: int) -> pd.DataFrame:
    if max_train_cells is None or len(train_cells) <= max_train_cells:
        return train_cells.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(train_cells.index.to_numpy(), size=max_train_cells, replace=False)
    return train_cells.loc[idx].reset_index(drop=True)


def load_split_rows_from_manifest(args: argparse.Namespace) -> tuple[list[dict], Path]:
    manifest_path = Path(args.split_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}. "
            "Run `python3 imputation_exp/scripts/make_splits.py --outdir imputation_exp/runs/splits` "
            "first, or pass --split-manifest pointing to an existing split_manifest.csv."
        )

    rows = load_split_manifest_rows(manifest_path, regimes=args.regimes, seeds=args.seeds)
    return rows, manifest_path


def model_score(metrics: dict[str, float], selection_metric: str) -> float:
    if selection_metric == "rmse":
        return -float(metrics["rmse"])
    if selection_metric == "macro_f1":
        return float(metrics["macro_f1"])
    raise ValueError(selection_metric)


def save_predictions(
    outdir: Path,
    variant: str,
    model_name: str,
    regime: str,
    seed: int,
    split: str,
    mask: pd.DataFrame,
    y_prob: np.ndarray,
) -> None:
    pred_dir = outdir / "predictions" / variant / model_name / regime
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred = mask.copy()
    pred["y_score"] = np.clip(y_prob, 0.0, 1.0)
    pred["y_pred"] = (pred["y_score"] >= 0.5).astype(int)
    pred.to_csv(pred_dir / f"seed_{seed}_{split}.csv", index=False)


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
    print(f"[data] matrix={dataset.X.shape}, observed={int(dataset.X.notna().sum().sum())}")
    print(f"[data] special high-missingness rows removed={dataset.n_special_filtered}")

    predictor_config = PredictorConfig(
        k_geo=args.k_geo,
        geo_backend=args.geo_backend,
        top_corr_features=args.top_corr_features,
        corr_shrinkage=args.corr_shrinkage,
        min_corr_overlap=args.min_corr_overlap,
    )
    variant = args.variant
    numeric_columns, categorical_columns = select_predictor_columns(variant)
    print(
        f"[predictors] set={variant} "
        f"numeric={len(numeric_columns)} categorical={len(categorical_columns)}"
    )

    metric_rows = []
    strata_rows = []
    diagnostic_coef_tables = []
    diagnostic_fit_tables = []
    diagnostic_vif_tables = []
    tree_importance_tables = []
    tree_stats_rows = []
    predictor_selection_rows = []
    manifest_split_rows, manifest_path = load_split_rows_from_manifest(args)
    print(f"[splits] reusing {len(manifest_split_rows)} split rows from {manifest_path}")
    manifest_copy_dir = outdir / "manifests"
    manifest_copy_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                **row,
                "split_path": str(row["split_path"]),
                "metadata_path": str(row["metadata_path"]),
            }
            for row in manifest_split_rows
        ]
    ).to_csv(manifest_copy_dir / "simple_split_manifest.csv", index=False)

    for row in manifest_split_rows:
        loaded = load_split(row, dataset.X, dataset.feature_types)
        regime = loaded.regime
        seed = loaded.seed
        print(f"[split] regime={regime} seed={seed} source=manifest")
        X_train = loaded.train_matrix
        val_mask = loaded.val_cells
        test_mask = loaded.test_cells
        languages = dataset.languages.reindex(X_train.index)
        feature_types = dataset.feature_types.reindex(X_train.columns)

        train_cells = observed_cells_df(X_train, feature_types)
        train_cells = maybe_sample_train_cells(train_cells, args.max_train_cells, seed)
        print(f"[features] train_cells={len(train_cells)} val={len(val_mask)} test={len(test_mask)}")

        builder = ProposalPredictorBuilder(
            X_train,
            languages,
            feature_types,
            config=predictor_config,
        ).fit()
        train_features = builder.build(train_cells)
        val_features = builder.build(val_mask)
        test_features = builder.build(test_mask)

        if args.save_predictor_table_heads:
            coef_dir = outdir / "predictor_tables" / variant / regime
            coef_dir.mkdir(parents=True, exist_ok=True)
            train_features.head(5000).to_csv(
                coef_dir / f"seed_{seed}_train_features_head.csv",
                index=False,
            )

        if "logistic_regression" in args.models and not args.skip_logit_diagnostics:
            print(f"[diagnostics] logistic_regression regime={regime} seed={seed}")
            coef_diag, fit_diag, vif_diag = write_logistic_diagnostics(
                outdir=outdir,
                variant=variant,
                regime=regime,
                seed=seed,
                train_features=train_features,
                val_features=val_features,
                test_features=test_features,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                max_cells=args.stats_max_cells,
                max_iter=args.stats_max_iter,
            )
            if len(coef_diag):
                diagnostic_coef_tables.append(coef_diag)
            if len(fit_diag):
                diagnostic_fit_tables.append(fit_diag)
            if len(vif_diag):
                diagnostic_vif_tables.append(vif_diag)

        for model_name in args.models:
            print(f"[model] tuning {model_name} regime={regime} seed={seed}")
            best = None
            tuning_rows = []
            for params in grid_for_model(model_name):
                params = dict(params)
                params["random_state"] = seed
                model = SklearnSimpleImputer(
                    model_name,
                    numeric_columns=numeric_columns,
                    categorical_columns=categorical_columns,
                    params=params,
                )
                model.fit(train_features, train_features["true_value"].to_numpy())
                val_prob = model.predict_proba(val_features)
                val_metrics = compute_binary_imputation_metrics(
                    val_mask["true_value"].to_numpy(),
                    val_prob,
                    threshold=0.5,
                )
                score = model_score(val_metrics, args.selection_metric)
                tuning_rows.append(
                    {
                        "model": model_name,
                        "variant": variant,
                        "regime": regime,
                        "seed": seed,
                        "params": params,
                        "threshold": 0.5,
                        **val_metrics,
                    }
                )
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "model": model,
                        "params": params,
                        "val_prob": val_prob,
                        "val_metrics": val_metrics,
                    }

            assert best is not None
            selected_model = best["model"]
            test_prob = selected_model.predict_proba(test_features)
            predictor_selection_rows.append(
                {
                    "model": model_name,
                    "variant": variant,
                    "regime": regime,
                    "seed": seed,
                    "numeric_predictors": json.dumps(numeric_columns),
                    "categorical_predictors": json.dumps(categorical_columns),
                    "selected_params": json.dumps(best["params"], sort_keys=True),
                }
            )

            if model_name == "decision_tree":
                importance_diag, tree_stats = write_decision_tree_diagnostics(
                    outdir=outdir,
                    variant=variant,
                    regime=regime,
                    seed=seed,
                    model=selected_model,
                    params=best["params"],
                    threshold=0.5,
                    plot_max_depth=(
                        args.tree_plot_max_depth
                        if args.tree_plot_max_depth > 0
                        else None
                    ),
                )
                if len(importance_diag):
                    tree_importance_tables.append(importance_diag)
                tree_stats_rows.append(tree_stats)

            tuning_dir = outdir / "tuning" / variant / model_name / regime
            tuning_dir.mkdir(parents=True, exist_ok=True)
            (tuning_dir / f"seed_{seed}_grid.json").write_text(
                json.dumps(tuning_rows, indent=2, default=str),
                encoding="utf-8",
            )

            for split_name, mask, probs in [
                ("val", val_mask, best["val_prob"]),
                ("test", test_mask, test_prob),
            ]:
                metrics = compute_binary_imputation_metrics(
                    mask["true_value"].to_numpy(),
                    probs,
                    threshold=0.5,
                )
                metrics.update(
                    {
                        "model": model_name,
                        "variant": variant,
                        "regime": regime,
                        "seed": seed,
                        "split": split_name,
                        "n": int(len(mask)),
                        "selection_metric": args.selection_metric,
                        "selected_params": json.dumps(best["params"], sort_keys=True),
                        "threshold": 0.5,
                        "n_train_observed": int(X_train.notna().sum().sum()),
                        "n_heldout": int(loaded.metadata["n_heldout"]),
                        "n_val": int(loaded.metadata["n_val"]),
                        "n_test": int(loaded.metadata["n_test"]),
                        "split_path": str(loaded.split_path),
                        "metadata_path": str(loaded.metadata_path),
                    }
                )
                metric_rows.append(metrics)
                save_predictions(
                    outdir,
                    variant,
                    model_name,
                    regime,
                    seed,
                    split_name,
                    mask,
                    probs,
                )

                strata = compute_stratified_metrics(
                    mask["true_value"].to_numpy(),
                    probs,
                    mask["feature_type"].to_numpy(),
                    threshold=0.5,
                )
                if len(strata):
                    strata["model"] = model_name
                    strata["variant"] = variant
                    strata["regime"] = regime
                    strata["seed"] = seed
                    strata["split"] = split_name
                    strata_rows.append(strata)

    metrics_df = pd.DataFrame(metric_rows)
    strata_df = (
        pd.concat(strata_rows, ignore_index=True)
        if strata_rows
        else pd.DataFrame()
    )

    if diagnostic_fit_tables or diagnostic_coef_tables or diagnostic_vif_tables:
        diag_root = outdir / "diagnostics" / "logistic_regression"
        diag_root.mkdir(parents=True, exist_ok=True)
        if diagnostic_fit_tables:
            pd.concat(diagnostic_fit_tables, ignore_index=True).to_csv(
                diag_root / "all_fit_stats.csv",
                index=False,
            )
        if diagnostic_coef_tables:
            pd.concat(diagnostic_coef_tables, ignore_index=True).to_csv(
                diag_root / "all_coefficients.csv",
                index=False,
            )
        if diagnostic_vif_tables:
            pd.concat(diagnostic_vif_tables, ignore_index=True).to_csv(
                diag_root / "all_vif.csv",
                index=False,
            )

    if tree_importance_tables or tree_stats_rows:
        tree_root = outdir / "diagnostics" / "decision_tree"
        tree_root.mkdir(parents=True, exist_ok=True)
        if tree_importance_tables:
            pd.concat(tree_importance_tables, ignore_index=True).to_csv(
                tree_root / "all_feature_importances.csv",
                index=False,
            )
        if tree_stats_rows:
            pd.DataFrame(tree_stats_rows).to_csv(
                tree_root / "all_tree_stats.csv",
                index=False,
            )

    diagnostics_root = outdir / "diagnostics"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    if predictor_selection_rows:
        pd.DataFrame(predictor_selection_rows).to_csv(
            diagnostics_root / "predictor_selections.csv",
            index=False,
        )
    (diagnostics_root / "README.md").write_text(
        "\n".join(
            [
                "# Simple-model diagnostics",
                "",
                "- `predictor_selections.csv`: selected predictor columns and tuned parameters.",
                "- `logistic_regression/`: coefficient, odds-ratio, p-value, fit, and VIF tables.",
                "- `decision_tree/`: tree rules, feature importances, plots, and tree-size tables.",
                "",
                "Prediction metrics are kept separately under `../metrics/`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result_outputs = write_result_tables(
        metrics_df,
        strata_df,
        metrics_dir,
    )
    print(f"[done] wrote metrics to {metrics_dir}")
    for name, path in result_outputs.items():
        print(f"[done] {name}: {path}")


if __name__ == "__main__":
    main()
