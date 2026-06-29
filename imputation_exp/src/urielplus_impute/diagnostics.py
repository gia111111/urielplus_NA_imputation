from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import compute_binary_imputation_metrics
from .simple import SklearnSimpleImputer


def _require_statsmodels():
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError(
            "Logistic diagnostics require statsmodels. Install dependencies with "
            "`python3 -m pip install -r imputation_exp/requirements.txt`, "
            "or pass --skip-logit-diagnostics."
        ) from exc
    return sm


@dataclass(frozen=True)
class LogisticDiagnosticDesign:
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_medians: pd.Series
    numeric_scales: pd.Series
    categorical_levels: dict[str, list[str]]
    feature_columns: list[str]


def fit_logistic_diagnostic_design(
    features: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> LogisticDiagnosticDesign:
    if numeric_columns:
        numeric = features[numeric_columns].apply(pd.to_numeric, errors="coerce")
        medians = numeric.median(axis=0).fillna(0.0)
        filled = numeric.fillna(medians)
        scales = filled.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    else:
        medians = pd.Series(dtype=float)
        scales = pd.Series(dtype=float)

    categorical_levels: dict[str, list[str]] = {}
    feature_columns = list(numeric_columns)
    for column in categorical_columns:
        observed_levels = sorted(
            features[column].fillna("Unknown").astype(str).unique().tolist()
        )
        levels = (
            ["Unknown"] + [level for level in observed_levels if level != "Unknown"]
            if "Unknown" in observed_levels
            else observed_levels
        )
        categorical_levels[column] = levels
        feature_columns.extend(f"{column}={level}" for level in levels[1:])
    return LogisticDiagnosticDesign(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_medians=medians,
        numeric_scales=scales,
        categorical_levels=categorical_levels,
        feature_columns=feature_columns,
    )


def transform_logistic_diagnostic_design(
    features: pd.DataFrame,
    design: LogisticDiagnosticDesign,
) -> pd.DataFrame:
    parts = []
    if design.numeric_columns:
        numeric = features[design.numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(design.numeric_medians)
        parts.append(numeric.divide(design.numeric_scales, axis=1).astype(float))
    for column in design.categorical_columns:
        values = features[column].fillna("Unknown").astype(str)
        dummies = {
            f"{column}={level}": (values == level).astype(float)
            for level in design.categorical_levels[column][1:]
        }
        if dummies:
            parts.append(pd.DataFrame(dummies, index=features.index))
    matrix = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=features.index)
    matrix = matrix.reindex(columns=design.feature_columns, fill_value=0.0)
    matrix.insert(0, "const", 1.0)
    return matrix.astype(float)


def _matrix_rank(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    return int(np.linalg.matrix_rank(values.T @ values))


def _drop_dependent_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    keep = ["const"] if "const" in X.columns else []
    dropped: list[str] = []
    current = X[keep].to_numpy(dtype=float) if keep else np.empty((len(X), 0))
    current_rank = _matrix_rank(current)
    for column in [name for name in X.columns if name != "const"]:
        values = X[column].to_numpy(dtype=float)
        if np.nanstd(values) == 0.0:
            dropped.append(column)
            continue
        candidate = np.column_stack([current, values])
        candidate_rank = _matrix_rank(candidate)
        if candidate_rank > current_rank:
            keep.append(column)
            current = candidate
            current_rank = candidate_rank
        else:
            dropped.append(column)
    return X[keep], dropped


def _compute_vif_table(X: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in X.columns if column != "const"]
    if not columns:
        return pd.DataFrame(columns=["term", "vif"])
    if len(columns) == 1:
        return pd.DataFrame({"term": columns, "vif": [1.0]})
    values = X[columns].to_numpy(dtype=float)
    standard_deviation = values.std(axis=0)
    valid = standard_deviation > 0.0
    rows = [
        {"term": column, "vif": np.nan}
        for column, is_valid in zip(columns, valid)
        if not is_valid
    ]
    valid_columns = [column for column, is_valid in zip(columns, valid) if is_valid]
    if valid_columns:
        standardized = (
            values[:, valid] - values[:, valid].mean(axis=0)
        ) / standard_deviation[valid]
        correlation = np.corrcoef(standardized, rowvar=False)
        vifs = (
            np.array([1.0])
            if correlation.ndim == 0
            else np.diag(np.linalg.pinv(correlation))
        )
        rows.extend(
            {"term": column, "vif": float(vif)}
            for column, vif in zip(valid_columns, vifs)
        )
    return pd.DataFrame(rows)


def _sample_diagnostic_rows(
    features: pd.DataFrame,
    max_cells: int | None,
    seed: int,
    split_name: str,
) -> pd.DataFrame:
    if max_cells is None or len(features) <= max_cells:
        return features.reset_index(drop=True)
    split_offset = {"train": 11, "val": 23, "test": 37}.get(split_name, 0)
    rng = np.random.default_rng(seed + split_offset)
    selected = rng.choice(features.index.to_numpy(), size=max_cells, replace=False)
    return features.loc[selected].reset_index(drop=True)


def fit_logistic_posthoc(
    *,
    features: pd.DataFrame,
    design: LogisticDiagnosticDesign,
    split_name: str,
    variant: str,
    regime: str,
    seed: int,
    max_iter: int,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    sm = _require_statsmodels()
    y = pd.to_numeric(features["true_value"], errors="coerce").round()
    valid = y.isin([0, 1])
    y_array = y.loc[valid].astype(int).to_numpy()
    raw_design = transform_logistic_diagnostic_design(features.loc[valid], design)
    X, dropped = _drop_dependent_columns(raw_design)
    vif = _compute_vif_table(X)
    for column, value in (
        ("variant", variant),
        ("regime", regime),
        ("seed", seed),
        ("split", split_name),
    ):
        vif.insert(0, column, value)

    fit_stats = {
        "variant": variant,
        "regime": regime,
        "seed": seed,
        "split": split_name,
        "n": int(len(y_array)),
        "n_terms": int(max(X.shape[1] - 1, 0)),
        "dropped_terms_count": int(len(dropped)),
        "dropped_terms": "|".join(dropped),
    }
    if len(y_array) == 0 or len(np.unique(y_array)) < 2:
        fit_stats.update(
            {"status": "failed", "message": "split has fewer than two outcome classes"}
        )
        return pd.DataFrame(), fit_stats, vif

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = sm.GLM(
                y_array,
                X,
                family=sm.families.Binomial(),
            ).fit(maxiter=max_iter, disp=0)
        warning_messages = [str(warning.message) for warning in caught]
    except Exception as exc:
        fit_stats.update(
            {"status": "failed", "message": f"{type(exc).__name__}: {exc}"}
        )
        return pd.DataFrame(), fit_stats, vif

    predicted = np.clip(np.asarray(result.predict(X), dtype=float), 0.0, 1.0)
    shared_metrics = compute_binary_imputation_metrics(y_array, predicted, threshold=0.5)
    log_likelihood = float(result.llf)
    n_observations = max(int(result.nobs), 1)
    n_parameters = int(len(result.params))
    y_mean = float(np.clip(y_array.mean(), 1e-12, 1.0 - 1e-12))
    null_log_likelihood = float(
        np.sum(
            y_array * np.log(y_mean)
            + (1 - y_array) * np.log(1.0 - y_mean)
        )
    )
    cox_snell = 1.0 - np.exp(
        (2.0 / n_observations) * (null_log_likelihood - log_likelihood)
    )
    nagelkerke_denominator = 1.0 - np.exp(
        (2.0 / n_observations) * null_log_likelihood
    )
    tjur = float(
        predicted[y_array == 1].mean() - predicted[y_array == 0].mean()
    )
    fit_stats.update(
        {
            "status": "ok",
            "message": "; ".join(warning_messages),
            "log_likelihood": log_likelihood,
            "null_log_likelihood": null_log_likelihood,
            "aic": float(-2.0 * log_likelihood + 2.0 * n_parameters),
            "bic": float(
                -2.0 * log_likelihood
                + np.log(n_observations) * n_parameters
            ),
            "pseudo_r2_mcfadden": float(
                1.0 - log_likelihood / null_log_likelihood
            )
            if null_log_likelihood != 0.0
            else np.nan,
            "adjusted_pseudo_r2_mcfadden": float(
                1.0 - (log_likelihood - n_parameters) / null_log_likelihood
            )
            if null_log_likelihood != 0.0
            else np.nan,
            "cox_snell_r2": float(cox_snell),
            "nagelkerke_r2": float(cox_snell / nagelkerke_denominator)
            if nagelkerke_denominator != 0.0
            else np.nan,
            "tjur_r2": tjur,
            **shared_metrics,
        }
    )

    parameters = pd.Series(np.asarray(result.params, dtype=float), index=X.columns)
    standard_errors = pd.Series(np.asarray(result.bse, dtype=float), index=X.columns)
    z_values = pd.Series(np.asarray(result.tvalues, dtype=float), index=X.columns)
    p_values = pd.Series(np.asarray(result.pvalues, dtype=float), index=X.columns)
    confidence = result.conf_int(alpha=0.05)
    confidence = pd.DataFrame(
        np.asarray(confidence),
        index=X.columns,
        columns=["ci_lower", "ci_upper"],
    )
    coefficients = pd.DataFrame(
        {
            "term": parameters.index,
            "coef": parameters.to_numpy(dtype=float),
            "std_err": standard_errors.to_numpy(dtype=float),
            "z": z_values.to_numpy(dtype=float),
            "p_value": p_values.to_numpy(dtype=float),
            "ci_lower": confidence["ci_lower"].to_numpy(dtype=float),
            "ci_upper": confidence["ci_upper"].to_numpy(dtype=float),
        }
    )
    coefficients["odds_ratio"] = np.exp(
        np.clip(coefficients["coef"], -700.0, 700.0)
    )
    coefficients["odds_ratio_ci_lower"] = np.exp(
        np.clip(coefficients["ci_lower"], -700.0, 700.0)
    )
    coefficients["odds_ratio_ci_upper"] = np.exp(
        np.clip(coefficients["ci_upper"], -700.0, 700.0)
    )
    for column, value in (
        ("variant", variant),
        ("regime", regime),
        ("seed", seed),
        ("split", split_name),
    ):
        coefficients.insert(0, column, value)
    return coefficients, fit_stats, vif


def write_logistic_diagnostics(
    *,
    outdir: Path,
    variant: str,
    regime: str,
    seed: int,
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    test_features: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    max_cells: int | None,
    max_iter: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    design = fit_logistic_diagnostic_design(
        train_features,
        numeric_columns,
        categorical_columns,
    )
    coefficient_tables = []
    fit_rows = []
    vif_tables = []
    for split_name, features in (
        ("train", train_features),
        ("val", val_features),
        ("test", test_features),
    ):
        sampled = _sample_diagnostic_rows(features, max_cells, seed, split_name)
        coefficients, fit_stats, vif = fit_logistic_posthoc(
            features=sampled,
            design=design,
            split_name=split_name,
            variant=variant,
            regime=regime,
            seed=seed,
            max_iter=max_iter,
        )
        if not coefficients.empty:
            coefficient_tables.append(coefficients)
        fit_rows.append(fit_stats)
        if not vif.empty:
            vif_tables.append(vif)

    diagnostic_dir = (
        outdir / "diagnostics" / "logistic_regression" / variant / regime
    )
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    coefficient_frame = (
        pd.concat(coefficient_tables, ignore_index=True)
        if coefficient_tables
        else pd.DataFrame()
    )
    fit_frame = pd.DataFrame(fit_rows)
    vif_frame = (
        pd.concat(vif_tables, ignore_index=True)
        if vif_tables
        else pd.DataFrame()
    )
    if not coefficient_frame.empty:
        coefficient_frame.to_csv(
            diagnostic_dir / f"seed_{seed}_coefficients.csv",
            index=False,
        )
    fit_frame.to_csv(
        diagnostic_dir / f"seed_{seed}_fit_stats.csv",
        index=False,
    )
    if not vif_frame.empty:
        vif_frame.to_csv(
            diagnostic_dir / f"seed_{seed}_vif.csv",
            index=False,
        )
    return coefficient_frame, fit_frame, vif_frame


def write_decision_tree_diagnostics(
    *,
    outdir: Path,
    variant: str,
    regime: str,
    seed: int,
    model: SklearnSimpleImputer,
    params: dict,
    threshold: float,
    plot_max_depth: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Write readable rules, feature importances, and shape stats for a tree."""
    diag_dir = outdir / "diagnostics" / "decision_tree" / variant / regime
    diag_dir.mkdir(parents=True, exist_ok=True)

    importances = model.decision_tree_feature_importances()
    if len(importances):
        importances.insert(0, "rank", np.arange(1, len(importances) + 1))
        importances.insert(0, "seed", seed)
        importances.insert(0, "regime", regime)
        importances.insert(0, "variant", variant)
        importances.to_csv(diag_dir / f"seed_{seed}_feature_importances.csv", index=False)

    tree_text = model.decision_tree_text()
    (diag_dir / f"seed_{seed}_tree.txt").write_text(tree_text, encoding="utf-8")
    if plot_max_depth is not None:
        model.save_decision_tree_plot(
            diag_dir / f"seed_{seed}_tree_depth{plot_max_depth}.png",
            max_depth=plot_max_depth,
        )
        model.save_decision_tree_plot(
            diag_dir / f"seed_{seed}_tree_depth{plot_max_depth}.pdf",
            max_depth=plot_max_depth,
        )

    stats = {
        "variant": variant,
        "regime": regime,
        "seed": seed,
        "tree_plot_depth": int(plot_max_depth) if plot_max_depth is not None else np.nan,
        "threshold": float(threshold),
        "selected_params": json.dumps(params, sort_keys=True),
        **model.decision_tree_stats(),
    }
    if len(importances):
        stats["top_feature"] = str(importances.iloc[0]["feature"])
        stats["top_importance"] = float(importances.iloc[0]["importance"])
    else:
        stats["top_feature"] = ""
        stats["top_importance"] = np.nan

    pd.DataFrame([stats]).to_csv(diag_dir / f"seed_{seed}_tree_stats.csv", index=False)
    return importances, stats
