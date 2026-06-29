from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _require_sklearn():
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as exc:
        raise ImportError(
            "Simple models require scikit-learn. Install it with "
            "`python3 -m pip install -r imputation_exp/requirements.txt`."
        ) from exc
    return {
        "ColumnTransformer": ColumnTransformer,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
        "DecisionTreeClassifier": DecisionTreeClassifier,
    }


def _one_hot_encoder(sklearn: dict[str, Any]):
    OneHotEncoder = sklearn["OneHotEncoder"]
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


@dataclass
class SimpleModelSpec:
    name: str
    params: dict[str, Any]


class SklearnSimpleImputer:
    """A sklearn-backed language-feature cell model."""

    def __init__(
        self,
        model_name: str,
        *,
        numeric_columns: list[str],
        categorical_columns: list[str],
        params: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.name = model_name
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.params = params or {}

    def fit(self, features: pd.DataFrame, y) -> "SklearnSimpleImputer":
        sklearn = _require_sklearn()
        ColumnTransformer = sklearn["ColumnTransformer"]
        SimpleImputer = sklearn["SimpleImputer"]
        Pipeline = sklearn["Pipeline"]
        StandardScaler = sklearn["StandardScaler"]
        LogisticRegression = sklearn["LogisticRegression"]
        DecisionTreeClassifier = sklearn["DecisionTreeClassifier"]

        transformers = []
        if self.numeric_columns:
            if self.model_name == "logistic_regression":
                numeric = Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                )
            else:
                numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
            transformers.append(("numeric", numeric, self.numeric_columns))
        if self.categorical_columns:
            categorical = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("onehot", _one_hot_encoder(sklearn)),
                ]
            )
            transformers.append(("categorical", categorical, self.categorical_columns))

        preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.3)

        if self.model_name == "logistic_regression":
            estimator = LogisticRegression(
                C=float(self.params.get("C", 1.0)),
                solver=str(self.params.get("solver", "lbfgs")),
                max_iter=int(self.params.get("max_iter", 1000)),
                class_weight=self.params.get("class_weight", "balanced"),
                n_jobs=None,
            )
        elif self.model_name == "decision_tree":
            estimator = DecisionTreeClassifier(
                max_depth=self.params.get("max_depth", 10),
                min_samples_leaf=int(self.params.get("min_samples_leaf", 100)),
                min_samples_split=int(self.params.get("min_samples_split", 200)),
                criterion=str(self.params.get("criterion", "gini")),
                class_weight=self.params.get("class_weight", "balanced"),
                random_state=int(self.params.get("random_state", 0)),
            )
        else:
            raise ValueError(f"Unknown simple model {self.model_name!r}.")

        self.pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        self.pipeline.fit(features[self.numeric_columns + self.categorical_columns], np.asarray(y).astype(int))
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        X = features[self.numeric_columns + self.categorical_columns]
        proba = self.pipeline.predict_proba(X)
        classes = list(self.pipeline.named_steps["model"].classes_)
        if 1 in classes:
            return np.clip(proba[:, classes.index(1)], 0.0, 1.0)
        return np.zeros(len(features), dtype=float)

    def transformed_feature_names(self) -> list[str]:
        """Return readable feature names after preprocessing."""
        preprocessor = self.pipeline.named_steps["preprocess"]
        raw_names = preprocessor.get_feature_names_out()
        return [self._clean_feature_name(str(name)) for name in raw_names]

    def decision_tree_stats(self) -> dict[str, int]:
        """Return compact shape diagnostics for a fitted decision tree."""
        estimator = self.pipeline.named_steps["model"]
        tree = estimator.tree_
        is_leaf = tree.children_left == -1
        return {
            "node_count": int(tree.node_count),
            "max_depth": int(estimator.get_depth()),
            "n_leaves": int(is_leaf.sum()),
            "n_features": int(estimator.n_features_in_),
        }

    def decision_tree_feature_importances(self) -> pd.DataFrame:
        """Return non-zero decision-tree importances with readable names."""
        estimator = self.pipeline.named_steps["model"]
        names = self.transformed_feature_names()
        importances = np.asarray(estimator.feature_importances_, dtype=float)
        table = pd.DataFrame({"feature": names, "importance": importances})
        table = table[table["importance"] > 0.0].copy()
        return table.sort_values("importance", ascending=False).reset_index(drop=True)

    def decision_tree_text(self, *, max_depth: int | None = None) -> str:
        """Export a fitted decision tree as readable nested rules."""
        try:
            from sklearn.tree import export_text
        except ImportError as exc:
            raise ImportError("Decision-tree text export requires scikit-learn.") from exc

        estimator = self.pipeline.named_steps["model"]
        depth = estimator.get_depth() if max_depth is None else max_depth
        return export_text(
            estimator,
            feature_names=self.transformed_feature_names(),
            max_depth=depth,
            decimals=4,
            show_weights=True,
        )

    def save_decision_tree_plot(
        self,
        path: str | Path,
        *,
        max_depth: int = 3,
        figsize: tuple[float, float] | None = None,
        dpi: int = 220,
    ) -> None:
        """Save a readable top-level decision-tree plot."""
        import os
        import tempfile

        cache_dir = Path(tempfile.gettempdir()) / "urielplus_mpl_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

        try:
            import matplotlib.pyplot as plt
            from sklearn.tree import plot_tree
        except ImportError as exc:
            raise ImportError("Decision-tree plotting requires matplotlib and scikit-learn.") from exc

        estimator = self.pipeline.named_steps["model"]
        depth = min(max_depth, estimator.get_depth())
        if figsize is None:
            figsize = (max(12.0, 4.8 * (2 ** min(depth, 3))), max(6.0, 2.4 * (depth + 1)))

        fig, ax = plt.subplots(figsize=figsize)
        plot_tree(
            estimator,
            feature_names=self.transformed_feature_names(),
            class_names=["0", "1"],
            max_depth=max_depth,
            filled=True,
            rounded=True,
            impurity=False,
            proportion=False,
            fontsize=8,
            ax=ax,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def _clean_feature_name(self, name: str) -> str:
        for prefix in ("numeric__", "categorical__"):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        for col in self.categorical_columns:
            prefix = f"{col}_"
            if name.startswith(prefix):
                return f"{col}={name[len(prefix):]}"
        return name


def grid_for_model(model_name: str) -> list[dict[str, Any]]:
    if model_name == "logistic_regression":
        return [
            {"C": 0.1, "max_iter": 1000, "class_weight": "balanced"},
            {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
            {"C": 10.0, "max_iter": 1000, "class_weight": "balanced"},
        ]
    if model_name == "decision_tree":
        return [
            {
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "min_samples_split": min_samples_split,
                "class_weight": "balanced",
            }
            for max_depth in [2, 4, 6, 8, 10]
            for min_samples_leaf in [25, 50, 100, 150, 200]
            for min_samples_split in [50, 100, 200, 300, 400]
        ]
    raise ValueError(f"Unknown simple model {model_name!r}.")
