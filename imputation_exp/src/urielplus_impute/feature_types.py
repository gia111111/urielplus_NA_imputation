from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_TYPES = ("S", "P", "M", "INV")
TYPE_ORDER = list(FEATURE_TYPES)

SPECIAL_FAMILY_IDS = {
    "unat1236",  # Unattested
    "sign1238",  # Sign
    "arti1236",  # Artificial
    "uncl1493",  # Unclassifiable
    "spee1234",  # Speech Register
    "book1242",  # Bookkeeping
}


def feature_type_series(columns: Iterable[str]) -> pd.Series:
    """Derive the four URIEL+ feature-type codes once from column prefixes."""
    feature_types: dict[str, str] = {}
    unsupported: list[str] = []
    for column in columns:
        feature = str(column)
        if feature.startswith("INV_"):
            feature_types[feature] = "INV"
        elif feature.startswith("S_"):
            feature_types[feature] = "S"
        elif feature.startswith("P_"):
            feature_types[feature] = "P"
        elif feature.startswith("M_"):
            feature_types[feature] = "M"
        else:
            unsupported.append(feature)

    if unsupported:
        preview = ", ".join(repr(value) for value in unsupported[:5])
        raise ValueError(
            "Unsupported typological feature columns. Expected prefixes S_, P_, M_, "
            f"or INV_; found {preview}."
        )
    return pd.Series(feature_types, name="feature_type", dtype="string")


def as_feature_type_array(
    feature_types: pd.Series | np.ndarray | Iterable[str],
    n_features: int,
) -> np.ndarray:
    """Validate and return one feature-type code per matrix column."""
    values = (
        feature_types.to_numpy(dtype=str)
        if isinstance(feature_types, pd.Series)
        else np.asarray(
            list(feature_types)
            if not isinstance(feature_types, np.ndarray)
            else feature_types,
            dtype=str,
        )
    )
    if values.ndim != 1 or len(values) != n_features:
        raise ValueError(
            f"feature_types must be one-dimensional with length {n_features}; "
            f"received shape {values.shape}."
        )
    unsupported = sorted(set(values).difference(FEATURE_TYPES))
    if unsupported:
        raise ValueError(
            f"feature_types contains unsupported values {unsupported}; "
            f"expected only {FEATURE_TYPES}."
        )
    return values


def feature_types_for_column_indices(
    feature_types: pd.Series | np.ndarray | Iterable[str],
    column_indices: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """Return validated feature-type codes for scored matrix cells."""
    types = as_feature_type_array(feature_types, n_features)
    indices = np.asarray(column_indices, dtype=int)
    if np.any(indices < 0) or np.any(indices >= n_features):
        raise ValueError("column_indices contains an index outside the feature matrix.")
    return types[indices]


def feature_type_from_regime(regime: str) -> str:
    for prefix in ("Local_block_", "Global_block_"):
        if regime.startswith(prefix):
            target_type = regime[len(prefix) :]
            if target_type in FEATURE_TYPES:
                return target_type
    raise ValueError(
        f"Unknown block regime {regime!r}. Expected Local_block_<type> or "
        f"Global_block_<type>, where type is one of {FEATURE_TYPES}."
    )
