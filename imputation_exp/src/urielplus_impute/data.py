from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .feature_types import feature_type_series


@dataclass(frozen=True)
class CoverageFilterSummary:
    language_min_coverage: float
    feature_min_coverage: float
    input_languages: int
    input_features: int
    input_observed_cells: int
    input_missingness: float
    output_languages: int
    output_features: int
    output_observed_cells: int
    output_missingness: float
    retained_languages: tuple[str, ...]
    dropped_languages: tuple[str, ...]
    retained_features: tuple[str, ...]
    dropped_features: tuple[str, ...]
    iterations: tuple[dict[str, int], ...]


@dataclass
class UrielDataset:
    X: pd.DataFrame
    languages: pd.DataFrame
    feature_types: pd.Series
    filter_summary: CoverageFilterSummary


def _clean_string_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.isin(["", "nan", "NaN", "None", "<NA>"]))


def _parse_bool(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def load_typological_matrix(
    path: str | Path,
    index_col: Optional[str] = None,
    keep_other_features: bool = False,
) -> pd.DataFrame:
    """Load the URIEL+ typological matrix with languages as the index."""
    path = Path(path)
    if index_col is None:
        df = pd.read_csv(path)
        first = str(df.columns[0]).strip()
        if first.lower() in {"language", "glottocode", "id", "lang"}:
            df = df.set_index(first)
    else:
        df = pd.read_csv(path, index_col=index_col)

    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    df = df.replace(-1, np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")

    if not keep_other_features:
        typed_cols = [
            col
            for col in df.columns
            if col.startswith("S_")
            or col.startswith("P_")
            or col.startswith("M_")
            or col.startswith("INV_")
        ]
        if not typed_cols:
            raise ValueError("No typed URIEL+ columns found. Expected S_, P_, M_, or INV_.")
        df = df[typed_cols].copy()

    return df


def load_language_metadata(path: str | Path) -> pd.DataFrame:
    """Load languages.csv and normalize the metadata fields used by experiments."""
    lang = pd.read_csv(path)
    lang.columns = [str(col).strip() for col in lang.columns]

    if "ID" in lang.columns:
        language_id = _clean_string_series(lang["ID"])
    elif "Glottocode" in lang.columns:
        language_id = _clean_string_series(lang["Glottocode"])
    else:
        raise ValueError("languages.csv must contain either Glottocode or ID.")

    if "Glottocode" in lang.columns:
        fallback_id = _clean_string_series(lang["Glottocode"])
        language_id = language_id.fillna(fallback_id)

    lang["language_id"] = language_id
    lang = lang.dropna(subset=["language_id"]).drop_duplicates(subset=["language_id"])
    lang = lang.set_index("language_id", drop=False)

    raw_family = _clean_string_series(lang["Family_ID"]) if "Family_ID" in lang.columns else pd.Series(pd.NA, index=lang.index)
    is_isolate = lang["Is_Isolate"].map(_parse_bool) if "Is_Isolate" in lang.columns else pd.Series(False, index=lang.index)

    family_id = raw_family.copy()
    missing_family = family_id.isna()
    own_family = np.where(is_isolate.to_numpy(), "isolate::" + lang["language_id"].astype(str), "unknown::" + lang["language_id"].astype(str))
    family_id.loc[missing_family] = own_family[missing_family.to_numpy()]
    lang["family_id"] = family_id.astype(str)
    lang["raw_family_id"] = raw_family
    lang["is_isolate"] = is_isolate.astype(bool)

    if "Macroarea" in lang.columns:
        macroarea = _clean_string_series(lang["Macroarea"]).fillna("Unknown")
    else:
        macroarea = pd.Series("Unknown", index=lang.index)
    lang["macroarea"] = macroarea.astype(str)

    for col in ["Latitude", "Longitude"]:
        if col in lang.columns:
            lang[col.lower()] = pd.to_numeric(lang[col], errors="coerce")
        else:
            lang[col.lower()] = np.nan

    return lang


def _validate_coverage_cutoff(value: float, *, name: str) -> float:
    cutoff = float(value)
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1; got {cutoff}.")
    return cutoff


def apply_joint_coverage_filter(
    X: pd.DataFrame,
    *,
    language_min_coverage: float = 0.05,
    feature_min_coverage: float = 0.05,
) -> tuple[pd.DataFrame, CoverageFilterSummary]:
    """Apply the joint sweep's alternating inclusive coverage cutoff.

    Each iteration filters languages first against the currently retained
    features, then filters features against the retained languages.  The
    process repeats until both axes are stable.  Coverage is natural-data
    coverage only; this function is never called after artificial masking.
    """
    language_cutoff = _validate_coverage_cutoff(
        language_min_coverage,
        name="language_min_coverage",
    )
    feature_cutoff = _validate_coverage_cutoff(
        feature_min_coverage,
        name="feature_min_coverage",
    )
    if X.empty or X.shape[1] == 0:
        raise ValueError("The typological matrix must have languages and features.")

    original = X.copy()
    filtered = original.copy()
    history: list[dict[str, int]] = []
    while True:
        before_languages, before_features = filtered.shape
        language_coverage = filtered.notna().mean(axis=1)
        filtered = filtered.loc[language_coverage >= language_cutoff].copy()
        if filtered.empty:
            raise ValueError(
                "The joint coverage cutoff removed every language after "
                f"language_min_coverage={language_cutoff}."
            )

        feature_coverage = filtered.notna().mean(axis=0)
        filtered = filtered.loc[:, feature_coverage >= feature_cutoff].copy()
        if filtered.shape[1] == 0:
            raise ValueError(
                "The joint coverage cutoff removed every feature after "
                f"feature_min_coverage={feature_cutoff}."
            )

        history.append(
            {
                "iteration": len(history) + 1,
                "input_languages": int(before_languages),
                "input_features": int(before_features),
                "output_languages": int(filtered.shape[0]),
                "output_features": int(filtered.shape[1]),
                "observed_cells": int(filtered.notna().to_numpy().sum()),
            }
        )
        if filtered.shape == (before_languages, before_features):
            break

    final_language_coverage = filtered.notna().mean(axis=1)
    final_feature_coverage = filtered.notna().mean(axis=0)
    if (final_language_coverage < language_cutoff).any():
        raise AssertionError("A retained language violates the finalized cutoff.")
    if (final_feature_coverage < feature_cutoff).any():
        raise AssertionError("A retained feature violates the finalized cutoff.")

    original_observed = int(original.notna().to_numpy().sum())
    filtered_observed = int(filtered.notna().to_numpy().sum())
    retained_languages = tuple(filtered.index.astype(str))
    retained_features = tuple(filtered.columns.astype(str))
    retained_language_set = set(retained_languages)
    retained_feature_set = set(retained_features)
    summary = CoverageFilterSummary(
        language_min_coverage=language_cutoff,
        feature_min_coverage=feature_cutoff,
        input_languages=int(original.shape[0]),
        input_features=int(original.shape[1]),
        input_observed_cells=original_observed,
        input_missingness=float(1.0 - original_observed / original.size),
        output_languages=int(filtered.shape[0]),
        output_features=int(filtered.shape[1]),
        output_observed_cells=filtered_observed,
        output_missingness=float(1.0 - filtered_observed / filtered.size),
        retained_languages=retained_languages,
        dropped_languages=tuple(
            value
            for value in original.index.astype(str)
            if value not in retained_language_set
        ),
        retained_features=retained_features,
        dropped_features=tuple(
            value
            for value in original.columns.astype(str)
            if value not in retained_feature_set
        ),
        iterations=tuple(history),
    )
    return filtered, summary


def load_dataset(
    typological_path: str | Path,
    languages_path: str | Path,
    *,
    index_col: Optional[str] = None,
    keep_other_features: bool = False,
    language_min_coverage: float = 0.05,
    feature_min_coverage: float = 0.05,
) -> UrielDataset:
    """Load, normalize, and freeze the post-cutoff benchmark matrix.

    The experiment matrix is defined by typological_data.csv. languages.csv is
    treated as side information and is left-joined by Glottolog code, using
    typological_data.csv's language column and languages.csv's ID column.
    """
    X = load_typological_matrix(
        typological_path,
        index_col=index_col,
        keep_other_features=keep_other_features,
    )
    X, filter_summary = apply_joint_coverage_filter(
        X,
        language_min_coverage=language_min_coverage,
        feature_min_coverage=feature_min_coverage,
    )
    languages = load_language_metadata(languages_path)

    overlap = X.index.intersection(languages.index)
    if len(overlap) == 0:
        raise ValueError("No overlapping language IDs between typological_data.csv and languages.csv.")

    languages = languages.reindex(X.index).copy()
    languages["language_id"] = X.index.astype(str)
    languages["metadata_available"] = languages.index.isin(overlap)

    if "raw_family_id" not in languages.columns:
        languages["raw_family_id"] = pd.NA
    if "is_isolate" not in languages.columns:
        languages["is_isolate"] = False
    languages["is_isolate"] = languages["is_isolate"].map(_parse_bool).astype(bool)
    languages["raw_family_id"] = languages["raw_family_id"].astype("string")

    missing_family = languages["raw_family_id"].isna()
    fallback_family = np.where(
        languages["is_isolate"].to_numpy(),
        "isolate::" + languages["language_id"].astype(str),
        "unknown::" + languages["language_id"].astype(str),
    )
    languages["family_id"] = languages["raw_family_id"].copy()
    languages.loc[missing_family, "family_id"] = fallback_family[missing_family.to_numpy()]
    languages["family_id"] = languages["family_id"].fillna("unknown").astype(str)

    if "macroarea" not in languages.columns:
        languages["macroarea"] = "Unknown"
    languages["macroarea"] = languages["macroarea"].fillna("Unknown").astype(str)

    for col in ["latitude", "longitude"]:
        if col not in languages.columns:
            languages[col] = np.nan
        languages[col] = pd.to_numeric(languages[col], errors="coerce")

    feature_types = feature_type_series(X.columns)
    languages["coverage"] = X.notna().mean(axis=1).astype(float)
    languages["missingness_rate"] = 1.0 - languages["coverage"]

    family_size = languages.groupby("family_id")["language_id"].transform("count")
    languages["family_size"] = family_size.astype(int)
    languages["log_family_size"] = np.log1p(languages["family_size"].astype(float))

    def family_size_group(size: int, isolate: bool) -> str:
        if isolate or size <= 1:
            return "isolate_or_unknown"
        if size <= 4:
            return "small_family_2_4"
        if size <= 19:
            return "medium_family_5_19"
        return "large_family_20_plus"

    languages["family_size_group"] = [
        family_size_group(int(size), bool(isolate))
        for size, isolate in zip(languages["family_size"], languages["is_isolate"])
    ]

    q75 = languages["missingness_rate"].quantile(0.75)
    languages["missingness_group"] = np.where(
        languages["missingness_rate"] >= q75,
        "high_missing_top_quartile",
        "lower_missing_bottom_75pct",
    )

    return UrielDataset(
        X=X,
        languages=languages,
        feature_types=feature_types,
        filter_summary=filter_summary,
    )


__all__ = [
    "CoverageFilterSummary",
    "UrielDataset",
    "apply_joint_coverage_filter",
    "load_dataset",
    "load_language_metadata",
    "load_typological_matrix",
]
