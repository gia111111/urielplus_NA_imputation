from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .feature_types import SPECIAL_FAMILY_IDS, feature_type_series


@dataclass
class UrielDataset:
    X: pd.DataFrame
    languages: pd.DataFrame
    feature_types: pd.Series
    n_special_filtered: int = 0


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


def load_dataset(
    typological_path: str | Path,
    languages_path: str | Path,
    *,
    index_col: Optional[str] = None,
    drop_empty_languages: bool = False,
    drop_empty_features: bool = True,
    keep_other_features: bool = False,
    filter_special_families: bool = True,
    special_family_ids: set[str] = SPECIAL_FAMILY_IDS,
) -> UrielDataset:
    """Load the typological matrix and append language metadata.

    The experiment matrix is defined by typological_data.csv. languages.csv is
    treated as side information and is left-joined by Glottolog code, using
    typological_data.csv's language column and languages.csv's ID column.
    """
    X = load_typological_matrix(
        typological_path,
        index_col=index_col,
        keep_other_features=keep_other_features,
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

    n_special_filtered = 0
    if filter_special_families:
        raw_family_for_filter = languages["raw_family_id"].astype("string")
        id_for_filter = languages["language_id"].astype("string")
        keep = ~raw_family_for_filter.isin(special_family_ids) & ~id_for_filter.isin(special_family_ids)
        n_special_filtered = int((~keep).sum())
        X = X.loc[keep].copy()
        languages = languages.loc[keep].copy()

    if drop_empty_languages:
        nonempty_rows = X.notna().any(axis=1)
        X = X.loc[nonempty_rows].copy()
        languages = languages.loc[nonempty_rows].copy()

    if drop_empty_features:
        X = X.loc[:, X.notna().any(axis=0)].copy()

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
        n_special_filtered=n_special_filtered,
    )
