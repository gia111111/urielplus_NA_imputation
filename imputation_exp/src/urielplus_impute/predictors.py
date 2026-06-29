from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .experiment import FULL_PREDICTOR_SET
from .feature_types import TYPE_ORDER


GeoBackend = Literal["knn", "macroarea"]


@dataclass
class PredictorConfig:
    k_geo: int = 50
    geo_backend: GeoBackend = "knn"
    distance_weighted_geo: bool = True
    top_corr_features: int = 32
    corr_shrinkage: float = 20.0
    min_corr_overlap: int = 20


NUMERIC_COLUMNS_FULL = [
    "feature_mean",
    "mu_phylo",
    "mu_geo",
    "mu_typcorr",
    "c_phylo",
    "c_geo",
    "c_typcorr",
    "log_family_size",
    "language_coverage",
    "language_missingness_rate",
    "has_coordinates",
    "mu_phylo_x_S",
    "mu_phylo_x_P",
    "mu_phylo_x_M",
    "mu_phylo_x_INV",
    "mu_geo_x_S",
    "mu_geo_x_P",
    "mu_geo_x_M",
    "mu_geo_x_INV",
    "mu_typcorr_x_S",
    "mu_typcorr_x_P",
    "mu_typcorr_x_M",
    "mu_typcorr_x_INV",
]

CATEGORICAL_COLUMNS_FULL = ["feature_type", "macroarea", "family_size_group", "missingness_group"]


def select_predictor_columns(predictor_set: str = FULL_PREDICTOR_SET) -> tuple[list[str], list[str]]:
    """Return the full predictor set described in the revised proposal."""
    predictor_set = predictor_set.strip().lower()
    if predictor_set == FULL_PREDICTOR_SET:
        return NUMERIC_COLUMNS_FULL.copy(), CATEGORICAL_COLUMNS_FULL.copy()
    raise ValueError(
        f"Unknown predictor set {predictor_set!r}. "
        f"The revised proposal keeps only {FULL_PREDICTOR_SET!r}."
    )


class ProposalPredictorBuilder:
    """Construct language-feature predictors from the revised proposal.

    The typological-correlation signal uses an efficient first-version
    conditional estimator: for target feature j, correlated observed features k
    in the same language vote via P(X_j = 1 | X_k = value). This keeps the
    proposal's feature-correlation information without the full O(p n^2)
    target-specific neighbor search.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        languages: pd.DataFrame,
        feature_types: pd.Series,
        config: PredictorConfig | None = None,
    ) -> None:
        self.X_train = X_train
        self.languages = languages.reindex(X_train.index)
        self.feature_types = feature_types.reindex(X_train.columns)
        self.config = config or PredictorConfig()

    def fit(self) -> "ProposalPredictorBuilder":
        self.columns = self.X_train.columns.to_numpy()
        self.index = self.X_train.index.to_numpy()
        self.X = self.X_train.to_numpy(dtype=float)
        self.obs = ~np.isnan(self.X)
        self.X_filled = np.nan_to_num(self.X, nan=0.0)
        self.n_rows, self.n_cols = self.X.shape

        observed_count = self.obs.sum(axis=0).astype(float)
        observed_sum = self.X_filled.sum(axis=0)
        overall = float(observed_sum.sum() / max(observed_count.sum(), 1.0))
        self.global_mean = np.divide(
            observed_sum,
            observed_count,
            out=np.full(self.n_cols, overall, dtype=float),
            where=observed_count > 0,
        )
        self.language_coverage = self.obs.mean(axis=1).astype(float)

        self._fit_metadata_arrays()
        self._fit_family_stats()
        self._fit_macroarea_stats()
        self._fit_geo_stats()
        self._fit_typcorr_stats()
        return self

    def _fit_metadata_arrays(self) -> None:
        meta = self.languages
        self.family = meta.get("family_id", pd.Series("unknown", index=meta.index)).fillna("unknown").astype(str).to_numpy()
        self.macroarea = meta.get("macroarea", pd.Series("Unknown", index=meta.index)).fillna("Unknown").astype(str).to_numpy()
        self.family_size_group = meta.get("family_size_group", pd.Series("unknown", index=meta.index)).fillna("unknown").astype(str).to_numpy()
        self.missingness_group = meta.get("missingness_group", pd.Series("unknown", index=meta.index)).fillna("unknown").astype(str).to_numpy()
        self.log_family_size = meta.get("log_family_size", pd.Series(0.0, index=meta.index)).fillna(0.0).astype(float).to_numpy()
        lat = meta.get("latitude", pd.Series(np.nan, index=meta.index))
        lon = meta.get("longitude", pd.Series(np.nan, index=meta.index))
        self.latitude = pd.to_numeric(lat, errors="coerce").to_numpy(dtype=float)
        self.longitude = pd.to_numeric(lon, errors="coerce").to_numpy(dtype=float)
        self.has_coordinates = (~np.isnan(self.latitude) & ~np.isnan(self.longitude)).astype(float)
        self.feature_type_by_col = self.feature_types.fillna("Other").astype(str).to_numpy()

    def _group_stats(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        codes, uniques = pd.factorize(labels, sort=False)
        sums = np.zeros((len(uniques), self.n_cols), dtype=float)
        counts = np.zeros((len(uniques), self.n_cols), dtype=float)
        for code in range(len(uniques)):
            rows = codes == code
            if not np.any(rows):
                continue
            sums[code] = self.X_filled[rows].sum(axis=0)
            counts[code] = self.obs[rows].sum(axis=0)
        return codes.astype(int), uniques.astype(str), sums, counts

    def _fit_family_stats(self) -> None:
        self.family_codes, self.family_labels, self.family_sums, self.family_counts = self._group_stats(self.family)
        self.family_sizes_by_row = np.bincount(self.family_codes, minlength=len(self.family_labels))[self.family_codes].astype(float)

    def _fit_macroarea_stats(self) -> None:
        self.macro_codes, self.macro_labels, self.macro_sums, self.macro_counts = self._group_stats(self.macroarea)

    def _group_mean_for_rows(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        codes: np.ndarray,
        sums: np.ndarray,
        counts: np.ndarray,
        exclude_self: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        group_codes = codes[rows]
        numer = sums[group_codes, cols].astype(float)
        denom = counts[group_codes, cols].astype(float)
        if exclude_self:
            values = self.X[rows, cols]
            self_observed = ~np.isnan(values)
            numer = numer - np.where(self_observed, values, 0.0)
            denom = denom - self_observed.astype(float)
        fallback = self.global_mean[cols]
        mean = np.divide(numer, denom, out=fallback.copy(), where=denom > 0)
        return np.clip(mean, 0.0, 1.0), np.maximum(denom, 0.0)

    def _fit_geo_stats(self) -> None:
        macro_rows = np.arange(self.n_rows)
        macro_cols = np.tile(np.arange(self.n_cols), self.n_rows)
        repeated_rows = np.repeat(macro_rows, self.n_cols)
        macro_mean_flat, macro_count_flat = self._group_mean_for_rows(
            repeated_rows,
            macro_cols,
            self.macro_codes,
            self.macro_sums,
            self.macro_counts,
            exclude_self=True,
        )
        macro_mean = macro_mean_flat.reshape(self.n_rows, self.n_cols)
        macro_count = macro_count_flat.reshape(self.n_rows, self.n_cols)

        if self.config.geo_backend == "macroarea" or self.config.k_geo <= 0:
            self.geo_mean = macro_mean
            self.geo_count = macro_count
            return

        valid = ~np.isnan(self.latitude) & ~np.isnan(self.longitude)
        valid_indices = np.where(valid)[0]
        if len(valid_indices) <= 1:
            self.geo_mean = macro_mean
            self.geo_count = macro_count
            return

        lat_rad = np.deg2rad(self.latitude)
        lon_rad = np.deg2rad(self.longitude)
        self.geo_mean = macro_mean.copy()
        self.geo_count = macro_count.copy()
        k_geo = int(min(self.config.k_geo, len(valid_indices) - 1))

        for row in range(self.n_rows):
            if not valid[row]:
                continue
            dists = self._haversine_km(lat_rad[row], lon_rad[row], lat_rad[valid_indices], lon_rad[valid_indices])
            not_self = valid_indices != row
            candidates = valid_indices[not_self]
            candidate_dists = dists[not_self]
            if len(candidates) == 0:
                continue
            if len(candidates) > k_geo:
                take = np.argpartition(candidate_dists, k_geo - 1)[:k_geo]
                candidates = candidates[take]
                candidate_dists = candidate_dists[take]
            if self.config.distance_weighted_geo:
                weights = 1.0 / (candidate_dists + 1.0)
            else:
                weights = np.ones(len(candidates), dtype=float)
            neighbor_obs = self.obs[candidates]
            denom = (neighbor_obs * weights[:, None]).sum(axis=0)
            numer = (self.X_filled[candidates] * weights[:, None]).sum(axis=0)
            mean = np.divide(numer, denom, out=self.global_mean.copy(), where=denom > 0)
            count = neighbor_obs.sum(axis=0).astype(float)
            self.geo_mean[row] = np.clip(mean, 0.0, 1.0)
            self.geo_count[row] = count

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2):
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    def _fit_typcorr_stats(self) -> None:
        p = self.n_cols
        top_k = int(max(0, self.config.top_corr_features))
        self.abs_corr = np.zeros((p, p), dtype=float)
        self.cond_mean_if_one = np.tile(self.global_mean[:, None], (1, p))
        self.cond_mean_if_zero = np.tile(self.global_mean[:, None], (1, p))
        self.top_corr_indices = [np.array([], dtype=int) for _ in range(p)]
        if top_k == 0:
            return

        obs_float = self.obs.astype(float)
        pair_count = obs_float.T @ obs_float
        sum_j = self.X_filled.T @ obs_float
        sum_k = obs_float.T @ self.X_filled
        sum_xy = self.X_filled.T @ self.X_filled

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_j = np.divide(sum_j, pair_count, out=np.zeros_like(sum_j), where=pair_count > 0)
            mean_k = np.divide(sum_k, pair_count, out=np.zeros_like(sum_k), where=pair_count > 0)
            cov = np.divide(sum_xy, pair_count, out=np.zeros_like(sum_xy), where=pair_count > 0) - mean_j * mean_k
            var_j = mean_j * (1.0 - mean_j)
            var_k = mean_k * (1.0 - mean_k)
            corr = np.divide(cov, np.sqrt(var_j * var_k), out=np.zeros_like(cov), where=(var_j > 0) & (var_k > 0))

        corr *= pair_count / (pair_count + float(self.config.corr_shrinkage))
        corr[pair_count < int(self.config.min_corr_overlap)] = 0.0
        np.fill_diagonal(corr, 0.0)
        self.abs_corr = np.abs(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))

        count_k1 = obs_float.T @ self.X_filled
        count_k0 = pair_count - count_k1
        sum_j_when_k1 = sum_xy
        sum_j_when_k0 = self.X_filled.T @ (obs_float - self.X_filled)
        lam = float(self.config.corr_shrinkage)
        global_by_target = self.global_mean[:, None]

        with np.errstate(divide="ignore", invalid="ignore"):
            raw_one = np.divide(sum_j_when_k1, count_k1, out=global_by_target.repeat(p, axis=1), where=count_k1 > 0)
            raw_zero = np.divide(sum_j_when_k0, count_k0, out=global_by_target.repeat(p, axis=1), where=count_k0 > 0)
        shrink_one = count_k1 / (count_k1 + lam)
        shrink_zero = count_k0 / (count_k0 + lam)
        self.cond_mean_if_one = shrink_one * raw_one + (1.0 - shrink_one) * global_by_target
        self.cond_mean_if_zero = shrink_zero * raw_zero + (1.0 - shrink_zero) * global_by_target

        for target_col in range(p):
            weights = self.abs_corr[target_col].copy()
            nonzero = np.flatnonzero(weights > 0)
            if len(nonzero) == 0:
                continue
            if len(nonzero) > top_k:
                take = np.argpartition(weights[nonzero], -top_k)[-top_k:]
                nonzero = nonzero[take]
            order = np.argsort(weights[nonzero])[::-1]
            self.top_corr_indices[target_col] = nonzero[order].astype(int)

    def build(self, cells: pd.DataFrame) -> pd.DataFrame:
        """Build predictors for a cell mask or observed-cell table."""
        rows, cols = self._cell_indices(cells)
        out = pd.DataFrame(index=np.arange(len(rows)))
        out["row_idx"] = rows
        out["col_idx"] = cols
        out["language"] = self.index[rows]
        out["feature"] = self.columns[cols]
        out["feature_type"] = self.feature_type_by_col[cols]
        if "true_value" in cells.columns:
            out["true_value"] = cells["true_value"].to_numpy()

        out["feature_mean"] = self.global_mean[cols]
        mu_phylo, c_phylo = self._group_mean_for_rows(
            rows, cols, self.family_codes, self.family_sums, self.family_counts, exclude_self=True
        )
        out["mu_phylo"] = mu_phylo
        out["c_phylo"] = np.log1p(c_phylo)
        out["mu_geo"] = self.geo_mean[rows, cols]
        out["c_geo"] = np.log1p(self.geo_count[rows, cols])
        mu_typcorr, c_typcorr = self._typcorr_for_cells(rows, cols)
        out["mu_typcorr"] = mu_typcorr
        out["c_typcorr"] = np.log1p(c_typcorr)

        out["log_family_size"] = self.log_family_size[rows]
        out["language_coverage"] = self.language_coverage[rows]
        out["language_missingness_rate"] = 1.0 - self.language_coverage[rows]
        out["has_coordinates"] = self.has_coordinates[rows]
        out["macroarea"] = self.macroarea[rows]
        out["family_size_group"] = self.family_size_group[rows]
        out["missingness_group"] = self.missingness_group[rows]

        for feature_type in TYPE_ORDER:
            mask = out["feature_type"].to_numpy() == feature_type
            out[f"mu_phylo_x_{feature_type}"] = np.where(mask, out["mu_phylo"], 0.0)
            out[f"mu_geo_x_{feature_type}"] = np.where(mask, out["mu_geo"], 0.0)
            out[f"mu_typcorr_x_{feature_type}"] = np.where(mask, out["mu_typcorr"], 0.0)

        return out

    def _cell_indices(self, cells: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if "row_idx" in cells.columns and "col_idx" in cells.columns:
            return cells["row_idx"].astype(int).to_numpy(), cells["col_idx"].astype(int).to_numpy()
        if "language" not in cells.columns or "feature" not in cells.columns:
            raise ValueError("Cells must contain row_idx/col_idx or language/feature columns.")
        row_lookup = pd.Series(np.arange(self.n_rows), index=self.X_train.index)
        col_lookup = pd.Series(np.arange(self.n_cols), index=self.X_train.columns)
        rows = row_lookup.loc[cells["language"].astype(str)].to_numpy(dtype=int)
        cols = col_lookup.loc[cells["feature"].astype(str)].to_numpy(dtype=int)
        return rows, cols

    def _typcorr_for_cells(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = self.global_mean[cols].copy()
        conf = np.zeros(len(rows), dtype=float)
        if self.config.top_corr_features <= 0:
            return mu, conf

        for target_col in np.unique(cols):
            loc = np.where(cols == target_col)[0]
            corr_cols = self.top_corr_indices[int(target_col)]
            if len(corr_cols) == 0:
                continue
            weights = self.abs_corr[int(target_col), corr_cols]
            row_subset = rows[loc]
            values = self.X[np.ix_(row_subset, corr_cols)]
            observed = ~np.isnan(values)
            probs = np.where(
                values >= 0.5,
                self.cond_mean_if_one[int(target_col), corr_cols],
                self.cond_mean_if_zero[int(target_col), corr_cols],
            )
            denom = (observed * weights).sum(axis=1)
            numer = np.where(observed, probs * weights, 0.0).sum(axis=1)
            valid = denom > 0
            mu[loc[valid]] = numer[valid] / denom[valid]
            conf[loc] = observed.sum(axis=1).astype(float)
        return np.clip(mu, 0.0, 1.0), conf
