from __future__ import annotations

import numpy as np
import pandas as pd


class SoftImpute:
    """NumPy SoftImpute implementation for the binary URIEL+ matrix."""

    name = "softimpute"

    def __init__(
        self,
        *,
        shrinkage: float = 1.0,
        max_rank: int | None = None,
        max_iters: int = 400,
        tol: float = 1e-4,
        verbose: bool = False,
    ) -> None:
        self.shrinkage = float(shrinkage)
        self.max_rank = max_rank
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.verbose = bool(verbose)

    def fit(self, X_train: pd.DataFrame) -> "SoftImpute":
        values = X_train.to_numpy(dtype=float)
        observed = ~np.isnan(values)
        missing = ~observed
        filled = np.nan_to_num(values, nan=0.0)
        counts = observed.sum(axis=0).astype(float)
        sums = filled.sum(axis=0)
        overall = float(sums.sum() / max(counts.sum(), 1.0))
        feature_means = np.divide(
            sums,
            counts,
            out=np.full(values.shape[1], overall),
            where=counts > 0,
        )
        current = np.clip(
            np.where(observed, values, feature_means[None, :]),
            0.0,
            1.0,
        )
        if not np.any(missing):
            self.imputed_matrix_ = current
            self.n_iters_ = 0
            self.relative_change_ = 0.0
            self.effective_rank_ = int(np.linalg.matrix_rank(current))
            return self

        previous_missing = current[missing].copy()
        relative_change = np.inf
        effective_rank = 0
        for iteration in range(1, self.max_iters + 1):
            u, singular_values, vt = np.linalg.svd(current, full_matrices=False)
            shrunk = np.maximum(singular_values - self.shrinkage, 0.0)
            keep = np.flatnonzero(shrunk > 0.0)
            if self.max_rank is not None:
                keep = keep[: int(self.max_rank)]
            effective_rank = int(len(keep))
            if len(keep):
                reconstructed = (u[:, keep] * shrunk[keep]) @ vt[keep, :]
            else:
                reconstructed = np.tile(feature_means[None, :], (values.shape[0], 1))
            reconstructed = np.clip(reconstructed, 0.0, 1.0)
            current = np.where(observed, values, reconstructed)

            current_missing = current[missing]
            denominator = max(float(np.linalg.norm(previous_missing)), 1e-12)
            relative_change = float(
                np.linalg.norm(current_missing - previous_missing) / denominator
            )
            previous_missing = current_missing.copy()
            if self.verbose:
                print(
                    f"[softimpute] iter={iteration} "
                    f"relative_change={relative_change:.6g} rank={effective_rank}"
                )
            if relative_change < self.tol:
                break

        self.imputed_matrix_ = np.clip(current, 0.0, 1.0)
        self.n_iters_ = int(iteration)
        self.relative_change_ = relative_change
        self.effective_rank_ = effective_rank
        return self

    def predict_cells(self, cells: pd.DataFrame) -> np.ndarray:
        rows = cells["row_idx"].astype(int).to_numpy()
        columns = cells["col_idx"].astype(int).to_numpy()
        return np.clip(self.imputed_matrix_[rows, columns], 0.0, 1.0)


def softimpute_parameter_grid(
    shrinkages: list[float],
    *,
    max_rank: int | None,
    max_iters: int,
    tol: float,
    verbose: bool,
) -> list[dict]:
    return [
        {
            "shrinkage": float(shrinkage),
            "max_rank": max_rank,
            "max_iters": int(max_iters),
            "tol": float(tol),
            "verbose": bool(verbose),
        }
        for shrinkage in shrinkages
    ]

