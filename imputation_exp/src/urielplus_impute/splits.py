from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping

import numpy as np
import pandas as pd

from .feature_types import FEATURE_TYPES, as_feature_type_array
from .masking import (
    ADAPTATION_BUDGETS,
    EVALUATION_SPLITS,
    InfeasibleMaskError,
    RESOURCE_GROUPS,
    SplitMasks,
    make_fewshot_local_masks,
    make_resource_conditioned_copy_mask,
    make_stratified_mcar_mask,
    validate_split_masks,
)


@dataclass(frozen=True)
class StratumQuotas:
    validation: int = 500
    calibration: int = 500
    test: int = 1_000

    def as_dict(self) -> dict[str, int]:
        values = {
            "validation": int(self.validation),
            "calibration": int(self.calibration),
            "test": int(self.test),
        }
        if any(value < 0 for value in values.values()):
            raise ValueError(f"Split quotas must be non-negative; got {values}.")
        if sum(values.values()) <= 0:
            raise ValueError("At least one split quota must be positive.")
        return values

    @property
    def scored_per_stratum(self) -> int:
        return sum(self.as_dict().values())


@dataclass(frozen=True)
class ResourceStratification:
    groups: np.ndarray
    observed_counts: np.ndarray
    post_mask_is_p2: np.ndarray
    minimum_remaining_counts: np.ndarray
    midpoint: int
    boundary_count: int
    boundary_tie_total: int
    boundary_tie_p1: int
    boundary_tie_p2: int

    def post_mask_group(self, row: int, observed_count: int) -> str:
        count = int(observed_count)
        if count < 0 or count >= self.post_mask_is_p2.shape[1]:
            raise ValueError(
                f"observed_count must be in [0, {self.post_mask_is_p2.shape[1]}); "
                f"got {count}."
            )
        return "P2" if self.post_mask_is_p2[int(row), count] else "P1"


def build_resource_stratification(observed_mask: np.ndarray) -> ResourceStratification:
    """Assign post-filter P1/P2 and build exact stable-tie post-mask lookup.

    Source-row order breaks equal-coverage ties.  With an odd population,
    ``n // 2`` languages enter P1 and the extra language enters P2.  For a
    hypothetical post-mask count, the lookup re-ranks that target against the
    frozen natural counts using the same stable source-row tie rule.
    """
    observed = np.asarray(observed_mask, dtype=bool)
    if observed.ndim != 2:
        raise ValueError(f"observed_mask must be two-dimensional; got {observed.shape}.")
    counts = observed.sum(axis=1).astype(int)
    if not len(counts):
        raise ValueError("Resource stratification requires at least one language.")
    if np.any(counts <= 0):
        raise ValueError(
            "Resource stratification requires the frozen post-cutoff matrix; "
            "zero-coverage languages are not part of the benchmark."
        )

    ordered = np.argsort(counts, kind="stable")
    midpoint = len(ordered) // 2
    groups = np.full(len(counts), "P2", dtype="<U2")
    groups[ordered[:midpoint]] = "P1"
    boundary_count = int(counts[ordered[midpoint]])

    n_features = observed.shape[1]
    row_indices = np.arange(len(counts))
    post_mask_is_p2 = np.zeros((len(counts), n_features + 1), dtype=bool)
    for hypothetical_count in range(n_features + 1):
        less_total = int((counts < hypothetical_count).sum())
        other_less = less_total - (counts < hypothetical_count).astype(int)
        equal_before = np.cumsum(counts == hypothetical_count).astype(int)
        equal_before = np.concatenate(([0], equal_before[:-1]))
        ranks = other_less + equal_before
        post_mask_is_p2[:, hypothetical_count] = ranks >= midpoint

    unchanged = np.where(
        post_mask_is_p2[row_indices, counts],
        "P2",
        "P1",
    )
    if not np.array_equal(unchanged, groups):
        raise AssertionError("Post-mask tie lookup disagrees with natural P1/P2 assignment.")

    minimum_remaining = np.ones(len(counts), dtype=int)
    for row in np.flatnonzero(groups == "P2"):
        eligible_counts = np.flatnonzero(post_mask_is_p2[row])
        eligible_counts = eligible_counts[eligible_counts > 0]
        if not len(eligible_counts):
            raise AssertionError(f"P2 language {row} has no positive P2 count.")
        minimum_remaining[row] = int(eligible_counts.min())

    boundary_tie = counts == boundary_count
    return ResourceStratification(
        groups=groups,
        observed_counts=counts,
        post_mask_is_p2=post_mask_is_p2,
        minimum_remaining_counts=minimum_remaining,
        midpoint=midpoint,
        boundary_count=boundary_count,
        boundary_tie_total=int(boundary_tie.sum()),
        boundary_tie_p1=int((boundary_tie & (groups == "P1")).sum()),
        boundary_tie_p2=int((boundary_tie & (groups == "P2")).sum()),
    )


def resource_group_summary(
    observed_mask: np.ndarray,
    feature_types: pd.Series | np.ndarray,
    stratification: ResourceStratification,
) -> dict[str, dict]:
    observed = np.asarray(observed_mask, dtype=bool)
    types = as_feature_type_array(feature_types, observed.shape[1])
    result: dict[str, dict] = {}
    for group in RESOURCE_GROUPS:
        rows = stratification.groups == group
        group_counts = stratification.observed_counts[rows]
        result[group] = {
            "languages": int(rows.sum()),
            "min_observed_cells": int(group_counts.min()),
            "max_observed_cells": int(group_counts.max()),
            "min_coverage": float(group_counts.min() / observed.shape[1]),
            "max_coverage": float(group_counts.max() / observed.shape[1]),
            "median_observed_cells": float(np.median(group_counts)),
            "median_coverage": float(np.median(group_counts) / observed.shape[1]),
            "observed_cells": int(group_counts.sum()),
            "observed_by_feature_type": {
                target_type: int(
                    observed[np.ix_(rows, types == target_type)].sum()
                )
                for target_type in FEATURE_TYPES
            },
        }
    result["boundary"] = {
        "rule": "stable ascending natural count; source-row ties; extra language in P2; hypothetical target re-ranked with the same stable rule",
        "midpoint": int(stratification.midpoint),
        "boundary_count": int(stratification.boundary_count),
        "boundary_tie_total": int(stratification.boundary_tie_total),
        "boundary_tie_p1": int(stratification.boundary_tie_p1),
        "boundary_tie_p2": int(stratification.boundary_tie_p2),
    }
    return result


@dataclass
class SeedSplitSummary:
    seed: int
    quotas_per_stratum: dict[str, int]
    resource_summary: dict[str, dict]
    generated_regimes: list[str] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)


def _expected_stratum_counts(
    masks: SplitMasks,
    quotas: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    if masks.resource_group is not None and masks.target_feature_type is not None:
        keys = [f"{masks.resource_group}:{masks.target_feature_type}"]
    else:
        keys = [
            f"{group}:{target_type}"
            for group in RESOURCE_GROUPS
            for target_type in FEATURE_TYPES
        ]
    return {
        split_name: {key: int(quotas[split_name]) for key in keys}
        for split_name in EVALUATION_SPLITS
    }


def validate_stratified_masks(
    masks: SplitMasks,
    feature_types: pd.Series | np.ndarray,
    quotas: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    expected = _expected_stratum_counts(masks, quotas)
    multiplier = 1 if masks.resource_group is not None else len(RESOURCE_GROUPS) * len(FEATURE_TYPES)
    validate_split_masks(
        masks,
        feature_types,
        expected_val_size=int(quotas["validation"]) * multiplier,
        expected_cal_size=int(quotas["calibration"]) * multiplier,
        expected_test_size=int(quotas["test"]) * multiplier,
        expected_stratum_counts=expected,
    )
    if masks.adaptation_budget is not None:
        if int(masks.adaptation_mask.sum()) != int(masks.adaptation_budget):
            raise AssertionError(
                f"{masks.regime} has {int(masks.adaptation_mask.sum())} restored "
                f"cells; expected {masks.adaptation_budget}."
            )
        if not np.all(masks.adaptation_mask <= masks.observed_mask):
            raise AssertionError(f"{masks.regime} restores naturally missing cells.")
    return expected


def iter_stratified_seed_masks(
    X: pd.DataFrame,
    feature_types: pd.Series,
    languages: pd.DataFrame,
    *,
    seed: int,
    quotas: StratumQuotas = StratumQuotas(),
    adaptation_budgets: Iterable[int] = ADAPTATION_BUDGETS,
    regimes: Iterable[str] = ("mcar", "resource_copy", "local_fewshot"),
    summary: SeedSplitSummary | None = None,
) -> Iterator[SplitMasks]:
    observed = X.notna().to_numpy(dtype=bool)
    types = as_feature_type_array(feature_types, X.shape[1])
    stratification = build_resource_stratification(observed)
    groups = stratification.groups
    quota_values = quotas.as_dict()
    requested = tuple(dict.fromkeys(str(value) for value in regimes))
    unsupported = set(requested).difference({"mcar", "resource_copy", "local_fewshot"})
    if unsupported:
        raise ValueError(
            f"Unsupported masking regimes {sorted(unsupported)}; expected mcar, "
            "resource_copy, and/or local_fewshot."
        )
    if summary is None:
        summary = SeedSplitSummary(
            seed=int(seed),
            quotas_per_stratum=quota_values,
            resource_summary=resource_group_summary(
                observed,
                types,
                stratification,
            ),
        )

    if "mcar" in requested:
        masks = make_stratified_mcar_mask(
            observed,
            types,
            groups,
            minimum_remaining_counts=stratification.minimum_remaining_counts,
            seed=seed,
            quotas=quota_values,
        )
        validate_stratified_masks(masks, types, quota_values)
        summary.generated_regimes.append(masks.regime)
        yield masks

    if "resource_copy" in requested:
        try:
            masks = make_resource_conditioned_copy_mask(
                observed,
                types,
                groups,
                languages.reindex(X.index),
                post_mask_is_p2=stratification.post_mask_is_p2,
                seed=seed,
                quotas=quota_values,
            )
        except InfeasibleMaskError as exc:
            summary.failures.append(
                {
                    "seed": int(seed),
                    "regime": "resource_copy",
                    "reason": str(exc),
                    "capacities": exc.capacities,
                }
            )
        else:
            validate_stratified_masks(masks, types, quota_values)
            summary.generated_regimes.append(masks.regime)
            yield masks

    if "local_fewshot" in requested:
        for masks in make_fewshot_local_masks(
            observed,
            types,
            groups,
            seed=seed,
            quotas=quota_values,
            adaptation_budgets=adaptation_budgets,
        ):
            validate_stratified_masks(masks, types, quota_values)
            summary.generated_regimes.append(masks.regime)
            yield masks


def make_seed_summary(
    X: pd.DataFrame,
    feature_types: pd.Series,
    *,
    seed: int,
    quotas: StratumQuotas,
) -> SeedSplitSummary:
    observed = X.notna().to_numpy(dtype=bool)
    stratification = build_resource_stratification(observed)
    return SeedSplitSummary(
        seed=int(seed),
        quotas_per_stratum=quotas.as_dict(),
        resource_summary=resource_group_summary(
            observed,
            feature_types,
            stratification,
        ),
    )


def apply_train_visible_mask(
    X: pd.DataFrame,
    train_visible_mask: np.ndarray,
) -> pd.DataFrame:
    visible = np.asarray(train_visible_mask, dtype=bool)
    if visible.shape != X.shape:
        raise ValueError(
            f"train_visible_mask has shape {visible.shape}; expected {X.shape}."
        )
    values = X.to_numpy(dtype=float, copy=True)
    values[~visible] = np.nan
    return pd.DataFrame(
        values,
        index=X.index.copy(),
        columns=X.columns.copy(),
    )


def mask_to_cell_df(
    mask: np.ndarray,
    X: pd.DataFrame,
    feature_types: pd.Series | np.ndarray,
    *,
    split: str | None = None,
    regime: str | None = None,
    seed: int | None = None,
    resource_groups: np.ndarray | None = None,
    scored_resource_groups: np.ndarray | None = None,
) -> pd.DataFrame:
    values_mask = np.asarray(mask, dtype=bool)
    if values_mask.shape != X.shape:
        raise ValueError(f"mask has shape {values_mask.shape}; expected {X.shape}.")
    types = as_feature_type_array(feature_types, X.shape[1])
    rows, columns = np.where(values_mask)
    values = X.to_numpy(dtype=float)
    cells = pd.DataFrame(
        {
            "row_idx": rows.astype(int),
            "col_idx": columns.astype(int),
            "language": X.index.to_numpy()[rows],
            "feature": X.columns.to_numpy()[columns],
            "feature_type": types[columns],
            "true_value": values[rows, columns],
        }
    )
    if resource_groups is not None:
        groups = np.asarray(resource_groups, dtype=str)
        if len(groups) != X.shape[0]:
            raise ValueError(
                f"resource_groups has length {len(groups)}; expected {X.shape[0]}."
            )
        cells["target_original_resource_group"] = groups[rows]
    if scored_resource_groups is not None:
        scoring = np.asarray(scored_resource_groups, dtype=str)
        if scoring.shape != X.shape:
            raise ValueError(
                "scored_resource_groups has shape "
                f"{scoring.shape}; expected {X.shape}."
            )
        cells["resource_group"] = scoring[rows, columns]
    if split is not None:
        cells["split"] = split
    if regime is not None:
        cells["regime"] = regime
    if seed is not None:
        cells["seed"] = int(seed)
    return cells.reset_index(drop=True)


def observed_cells_df(
    X: pd.DataFrame,
    feature_types: pd.Series,
) -> pd.DataFrame:
    return mask_to_cell_df(
        X.notna().to_numpy(dtype=bool),
        X,
        feature_types,
    )


__all__ = [
    "ResourceStratification",
    "SeedSplitSummary",
    "StratumQuotas",
    "apply_train_visible_mask",
    "build_resource_stratification",
    "iter_stratified_seed_masks",
    "make_seed_summary",
    "mask_to_cell_df",
    "observed_cells_df",
    "resource_group_summary",
    "validate_stratified_masks",
]
