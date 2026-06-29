from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .feature_types import FEATURE_TYPES, as_feature_type_array, feature_type_from_regime
from .masking import (
    SplitMasks,
    TargetLanguageSelection,
    derived_seed,
    equalize_training_input_size,
    make_empirical_copy_mask,
    make_global_block_mask,
    make_local_block_mask,
    make_mcar_mask,
    select_shared_target_languages,
    validate_split_masks,
)


@dataclass(frozen=True)
class SeedSplitSummary:
    seed: int
    requested_heldout_budget: int
    shared_heldout_budget: int
    n_val: int
    n_test: int
    scoring_capacities: dict[str, int]


def requested_heldout_budget(n_observed: int, val_frac: float, test_frac: float) -> int:
    if n_observed <= 0:
        raise ValueError("No observed typological cells are available for masking.")
    if val_frac < 0 or test_frac < 0:
        raise ValueError("val_frac and test_frac must be non-negative.")
    if val_frac + test_frac <= 0:
        raise ValueError("At least one of val_frac or test_frac must be positive.")
    if val_frac + test_frac >= 1:
        raise ValueError("val_frac + test_frac must be less than 1.0.")
    return max(1, int(round((val_frac + test_frac) * n_observed)))


def split_budget(total_budget: int, val_frac: float, test_frac: float) -> tuple[int, int]:
    if total_budget <= 0:
        raise ValueError("total_budget must be positive.")
    n_val = int(round(total_budget * val_frac / (val_frac + test_frac)))
    if val_frac > 0 and test_frac > 0 and total_budget > 1:
        n_val = min(max(n_val, 1), total_budget - 1)
    return n_val, total_budget - n_val


def _target_row_mask(n_languages: int, target_languages: np.ndarray) -> np.ndarray:
    rows = np.zeros(n_languages, dtype=bool)
    rows[np.asarray(target_languages, dtype=int)] = True
    return rows


def _target_pool_source(selection: TargetLanguageSelection) -> str:
    return "shared" if selection.shared_pool_used else "type_specific_fallback_union"


def _regime_scoring_capacity(
    regime: str,
    observed_mask: np.ndarray,
    feature_types: np.ndarray,
    selection: TargetLanguageSelection,
    min_remaining_input: int,
) -> int:
    if regime == "mcar":
        rows = _target_row_mask(observed_mask.shape[0], selection.matched_target_languages)
        return int((observed_mask & rows[:, None]).sum())
    if regime == "empirical_copy":
        copyable_columns = (~observed_mask).any(axis=0)
        capacity = 0
        for row in selection.matched_target_languages:
            known_count = int(observed_mask[row].sum())
            copyable_count = int((observed_mask[row] & copyable_columns).sum())
            capacity += min(copyable_count, max(known_count - min_remaining_input, 0))
        return capacity
    target_type = feature_type_from_regime(regime)
    rows = _target_row_mask(observed_mask.shape[0], selection.for_type(target_type))
    return int((observed_mask[:, feature_types == target_type] & rows[:, None]).sum())


def compute_shared_heldout_budget(
    X: pd.DataFrame,
    feature_types: pd.Series,
    regimes: Iterable[str],
    *,
    val_frac: float,
    test_frac: float,
    min_cells_per_unit: int,
    min_remaining_input: int = 2,
    n_target_languages: int | None = None,
    seed: int = 0,
) -> tuple[int, dict[str, int], int, TargetLanguageSelection]:
    observed = X.notna().to_numpy(dtype=bool)
    types = as_feature_type_array(feature_types, X.shape[1])
    selection = select_shared_target_languages(
        observed,
        types,
        seed,
        min_cells_per_type=min_cells_per_unit,
        min_remaining_input=min_remaining_input,
        n_target_languages=n_target_languages,
    )
    regime_names = list(dict.fromkeys(regimes))
    if not regime_names:
        raise ValueError("At least one masking regime is required.")
    capacities = {
        regime: _regime_scoring_capacity(
            regime,
            observed,
            types,
            selection,
            min_remaining_input,
        )
        for regime in regime_names
    }
    requested = requested_heldout_budget(int(observed.sum()), val_frac, test_frac)
    shared_budget = min(requested, min(capacities.values()))
    if shared_budget <= 0:
        raise ValueError(f"No requested masking regime has positive capacity: {capacities}")
    return int(shared_budget), capacities, int(requested), selection


def _make_regime_masks(
    regime: str,
    observed_mask: np.ndarray,
    feature_types: pd.Series | np.ndarray,
    selection: TargetLanguageSelection,
    *,
    seed: int,
    n_val: int,
    n_test: int,
    min_cells_per_unit: int,
    min_remaining_input: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    regime_seed = derived_seed(seed, regime)
    if regime == "mcar":
        targets = selection.matched_target_languages
        val, test = make_mcar_mask(
            observed_mask,
            n_val,
            n_test,
            regime_seed,
            target_languages=targets,
        )
        return val | test, val, test, targets, _target_pool_source(selection)
    if regime == "empirical_copy":
        targets = selection.matched_target_languages
        removed, val, test = make_empirical_copy_mask(
            observed_mask,
            targets,
            n_val,
            n_test,
            regime_seed,
            min_cells_per_unit=min_cells_per_unit,
            min_remaining_input=min_remaining_input,
        )
        return removed, val, test, targets, _target_pool_source(selection)
    if regime.startswith("Local_block_"):
        target_type = feature_type_from_regime(regime)
        targets = selection.for_type(target_type)
        removed, val, test = make_local_block_mask(
            observed_mask,
            feature_types,
            target_type,
            targets,
            n_val,
            n_test,
            regime_seed,
            min_cells_per_unit=min_cells_per_unit,
            min_remaining_input=min_remaining_input,
        )
        return removed, val, test, targets, selection.pool_source_by_type[target_type]
    if regime.startswith("Global_block_"):
        target_type = feature_type_from_regime(regime)
        targets = selection.for_type(target_type)
        removed, val, test = make_global_block_mask(
            observed_mask,
            feature_types,
            target_type,
            n_val,
            n_test,
            regime_seed,
            target_languages_for_scoring=targets,
        )
        return removed, val, test, targets, selection.pool_source_by_type[target_type]
    raise ValueError(
        f"Unknown regime {regime!r}; expected mcar, empirical_copy, "
        "Local_block_<type>, or Global_block_<type>."
    )


def generate_equalized_regime_masks(
    observed_mask: np.ndarray,
    feature_types: pd.Series | np.ndarray,
    regimes: Iterable[str],
    *,
    seed: int,
    n_val: int,
    n_test: int,
    min_cells_per_unit: int = 3,
    min_remaining_input: int = 2,
    n_target_languages: int | None = None,
    selection: TargetLanguageSelection | None = None,
) -> tuple[dict[str, SplitMasks], TargetLanguageSelection]:
    observed = np.asarray(observed_mask, dtype=bool)
    if observed.ndim != 2:
        raise ValueError(f"observed_mask must be two-dimensional; got {observed.shape}.")
    as_feature_type_array(feature_types, observed.shape[1])
    regime_names = list(dict.fromkeys(regimes))
    if not regime_names:
        raise ValueError("At least one masking regime is required.")
    if selection is None:
        selection = select_shared_target_languages(
            observed,
            feature_types,
            seed,
            min_cells_per_type=min_cells_per_unit,
            min_remaining_input=min_remaining_input,
            n_target_languages=n_target_languages,
        )

    base_masks = {
        regime: _make_regime_masks(
            regime,
            observed,
            feature_types,
            selection,
            seed=seed,
            n_val=n_val,
            n_test=n_test,
            min_cells_per_unit=min_cells_per_unit,
            min_remaining_input=min_remaining_input,
        )
        for regime in regime_names
    }
    equalized_removed_count = max(int(values[0].sum()) for values in base_masks.values())
    masks_by_regime: dict[str, SplitMasks] = {}
    for regime, (regime_removed, val, test, targets, pool_source) in base_masks.items():
        equalization_removed = equalize_training_input_size(
            observed,
            regime_removed,
            val,
            test,
            equalized_removed_count,
            derived_seed(seed, f"{regime}:equalization"),
            targets,
        )
        train_removed = regime_removed | equalization_removed
        train_visible = observed & ~train_removed
        masks = SplitMasks(
            regime=regime,
            seed=int(seed),
            observed_mask=observed.copy(),
            regime_removed_mask=regime_removed,
            equalization_removed_mask=equalization_removed,
            train_removed_mask=train_removed,
            train_visible_mask=train_visible,
            val_mask=val,
            test_mask=test,
            unscored_removed_mask=train_removed & ~val & ~test,
            target_languages=np.asarray(targets, dtype=int),
            target_language_pool_source=pool_source,
        )
        validate_split_masks(
            masks,
            feature_types,
            expected_val_size=n_val,
            expected_test_size=n_test,
        )
        masks_by_regime[regime] = masks

    train_sizes = {
        int(split_masks.train_visible_mask.sum())
        for split_masks in masks_by_regime.values()
    }
    if len(train_sizes) != 1:
        details = {
            regime: int(split_masks.train_visible_mask.sum())
            for regime, split_masks in masks_by_regime.items()
        }
        raise AssertionError(
            f"All regimes in seed {seed} must have equal train_visible_mask sums; "
            f"got {details}."
        )
    return masks_by_regime, selection


def build_equalized_seed_masks(
    X: pd.DataFrame,
    feature_types: pd.Series,
    regimes: Iterable[str],
    *,
    seed: int,
    val_frac: float,
    test_frac: float,
    min_cells_per_unit: int = 3,
    min_remaining_input: int = 2,
    n_target_languages: int | None = None,
) -> tuple[dict[str, SplitMasks], SeedSplitSummary]:
    regime_names = list(dict.fromkeys(regimes))
    shared_budget, capacities, requested_budget, selection = compute_shared_heldout_budget(
        X,
        feature_types,
        regime_names,
        val_frac=val_frac,
        test_frac=test_frac,
        min_cells_per_unit=min_cells_per_unit,
        min_remaining_input=min_remaining_input,
        n_target_languages=n_target_languages,
        seed=seed,
    )
    n_val, n_test = split_budget(shared_budget, val_frac, test_frac)
    masks_by_regime, _ = generate_equalized_regime_masks(
        X.notna().to_numpy(dtype=bool),
        feature_types,
        regime_names,
        seed=seed,
        n_val=n_val,
        n_test=n_test,
        min_cells_per_unit=min_cells_per_unit,
        min_remaining_input=min_remaining_input,
        n_target_languages=n_target_languages,
        selection=selection,
    )
    return masks_by_regime, SeedSplitSummary(
        seed=int(seed),
        requested_heldout_budget=requested_budget,
        shared_heldout_budget=shared_budget,
        n_val=n_val,
        n_test=n_test,
        scoring_capacities=capacities,
    )


def apply_train_visible_mask(X: pd.DataFrame, train_visible_mask: np.ndarray) -> pd.DataFrame:
    visible = np.asarray(train_visible_mask, dtype=bool)
    if visible.shape != X.shape:
        raise ValueError(
            f"train_visible_mask has shape {visible.shape}; expected {X.shape}."
        )
    values = X.to_numpy(dtype=float, copy=True)
    values[~visible] = np.nan
    return pd.DataFrame(values, index=X.index.copy(), columns=X.columns.copy())


def mask_to_cell_df(
    mask: np.ndarray,
    X: pd.DataFrame,
    feature_types: pd.Series | np.ndarray,
    *,
    split: str | None = None,
    regime: str | None = None,
    seed: int | None = None,
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
    if split is not None:
        cells["split"] = split
    if regime is not None:
        cells["regime"] = regime
    if seed is not None:
        cells["seed"] = int(seed)
    return cells.reset_index(drop=True)


def observed_cells_df(X: pd.DataFrame, feature_types: pd.Series) -> pd.DataFrame:
    return mask_to_cell_df(X.notna().to_numpy(dtype=bool), X, feature_types)


__all__ = [
    "FEATURE_TYPES",
    "SeedSplitSummary",
    "SplitMasks",
    "apply_train_visible_mask",
    "build_equalized_seed_masks",
    "compute_shared_heldout_budget",
    "generate_equalized_regime_masks",
    "mask_to_cell_df",
    "observed_cells_df",
    "requested_heldout_budget",
    "split_budget",
]
