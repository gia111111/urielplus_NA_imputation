from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .feature_types import FEATURE_TYPES, as_feature_type_array, feature_type_from_regime


@dataclass(frozen=True)
class TargetLanguageSelection:
    shared_eligible_languages: np.ndarray
    shared_target_languages: np.ndarray
    target_languages_by_type: dict[str, np.ndarray]
    pool_source_by_type: dict[str, str]
    shared_pool_used: bool

    def for_type(self, target_type: str) -> np.ndarray:
        return self.target_languages_by_type[target_type]

    @property
    def matched_target_languages(self) -> np.ndarray:
        if self.shared_pool_used:
            return self.shared_target_languages
        arrays = [self.target_languages_by_type[target_type] for target_type in FEATURE_TYPES]
        return np.unique(np.concatenate(arrays)).astype(int)


@dataclass(frozen=True)
class SplitMasks:
    regime: str
    seed: int
    observed_mask: np.ndarray
    regime_removed_mask: np.ndarray
    equalization_removed_mask: np.ndarray
    train_removed_mask: np.ndarray
    train_visible_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    unscored_removed_mask: np.ndarray
    target_languages: np.ndarray
    target_language_pool_source: str


def _boolean_mask(mask: np.ndarray, *, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional boolean mask; got {values.shape}.")
    if shape is not None and values.shape != shape:
        raise ValueError(f"{name} has shape {values.shape}; expected {shape}.")
    return values


def _target_row_mask(n_languages: int, target_languages: Iterable[int] | None) -> np.ndarray:
    if target_languages is None:
        return np.ones(n_languages, dtype=bool)
    indices = np.unique(np.asarray(list(target_languages), dtype=int))
    if np.any(indices < 0) or np.any(indices >= n_languages):
        raise ValueError(
            f"target_languages contains an index outside [0, {n_languages}); "
            f"min={indices.min(initial=-1)}, max={indices.max(initial=-1)}."
        )
    rows = np.zeros(n_languages, dtype=bool)
    rows[indices] = True
    return rows


def derived_seed(seed: int, label: str) -> int:
    digest = hashlib.blake2b(f"{int(seed)}:{label}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def get_language_counts_by_type(
    observed_mask: np.ndarray,
    feature_types: pd.Series | np.ndarray,
) -> pd.DataFrame:
    observed = _boolean_mask(observed_mask, name="observed_mask")
    types = as_feature_type_array(feature_types, observed.shape[1])
    counts: dict[str, np.ndarray] = {
        "total_known_cells": observed.sum(axis=1).astype(int)
    }
    for target_type in FEATURE_TYPES:
        slug = {
            "S": "syntax",
            "P": "phonology",
            "M": "morphology",
            "INV": "inventory",
        }[target_type]
        known = observed[:, types == target_type].sum(axis=1).astype(int)
        counts[f"known_{slug}_cells"] = known
        counts[f"known_cells_outside_{slug}"] = counts["total_known_cells"] - known
    return pd.DataFrame(
        counts,
        index=pd.RangeIndex(observed.shape[0], name="language_index"),
    )


def _eligible_languages(
    counts: pd.DataFrame,
    target_type: str,
    min_cells_per_type: int,
    min_remaining_input: int,
) -> np.ndarray:
    slug = {
        "S": "syntax",
        "P": "phonology",
        "M": "morphology",
        "INV": "inventory",
    }[target_type]
    eligible = (
        (counts[f"known_{slug}_cells"] >= min_cells_per_type)
        & (counts[f"known_cells_outside_{slug}"] >= min_remaining_input)
    )
    return counts.index[eligible].to_numpy(dtype=int)


def _sample_languages(
    pool: np.ndarray,
    n_target_languages: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    pool = np.asarray(pool, dtype=int)
    if n_target_languages is None:
        return rng.permutation(pool).astype(int)
    if n_target_languages <= 0:
        raise ValueError("n_target_languages must be positive when provided.")
    if len(pool) < n_target_languages:
        raise ValueError(
            f"Requested {n_target_languages} target languages, but only {len(pool)} are eligible."
        )
    return rng.choice(pool, size=n_target_languages, replace=False).astype(int)


def select_shared_target_languages(
    observed_mask: np.ndarray,
    feature_types: pd.Series | np.ndarray,
    seed: int,
    min_cells_per_type: int = 3,
    min_remaining_input: int = 2,
    n_target_languages: int | None = None,
) -> TargetLanguageSelection:
    observed = _boolean_mask(observed_mask, name="observed_mask")
    counts = get_language_counts_by_type(observed, feature_types)
    eligible_by_type = {
        target_type: _eligible_languages(
            counts,
            target_type,
            min_cells_per_type,
            min_remaining_input,
        )
        for target_type in FEATURE_TYPES
    }
    shared = eligible_by_type["S"]
    for target_type in FEATURE_TYPES[1:]:
        shared = np.intersect1d(shared, eligible_by_type[target_type], assume_unique=True)

    enough_shared = len(shared) > 0 and (
        n_target_languages is None or len(shared) >= n_target_languages
    )
    rng = np.random.default_rng(seed)
    if enough_shared:
        targets = _sample_languages(shared, n_target_languages, rng)
        return TargetLanguageSelection(
            shared_eligible_languages=shared,
            shared_target_languages=targets,
            target_languages_by_type={
                target_type: targets.copy() for target_type in FEATURE_TYPES
            },
            pool_source_by_type={target_type: "shared" for target_type in FEATURE_TYPES},
            shared_pool_used=True,
        )

    by_type: dict[str, np.ndarray] = {}
    for target_type in FEATURE_TYPES:
        pool = eligible_by_type[target_type]
        if len(pool) == 0:
            raise ValueError(
                f"No languages are eligible for {target_type}; "
                f"min_cells_per_type={min_cells_per_type}, "
                f"min_remaining_input={min_remaining_input}."
            )
        sample_size = None if n_target_languages is None else min(n_target_languages, len(pool))
        by_type[target_type] = _sample_languages(pool, sample_size, rng)
    return TargetLanguageSelection(
        shared_eligible_languages=shared,
        shared_target_languages=np.array([], dtype=int),
        target_languages_by_type=by_type,
        pool_source_by_type={
            target_type: "type_specific_fallback" for target_type in FEATURE_TYPES
        },
        shared_pool_used=False,
    )


def _sample_scored_masks(
    candidate_mask: np.ndarray,
    n_val: int,
    n_test: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    candidates_mask = _boolean_mask(candidate_mask, name="candidate_mask")
    if n_val < 0 or n_test < 0:
        raise ValueError("n_val and n_test must be non-negative.")
    requested = int(n_val + n_test)
    candidates = np.flatnonzero(candidates_mask)
    if len(candidates) < requested:
        raise ValueError(
            f"Requested {requested} validation/test cells, but only "
            f"{len(candidates)} candidates are available."
        )
    selected = rng.choice(candidates, size=requested, replace=False)
    rng.shuffle(selected)
    val = np.zeros(candidates_mask.shape, dtype=bool)
    test = np.zeros(candidates_mask.shape, dtype=bool)
    val.flat[selected[:n_val]] = True
    test.flat[selected[n_val:]] = True
    return val, test


def make_mcar_mask(
    observed_mask: np.ndarray,
    n_val: int,
    n_test: int,
    seed: int,
    target_languages: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    observed = _boolean_mask(observed_mask, name="observed_mask")
    target_rows = _target_row_mask(observed.shape[0], target_languages)
    return _sample_scored_masks(
        observed & target_rows[:, None],
        n_val,
        n_test,
        np.random.default_rng(seed),
    )


def make_empirical_copy_mask(
    observed_mask: np.ndarray,
    target_languages: Iterable[int],
    n_val: int,
    n_test: int,
    seed: int,
    min_cells_per_unit: int = 3,
    min_remaining_input: int = 2,
    max_attempts: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = _boolean_mask(observed_mask, name="observed_mask")
    targets = np.unique(np.asarray(list(target_languages), dtype=int))
    _target_row_mask(observed.shape[0], targets)
    if len(targets) == 0:
        raise ValueError("empirical_copy requires at least one target language.")
    budget = int(n_val + n_test)
    if budget < min_cells_per_unit:
        raise ValueError(
            "The empirical_copy held-out budget must be at least min_cells_per_unit."
        )

    rng = np.random.default_rng(seed)
    removed = np.zeros_like(observed)
    pairs = np.array(
        [
            (int(target), int(donor))
            for target in targets
            for donor in range(observed.shape[0])
            if int(target) != int(donor)
        ],
        dtype=int,
    )
    if len(pairs) == 0:
        raise ValueError("empirical_copy has no non-self target/donor pairs to sample.")
    pairs = pairs[rng.permutation(len(pairs))]
    pair_limit = min(int(max_attempts), len(pairs))

    for attempt, (target, donor) in enumerate(pairs[:pair_limit]):
        removed_count = int(removed.sum())
        if removed_count == budget:
            break
        target = int(target)
        donor = int(donor)
        columns = np.flatnonzero((~observed[donor]) & observed[target] & ~removed[target])
        remaining_budget = budget - removed_count
        if len(columns) < min_cells_per_unit:
            continue
        if len(columns) < remaining_budget and remaining_budget - len(columns) < min_cells_per_unit:
            continue
        if len(columns) > remaining_budget:
            columns = rng.choice(columns, size=remaining_budget, replace=False)
        if (
            int(observed[target].sum())
            - int(removed[target].sum())
            - len(columns)
            < min_remaining_input
        ):
            continue
        removed[target, columns] = True

    if int(removed.sum()) != budget:
        raise ValueError(
            f"empirical_copy produced {int(removed.sum())}/{budget} cells after "
            f"{min(pair_limit, len(pairs))} unique target/donor pair attempts; "
            f"targets={len(targets)}, donors={observed.shape[0]}, "
            f"min_cells_per_unit={min_cells_per_unit}, "
            f"min_remaining_input={min_remaining_input}."
        )
    selected = np.flatnonzero(removed)
    rng.shuffle(selected)
    val = np.zeros_like(observed)
    test = np.zeros_like(observed)
    val.flat[selected[:n_val]] = True
    test.flat[selected[n_val:]] = True
    return removed, val, test


def make_local_block_mask(
    observed_mask: np.ndarray,
    feature_types: pd.Series | np.ndarray,
    target_type: str,
    target_languages: Iterable[int],
    n_val: int,
    n_test: int,
    seed: int,
    min_cells_per_unit: int = 3,
    min_remaining_input: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = _boolean_mask(observed_mask, name="observed_mask")
    types = as_feature_type_array(feature_types, observed.shape[1])
    if target_type not in FEATURE_TYPES:
        raise ValueError(f"target_type must be one of {FEATURE_TYPES}; got {target_type!r}.")
    targets = np.unique(np.asarray(list(target_languages), dtype=int))
    target_rows = _target_row_mask(observed.shape[0], targets)
    target_columns = types == target_type
    for row in targets:
        known_target = int(observed[row, target_columns].sum())
        known_remaining = int(observed[row, ~target_columns].sum())
        if known_target < min_cells_per_unit or known_remaining < min_remaining_input:
            raise ValueError(
                f"Language {row} is ineligible for Local_block_{target_type}: "
                f"known_target={known_target}, known_remaining={known_remaining}, "
                f"required_target={min_cells_per_unit}, "
                f"required_remaining={min_remaining_input}."
            )
    removed = np.zeros_like(observed)
    removed[:, target_columns] = observed[:, target_columns] & target_rows[:, None]
    val, test = _sample_scored_masks(
        removed,
        n_val,
        n_test,
        np.random.default_rng(seed),
    )
    return removed, val, test


def make_global_block_mask(
    observed_mask: np.ndarray,
    feature_types: pd.Series | np.ndarray,
    target_type: str,
    n_val: int,
    n_test: int,
    seed: int,
    target_languages_for_scoring: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = _boolean_mask(observed_mask, name="observed_mask")
    types = as_feature_type_array(feature_types, observed.shape[1])
    if target_type not in FEATURE_TYPES:
        raise ValueError(f"target_type must be one of {FEATURE_TYPES}; got {target_type!r}.")
    target_columns = types == target_type
    removed = np.zeros_like(observed)
    removed[:, target_columns] = observed[:, target_columns]
    scoring_rows = _target_row_mask(observed.shape[0], target_languages_for_scoring)
    val, test = _sample_scored_masks(
        removed & scoring_rows[:, None],
        n_val,
        n_test,
        np.random.default_rng(seed),
    )
    return removed, val, test


def equalize_training_input_size(
    observed_mask: np.ndarray,
    regime_removed_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    target_removed_count: int,
    seed: int,
    target_languages: Iterable[int] | None,
) -> np.ndarray:
    observed = _boolean_mask(observed_mask, name="observed_mask")
    removed = _boolean_mask(
        regime_removed_mask,
        name="regime_removed_mask",
        shape=observed.shape,
    )
    val = _boolean_mask(val_mask, name="val_mask", shape=observed.shape)
    test = _boolean_mask(test_mask, name="test_mask", shape=observed.shape)
    extra_needed = int(target_removed_count - removed.sum())
    if extra_needed < 0:
        raise ValueError(
            f"target_removed_count={target_removed_count} is smaller than "
            f"regime_removed_count={int(removed.sum())}."
        )
    equalization = np.zeros_like(observed)
    if extra_needed == 0:
        return equalization

    available = observed & ~removed & ~val & ~test
    target_rows = _target_row_mask(observed.shape[0], target_languages)
    preferred = np.flatnonzero(available & ~target_rows[:, None])
    fallback = np.flatnonzero(available & target_rows[:, None])
    rng = np.random.default_rng(seed)
    take_preferred = min(extra_needed, len(preferred))
    if take_preferred:
        equalization.flat[
            rng.choice(preferred, size=take_preferred, replace=False)
        ] = True
    remaining = extra_needed - take_preferred
    if remaining:
        if len(fallback) < remaining:
            raise ValueError(
                f"Equalization needs {extra_needed} cells, but only "
                f"{take_preferred + len(fallback)} non-overlapping observed cells are available."
            )
        equalization.flat[rng.choice(fallback, size=remaining, replace=False)] = True
    return equalization


def _check(condition: bool, message: str, masks: SplitMasks) -> None:
    if not condition:
        raise AssertionError(
            f"{message} [regime={masks.regime}, seed={masks.seed}, "
            f"observed={int(masks.observed_mask.sum())}, "
            f"regime_removed={int(masks.regime_removed_mask.sum())}, "
            f"equalization_removed={int(masks.equalization_removed_mask.sum())}, "
            f"train_visible={int(masks.train_visible_mask.sum())}, "
            f"val={int(masks.val_mask.sum())}, test={int(masks.test_mask.sum())}]"
        )


def validate_split_masks(
    masks: SplitMasks,
    feature_types: pd.Series | np.ndarray,
    *,
    expected_val_size: int,
    expected_test_size: int,
) -> None:
    """Validate split invariants with regime/seed/count context in every failure."""
    shape = masks.observed_mask.shape
    for name in (
        "regime_removed_mask",
        "equalization_removed_mask",
        "train_removed_mask",
        "train_visible_mask",
        "val_mask",
        "test_mask",
        "unscored_removed_mask",
    ):
        _boolean_mask(getattr(masks, name), name=name, shape=shape)

    observed = masks.observed_mask
    val = masks.val_mask
    test = masks.test_mask
    scored = val | test
    _check(int(val.sum()) == expected_val_size, "validation mask has the wrong size", masks)
    _check(int(test.sum()) == expected_test_size, "test mask has the wrong size", masks)
    _check(not np.any(val & test), "validation and test masks overlap", masks)
    _check(np.all(scored <= observed), "validation/test contains naturally missing cells", masks)
    _check(
        np.all(scored <= masks.train_removed_mask),
        "validation/test cells remain in the training input",
        masks,
    )
    _check(
        not np.any(masks.equalization_removed_mask & scored),
        "equalization cells overlap validation/test cells",
        masks,
    )
    _check(
        not np.any(masks.equalization_removed_mask & masks.regime_removed_mask),
        "equalization cells overlap regime-removed cells",
        masks,
    )
    _check(
        np.array_equal(
            masks.train_removed_mask,
            masks.regime_removed_mask | masks.equalization_removed_mask,
        ),
        "train_removed_mask is not the union of regime and equalization removals",
        masks,
    )
    _check(
        np.array_equal(masks.train_visible_mask, observed & ~masks.train_removed_mask),
        "train_visible_mask is inconsistent with observed/train-removed masks",
        masks,
    )
    _check(
        np.array_equal(
            masks.unscored_removed_mask,
            masks.train_removed_mask & ~val & ~test,
        ),
        "unscored_removed_mask is inconsistent",
        masks,
    )

    if masks.regime.startswith("Local_block_"):
        types = as_feature_type_array(feature_types, shape[1])
        target_type = feature_type_from_regime(masks.regime)
        target_columns = types == target_type
        target_rows = _target_row_mask(shape[0], masks.target_languages)
        expected_removed = np.zeros(shape, dtype=bool)
        expected_removed[:, target_columns] = observed[:, target_columns] & target_rows[:, None]
        _check(
            np.array_equal(masks.regime_removed_mask, expected_removed),
            f"local block has leakage or removes cells outside target type {target_type}",
            masks,
        )
        _check(
            not np.any(masks.train_visible_mask[np.ix_(target_rows, target_columns)]),
            f"local block leaves target type {target_type} visible for target languages",
            masks,
        )
