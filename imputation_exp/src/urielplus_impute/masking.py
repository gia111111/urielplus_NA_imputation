from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping

import numpy as np
import pandas as pd

from .feature_types import FEATURE_TYPES, as_feature_type_array


RESOURCE_GROUPS = ("P1", "P2")
EVALUATION_SPLITS = ("validation", "calibration", "test")
SPLIT_CODES = {"none": 0, "validation": 1, "calibration": 2, "test": 3}
ADAPTATION_BUDGETS = (0, 2, 4, 8, 16, 32, 64, 128)


class InfeasibleMaskError(ValueError):
    """Raised when an exact masking design cannot be constructed."""

    def __init__(self, message: str, *, capacities: dict | None = None) -> None:
        super().__init__(message)
        self.capacities = capacities or {}


@dataclass(frozen=True)
class CopyMaskCase:
    """One scored cell from a full-pattern donor-target copy event."""

    event_id: int
    seed: int
    split: str
    donor_index: int
    donor_id: str
    target_index: int
    target_id: str
    preselected_feature_index: int
    scored_feature_index: int
    feature_type: str
    donor_group: str
    target_original_group: str
    target_post_mask_group: str
    matching_tier: str
    same_family: bool
    same_macroarea: bool
    donor_observed_count: int
    intersection_count: int
    hidden_count: int
    similarity: float
    distance_km: float


@dataclass(frozen=True)
class SplitMasks:
    regime: str
    seed: int
    observed_mask: np.ndarray
    regime_removed_mask: np.ndarray
    train_removed_mask: np.ndarray
    train_visible_mask: np.ndarray
    val_mask: np.ndarray
    cal_mask: np.ndarray
    test_mask: np.ndarray
    unscored_removed_mask: np.ndarray
    adaptation_mask: np.ndarray
    target_languages: np.ndarray
    resource_groups: np.ndarray
    scored_resource_groups: np.ndarray
    language_split: np.ndarray
    target_language_pool_source: str
    resource_group: str | None = None
    target_feature_type: str | None = None
    adaptation_budget: int | None = None
    copy_provenance: tuple[CopyMaskCase, ...] = ()


def derived_seed(seed: int, label: str) -> int:
    digest = hashlib.blake2b(
        f"{int(seed)}:{label}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def _boolean_mask(
    mask: np.ndarray,
    *,
    name: str,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got {values.shape}.")
    if shape is not None and values.shape != shape:
        raise ValueError(f"{name} has shape {values.shape}; expected {shape}.")
    return values


def resource_domain_counts(
    mask: np.ndarray,
    feature_types: pd.Series | np.ndarray,
    scored_resource_groups: np.ndarray,
) -> dict[str, int]:
    """Count scored cells by their scoring resource group and feature domain."""
    values = _boolean_mask(mask, name="mask")
    types = as_feature_type_array(feature_types, values.shape[1])
    scoring_groups = np.asarray(scored_resource_groups, dtype=str)
    if scoring_groups.shape != values.shape:
        raise ValueError(
            "scored_resource_groups must have the same shape as mask; "
            f"got {scoring_groups.shape} and {values.shape}."
        )
    return {
        f"{group}:{target_type}": int(
            (values & (scoring_groups == group))[:, types == target_type].sum()
        )
        for group in RESOURCE_GROUPS
        for target_type in FEATURE_TYPES
    }


def _split_sizes(n_rows: int, quotas: Mapping[str, int]) -> dict[str, int]:
    total = int(sum(quotas.values()))
    if total <= 0:
        raise ValueError("At least one split quota must be positive.")
    raw = {name: n_rows * int(quotas[name]) / total for name in EVALUATION_SPLITS}
    sizes = {name: int(math.floor(raw[name])) for name in EVALUATION_SPLITS}
    remainder = n_rows - sum(sizes.values())
    order = sorted(
        EVALUATION_SPLITS,
        key=lambda name: (raw[name] - sizes[name], quotas[name]),
        reverse=True,
    )
    for name in order[:remainder]:
        sizes[name] += 1
    return sizes


def _partition_rows(
    rows: np.ndarray,
    quotas: Mapping[str, int],
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rows = np.asarray(rows, dtype=int)
    shuffled = rng.permutation(rows)
    sizes = _split_sizes(len(rows), quotas)
    pools: dict[str, np.ndarray] = {}
    start = 0
    for name in EVALUATION_SPLITS:
        stop = start + sizes[name]
        pools[name] = shuffled[start:stop]
        start = stop
    return pools, shuffled


def _pool_domain_capacity(
    observed: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    row_capacity: np.ndarray | None,
) -> int:
    counts = observed[np.ix_(rows, columns)].sum(axis=1).astype(int)
    if row_capacity is not None:
        counts = np.minimum(counts, row_capacity[rows])
    return int(counts.sum())


def _partition_with_capacity(
    observed: np.ndarray,
    rows: np.ndarray,
    types: np.ndarray,
    quotas: Mapping[str, int],
    rng: np.random.Generator,
    *,
    row_capacity: np.ndarray | None = None,
    required_types: Iterable[str] = FEATURE_TYPES,
    max_attempts: int = 500,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    last_capacities: dict[str, int] = {}
    for _ in range(max_attempts):
        pools, shuffled = _partition_rows(rows, quotas, rng)
        last_capacities = {}
        for split_name, pool in pools.items():
            for target_type in required_types:
                columns = np.flatnonzero(types == target_type)
                capacity = _pool_domain_capacity(
                    observed,
                    pool,
                    columns,
                    row_capacity,
                )
                last_capacities[f"{split_name}:{target_type}"] = capacity
        if all(
            last_capacities[f"{split_name}:{target_type}"]
            >= int(quotas[split_name])
            for split_name in EVALUATION_SPLITS
            for target_type in required_types
        ):
            return pools, shuffled
    raise InfeasibleMaskError(
        "Could not partition target languages into disjoint validation, "
        "calibration, and test pools with the requested per-domain capacities.",
        capacities=last_capacities,
    )


def _balanced_sample_cells(
    observed: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    quota: int,
    rng: np.random.Generator,
    selected: np.ndarray,
    *,
    row_remaining_capacity: np.ndarray | None = None,
) -> None:
    """Sample languages in balanced reuse cycles, then one cell per language."""
    rows = np.asarray(rows, dtype=int)
    columns = np.asarray(columns, dtype=int)
    available_by_row: dict[int, np.ndarray] = {}
    used_by_row: dict[int, int] = {}
    for row in rows:
        available = columns[observed[row, columns] & ~selected[row, columns]]
        if len(available):
            available_by_row[int(row)] = rng.permutation(available)
            used_by_row[int(row)] = 0

    sampled = 0
    while sampled < int(quota):
        candidate_rows = np.asarray(
            [
                row
                for row, available in available_by_row.items()
                if used_by_row[row] < len(available)
                and (
                    row_remaining_capacity is None
                    or row_remaining_capacity[row] > 0
                )
            ],
            dtype=int,
        )
        if not len(candidate_rows):
            raise InfeasibleMaskError(
                f"Balanced cell sampling stopped after {sampled} cells; "
                f"the exact quota {quota} cannot be filled."
            )
        for row in rng.permutation(candidate_rows):
            row = int(row)
            column = int(available_by_row[row][used_by_row[row]])
            used_by_row[row] += 1
            selected[row, column] = True
            sampled += 1
            if row_remaining_capacity is not None:
                row_remaining_capacity[row] -= 1
            if sampled == int(quota):
                break


def _scoring_groups_from_targets(
    scored: np.ndarray,
    resource_groups: np.ndarray,
) -> np.ndarray:
    scoring = np.full(scored.shape, "", dtype="<U2")
    for group in RESOURCE_GROUPS:
        scoring[scored & (resource_groups[:, None] == group)] = group
    return scoring


def _new_split_masks(
    *,
    regime: str,
    seed: int,
    observed: np.ndarray,
    removed: np.ndarray,
    val: np.ndarray,
    cal: np.ndarray,
    test: np.ndarray,
    adaptation: np.ndarray,
    target_languages: Iterable[int],
    resource_groups: np.ndarray,
    language_split: np.ndarray,
    pool_source: str,
    scored_resource_groups: np.ndarray | None = None,
    resource_group: str | None = None,
    target_feature_type: str | None = None,
    adaptation_budget: int | None = None,
    copy_provenance: tuple[CopyMaskCase, ...] = (),
) -> SplitMasks:
    train_removed = np.asarray(removed, dtype=bool)
    scored = val | cal | test
    if scored_resource_groups is None:
        scored_resource_groups = _scoring_groups_from_targets(
            scored,
            np.asarray(resource_groups, dtype=str),
        )
    return SplitMasks(
        regime=regime,
        seed=int(seed),
        observed_mask=observed.copy(),
        regime_removed_mask=train_removed.copy(),
        train_removed_mask=train_removed.copy(),
        train_visible_mask=observed & ~train_removed,
        val_mask=val,
        cal_mask=cal,
        test_mask=test,
        unscored_removed_mask=train_removed & ~scored,
        adaptation_mask=adaptation,
        target_languages=np.unique(np.asarray(list(target_languages), dtype=int)),
        resource_groups=np.asarray(resource_groups, dtype="<U2"),
        scored_resource_groups=np.asarray(scored_resource_groups, dtype="<U2"),
        language_split=np.asarray(language_split, dtype=np.int8),
        target_language_pool_source=pool_source,
        resource_group=resource_group,
        target_feature_type=target_feature_type,
        adaptation_budget=adaptation_budget,
        copy_provenance=copy_provenance,
    )


def make_stratified_mcar_mask(
    observed_mask: np.ndarray,
    feature_types: pd.Series | np.ndarray,
    resource_groups: np.ndarray,
    *,
    minimum_remaining_counts: np.ndarray,
    seed: int,
    quotas: Mapping[str, int],
) -> SplitMasks:
    observed = _boolean_mask(observed_mask, name="observed_mask")
    types = as_feature_type_array(feature_types, observed.shape[1])
    groups = np.asarray(resource_groups, dtype=str)
    minimum_remaining = np.asarray(minimum_remaining_counts, dtype=int)
    if minimum_remaining.shape != (observed.shape[0],):
        raise ValueError("minimum_remaining_counts has the wrong shape.")
    rng = np.random.default_rng(derived_seed(seed, "stratified_mcar"))
    known = observed.sum(axis=1).astype(int)
    selected_by_split = {
        name: np.zeros_like(observed) for name in EVALUATION_SPLITS
    }
    language_split = np.zeros(observed.shape[0], dtype=np.int8)
    all_targets: list[int] = []

    for group in RESOURCE_GROUPS:
        rows = np.flatnonzero(groups == group)
        base_capacity = np.maximum(known - minimum_remaining, 0)
        pools, ordered = _partition_with_capacity(
            observed,
            rows,
            types,
            quotas,
            rng,
            row_capacity=base_capacity,
        )
        all_targets.extend(ordered.tolist())
        for split_name, pool in pools.items():
            language_split[pool] = SPLIT_CODES[split_name]
            remaining = base_capacity.copy()
            domain_order = sorted(
                FEATURE_TYPES,
                key=lambda target_type: int(
                    observed[np.ix_(pool, types == target_type)].sum()
                ),
            )
            for target_type in domain_order:
                _balanced_sample_cells(
                    observed,
                    pool,
                    np.flatnonzero(types == target_type),
                    int(quotas[split_name]),
                    rng,
                    selected_by_split[split_name],
                    row_remaining_capacity=remaining,
                )

    val = selected_by_split["validation"]
    cal = selected_by_split["calibration"]
    test = selected_by_split["test"]
    removed = val | cal | test
    return _new_split_masks(
        regime="mcar",
        seed=seed,
        observed=observed,
        removed=removed,
        val=val,
        cal=cal,
        test=test,
        adaptation=np.zeros_like(observed),
        target_languages=all_targets,
        resource_groups=groups,
        language_split=language_split,
        pool_source="all_languages_partitioned_within_natural_resource_group",
    )


def _haversine_km(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    if not np.isfinite(lat1) or not np.isfinite(lon1):
        return np.full(len(lat2), np.inf)
    valid = np.isfinite(lat2) & np.isfinite(lon2)
    result = np.full(len(lat2), np.inf)
    if not valid.any():
        return result
    lat1r = math.radians(lat1)
    lat2r = np.radians(lat2[valid])
    dlat = lat2r - lat1r
    dlon = np.radians(lon2[valid] - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + math.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    result[valid] = 6371.0088 * 2.0 * np.arcsin(np.sqrt(a))
    return result


def _metadata_array(
    metadata: pd.DataFrame,
    names: tuple[str, ...],
    default: str,
) -> np.ndarray:
    for name in names:
        if name in metadata.columns:
            return metadata[name].fillna(default).astype(str).to_numpy()
    return np.full(len(metadata), default, dtype=object)


def _copy_candidate(
    *,
    donor: int,
    anchor_feature: int,
    candidates: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
    donor_group: str,
    post_mask_is_p2: np.ndarray,
    family: np.ndarray,
    macroarea: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    language_ids: np.ndarray,
) -> tuple[int, np.ndarray, int, str, float] | None:
    candidates = candidates[
        (candidates != donor) & observed[candidates, anchor_feature]
    ]
    if not len(candidates):
        return None
    shared_counts = (observed[candidates] & observed[donor][None, :]).sum(axis=1)
    positive = shared_counts > 0
    candidates = candidates[positive]
    shared_counts = shared_counts[positive]
    if not len(candidates):
        return None
    wants_p2 = donor_group == "P2"
    group_match = post_mask_is_p2[candidates, shared_counts] == wants_p2
    candidates = candidates[group_match]
    shared_counts = shared_counts[group_match]
    if not len(candidates):
        return None

    same_family = family[candidates] == family[donor]
    same_macroarea = macroarea[candidates] == macroarea[donor]
    tiers = np.where(same_family, 0, np.where(same_macroarea, 1, 2))
    best_tier = int(tiers.min())
    keep = tiers == best_tier
    candidates = candidates[keep]
    shared_counts = shared_counts[keep]
    target_group_preference = groups[candidates] == donor_group
    if target_group_preference.any():
        candidates = candidates[target_group_preference]
        shared_counts = shared_counts[target_group_preference]

    donor_count = int(observed[donor].sum())
    similarity = shared_counts / float(donor_count)
    best_similarity = float(similarity.max())
    keep = np.isclose(similarity, best_similarity)
    candidates = candidates[keep]
    shared_counts = shared_counts[keep]
    distances = _haversine_km(
        latitude[donor],
        longitude[donor],
        latitude[candidates],
        longitude[candidates],
    )
    order = np.lexsort((language_ids[candidates], distances))
    position = int(order[0])
    target = int(candidates[position])
    tier = ("family", "macroarea", "other")[best_tier]
    return (
        target,
        observed[target] & ~observed[donor],
        int(shared_counts[position]),
        tier,
        float(distances[position]),
    )


def _copy_attempt(
    *,
    observed: np.ndarray,
    types: np.ndarray,
    groups: np.ndarray,
    metadata: pd.DataFrame,
    post_mask_is_p2: np.ndarray,
    quotas: Mapping[str, int],
    seed: int,
    rng: np.random.Generator,
) -> SplitMasks | dict[str, int]:
    all_rows = np.arange(observed.shape[0], dtype=int)
    pools, _ = _partition_rows(all_rows, quotas, rng)
    language_split = np.zeros(observed.shape[0], dtype=np.int8)
    for split_name, pool in pools.items():
        language_split[pool] = SPLIT_CODES[split_name]

    family = _metadata_array(metadata, ("family_id", "family"), "unknown")
    macroarea = _metadata_array(metadata, ("macroarea",), "Unknown")
    language_ids = _metadata_array(
        metadata,
        ("language_id", "glottocode", "GLOTTOCODE"),
        "",
    )
    empty_ids = language_ids == ""
    language_ids[empty_ids] = np.arange(len(metadata)).astype(str)[empty_ids]
    latitude = pd.to_numeric(
        metadata.get("latitude", pd.Series(np.nan, index=metadata.index)),
        errors="coerce",
    ).to_numpy()
    longitude = pd.to_numeric(
        metadata.get("longitude", pd.Series(np.nan, index=metadata.index)),
        errors="coerce",
    ).to_numpy()

    queues: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for group in RESOURCE_GROUPS:
        donor_rows = np.flatnonzero(groups == group)
        for target_type in FEATURE_TYPES:
            columns = np.flatnonzero(types == target_type)
            donor_pos, column_pos = np.nonzero(~observed[np.ix_(donor_rows, columns)])
            cells = list(
                zip(
                    donor_rows[donor_pos].astype(int).tolist(),
                    columns[column_pos].astype(int).tolist(),
                )
            )
            if cells:
                order = rng.permutation(len(cells))
                cells = [cells[int(index)] for index in order]
            queues[(group, target_type)] = cells

    deficits = {
        split_name: {
            (group, target_type): int(quotas[split_name])
            for group in RESOURCE_GROUPS
            for target_type in FEATURE_TYPES
        }
        for split_name in EVALUATION_SPLITS
    }
    used_targets = np.zeros(observed.shape[0], dtype=bool)
    removed = np.zeros_like(observed)
    scored_by_split = {
        name: np.zeros_like(observed) for name in EVALUATION_SPLITS
    }
    scoring_groups = np.full(observed.shape, "", dtype="<U2")
    provenance: list[CopyMaskCase] = []
    consumed_donor_cells: set[tuple[int, int]] = set()
    event_id = 0

    while True:
        unfinished = [
            (split_name, group, target_type)
            for split_name in EVALUATION_SPLITS
            for group in RESOURCE_GROUPS
            for target_type in FEATURE_TYPES
            if deficits[split_name][(group, target_type)] > 0
        ]
        if not unfinished:
            break
        unfinished.sort(
            key=lambda item: (
                -deficits[item[0]][(item[1], item[2])]
                / max(int(quotas[item[0]]), 1),
                len(queues[(item[1], item[2])]),
                EVALUATION_SPLITS.index(item[0]),
                RESOURCE_GROUPS.index(item[1]),
                FEATURE_TYPES.index(item[2]),
            )
        )
        progressed = False
        for split_name, donor_group, focus_type in unfinished:
            if deficits[split_name][(donor_group, focus_type)] <= 0:
                continue
            queue = queues[(donor_group, focus_type)]
            pool = pools[split_name]
            candidates = pool[~used_targets[pool]]
            while queue and len(candidates):
                donor, anchor_feature = queue.pop()
                if (donor, anchor_feature) in consumed_donor_cells:
                    continue
                match = _copy_candidate(
                    donor=donor,
                    anchor_feature=anchor_feature,
                    candidates=candidates,
                    observed=observed,
                    groups=groups,
                    donor_group=donor_group,
                    post_mask_is_p2=post_mask_is_p2,
                    family=family,
                    macroarea=macroarea,
                    latitude=latitude,
                    longitude=longitude,
                    language_ids=language_ids,
                )
                if match is None:
                    continue
                target, hidden, intersection_count, tier, distance = match
                used_targets[target] = True
                removed[target, hidden] = True
                event_id += 1

                selected_columns: list[int] = []
                for target_type in FEATURE_TYPES:
                    need = deficits[split_name][(donor_group, target_type)]
                    if need <= 0:
                        continue
                    columns = np.flatnonzero(hidden & (types == target_type))
                    columns = np.asarray(
                        [
                            column
                            for column in columns
                            if (donor, int(column)) not in consumed_donor_cells
                        ],
                        dtype=int,
                    )
                    if target_type == focus_type and anchor_feature in columns:
                        columns = np.concatenate(
                            ([anchor_feature], columns[columns != anchor_feature])
                        )
                    if len(columns) > 1:
                        if target_type == focus_type:
                            head = columns[:1]
                            tail = rng.permutation(columns[1:])
                            columns = np.concatenate((head, tail))
                        else:
                            columns = rng.permutation(columns)
                    selected_columns.extend(columns[:need].astype(int).tolist())

                if not selected_columns:
                    raise AssertionError("A matched copy event did not score its anchor.")
                target_post_group = (
                    "P2" if post_mask_is_p2[target, intersection_count] else "P1"
                )
                for column in selected_columns:
                    consumed_donor_cells.add((int(donor), int(column)))
                    target_type = str(types[column])
                    scored_by_split[split_name][target, column] = True
                    scoring_groups[target, column] = donor_group
                    deficits[split_name][(donor_group, target_type)] -= 1
                    provenance.append(
                        CopyMaskCase(
                            event_id=event_id,
                            seed=int(seed),
                            split=split_name,
                            donor_index=int(donor),
                            donor_id=str(language_ids[donor]),
                            target_index=int(target),
                            target_id=str(language_ids[target]),
                            preselected_feature_index=int(anchor_feature),
                            scored_feature_index=int(column),
                            feature_type=target_type,
                            donor_group=donor_group,
                            target_original_group=str(groups[target]),
                            target_post_mask_group=target_post_group,
                            matching_tier=tier,
                            same_family=bool(family[target] == family[donor]),
                            same_macroarea=bool(
                                macroarea[target] == macroarea[donor]
                            ),
                            donor_observed_count=int(observed[donor].sum()),
                            intersection_count=int(intersection_count),
                            hidden_count=int(hidden.sum()),
                            similarity=float(
                                intersection_count / observed[donor].sum()
                            ),
                            distance_km=float(distance),
                        )
                    )
                progressed = True
                break
        if not progressed:
            return {
                f"{split_name}:{group}:{target_type}": int(value)
                for split_name in EVALUATION_SPLITS
                for (group, target_type), value in deficits[split_name].items()
                if value > 0
            }

    return _new_split_masks(
        regime="resource_copy",
        seed=seed,
        observed=observed,
        removed=removed,
        val=scored_by_split["validation"],
        cal=scored_by_split["calibration"],
        test=scored_by_split["test"],
        adaptation=np.zeros_like(observed),
        target_languages=np.flatnonzero(used_targets),
        resource_groups=groups,
        scored_resource_groups=scoring_groups,
        language_split=language_split,
        pool_source=(
            "all_languages_partitioned_25_25_50; each target used once; "
            "target natural group used only as a within-tier preference"
        ),
        copy_provenance=tuple(provenance),
    )


def make_resource_conditioned_copy_mask(
    observed_mask: np.ndarray,
    feature_types: pd.Series | np.ndarray,
    resource_groups: np.ndarray,
    languages: pd.DataFrame,
    *,
    post_mask_is_p2: np.ndarray,
    seed: int,
    quotas: Mapping[str, int],
    max_partition_attempts: int = 20,
) -> SplitMasks:
    """Transfer full donor patterns and score multiple hidden cells per target.

    Donor missing cells are preselected without replacement within each donor
    resource-group/domain queue.  A target is assigned to exactly one evaluation
    split and used in at most one donor-target event.  The full eligible hidden
    set is removed, while as many members as needed are scored toward exact
    donor-group/domain quotas.
    """
    observed = _boolean_mask(observed_mask, name="observed_mask")
    types = as_feature_type_array(feature_types, observed.shape[1])
    groups = np.asarray(resource_groups, dtype=str)
    post_lookup = np.asarray(post_mask_is_p2, dtype=bool)
    if post_lookup.shape != (observed.shape[0], observed.shape[1] + 1):
        raise ValueError("post_mask_is_p2 has the wrong shape.")
    metadata = languages.reset_index(drop=True)
    if len(metadata) != observed.shape[0]:
        raise ValueError("languages must align one-to-one with observed_mask rows.")

    last_remaining: dict[str, int] = {}
    for attempt in range(int(max_partition_attempts)):
        rng = np.random.default_rng(
            derived_seed(seed, f"resource_copy:partition:{attempt}")
        )
        result = _copy_attempt(
            observed=observed,
            types=types,
            groups=groups,
            metadata=metadata,
            post_mask_is_p2=post_lookup,
            quotas=quotas,
            seed=seed,
            rng=rng,
        )
        if isinstance(result, SplitMasks):
            return result
        last_remaining = result
    raise InfeasibleMaskError(
        "Resource-conditioned copy masking could not fill the exact quotas "
        f"after {max_partition_attempts} deterministic target-pool attempts.",
        capacities={"unfilled_quotas": last_remaining},
    )


def _nested_adaptation_order(
    candidate_mask: np.ndarray,
    target_rows: np.ndarray,
    max_budget: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    candidates = _boolean_mask(candidate_mask, name="candidate_mask").copy()
    row_usage = np.zeros(candidates.shape[0], dtype=int)
    column_usage = np.zeros(candidates.shape[1], dtype=int)
    selected: list[tuple[int, int]] = []
    for _ in range(int(max_budget)):
        eligible_rows = target_rows[candidates[target_rows].any(axis=1)]
        if not len(eligible_rows):
            raise InfeasibleMaskError(
                f"Only {len(selected)} non-scored adaptation cells are available; "
                f"requested {max_budget}."
            )
        minimum_row_usage = int(row_usage[eligible_rows].min())
        eligible_rows = eligible_rows[row_usage[eligible_rows] == minimum_row_usage]
        row = int(rng.choice(eligible_rows))
        columns = np.flatnonzero(candidates[row])
        minimum_column_usage = int(column_usage[columns].min())
        columns = columns[column_usage[columns] == minimum_column_usage]
        column = int(rng.choice(columns))
        selected.append((row, column))
        candidates[row, column] = False
        row_usage[row] += 1
        column_usage[column] += 1
    return selected


def make_fewshot_local_masks(
    observed_mask: np.ndarray,
    feature_types: pd.Series | np.ndarray,
    resource_groups: np.ndarray,
    *,
    seed: int,
    quotas: Mapping[str, int],
    adaptation_budgets: Iterable[int] = ADAPTATION_BUDGETS,
) -> Iterator[SplitMasks]:
    observed = _boolean_mask(observed_mask, name="observed_mask")
    types = as_feature_type_array(feature_types, observed.shape[1])
    groups = np.asarray(resource_groups, dtype=str)
    known = observed.sum(axis=1).astype(int)
    budgets = tuple(sorted(set(int(value) for value in adaptation_budgets)))
    if not budgets or budgets[0] < 0:
        raise ValueError("adaptation_budgets must contain non-negative integers.")
    max_budget = budgets[-1]
    for group in RESOURCE_GROUPS:
        for target_type in FEATURE_TYPES:
            rng = np.random.default_rng(
                derived_seed(seed, f"local:{group}:{target_type}")
            )
            target_columns = np.flatnonzero(types == target_type)
            domain_known = observed[:, target_columns].sum(axis=1).astype(int)
            eligible_rows = np.flatnonzero(
                (groups == group)
                & (domain_known > 0)
                & ((known - domain_known) > 0)
            )
            pools, ordered_targets = _partition_with_capacity(
                observed,
                eligible_rows,
                types,
                quotas,
                rng,
                required_types=(target_type,),
            )
            language_split = np.zeros(observed.shape[0], dtype=np.int8)
            scored_by_split = {
                name: np.zeros_like(observed) for name in EVALUATION_SPLITS
            }
            for split_name, pool in pools.items():
                language_split[pool] = SPLIT_CODES[split_name]
                _balanced_sample_cells(
                    observed,
                    pool,
                    target_columns,
                    int(quotas[split_name]),
                    rng,
                    scored_by_split[split_name],
                )
            val = scored_by_split["validation"]
            cal = scored_by_split["calibration"]
            test = scored_by_split["test"]
            scored = val | cal | test
            blocked = np.zeros_like(observed)
            blocked[np.ix_(ordered_targets, target_columns)] = observed[
                np.ix_(ordered_targets, target_columns)
            ]
            adaptation_order = _nested_adaptation_order(
                blocked & ~scored,
                ordered_targets,
                max_budget,
                rng,
            )
            for budget in budgets:
                adaptation = np.zeros_like(observed)
                if budget:
                    positions = adaptation_order[:budget]
                    adaptation[
                        np.asarray([row for row, _ in positions], dtype=int),
                        np.asarray([column for _, column in positions], dtype=int),
                    ] = True
                removed = blocked & ~adaptation
                regime = f"local_fewshot_{target_type}_{group}_n{budget}"
                yield _new_split_masks(
                    regime=regime,
                    seed=seed,
                    observed=observed,
                    removed=removed,
                    val=val.copy(),
                    cal=cal.copy(),
                    test=test.copy(),
                    adaptation=adaptation,
                    target_languages=ordered_targets,
                    resource_groups=groups,
                    language_split=language_split,
                    pool_source=(
                        f"{group}_{target_type}_eligible_languages_"
                        "partitioned_into_disjoint_split_pools"
                    ),
                    resource_group=group,
                    target_feature_type=target_type,
                    adaptation_budget=budget,
                )


def _check(condition: bool, message: str, masks: SplitMasks) -> None:
    if not condition:
        raise AssertionError(
            f"{message} [regime={masks.regime}, seed={masks.seed}, "
            f"train_visible={int(masks.train_visible_mask.sum())}, "
            f"val={int(masks.val_mask.sum())}, cal={int(masks.cal_mask.sum())}, "
            f"test={int(masks.test_mask.sum())}]"
        )


def validate_split_masks(
    masks: SplitMasks,
    feature_types: pd.Series | np.ndarray,
    *,
    expected_val_size: int,
    expected_cal_size: int,
    expected_test_size: int,
    expected_stratum_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> None:
    shape = masks.observed_mask.shape
    for name in (
        "regime_removed_mask",
        "train_removed_mask",
        "train_visible_mask",
        "val_mask",
        "cal_mask",
        "test_mask",
        "unscored_removed_mask",
        "adaptation_mask",
    ):
        _boolean_mask(getattr(masks, name), name=name, shape=shape)
    _check(len(masks.resource_groups) == shape[0], "wrong resource group length", masks)
    _check(masks.scored_resource_groups.shape == shape, "wrong scoring group shape", masks)
    _check(len(masks.language_split) == shape[0], "wrong language split length", masks)

    observed = masks.observed_mask
    val, cal, test = masks.val_mask, masks.cal_mask, masks.test_mask
    scored = val | cal | test
    _check(int(val.sum()) == expected_val_size, "wrong validation size", masks)
    _check(int(cal.sum()) == expected_cal_size, "wrong calibration size", masks)
    _check(int(test.sum()) == expected_test_size, "wrong test size", masks)
    _check(not np.any(val & cal), "validation/calibration overlap", masks)
    _check(not np.any(val & test), "validation/test overlap", masks)
    _check(not np.any(cal & test), "calibration/test overlap", masks)
    _check(np.all(scored <= observed), "scored mask contains missing cells", masks)
    _check(np.all(scored <= masks.train_removed_mask), "scored cells are train-visible", masks)
    _check(
        np.array_equal(masks.train_removed_mask, masks.regime_removed_mask),
        "train_removed is inconsistent",
        masks,
    )
    _check(
        np.array_equal(masks.train_visible_mask, observed & ~masks.train_removed_mask),
        "train_visible is inconsistent",
        masks,
    )
    _check(
        np.array_equal(masks.unscored_removed_mask, masks.train_removed_mask & ~scored),
        "unscored_removed is inconsistent",
        masks,
    )
    _check(not np.any(masks.adaptation_mask & masks.train_removed_mask), "adaptation remains masked", masks)
    _check(not np.any(masks.adaptation_mask & scored), "adaptation overlaps scored cells", masks)
    _check(
        np.all(np.isin(masks.scored_resource_groups[scored], RESOURCE_GROUPS)),
        "a scored cell lacks a P1/P2 scoring group",
        masks,
    )
    _check(
        not np.any(masks.scored_resource_groups[~scored] != ""),
        "an unscored cell has a scoring group",
        masks,
    )

    split_row_sets: list[set[int]] = []
    for split_name, split_mask in (
        ("validation", val),
        ("calibration", cal),
        ("test", test),
    ):
        rows = np.flatnonzero(split_mask.any(axis=1))
        split_row_sets.append(set(rows.tolist()))
        _check(
            np.all(masks.language_split[rows] == SPLIT_CODES[split_name]),
            f"{split_name} contains a language assigned to another split",
            masks,
        )
    _check(not (split_row_sets[0] & split_row_sets[1]), "validation/calibration share languages", masks)
    _check(not (split_row_sets[0] & split_row_sets[2]), "validation/test share languages", masks)
    _check(not (split_row_sets[1] & split_row_sets[2]), "calibration/test share languages", masks)

    if masks.regime == "resource_copy":
        events = [case.event_id for case in masks.copy_provenance]
        target_by_event = {
            event: {case.target_index for case in masks.copy_provenance if case.event_id == event}
            for event in set(events)
        }
        event_targets = [next(iter(values)) for values in target_by_event.values()]
        _check(all(len(values) == 1 for values in target_by_event.values()), "copy event has multiple targets", masks)
        _check(len(event_targets) == len(set(event_targets)), "copy target is used more than once", masks)
        _check(len(masks.copy_provenance) == int(scored.sum()), "copy provenance is not cell-complete", masks)
        by_event = {
            event: [case for case in masks.copy_provenance if case.event_id == event]
            for event in set(events)
        }
        for cases in by_event.values():
            first = cases[0]
            donor, target = first.donor_index, first.target_index
            hidden = observed[target] & ~observed[donor]
            intersection_count = int((observed[target] & observed[donor]).sum())
            _check(
                np.array_equal(
                    masks.train_visible_mask[target],
                    observed[target] & observed[donor],
                ),
                "copy event did not transfer the complete donor pattern",
                masks,
            )
            _check(
                all(hidden[case.scored_feature_index] for case in cases),
                "copy provenance includes a cell outside H_dt",
                masks,
            )
            _check(
                all(case.intersection_count == intersection_count for case in cases),
                "copy provenance has an incorrect intersection count",
                masks,
            )
            post_group = "P2" if first.target_post_mask_group == "P2" else "P1"
            _check(
                all(
                    case.donor_group == post_group
                    and masks.scored_resource_groups[
                        target, case.scored_feature_index
                    ] == case.donor_group
                    for case in cases
                ),
                "copy scoring group does not match the target post-mask group",
                masks,
            )

    if expected_stratum_counts is not None:
        for split_name, split_mask in (
            ("validation", val),
            ("calibration", cal),
            ("test", test),
        ):
            actual = resource_domain_counts(
                split_mask,
                feature_types,
                masks.scored_resource_groups,
            )
            for key, value in expected_stratum_counts[split_name].items():
                _check(
                    actual.get(key, 0) == int(value),
                    f"{split_name} stratum {key} has {actual.get(key, 0)} cells; expected {value}",
                    masks,
                )


__all__ = [
    "ADAPTATION_BUDGETS",
    "CopyMaskCase",
    "EVALUATION_SPLITS",
    "InfeasibleMaskError",
    "RESOURCE_GROUPS",
    "SPLIT_CODES",
    "SplitMasks",
    "derived_seed",
    "make_fewshot_local_masks",
    "make_resource_conditioned_copy_mask",
    "make_stratified_mcar_mask",
    "resource_domain_counts",
    "validate_split_masks",
]
