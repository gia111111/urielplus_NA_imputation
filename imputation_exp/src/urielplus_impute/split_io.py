from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_types import FEATURE_TYPES, as_feature_type_array
from .masking import CopyMaskCase, RESOURCE_GROUPS, SplitMasks, resource_domain_counts
from .splits import (
    StratumQuotas,
    apply_train_visible_mask,
    mask_to_cell_df,
    validate_stratified_masks,
)


SPLIT_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class SplitPaths:
    regime: str
    seed: int
    split_path: str
    metadata_path: str
    copy_provenance_path: str | None
    n_target_languages: int
    n_train_observed: int
    n_regime_removed: int
    n_heldout: int
    n_val: int
    n_cal: int
    n_test: int
    target_language_pool_source: str


@dataclass(frozen=True)
class LoadedSplit:
    regime: str
    seed: int
    masks: SplitMasks
    train_matrix: pd.DataFrame
    val_cells: pd.DataFrame
    cal_cells: pd.DataFrame
    test_cells: pd.DataFrame
    metadata: dict
    split_path: Path
    metadata_path: Path


def _dataset_fingerprint(X: pd.DataFrame) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update("\0".join(X.index.astype(str)).encode("utf-8"))
    digest.update(b"\1")
    digest.update("\0".join(X.columns.astype(str)).encode("utf-8"))
    digest.update(b"\2")
    digest.update(np.packbits(X.notna().to_numpy(dtype=bool)).tobytes())
    return digest.hexdigest()


def _feature_type_counts(mask: np.ndarray, feature_types: np.ndarray) -> dict[str, int]:
    return {
        target_type: int(mask[:, feature_types == target_type].sum())
        for target_type in FEATURE_TYPES
    }


def _per_language_scored_counts(
    masks: SplitMasks,
    language_ids: np.ndarray,
) -> list[dict[str, int | str]]:
    split_counts = {
        "validation_cells": masks.val_mask.sum(axis=1).astype(int),
        "calibration_cells": masks.cal_mask.sum(axis=1).astype(int),
        "test_cells": masks.test_mask.sum(axis=1).astype(int),
    }
    total = sum(split_counts.values())
    return [
        {
            "language_index": int(row),
            "language_id": str(language_ids[row]),
            "target_original_resource_group": str(masks.resource_groups[row]),
            **{name: int(counts[row]) for name, counts in split_counts.items()},
        }
        for row in np.flatnonzero(total > 0)
    ]


def _actual_stratum_counts(
    masks: SplitMasks,
    feature_types: np.ndarray,
) -> dict[str, dict[str, int]]:
    return {
        split_name: resource_domain_counts(
            split_mask,
            feature_types,
            masks.scored_resource_groups,
        )
        for split_name, split_mask in (
            ("validation", masks.val_mask),
            ("calibration", masks.cal_mask),
            ("test", masks.test_mask),
        )
    }


def _write_copy_provenance(
    path: Path,
    cases: tuple[CopyMaskCase, ...],
    feature_ids: np.ndarray,
) -> None:
    records: list[dict] = []
    for case in cases:
        record = asdict(case)
        record["preselected_feature"] = str(
            feature_ids[case.preselected_feature_index]
        )
        record["scored_feature"] = str(feature_ids[case.scored_feature_index])
        records.append(record)
    pd.DataFrame(records).to_csv(path, index=False)


def write_split(
    outdir: str | Path,
    X: pd.DataFrame,
    feature_types: pd.Series,
    masks: SplitMasks,
    *,
    quotas: StratumQuotas,
) -> SplitPaths:
    """Write one schema-v4 split artifact and its complete audit metadata."""
    quota_values = quotas.as_dict()
    expected_strata = validate_stratified_masks(masks, feature_types, quota_values)
    split_dir = Path(outdir) / "splits" / masks.regime / f"seed_{masks.seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / "split.npz"
    metadata_path = split_dir / "metadata.json"
    types = as_feature_type_array(feature_types, X.shape[1])

    np.savez_compressed(
        split_path,
        observed_mask=masks.observed_mask,
        regime_removed_mask=masks.regime_removed_mask,
        train_removed_mask=masks.train_removed_mask,
        train_visible_mask=masks.train_visible_mask,
        val_mask=masks.val_mask,
        cal_mask=masks.cal_mask,
        test_mask=masks.test_mask,
        target_languages=masks.target_languages,
        resource_groups=masks.resource_groups,
        scored_resource_groups=masks.scored_resource_groups,
        language_split=masks.language_split,
    )

    provenance_path: Path | None = None
    if masks.copy_provenance:
        provenance_path = split_dir / "copy_provenance.csv"
        _write_copy_provenance(
            provenance_path,
            masks.copy_provenance,
            X.columns.to_numpy(),
        )

    scored = masks.val_mask | masks.cal_mask | masks.test_mask
    metadata = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "regime": masks.regime,
        "seed": int(masks.seed),
        "matrix_shape": [int(X.shape[0]), int(X.shape[1])],
        "dataset_fingerprint": _dataset_fingerprint(X),
        "quotas_per_stratum": quota_values,
        "expected_stratum_counts": expected_strata,
        "actual_stratum_counts": _actual_stratum_counts(masks, types),
        "n_observed": int(masks.observed_mask.sum()),
        "n_target_languages": int(len(masks.target_languages)),
        "target_language_indices": masks.target_languages.astype(int).tolist(),
        "target_language_ids": (
            X.index.to_numpy()[masks.target_languages].astype(str).tolist()
        ),
        "target_language_pool_source": masks.target_language_pool_source,
        "n_regime_removed": int(masks.regime_removed_mask.sum()),
        "n_train_removed": int(masks.train_removed_mask.sum()),
        "n_train_observed": int(masks.train_visible_mask.sum()),
        "n_val": int(masks.val_mask.sum()),
        "n_cal": int(masks.cal_mask.sum()),
        "n_test": int(masks.test_mask.sum()),
        "n_heldout": int(scored.sum()),
        "copy_events": int(
            len({case.event_id for case in masks.copy_provenance})
        ),
        "copy_provenance_path": (
            provenance_path.name if provenance_path is not None else None
        ),
        "language_counts_by_resource_group": {
            group: int((masks.resource_groups == group).sum())
            for group in RESOURCE_GROUPS
        },
        "feature_type_counts": {
            "validation": _feature_type_counts(masks.val_mask, types),
            "calibration": _feature_type_counts(masks.cal_mask, types),
            "test": _feature_type_counts(masks.test_mask, types),
            "regime_removed": _feature_type_counts(masks.regime_removed_mask, types),
        },
        "overlap_checks": {
            "validation_calibration": int((masks.val_mask & masks.cal_mask).sum()),
            "validation_test": int((masks.val_mask & masks.test_mask).sum()),
            "calibration_test": int((masks.cal_mask & masks.test_mask).sum()),
            "scored_visible_in_training": int(
                (scored & masks.train_visible_mask).sum()
            ),
        },
        "per_language_scored_counts": _per_language_scored_counts(
            masks,
            X.index.to_numpy(),
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return SplitPaths(
        regime=masks.regime,
        seed=int(masks.seed),
        split_path=str(split_path),
        metadata_path=str(metadata_path),
        copy_provenance_path=(
            str(provenance_path) if provenance_path is not None else None
        ),
        n_target_languages=int(len(masks.target_languages)),
        n_train_observed=int(masks.train_visible_mask.sum()),
        n_regime_removed=int(masks.regime_removed_mask.sum()),
        n_heldout=int(scored.sum()),
        n_val=int(masks.val_mask.sum()),
        n_cal=int(masks.cal_mask.sum()),
        n_test=int(masks.test_mask.sum()),
        target_language_pool_source=masks.target_language_pool_source,
    )


def write_split_manifest(outdir: str | Path, rows: list[SplitPaths]) -> Path:
    manifest_dir = Path(outdir) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "split_manifest.csv"
    pd.DataFrame([asdict(row) for row in rows]).to_csv(manifest_path, index=False)
    return manifest_path


def resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    for candidate in (
        Path.cwd() / path,
        manifest_path.parent / path,
        manifest_path.parent.parent / path,
    ):
        if candidate.exists():
            return candidate
    return path


def load_split_manifest_rows(
    manifest_path: str | Path,
    *,
    regimes: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[dict]:
    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path)
    required = {"regime", "seed", "split_path", "metadata_path"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(
            f"Split manifest {manifest_path} is missing columns: {sorted(missing)}"
        )
    if regimes is not None:
        manifest = manifest[manifest["regime"].astype(str).isin(regimes)]
    if seeds is not None:
        manifest = manifest[manifest["seed"].astype(int).isin(seeds)]
    if manifest.empty:
        raise ValueError("No split rows remain after applying regime/seed filters.")

    rows = []
    for _, row in manifest.iterrows():
        record = row.to_dict()
        record["regime"] = str(row["regime"])
        record["seed"] = int(row["seed"])
        record["split_path"] = resolve_manifest_path(
            str(row["split_path"]), manifest_path
        )
        record["metadata_path"] = resolve_manifest_path(
            str(row["metadata_path"]), manifest_path
        )
        rows.append(record)
    return rows


def _load_copy_provenance(metadata: dict, metadata_path: Path) -> tuple[CopyMaskCase, ...]:
    value = metadata.get("copy_provenance_path")
    if not value:
        return ()
    path = resolve_manifest_path(str(value), metadata_path)
    frame = pd.read_csv(path)
    fields = CopyMaskCase.__dataclass_fields__
    return tuple(
        CopyMaskCase(**{name: row[name] for name in fields})
        for _, row in frame.iterrows()
    )


def _load_masks(split_path: Path, metadata: dict, metadata_path: Path) -> SplitMasks:
    required = {
        "observed_mask",
        "regime_removed_mask",
        "train_removed_mask",
        "train_visible_mask",
        "val_mask",
        "cal_mask",
        "test_mask",
        "target_languages",
        "resource_groups",
        "scored_resource_groups",
        "language_split",
    }
    with np.load(split_path) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Split {split_path} is missing arrays: {sorted(missing)}")
        arrays = {name: archive[name] for name in required}
    return SplitMasks(
        regime=str(metadata["regime"]),
        seed=int(metadata["seed"]),
        observed_mask=arrays["observed_mask"].astype(bool),
        regime_removed_mask=arrays["regime_removed_mask"].astype(bool),
        train_removed_mask=arrays["train_removed_mask"].astype(bool),
        train_visible_mask=arrays["train_visible_mask"].astype(bool),
        val_mask=arrays["val_mask"].astype(bool),
        cal_mask=arrays["cal_mask"].astype(bool),
        test_mask=arrays["test_mask"].astype(bool),
        target_languages=arrays["target_languages"].astype(int),
        resource_groups=arrays["resource_groups"].astype(str),
        scored_resource_groups=arrays["scored_resource_groups"].astype(str),
        language_split=arrays["language_split"].astype(np.int8),
        target_language_pool_source=str(metadata["target_language_pool_source"]),
        copy_provenance=_load_copy_provenance(metadata, metadata_path),
    )


def load_split(
    row: dict,
    X: pd.DataFrame,
    feature_types: pd.Series,
) -> LoadedSplit:
    split_path = Path(row["split_path"])
    metadata_path = Path(row["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", -1)) != SPLIT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported split schema in {metadata_path}: "
            f"{metadata.get('schema_version')!r}."
        )
    if metadata.get("dataset_fingerprint") != _dataset_fingerprint(X):
        raise ValueError(
            f"Split {split_path} was built for a different dataset universe or "
            "observed-cell pattern. Regenerate it with matching cutoff options."
        )
    masks = _load_masks(split_path, metadata, metadata_path)
    quotas = StratumQuotas(**metadata["quotas_per_stratum"])
    validate_stratified_masks(masks, feature_types, quotas.as_dict())
    train_matrix = apply_train_visible_mask(X, masks.train_visible_mask)
    if int(train_matrix.notna().sum().sum()) != int(metadata["n_train_observed"]):
        raise AssertionError(
            f"Loaded train count mismatch for {masks.regime}/seed {masks.seed}."
        )

    cell_kwargs = {
        "X": X,
        "feature_types": feature_types,
        "regime": masks.regime,
        "seed": masks.seed,
        "resource_groups": masks.resource_groups,
        "scored_resource_groups": masks.scored_resource_groups,
    }
    return LoadedSplit(
        regime=masks.regime,
        seed=masks.seed,
        masks=masks,
        train_matrix=train_matrix,
        val_cells=mask_to_cell_df(masks.val_mask, split="val", **cell_kwargs),
        cal_cells=mask_to_cell_df(
            masks.cal_mask, split="calibration", **cell_kwargs
        ),
        test_cells=mask_to_cell_df(masks.test_mask, split="test", **cell_kwargs),
        metadata=metadata,
        split_path=split_path,
        metadata_path=metadata_path,
    )


__all__ = [
    "LoadedSplit",
    "SPLIT_SCHEMA_VERSION",
    "SplitPaths",
    "load_split",
    "load_split_manifest_rows",
    "resolve_manifest_path",
    "write_split",
    "write_split_manifest",
]
