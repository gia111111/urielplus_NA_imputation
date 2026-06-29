from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_types import FEATURE_TYPES, as_feature_type_array
from .masking import SplitMasks, validate_split_masks
from .splits import apply_train_visible_mask, mask_to_cell_df


SPLIT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SplitPaths:
    regime: str
    seed: int
    split_path: str
    metadata_path: str
    n_target_languages: int
    n_train_observed: int
    n_regime_removed: int
    n_equalization_removed: int
    n_heldout: int
    n_val: int
    n_test: int
    target_language_pool_source: str


@dataclass(frozen=True)
class LoadedSplit:
    regime: str
    seed: int
    masks: SplitMasks
    train_matrix: pd.DataFrame
    val_cells: pd.DataFrame
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
    val_counts = masks.val_mask.sum(axis=1).astype(int)
    test_counts = masks.test_mask.sum(axis=1).astype(int)
    rows = np.flatnonzero((val_counts + test_counts) > 0)
    return [
        {
            "language_index": int(row),
            "language_id": str(language_ids[row]),
            "validation_cells": int(val_counts[row]),
            "test_cells": int(test_counts[row]),
        }
        for row in rows
    ]


def write_split(
    outdir: str | Path,
    X: pd.DataFrame,
    feature_types: pd.Series,
    masks: SplitMasks,
) -> SplitPaths:
    """Write one compressed split artifact and one debug metadata file."""
    split_dir = Path(outdir) / "splits" / masks.regime / f"seed_{masks.seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / "split.npz"
    metadata_path = split_dir / "metadata.json"
    types = as_feature_type_array(feature_types, X.shape[1])

    validate_split_masks(
        masks,
        types,
        expected_val_size=int(masks.val_mask.sum()),
        expected_test_size=int(masks.test_mask.sum()),
    )
    np.savez_compressed(
        split_path,
        observed_mask=masks.observed_mask,
        regime_removed_mask=masks.regime_removed_mask,
        equalization_removed_mask=masks.equalization_removed_mask,
        train_removed_mask=masks.train_removed_mask,
        train_visible_mask=masks.train_visible_mask,
        val_mask=masks.val_mask,
        test_mask=masks.test_mask,
        unscored_removed_mask=masks.unscored_removed_mask,
        target_languages=masks.target_languages,
    )

    metadata = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "regime": masks.regime,
        "seed": int(masks.seed),
        "matrix_shape": [int(X.shape[0]), int(X.shape[1])],
        "dataset_fingerprint": _dataset_fingerprint(X),
        "n_observed": int(masks.observed_mask.sum()),
        "n_target_languages": int(len(masks.target_languages)),
        "target_language_indices": masks.target_languages.astype(int).tolist(),
        "target_language_ids": X.index.to_numpy()[masks.target_languages].astype(str).tolist(),
        "target_language_pool_source": masks.target_language_pool_source,
        "n_regime_removed": int(masks.regime_removed_mask.sum()),
        "n_equalization_removed": int(masks.equalization_removed_mask.sum()),
        "n_train_removed": int(masks.train_removed_mask.sum()),
        "n_train_observed": int(masks.train_visible_mask.sum()),
        "n_val": int(masks.val_mask.sum()),
        "n_test": int(masks.test_mask.sum()),
        "n_heldout": int(masks.val_mask.sum() + masks.test_mask.sum()),
        "n_unscored_removed": int(masks.unscored_removed_mask.sum()),
        "feature_type_counts": {
            "validation": _feature_type_counts(masks.val_mask, types),
            "test": _feature_type_counts(masks.test_mask, types),
            "regime_removed": _feature_type_counts(masks.regime_removed_mask, types),
            "equalization_removed": _feature_type_counts(
                masks.equalization_removed_mask,
                types,
            ),
        },
        "overlap_checks": {
            "validation_test": int((masks.val_mask & masks.test_mask).sum()),
            "equalization_validation_test": int(
                (
                    masks.equalization_removed_mask
                    & (masks.val_mask | masks.test_mask)
                ).sum()
            ),
            "equalization_regime_removed": int(
                (
                    masks.equalization_removed_mask
                    & masks.regime_removed_mask
                ).sum()
            ),
            "heldout_visible_in_training": int(
                (
                    (masks.val_mask | masks.test_mask)
                    & masks.train_visible_mask
                ).sum()
            ),
        },
        "per_language_validation_test_counts": _per_language_scored_counts(
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
        n_target_languages=int(len(masks.target_languages)),
        n_train_observed=int(masks.train_visible_mask.sum()),
        n_regime_removed=int(masks.regime_removed_mask.sum()),
        n_equalization_removed=int(masks.equalization_removed_mask.sum()),
        n_heldout=int(masks.val_mask.sum() + masks.test_mask.sum()),
        n_val=int(masks.val_mask.sum()),
        n_test=int(masks.test_mask.sum()),
        target_language_pool_source=masks.target_language_pool_source,
    )


def write_split_manifest(outdir: str | Path, rows: list[SplitPaths]) -> Path:
    manifest_dir = Path(outdir) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    records = [asdict(row) for row in rows]
    manifest_path = manifest_dir / "split_manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)
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
        record["split_path"] = resolve_manifest_path(str(row["split_path"]), manifest_path)
        record["metadata_path"] = resolve_manifest_path(
            str(row["metadata_path"]),
            manifest_path,
        )
        rows.append(record)
    return rows


def _load_masks(split_path: Path, metadata: dict) -> SplitMasks:
    with np.load(split_path) as archive:
        required = {
            "observed_mask",
            "regime_removed_mask",
            "equalization_removed_mask",
            "train_removed_mask",
            "train_visible_mask",
            "val_mask",
            "test_mask",
            "unscored_removed_mask",
            "target_languages",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Split {split_path} is missing arrays: {sorted(missing)}")
        arrays = {name: archive[name] for name in required}
    return SplitMasks(
        regime=str(metadata["regime"]),
        seed=int(metadata["seed"]),
        observed_mask=arrays["observed_mask"].astype(bool),
        regime_removed_mask=arrays["regime_removed_mask"].astype(bool),
        equalization_removed_mask=arrays["equalization_removed_mask"].astype(bool),
        train_removed_mask=arrays["train_removed_mask"].astype(bool),
        train_visible_mask=arrays["train_visible_mask"].astype(bool),
        val_mask=arrays["val_mask"].astype(bool),
        test_mask=arrays["test_mask"].astype(bool),
        unscored_removed_mask=arrays["unscored_removed_mask"].astype(bool),
        target_languages=arrays["target_languages"].astype(int),
        target_language_pool_source=str(metadata["target_language_pool_source"]),
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
            "observed-cell pattern. Regenerate splits with matching data/filter options."
        )
    masks = _load_masks(split_path, metadata)
    validate_split_masks(
        masks,
        feature_types,
        expected_val_size=int(metadata["n_val"]),
        expected_test_size=int(metadata["n_test"]),
    )
    train_matrix = apply_train_visible_mask(X, masks.train_visible_mask)
    if int(train_matrix.notna().sum().sum()) != int(metadata["n_train_observed"]):
        raise AssertionError(
            f"Loaded train matrix count mismatch for regime={masks.regime}, seed={masks.seed}: "
            f"actual={int(train_matrix.notna().sum().sum())}, "
            f"metadata={int(metadata['n_train_observed'])}."
        )
    val_cells = mask_to_cell_df(
        masks.val_mask,
        X,
        feature_types,
        split="val",
        regime=masks.regime,
        seed=masks.seed,
    )
    test_cells = mask_to_cell_df(
        masks.test_mask,
        X,
        feature_types,
        split="test",
        regime=masks.regime,
        seed=masks.seed,
    )
    return LoadedSplit(
        regime=masks.regime,
        seed=masks.seed,
        masks=masks,
        train_matrix=train_matrix,
        val_cells=val_cells,
        test_cells=test_cells,
        metadata=metadata,
        split_path=split_path,
        metadata_path=metadata_path,
    )
