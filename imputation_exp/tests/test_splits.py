from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urielplus_impute.feature_types import feature_type_series
from urielplus_impute.masking import resource_domain_counts
from urielplus_impute.split_io import (
    load_split,
    load_split_manifest_rows,
    write_split,
    write_split_manifest,
)
from urielplus_impute.splits import (
    StratumQuotas,
    build_resource_stratification,
    iter_stratified_seed_masks,
    make_seed_summary,
)


class StratifiedSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        columns = (
            [f"S_{index}" for index in range(474)]
            + [f"P_{index}" for index in range(30)]
            + [f"M_{index}" for index in range(133)]
            + [f"INV_{index}" for index in range(163)]
        )
        values = np.full((14, 800), np.nan)
        # Two Z0 rows, six low-resource rows, and six high-resource rows.
        for offset, row in enumerate(range(2, 8)):
            count = 20 + offset
            domain_columns = (
                list(range(4))
                + list(range(474, 478))
                + list(range(504, 508))
                + list(range(637, 641))
            )
            chosen = domain_columns + list(range(4, count - 12))
            values[row, chosen] = (np.arange(len(chosen)) % 2).astype(float)
        for offset, row in enumerate(range(8, 14)):
            count = 200 + 5 * offset
            domain_columns = (
                list(range(4))
                + list(range(474, 478))
                + list(range(504, 508))
                + list(range(637, 641))
            )
            chosen = domain_columns + list(range(4, count - 12))
            values[row, chosen] = (np.arange(len(chosen)) % 2).astype(float)
        index = [f"lang{row:02d}" for row in range(14)]
        self.X = pd.DataFrame(values, index=index, columns=columns).iloc[2:]
        self.feature_types = feature_type_series(columns)
        self.languages = pd.DataFrame(
            {
                "language_id": index,
                "family_id": [f"fam{row // 3}" for row in range(14)],
                "macroarea": ["A" if row % 2 else "B" for row in range(14)],
                "latitude": np.linspace(-30, 30, 14),
                "longitude": np.linspace(-60, 60, 14),
            },
            index=index,
        ).reindex(self.X.index)
        self.quotas = StratumQuotas(validation=1, calibration=1, test=2)

    def _masks(self, regimes: list[str]):
        summary = make_seed_summary(
            self.X,
            self.feature_types,
            seed=7,
            quotas=self.quotas,
        )
        masks = list(
            iter_stratified_seed_masks(
                self.X,
                self.feature_types,
                self.languages,
                seed=7,
                quotas=self.quotas,
                regimes=regimes,
                summary=summary,
            )
        )
        return masks, summary

    def test_resource_groups_are_equal_positive_halves(self) -> None:
        groups = build_resource_stratification(
            self.X.notna().to_numpy()
        ).groups
        self.assertEqual(int((groups == "P1").sum()), 6)
        self.assertEqual(int((groups == "P2").sum()), 6)

    def test_mcar_has_balanced_strata_and_disjoint_language_pools(self) -> None:
        masks, summary = self._masks(["mcar"])
        self.assertEqual(len(masks), 1)
        split = masks[0]
        self.assertEqual(int(split.val_mask.sum()), 8)
        self.assertEqual(int(split.cal_mask.sum()), 8)
        self.assertEqual(int(split.test_mask.sum()), 16)
        for mask, expected in (
            (split.val_mask, 1),
            (split.cal_mask, 1),
            (split.test_mask, 2),
        ):
            counts = resource_domain_counts(
                mask,
                self.feature_types,
                split.scored_resource_groups,
            )
            self.assertTrue(all(value == expected for value in counts.values()))
        val_rows = set(np.flatnonzero(split.val_mask.any(axis=1)))
        cal_rows = set(np.flatnonzero(split.cal_mask.any(axis=1)))
        test_rows = set(np.flatnonzero(split.test_mask.any(axis=1)))
        self.assertFalse(val_rows & cal_rows)
        self.assertFalse(val_rows & test_rows)
        self.assertFalse(cal_rows & test_rows)
        post_known = split.train_visible_mask.sum(axis=1)
        self.assertTrue(np.all(post_known[split.resource_groups == "P1"] >= 1))
        self.assertTrue(np.all(post_known[split.resource_groups == "P2"] >= 175))
        self.assertEqual(summary.failures, [])

    def test_copy_capacity_failure_is_recorded(self) -> None:
        impossible = StratumQuotas(validation=100, calibration=100, test=200)
        summary = make_seed_summary(
            self.X,
            self.feature_types,
            seed=3,
            quotas=impossible,
        )
        masks = list(
            iter_stratified_seed_masks(
                self.X,
                self.feature_types,
                self.languages,
                seed=3,
                quotas=impossible,
                regimes=["resource_copy"],
                summary=summary,
            )
        )
        self.assertEqual(masks, [])
        self.assertEqual(len(summary.failures), 1)
        self.assertIn("exact quotas", summary.failures[0]["reason"])

    def test_schema_v4_round_trip_includes_calibration(self) -> None:
        masks, _ = self._masks(["mcar"])
        with tempfile.TemporaryDirectory() as tmp:
            record = write_split(
                tmp,
                self.X,
                self.feature_types,
                masks[0],
                quotas=self.quotas,
            )
            manifest = write_split_manifest(tmp, [record])
            row = load_split_manifest_rows(manifest)[0]
            loaded = load_split(row, self.X, self.feature_types)
            self.assertEqual(len(loaded.val_cells), 8)
            self.assertEqual(len(loaded.cal_cells), 8)
            self.assertEqual(len(loaded.test_cells), 16)
            metadata = json.loads(Path(record.metadata_path).read_text())
            self.assertEqual(metadata["schema_version"], 4)
            self.assertEqual(metadata["overlap_checks"]["validation_calibration"], 0)


if __name__ == "__main__":
    unittest.main()
