from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from urielplus_impute.data import apply_joint_coverage_filter, load_dataset
from urielplus_impute.feature_types import FEATURE_TYPES
from urielplus_impute.splits import build_resource_stratification


class CoverageFilterTests(unittest.TestCase):
    def test_joint_filter_alternates_languages_then_features_until_stable(self) -> None:
        matrix = pd.DataFrame(
            {
                "S_a": [1.0, 1.0, np.nan, np.nan],
                "P_a": [1.0, np.nan, 1.0, np.nan],
                "M_a": [np.nan, np.nan, 1.0, np.nan],
                "INV_a": [np.nan, np.nan, np.nan, 1.0],
            },
            index=["a", "b", "c", "d"],
        )
        filtered, summary = apply_joint_coverage_filter(
            matrix,
            language_min_coverage=0.5,
            feature_min_coverage=0.5,
        )
        self.assertEqual(filtered.index.tolist(), ["a", "c"])
        self.assertEqual(filtered.columns.tolist(), ["S_a", "P_a", "M_a"])
        self.assertGreaterEqual(len(summary.iterations), 2)
        self.assertTrue((filtered.notna().mean(axis=1) >= 0.5).all())
        self.assertTrue((filtered.notna().mean(axis=0) >= 0.5).all())

    def test_reference_cutoff_regression(self) -> None:
        dataset = load_dataset(
            REPOSITORY_ROOT / "urielplus_analysis" / "typological_data.csv",
            REPOSITORY_ROOT / "urielplus_analysis" / "languages.csv",
        )
        self.assertEqual(dataset.filter_summary.input_languages, 7723)
        self.assertEqual(dataset.X.shape, (3802, 456))
        self.assertEqual(int(dataset.X.notna().to_numpy().sum()), 834850)
        self.assertAlmostEqual(
            100.0 * dataset.filter_summary.output_missingness,
            51.84609669887501,
            places=10,
        )
        self.assertEqual(
            dataset.feature_types.value_counts().reindex(FEATURE_TYPES).to_dict(),
            {"S": 195, "P": 28, "M": 75, "INV": 158},
        )
        self.assertTrue((dataset.X.notna().mean(axis=1) >= 0.05).all())
        self.assertTrue((dataset.X.notna().mean(axis=0) >= 0.05).all())
        self.assertTrue((dataset.X.notna().sum(axis=1) > 0).all())
        self.assertFalse(
            dataset.languages["raw_family_id"].isin(
                {
                    "unat1236",
                    "sign1238",
                    "arti1236",
                    "uncl1493",
                    "spee1234",
                    "book1242",
                }
            ).any()
        )

        repeated = load_dataset(
            REPOSITORY_ROOT / "urielplus_analysis" / "typological_data.csv",
            REPOSITORY_ROOT / "urielplus_analysis" / "languages.csv",
        )
        pd.testing.assert_frame_equal(dataset.X, repeated.X)


class ResourceStratificationTests(unittest.TestCase):
    def test_odd_population_and_boundary_ties_are_stable(self) -> None:
        observed = np.asarray(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        )
        stratification = build_resource_stratification(observed)
        self.assertEqual(stratification.groups.tolist(), ["P1", "P1", "P2", "P2", "P2"])
        self.assertEqual(stratification.midpoint, 2)
        self.assertEqual(stratification.boundary_count, 2)
        self.assertEqual(stratification.boundary_tie_total, 2)
        self.assertEqual(stratification.boundary_tie_p1, 1)
        self.assertEqual(stratification.boundary_tie_p2, 1)
        self.assertEqual(stratification.post_mask_group(1, 2), "P1")
        self.assertEqual(stratification.post_mask_group(2, 2), "P2")
        self.assertEqual(stratification.post_mask_group(2, 1), "P1")
        self.assertEqual(int((stratification.groups == "P1").sum()), 2)
        self.assertEqual(int((stratification.groups == "P2").sum()), 3)


if __name__ == "__main__":
    unittest.main()
