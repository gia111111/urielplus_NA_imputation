from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from urielplus_impute.data import load_dataset
from urielplus_impute.split_io import load_split, write_split
from urielplus_impute.splits import StratumQuotas, iter_stratified_seed_masks


class SplitRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(
            REPOSITORY_ROOT / "urielplus_analysis" / "typological_data.csv",
            REPOSITORY_ROOT / "urielplus_analysis" / "languages.csv",
        )

    def test_copy_artifact_and_provenance_round_trip(self) -> None:
        quotas = StratumQuotas()
        masks = next(
            iter_stratified_seed_masks(
                self.dataset.X,
                self.dataset.feature_types,
                self.dataset.languages,
                seed=0,
                quotas=quotas,
                regimes=("resource_copy",),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_split(
                directory,
                self.dataset.X,
                self.dataset.feature_types,
                masks,
                quotas=quotas,
            )
            self.assertIsNotNone(paths.copy_provenance_path)
            self.assertTrue(Path(paths.copy_provenance_path).is_file())
            loaded = load_split(
                {
                    "split_path": paths.split_path,
                    "metadata_path": paths.metadata_path,
                },
                self.dataset.X,
                self.dataset.feature_types,
            )
            self.assertEqual(len(loaded.val_cells), 4000)
            self.assertEqual(len(loaded.cal_cells), 4000)
            self.assertEqual(len(loaded.test_cells), 8000)
            self.assertEqual(len(loaded.masks.copy_provenance), 16000)
            self.assertIn("resource_group", loaded.test_cells.columns)
            self.assertIn(
                "target_original_resource_group",
                loaded.test_cells.columns,
            )


if __name__ == "__main__":
    unittest.main()
