from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from urielplus_impute.data import load_dataset
from urielplus_impute.masking import (
    EVALUATION_SPLITS,
    RESOURCE_GROUPS,
    _copy_candidate,
    resource_domain_counts,
)
from urielplus_impute.splits import (
    StratumQuotas,
    iter_stratified_seed_masks,
)


class CopyCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observed = np.asarray(
            [
                [1, 0, 1, 0, 0],  # donor
                [1, 1, 1, 1, 1],  # same family
                [1, 1, 1, 1, 0],  # same macroarea only
                [1, 1, 1, 0, 0],  # other
            ],
            dtype=bool,
        )
        self.groups = np.asarray(["P1", "P2", "P1", "P1"])
        self.post_lookup = np.zeros((4, 6), dtype=bool)
        self.family = np.asarray(["f", "f", "g", "h"])
        self.macroarea = np.asarray(["m", "m", "m", "n"])
        self.latitude = np.asarray([0.0, 0.0, 0.0, 0.0])
        self.longitude = np.asarray([0.0, 1.0, 2.0, 3.0])
        self.language_ids = np.asarray(["donor", "family", "macro", "other"])

    def test_complete_pattern_and_matching_tier(self) -> None:
        match = _copy_candidate(
            donor=0,
            anchor_feature=3,
            candidates=np.asarray([1, 2, 3]),
            observed=self.observed,
            groups=self.groups,
            donor_group="P1",
            post_mask_is_p2=self.post_lookup,
            family=self.family,
            macroarea=self.macroarea,
            latitude=self.latitude,
            longitude=self.longitude,
            language_ids=self.language_ids,
        )
        self.assertIsNotNone(match)
        target, hidden, intersection, tier, _ = match
        self.assertEqual(target, 1)
        self.assertEqual(tier, "family")
        self.assertEqual(intersection, 2)
        self.assertEqual(hidden.tolist(), [False, True, False, True, True])
        self.assertGreater(int(hidden.sum()), 1)

    def test_post_mask_resource_mismatch_rejects_candidate(self) -> None:
        self.post_lookup[1, 2] = True
        match = _copy_candidate(
            donor=0,
            anchor_feature=4,
            candidates=np.asarray([1]),
            observed=self.observed,
            groups=self.groups,
            donor_group="P1",
            post_mask_is_p2=self.post_lookup,
            family=self.family,
            macroarea=self.macroarea,
            latitude=self.latitude,
            longitude=self.longitude,
            language_ids=self.language_ids,
        )
        self.assertIsNone(match)


class FullBenchmarkMaskingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(
            REPOSITORY_ROOT / "urielplus_analysis" / "typological_data.csv",
            REPOSITORY_ROOT / "urielplus_analysis" / "languages.csv",
        )
        cls.quotas = StratumQuotas()

    def _assert_exact_quotas(self, masks) -> None:
        expected_split = {
            "validation": self.quotas.validation,
            "calibration": self.quotas.calibration,
            "test": self.quotas.test,
        }
        for name, split_mask in (
            ("validation", masks.val_mask),
            ("calibration", masks.cal_mask),
            ("test", masks.test_mask),
        ):
            counts = resource_domain_counts(
                split_mask,
                self.dataset.feature_types,
                masks.scored_resource_groups,
            )
            self.assertEqual(
                counts,
                {
                    f"{group}:{domain}": expected_split[name]
                    for group in RESOURCE_GROUPS
                    for domain in ("S", "P", "M", "INV")
                },
            )
        self.assertEqual(int(masks.val_mask.sum()), 4000)
        self.assertEqual(int(masks.cal_mask.sum()), 4000)
        self.assertEqual(int(masks.test_mask.sum()), 8000)

    def _assert_no_leakage(self, masks) -> None:
        scored = masks.val_mask | masks.cal_mask | masks.test_mask
        self.assertFalse(np.any(masks.val_mask & masks.cal_mask))
        self.assertFalse(np.any(masks.val_mask & masks.test_mask))
        self.assertFalse(np.any(masks.cal_mask & masks.test_mask))
        self.assertFalse(np.any(scored & masks.train_visible_mask))
        self.assertTrue(np.all(scored <= masks.observed_mask))
        row_sets = [
            set(np.flatnonzero(split_mask.any(axis=1)).tolist())
            for split_mask in (masks.val_mask, masks.cal_mask, masks.test_mask)
        ]
        self.assertFalse(row_sets[0] & row_sets[1])
        self.assertFalse(row_sets[0] & row_sets[2])
        self.assertFalse(row_sets[1] & row_sets[2])

    def test_mcar_and_copy_exact_for_all_seeds(self) -> None:
        for seed in (0, 1, 2):
            generated = list(
                iter_stratified_seed_masks(
                    self.dataset.X,
                    self.dataset.feature_types,
                    self.dataset.languages,
                    seed=seed,
                    quotas=self.quotas,
                    regimes=("mcar", "resource_copy"),
                )
            )
            self.assertEqual([item.regime for item in generated], ["mcar", "resource_copy"])
            for masks in generated:
                self._assert_exact_quotas(masks)
                self._assert_no_leakage(masks)

            copy_masks = generated[1]
            event_targets: dict[int, int] = {}
            event_sizes: dict[int, int] = {}
            donor_cells: set[tuple[int, int]] = set()
            for case in copy_masks.copy_provenance:
                event_targets.setdefault(case.event_id, case.target_index)
                self.assertEqual(event_targets[case.event_id], case.target_index)
                event_sizes[case.event_id] = event_sizes.get(case.event_id, 0) + 1
                self.assertEqual(case.seed, seed)
                self.assertEqual(case.donor_group, case.target_post_mask_group)
                self.assertAlmostEqual(
                    case.similarity,
                    case.intersection_count / case.donor_observed_count,
                )
                donor_cell = (case.donor_index, case.scored_feature_index)
                self.assertNotIn(donor_cell, donor_cells)
                donor_cells.add(donor_cell)
            self.assertEqual(len(event_targets), len(set(event_targets.values())))
            self.assertTrue(any(size > 1 for size in event_sizes.values()))
            self.assertGreater(int(copy_masks.unscored_removed_mask.sum()), 0)

    def test_fewshot_adaptation_sets_are_nested(self) -> None:
        generated = list(
            iter_stratified_seed_masks(
                self.dataset.X,
                self.dataset.feature_types,
                self.dataset.languages,
                seed=0,
                quotas=self.quotas,
                regimes=("local_fewshot",),
            )
        )
        self.assertTrue(
            all(masks.regime.startswith("local_fewshot_") for masks in generated)
        )
        selected = {
            masks.adaptation_budget: masks
            for masks in generated
            if masks.resource_group == "P1" and masks.target_feature_type == "S"
        }
        self.assertEqual(tuple(sorted(selected)), (0, 2, 4, 8, 16, 32, 64, 128))
        previous = np.zeros_like(next(iter(selected.values())).adaptation_mask)
        for budget in sorted(selected):
            masks = selected[budget]
            self.assertEqual(int(masks.adaptation_mask.sum()), budget)
            self.assertTrue(np.all(previous <= masks.adaptation_mask))
            self.assertFalse(
                np.any(
                    masks.adaptation_mask
                    & (masks.val_mask | masks.cal_mask | masks.test_mask)
                )
            )
            previous = masks.adaptation_mask


if __name__ == "__main__":
    unittest.main()
