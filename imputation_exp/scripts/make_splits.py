#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urielplus_impute.data import load_dataset
from urielplus_impute.experiment import DEFAULT_REGIMES
from urielplus_impute.split_io import write_split, write_split_manifest
from urielplus_impute.splits import build_equalized_seed_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed URIEL+ validation/test masks.")
    parser.add_argument("--typological", default="urielplus_analysis/typological_data.csv")
    parser.add_argument("--languages", default="urielplus_analysis/languages.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--min-cells-per-unit", type=int, default=3)
    parser.add_argument("--min-remaining-input", type=int, default=2)
    parser.add_argument("--n-target-languages", type=int, default=None)
    parser.add_argument("--index-col", default=None)
    parser.add_argument("--drop-empty-languages", action="store_true")
    parser.add_argument("--keep-special-languages", action="store_true", help="Keep special high-missingness groups instead of filtering them before masking.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(
        args.typological,
        args.languages,
        index_col=args.index_col,
        drop_empty_languages=args.drop_empty_languages,
        filter_special_families=not args.keep_special_languages,
    )
    n_observed = int(dataset.X.notna().sum().sum())
    missing_rate = float(dataset.X.isna().mean().mean())
    print(f"[data] matrix={dataset.X.shape}, observed={n_observed}, missing_rate={missing_rate:.4f}")
    print(f"[data] special high-missingness rows removed={dataset.n_special_filtered}")
    print(f"[data] total intentional mask fraction={args.val_frac + args.test_frac:.4f} "
          f"(val={args.val_frac:.4f}, test={args.test_frac:.4f})")
    split_records = []
    for seed in args.seeds:
        masks_by_regime, summary = build_equalized_seed_masks(
            dataset.X,
            dataset.feature_types,
            args.regimes,
            seed=seed,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            min_cells_per_unit=args.min_cells_per_unit,
            min_remaining_input=args.min_remaining_input,
            n_target_languages=args.n_target_languages,
        )
        print(f"[split] seed={seed} requested held-out budget={summary.requested_heldout_budget}")
        print(
            f"[split] seed={seed} shared held-out budget={summary.shared_heldout_budget} "
            f"(val={summary.n_val}, test={summary.n_test})"
        )
        for regime, capacity in summary.scoring_capacities.items():
            print(f"[split] capacity[{regime}]={capacity}")
        for regime in args.regimes:
            print(f"[split] regime={regime} seed={seed}")
            split_records.append(
                write_split(
                    args.outdir,
                    dataset.X,
                    dataset.feature_types,
                    masks_by_regime[regime],
                )
            )
    manifest_path = write_split_manifest(args.outdir, split_records)
    print(f"[done] wrote {len(split_records)} split records")
    print(f"[done] manifest={manifest_path}")


if __name__ == "__main__":
    main()
