#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urielplus_impute.data import load_dataset
from urielplus_impute.experiment import DEFAULT_REGIMES
from urielplus_impute.masking import ADAPTATION_BUDGETS
from urielplus_impute.split_io import write_split, write_split_manifest
from urielplus_impute.splits import (
    StratumQuotas,
    iter_stratified_seed_masks,
    make_seed_summary,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create P1/P2 × feature-domain stratified URIEL+ masks with "
            "disjoint validation/calibration/test language pools."
        )
    )
    parser.add_argument("--typological", default="urielplus_analysis/typological_data.csv")
    parser.add_argument("--languages", default="urielplus_analysis/languages.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--validation-per-stratum", type=int, default=500)
    parser.add_argument("--calibration-per-stratum", type=int, default=500)
    parser.add_argument("--test-per-stratum", type=int, default=1_000)
    parser.add_argument(
        "--adaptation-budgets",
        nargs="+",
        type=int,
        default=list(ADAPTATION_BUDGETS),
    )
    parser.add_argument("--index-col", default=None)
    parser.add_argument(
        "--language-min-coverage",
        type=float,
        default=0.05,
        help="Inclusive natural-data language coverage cutoff (default: 0.05).",
    )
    parser.add_argument(
        "--feature-min-coverage",
        type=float,
        default=0.05,
        help="Inclusive natural-data feature coverage cutoff (default: 0.05).",
    )
    return parser.parse_args()


def _write_run_summary(
    outdir: Path,
    summaries: list,
    split_records: list,
    filter_summary,
    feature_types: pd.Series,
    languages: pd.DataFrame,
) -> tuple[Path, Path]:
    manifest_dir = outdir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    summary_json = manifest_dir / "split_generation_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "seeds": [summary.seed for summary in summaries],
                "quotas_per_stratum": summaries[0].quotas_per_stratum,
                "resource_summary": summaries[0].resource_summary,
                "coverage_filter": asdict(filter_summary),
                "domain_feature_counts": {
                    str(name): int(value)
                    for name, value in feature_types.value_counts().items()
                },
                "metadata_summary": {
                    "families": int(languages["family_id"].nunique()),
                    "macroareas": int(languages["macroarea"].nunique()),
                    "languages_with_metadata": int(
                        languages["metadata_available"].sum()
                    ),
                },
                "generated_split_artifacts": len(split_records),
                "split_artifacts": [asdict(record) for record in split_records],
                "generated_regimes_by_seed": {
                    str(summary.seed): summary.generated_regimes
                    for summary in summaries
                },
                "failures": [
                    failure
                    for summary in summaries
                    for failure in summary.failures
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    failures = [
        {
            "seed": failure["seed"],
            "regime": failure["regime"],
            "reason": failure["reason"],
            "capacities": json.dumps(failure["capacities"], sort_keys=True),
        }
        for summary in summaries
        for failure in summary.failures
    ]
    failure_csv = manifest_dir / "infeasible_regimes.csv"
    pd.DataFrame(
        failures,
        columns=["seed", "regime", "reason", "capacities"],
    ).to_csv(failure_csv, index=False)

    quota = summaries[0].quotas_per_stratum
    resource = summaries[0].resource_summary
    lines = [
        "# Stratified split-generation summary",
        "",
        (
            f"- Seeds: {', '.join(str(summary.seed) for summary in summaries)}"
        ),
        (
            "- Per resource-domain stratum: "
            f"{quota['validation']} validation, "
            f"{quota['calibration']} calibration, "
            f"{quota['test']} test."
        ),
        (
            f"- Dataset: {resource['P1']['languages']:,} P1, "
            f"{resource['P2']['languages']:,} P2 languages."
        ),
        (
            "- Frozen matrix: "
            f"{filter_summary.output_languages:,} languages × "
            f"{filter_summary.output_features:,} features; "
            f"{filter_summary.output_observed_cells:,} observed cells; "
            f"{100 * filter_summary.output_missingness:.6f}% missing."
        ),
        (
            "- Resource boundary: stable ascending natural coverage with "
            "source-row tie order; the odd-population extra language is in P2."
        ),
        f"- Written split artifacts: {len(split_records):,}.",
        f"- Infeasible regime-seed combinations: {len(failures):,}.",
    ]
    if failures:
        lines.extend(
            [
                "",
                "## Infeasible conditions",
                "",
            ]
        )
        for failure in failures:
            lines.append(
                f"- Seed {failure['seed']} / {failure['regime']}: "
                f"{failure['reason']}"
            )
    summary_md = manifest_dir / "split_generation_summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_json, summary_md


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    quotas = StratumQuotas(
        validation=args.validation_per_stratum,
        calibration=args.calibration_per_stratum,
        test=args.test_per_stratum,
    )
    dataset = load_dataset(
        args.typological,
        args.languages,
        index_col=args.index_col,
        language_min_coverage=args.language_min_coverage,
        feature_min_coverage=args.feature_min_coverage,
    )
    print(
        f"[data] matrix={dataset.X.shape}, "
        f"observed={int(dataset.X.notna().sum().sum())}"
    )
    print(f"[quota] per_stratum={quotas.as_dict()}")
    split_records = []
    summaries = []
    for seed in args.seeds:
        summary = make_seed_summary(
            dataset.X,
            dataset.feature_types,
            seed=seed,
            quotas=quotas,
        )
        summaries.append(summary)
        for masks in iter_stratified_seed_masks(
            dataset.X,
            dataset.feature_types,
            dataset.languages,
            seed=seed,
            quotas=quotas,
            adaptation_budgets=args.adaptation_budgets,
            regimes=args.regimes,
            summary=summary,
        ):
            print(
                f"[split] seed={seed} regime={masks.regime} "
                f"val={int(masks.val_mask.sum())} "
                f"cal={int(masks.cal_mask.sum())} "
                f"test={int(masks.test_mask.sum())} "
                f"train={int(masks.train_visible_mask.sum())}"
            )
            split_records.append(
                write_split(
                    outdir,
                    dataset.X,
                    dataset.feature_types,
                    masks,
                    quotas=quotas,
                )
            )
        for failure in summary.failures:
            print(
                f"[infeasible] seed={seed} regime={failure['regime']} "
                f"reason={failure['reason']}"
            )

    manifest_path = write_split_manifest(outdir, split_records)
    summary_json, summary_md = _write_run_summary(
        outdir,
        summaries,
        split_records,
        dataset.filter_summary,
        dataset.feature_types,
        dataset.languages,
    )
    print(f"[done] manifest={manifest_path}")
    print(f"[done] summary={summary_json}")
    print(f"[done] report={summary_md}")
    failures = [failure for summary in summaries for failure in summary.failures]
    if failures:
        raise SystemExit(
            f"Split generation failed for {len(failures)} required regime-seed "
            "combination(s); see manifests/infeasible_regimes.csv."
        )


if __name__ == "__main__":
    main()
