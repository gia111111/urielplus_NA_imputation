# URIEL+ imputation split methodology

## Canonical pipeline and ownership

The experiment has one path:

1. `data.py` loads natural values and applies the alternating 5% language / 5%
   feature coverage cutoff until both axes are stable.
2. `splits.py` freezes the retained population, derives P1/P2, and validates
   exact validation/calibration/test quotas.
3. `masking.py` implements stratified MCAR and resource-conditioned copy
   masking.
4. `split_io.py` writes and reloads experiment artifacts, manifests, metadata,
   and copy provenance.
5. `scripts/make_splits.py` is the only split-generation entry point.

Coverage filtering is applied only to natural data. It is never recomputed
after an artificial mask.

## Resource boundary

Retained languages are stably sorted by their natural observed-cell count.
Source-row order breaks ties. The first `n // 2` languages form P1 and the
remainder form P2, so an odd-population extra language is assigned to P2.

For copy masking, the complete simulated donor-pattern hidden set is
`H_dt = observed[target] & ~observed[donor]`. The target is hypothetically
re-ranked after hiding all of `H_dt`; this *pattern group* must match the donor
group and defines the scoring stratum. Only quota-selected members of `H_dt`
are actually removed from training. The provenance records both the simulated
pattern group and the target's actual group after those selected removals. The
two groups can differ. The boundary count and both sides of a boundary tie are
recorded in run metadata.

## Copy-mask algorithm checklist

The authoritative copy rule permits multiple scored features from one
donor-target event. One target is still used at most once.

| Method step | Implementation |
| --- | --- |
| Construct naturally missing donor cells | `_copy_attempt` builds one shuffled queue for every donor-group/domain stratum. |
| Use donor cells without replacement | `_copy_attempt` pops queue entries and tracks every anchor or additional scored donor cell in `consumed_donor_cells`. |
| Keep evaluation languages split-disjoint | `_partition_rows` assigns every retained language to one validation, calibration, or test target pool before matching. |
| Require the target gold value to exist | `_copy_candidate` requires `observed[target, anchor_feature]`. |
| Simulate the complete donor pattern | `_copy_candidate` returns every member of `H_dt = observed[target] & ~observed[donor]`; this full set defines eligibility, similarity, and pattern-group matching. |
| Compute simulated pattern coverage | `_copy_candidate` computes the target/donor observed intersection count after hiding all of `H_dt`. |
| Require donor/pattern-group agreement | `_copy_candidate` filters candidates through the frozen `post_mask_is_p2` lookup using the full simulated pattern. |
| Prefer family, then macroarea, then other | `_copy_candidate` selects the best available matching tier first. |
| Prefer the target's original donor-matching group within that tier | `_copy_candidate` applies `target_group_preference` only after tier selection. |
| Maximize full-pattern similarity | `_copy_candidate` maximizes the intersection divided by the donor's natural observed count. |
| Break ties deterministically | `_copy_candidate` orders by geographic distance and then language identifier. |
| Score multiple eligible members of the hidden set | `_copy_attempt` selects needed hidden cells across domains for the event's donor group. |
| Use each target once | `_copy_attempt` maintains `used_targets` across all donor groups and domains. |
| Remove exactly the scored cells | Only quota-selected members of `H_dt` enter `regime_removed_mask` and `train_removed_mask`; every other member stays visible in training. Consequently, every artificial removal is scored and MCAR/copy have the same 16,000-cell budget under the default quotas. |
| Never copy donor values | Mask construction receives only the natural observation mask and metadata, never donor feature values. |
| Make every scored cell auditable | `CopyMaskCase` and `copy_provenance.csv` record donor, target, anchor and scored features, matching tier, coverage counts, distance, split, and seed. |

## Relation to CACTI

The implementation uses a donor's complete observation pattern to construct
`H_dt`, choose a compatible target, and determine the copy scoring stratum. It
does not copy donor feature values. Under the revised equal-budget URIEL+
design, it also does not remove non-selected members of `H_dt`: they remain
ordinary training observations.

URIEL+ adds exact P1/P2-domain quotas, a frozen post-cutoff population,
family/macroarea matching, deterministic post-mask resource matching,
held-out gold labels, language-level evaluation isolation, one-use targets,
and per-cell provenance. These additions are necessary for the benchmark's
stratified and independently auditable evaluation design.
