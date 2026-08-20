# URIEL+ imputation split methodology

## Canonical pipeline and ownership

The experiment has one path:

1. `data.py` loads natural values and applies the alternating 5% language / 5%
   feature coverage cutoff until both axes are stable.
2. `splits.py` freezes the retained population, derives P1/P2, and validates
   exact validation/calibration/test quotas.
3. `masking.py` implements stratified MCAR, resource-conditioned copy masking,
   and nested local/few-shot masking.
4. `split_io.py` writes and reloads experiment artifacts, manifests, metadata,
   and copy provenance.
5. `scripts/make_splits.py` is the only split-generation entry point.

Coverage filtering is applied only to natural data. It is never recomputed
after an artificial mask.

## Resource boundary

Retained languages are stably sorted by their natural observed-cell count.
Source-row order breaks ties. The first `n // 2` languages form P1 and the
remainder form P2, so an odd-population extra language is assigned to P2.

For copy masking, a hypothetical target is re-ranked against every other
language's frozen natural count using the same source-row tie rule. This exact
lookup defines its post-mask group. The boundary count and both sides of a
boundary tie are recorded in run metadata.

## Copy-mask algorithm checklist

The authoritative copy rule permits multiple scored features from one
donor-target event. One target is still used at most once.

| Method step | Implementation |
| --- | --- |
| Construct naturally missing donor cells | `_copy_attempt` builds one shuffled queue for every donor-group/domain stratum. |
| Use donor cells without replacement | `_copy_attempt` pops queue entries and tracks every anchor or additional scored donor cell in `consumed_donor_cells`. |
| Keep evaluation languages split-disjoint | `_partition_rows` assigns every retained language to one validation, calibration, or test target pool before matching. |
| Require the target gold value to exist | `_copy_candidate` requires `observed[target, anchor_feature]`. |
| Apply the complete donor pattern | `_copy_candidate` returns `observed[target] & ~observed[donor]`; `_copy_attempt` removes all of it. |
| Compute post-mask coverage | `_copy_candidate` computes the target/donor observed intersection count. |
| Require donor/post-mask group agreement | `_copy_candidate` filters candidates through the frozen `post_mask_is_p2` lookup. |
| Prefer family, then macroarea, then other | `_copy_candidate` selects the best available matching tier first. |
| Prefer the target's original donor-matching group within that tier | `_copy_candidate` applies `target_group_preference` only after tier selection. |
| Maximize full-pattern similarity | `_copy_candidate` maximizes the intersection divided by the donor's natural observed count. |
| Break ties deterministically | `_copy_candidate` orders by geographic distance and then language identifier. |
| Score multiple eligible members of the hidden set | `_copy_attempt` selects needed hidden cells across domains for the event's donor group. |
| Use each target once | `_copy_attempt` maintains `used_targets` across all donor groups and domains. |
| Separate gold from context removal | `SplitMasks` stores scored split masks independently of `regime_removed_mask` and `unscored_removed_mask`. |
| Never copy donor values | Mask construction receives only the natural observation mask and metadata, never donor feature values. |
| Make every scored cell auditable | `CopyMaskCase` and `copy_provenance.csv` record donor, target, anchor and scored features, matching tier, coverage counts, distance, split, and seed. |

## Relation to CACTI

The implementation adopts CACTI's conceptual separation between natural
missingness, an artificial copied observation pattern, and the model-visible
matrix. It also transfers a donor's complete pattern without transferring its
feature values.

URIEL+ adds exact P1/P2-domain quotas, a frozen post-cutoff population,
family/macroarea matching, deterministic post-mask resource matching,
held-out gold labels, language-level evaluation isolation, one-use targets,
and per-cell provenance. These additions are necessary for the benchmark's
stratified and independently auditable evaluation design.
