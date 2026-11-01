# R1 rule protocol (frozen before computation, 2026-09-28)

## Algorithm versions

Historical `scrl-legacy-20260725` is reproduced using `support_mode="legacy"`.
Revised `scrl-r1-hard-support-v1` uses `support_mode="hard"`.
Both preserve the historical base score:

`abs(mean(tau_R)-mean(tau_all_train))*sqrt(p_R) + 0.05*std(tau_R) - std(bootstrap_mean(tau_R))`.

The sequential selection score additionally adds `0.3 * newly_covered_fraction`
and subtracts `0.25 * sum(previous_selection_count(feature))` when diversity is enabled.
The standard-deviation term is a **reward**, not the negative penalty in the submitted formula.
The historical finite overlap penalty is `-1e6`; the revised implementation explicitly rejects
invalid selection candidates and may return fewer than K rules. Bootstrap samples are consumed
for all historically scored candidates before filtering to isolate the change from RNG drift.

Discovery support: `n_R >= min_support * n_discovery` and
`min(n_treated,n_control) >= overlap_ratio*n_R`.
Selection support: the same inequalities on the selection set, plus
`n_R/n_selection <= max_coverage`. Test membership is never filtered to improve support.
These are empirical sample checks, not positivity, causal identification, population support,
or post-selection uncertainty certificates. Global tau mean uses all training observations.

## Data roles

The CATE teacher is fitted on the entire outer training set. SCRL then partitions this set
50/50 for candidate discovery and adaptive scoring/selection (no split when n<100).
The selection half is not independent of teacher training. No nuisance cross-fitting is used.
The outer test outcomes are reserved for RCT evaluation. Test-X preprocessing in historical
real-data loaders used full-sample statistics; new E3 runs explicitly request train-only fitting.
E1 retains full-skeleton standardization as part of the original synthetic DGP definition.

## Parameters and runs

| Family | Seeds | K/L/Q/bootstrap | min support / max coverage / treatment fraction |
|---|---|---|---|
| E1 Hillstrom/Criteo skeleton, n=5000 | 0..4 | 5/2/10/5 | .10/.40/.10 |
| E3 Hillstrom RCT, original full data | 0..4 | 5/2/20/10 | .05/.40/.10 |
| E3 MegaFon synthetic competition, n=100000 | 0..9 | 5/2/20/10 | .05/.40/.10 |
| IHDP main rule reconstruction | 42..51 | 5/3/20/10 | .05/.40/.10 |
| IHDP component ablation / knowledge | 42..46 | 5/3/20/10 | .05/.40/.10 |

E1: fit one teacher per seed; evaluate legacy and revised SCRL using the same inputs and
randomness; compare Causal Tree top-5 training-ranked leaves and record all-leaf degeneracy.
Record old coverage-count AUUC-like score separately from normalized coverage fidelity.
E3: pair legacy-full and train-only preprocessing with identical raw row splits;
record subgroup/outcome ATEs, population ATE, empirical support and coverage, model top-20%,
and model-referenced fidelity. Causal Tree is a pseudo-outcome partition, not an honest tree.
No tuning, seed replacement, or test-dependent filtering is allowed.

## Metric and reporting conventions

Normalized coverage fidelity is `(mean_tau_covered-mean_tau_all)/(mean_tau_top_ncovered-mean_tau_all)`.
Empty/full coverage, non-finite predictions and near-zero denominators are flagged; value is null.
Summaries report valid/degenerate seed counts and do not impute zero. Negative values remain.
Seed dispersion describes repeated algorithm/split variation, not independent randomized trials.
RCT standard errors are conditional on a fitted, selected subgroup. Each report preserves T/Y,
raw row indices, predictions, rules, discovery/selection indices, test masks, config and environment.

## Unverified historical assertions

No executable or saved provenance for claimed IHDP 5-fold lambda tuning was located.
Hillstrom lambda sensitivity monkey-patches an alternative negative-variance score; it is not
hyperparameter tuning of the historical default algorithm. No `r=0.73` recourse truth-correlation
calculation was located. These assertions must not be represented as verified historical evidence.

MegaFon provenance correction: the official competition describes synthetically generated data approximating a real case (https://ods.ai/tracks/df21-megafon/competitions/megafon-df21-comp). Outcome differences there are descriptive benchmark contrasts; random treatment assignment is not independently verified. Only Hillstrom is described here as a real RCT.

## Completed supporting evidence and precise sampling roles

IHDP main rules use the fixed official replication-0 partition (672 train / 75 test).
Seeds 42..51 change modifier choice, simulated individual-effect noise and SCRL's internal
split; they do not re-randomize the outer IHDP rows. The component and targeting comparison
use seeds 42..46 on this same replication-0 partition. The nonlinear extension uses
replications 0,1,2 with seeds 42,43,44 and fixed modifier identities. The saved `tau` in this
construction is the simulated individual effect including its individual noise term;
it is not the noise-free conditional expectation.

The three component variants share K=5,L=3,Q=20,B=10, minimum support .05, maximum coverage
.4, treatment fraction .1, and the same teacher/data per seed. `no_div_dedup` disables both
feature diversity and condition deduplication together; it does not isolate each separately.
`variance_based` replaces only the absolute mean-deviation magnitude by within-rule standard
deviation times sqrt(coverage), retaining +.05 standard deviation, bootstrap penalty,
coverage bonus and feature penalty. The full variant is reused from the matching main cache.

The rule-assisted targeting comparison now uses direct teacher top-k at the same selected
count, with a nominal 20% budget; it does not use attribution magnitude as a teacher-ranking
baseline or combine historical recourse-success definitions. All five matched runs select
15 of 75 test rows per method. Predicted ranks alone determine membership; the stored
simulated individual effect is used only for evaluation.
