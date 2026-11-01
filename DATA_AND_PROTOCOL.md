# Data generation and interpretation

## Dataset roles

IHDP uses the supplied NPCI training/test split. The semi-synthetic construction
retains the supplied treatment and baseline outcome mean, selects effect-modifier
coordinates using the experiment seed and constructs effects from those
coordinates. The stored `tau` is a realized individual effect including added
individual-effect noise; it is not the noise-free conditional average treatment
effect. Scaling uses training rows. Main attribution/rules/ablation use signal
coefficient 2 and noise SD 0.3; recourse and membership use 1 and 0.5. E4 and
Figure 2 reuse E2 teachers and inherit the latter settings. Main IHDP repetitions
use replication 0's fixed outer split while varying DGP/internal randomness;
nonlinear runs use replications 0–2. Exact formulas, selected features and
interactions are implemented in the supplied generator and experiment modules.

Hillstrom pools the two email arms against no email and uses visit as outcome.
The loader excludes treatment and post-campaign visit/conversion/spend from
covariates. Historical settings fit covariate transforms before the split;
revised E3 first splits raw rows and fits preprocessing on training rows. These
are distinct protocols, not interchangeable results. Historical recourse uses
an 8,000-row training subsample and retains its historical preprocessing scope.

Criteo uses only released f-prefixed covariates; treatment/outcomes are separate.
Use the corrected v2.1 release. Anonymous feature names do not establish their
individual measurement times or actionable semantics. Cross-skeleton experiments
retain full-skeleton standardization as part of their synthetic DGP definition.

MegaFon is provider-generated synthetic competition data. Each E3 seed draws
100,000 source records and uses an 80/20 split with train-only preprocessing.
Observed treated-minus-control differences are descriptive benchmark contrasts;
this dataset is not presented as a second verified real-world randomized trial.

No complete third-party raw release is redistributed in this archive. Historical
entry points under `legacy/` need the same data filenames relative to that working
directory. Supply that data tree explicitly before historical reproduction.

## Algorithm and evaluation boundaries

Current SCRL defaults to hard empirical support filtering and may return fewer
than K rules. `support_mode="legacy"` retains the historical finite-penalty path.
The base score is absolute subgroup prediction deviation times square-root
coverage, plus 0.05 times within-rule prediction SD, minus fixed-rule bootstrap
mean SD. Sequential selection adds 0.3 times newly covered fraction and subtracts
0.25 times accumulated feature reuse. The teacher sees the entire outer training
set; discovery/selection splitting is not independent teacher estimation.

The historical API `CausalTreeBaseline` is a residual-product CART comparator;
`CURLSBaseline` is a bootstrap-pruned teacher-prediction tree, not the published
CURLS optimizer. `CausalSHAPBaseline` is a neighbor-replacement/heuristic-ordering
implementation, not graph-based causal identification. TESV computes direct
marginal-replacement Shapley attribution of the fitted CATE function.

Rule summaries use population SD (`ddof=0`); recourse summaries use sample SD of
five seed means (`ddof=1`). Empty/full rule-union coverage and near-zero normalized
fidelity denominators remain undefined. Test support failures stay in evaluation.
Repeated splits are not independent randomized trials. Model-referenced fidelity
and attribution equivalence do not establish outcome-based targeting improvement.

Recourse diagnostic success requires a changed, legal state and predicted gain
above 1e-9. No-op returns and initially invalid inputs remain in all-input
denominators. Fifth-neighbor support is a training-calibrated proximity diagnostic,
not evidence that a real intervention can reach the state. See the unchanged
`audit/protocol_recourse.md` for schema, budgets and tolerances; the runner records
that file's SHA256. `audit/protocol_rules.md` and
`audit/protocol_shap_equivalence.md` provide the other computation protocols.
These protocols were internally specified, not externally preregistered.

Historical numerical results remain under `historical_results/`. Some historical
row identities, point pairs and environment records are unavailable. Existing
summary files and source-level protocols do not fill those evidence gaps.
