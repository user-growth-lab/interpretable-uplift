# Interpretable Uplift Modelling — companion code

Code, experimental protocols and numerical summaries accompanying:

**Interpretable Uplift Modelling: Conditional Average Treatment Effect Attribution,
Subgroup Rules and Constrained Recourse**  
Yu Zhang, Yueguang Shan and Wanlong Ma  
*Expert Systems* · Article ID **EXSY70471** · DOI [10.1111/exsy.70471](https://doi.org/10.1111/exsy.70471)

This package is prepared for the promised public release of the accepted article.
It is based on the code supplement submitted to Wiley on 13 October 2026.
The journal production title uses British spelling; the submitted title used
“Modeling” and a comma after “Rules”. These identify the same article.

The three components explain a fitted treatment-effect model: **TESV** attributes
its predictions to features, **SCRL** describes subgroups using rules, and
**constrained recourse** searches for alternative feature states with greater
predicted uplift. Their outputs describe the fitted model; they do not establish
that changing a feature causes the predicted improvement.

## Start here

Use Python **3.9.6** to match the recorded R1 runtime. Create an isolated environment
and install the pinned numerical requirements from this directory:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-r1.txt
python scripts/smoke_test.py
```

The smoke check uses small in-memory examples, needs no dataset download, and
checks attribution, rule support and recourse constraints. It is an installation
check, not a reproduction of the paper's experiments. For the scientific runs,
read [DATA_AND_PROTOCOL.md](DATA_AND_PROTOCOL.md), obtain the datasets below, and
follow the fresh-reproduction steps. Those runs require substantially more time
and memory than the smoke check.

`requirements-r1.txt` is the install target for the revised experiments.
`requirements.txt` preserves the broader original dependency list for historical
reference; it is not an additional install step. `requirements-artifacts.txt` is
optional and concerns manuscript/PDF utilities, not the three core methods.

## Contents and limits

- `src/`: current implementations, data generators, experiment entry points,
  aggregation code and numerical validators.
- `configs/`: preserved historical configuration template. Current runners do
  not load `base.yaml`; use their CLI arguments and implemented defaults.
  Checkpoint JSONs record effective parameters and seeds.
- `legacy/`: preserved historical executable implementation. Run its entry
  points with `legacy/` as the working directory when studying historical results.
- `historical_results/`: preserved numerical summaries of historical experiments.
- `results/`: JSON outputs, configurations, type schemas, seed-level summaries
  and current aggregate results for comparison with the reported numbers.
- `audit/`: selected computational protocols, environment and numerical
  validation records, plus execution snapshots needed to trace source versions.
- `scripts/`: recourse, SHAP-equivalence and scientific-figure entry points.

This archive contains no third-party raw datasets, sample-level recourse vectors,
NPZ evaluation arrays, cached teachers, pickles or joblib model files. The included JSONs support inspection of
reported configurations and summaries. Saved-array validators and figure scripts
cannot run directly from those JSONs: first regenerate their arrays and teachers
from the original datasets using the commands below. The saved validation reports
describe checks on the authors' complete computational artifacts, not checks that
can be repeated on this lightweight archive alone. Some historical summaries lack
retained point-level outputs; re-running code is not recovery of missing historical
execution evidence.

## Environment and data

The recorded numerical runtime is Python 3.9.6. `requirements-r1.txt` gives its
package versions; `audit/runtime_environment.json` records the R1 environment,
not historical hardware. `requirements.txt` preserves the broader original
requirements. `requirements-artifacts.txt` lists optional artifact utilities.
Install dependencies in a separate Python environment before running experiments.

Provide these paths under the extracted archive root:

```text
data/ihdp/ihdp_npci_1-100.train.npz
data/ihdp/ihdp_npci_1-100.test.npz
data/hillstrom/hillstrom.csv
data/criteo/criteo-uplift-v2.1.csv.gz
```

Use the IHDP NPCI 1–100 archive linked by the authors' CFR repository
(https://github.com/clinicalml/cfrnet#examples), not the separate 1,000-replication
release. Hillstrom is documented by its original author
(https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html).
Criteo requires the corrected v2.1 release
(https://ailab.criteo.com/criteo-uplift-prediction-dataset/).
MegaFon is loaded through `sklift.datasets.fetch_megafon()` and its normal cache;
its provider describes synthetic competition data
(https://ods.ai/tracks/df21-megafon/competitions/megafon-df21-comp).
Raw row order and release versions matter for stored row identifiers.

`DATA_AND_PROTOCOL.md` explains generation, preprocessing and interpretation.
The exact implemented DGPs are in `src/data/datasets.py`,
`src/experiments/run_cross_skeleton.py` and
`src/experiments/run_nonlinear_modifiers.py`.

## Fresh reproduction

Make a separate working copy of the extracted directory so the distribution and
its manifest remain intact. Run the following commands from that working copy's
root, once before the first scientific run:

```bash
mv results reference_results
mkdir results
```

**Keep the included summaries in `reference_results/` for comparison and start
with the new, empty `results/` directory.** Runners skip existing checkpoint JSONs; leaving the
lightweight JSONs in place could falsely suggest that missing arrays/teachers
were regenerated. Keep `historical_results/` unchanged. Runs can be expensive;
retain the prescribed sample sizes and seed lists. These packaging checks did
not rerun scientific experiments.

The default NumPy bootstrap backend is sufficient. The optional Numba backend
can be checked with `SCRL_BOOTSTRAP_BACKEND=numba python3 -m
src.experiments.verify_bootstrap_acceleration`; its required implementation
snapshots are included. Exact-path legacy-rule reuse is allowed only when the
recorded support/RNG certificate holds; `--force-hard` requests an explicit fit.

1. Generate E2 recourse artifacts and E4 equivalence results:

```bash
python3 scripts/run_recourse_diagnostic.py --dataset all --seeds 42 43 44 45 46 --n-test 50
python3 scripts/verify_recourse_artifacts.py
python3 scripts/check_shap_equivalence.py
```

2. Generate revised rule and outcome checkpoints:

```bash
python3 -m src.experiments.run_r1_rules --experiment cross --dataset hillstrom --seeds 0 1 2 3 4 --scopes legacy_full
python3 -m src.experiments.run_r1_rules --experiment cross --dataset criteo --seeds 0 1 2 3 4 --scopes legacy_full
python3 -m src.experiments.run_r1_rules --experiment ihdp --dataset ihdp --seeds 42 43 44 45 46 47 48 49 50 51 --scopes legacy_full
python3 -m src.experiments.run_r1_rule_extensions --family ablation --seeds 42 43 44 45 46
python3 -m src.experiments.run_r1_rule_extensions --family nonlinear --seeds 42 43 44
python3 -m src.experiments.run_r1_rule_extensions --family stability --dataset ihdp
python3 -m src.experiments.run_r1_rule_extensions --family stability --dataset hillstrom
python3 -m src.experiments.run_r1_rule_extensions --family stability --dataset criteo
python3 -m src.experiments.run_r1_rules --experiment rct --dataset hillstrom --seeds 0 1 2 3 4 --scopes legacy_full train_only
python3 -m src.experiments.run_r1_rules --experiment rct --dataset megafon --seeds 0 1 2 3 4 5 6 7 8 9 --scopes train_only
```

3. Rebuild summaries and validate the newly generated arrays:

```bash
python3 -m src.experiments.rebuild_r1_ihdp_table
python3 -m src.experiments.rebuild_r1_targeting
python3 -m src.experiments.summarize_r1_rules
python3 -m src.experiments.summarize_r1_e3
python3 -m src.experiments.summarize_r1_extensions
python3 -m src.experiments.validate_r1_rules --require-complete
```

The strict rule validator expects 62 ordinary checkpoints: 40 main/E1/E3,
12 nonlinear and 10 non-full ablation. Full ablation reuses five main IHDP runs.
Membership summaries separately check six teacher refits on each of three datasets.
Recourse checks require 3,000 regenerated point pairs across ten dataset–seed cases.
A summary script exiting successfully does not establish that every seed is present:
check its `complete` field and run the strict validator above. The IHDP table rebuild
refits the bootstrap-pruned comparator to restore training masks and checks its test
masks against saved values; the targeting rebuild uses saved predictions without
refitting a model.

After regenerating the prerequisites, create the figure output directory and
rebuild the scientific figures:

```bash
mkdir -p paper/figures
python scripts/build_revision_figures.py
python scripts/build_rule_illustration.py
python scripts/build_recourse_plot.py
```

Outputs go to `paper/figures/`. The plotting scripts require the regenerated
arrays/teachers as well as the JSON summaries; the distributed summaries alone
are insufficient.

## Source and metadata provenance

The original algorithm implementations, experiment scripts and aggregate numerical
results are preserved from the submitted R1 archive. The 60 recourse method JSONs
omit their sample-level `rows` for public distribution; their other fields and
summary values are retained. Obtain the original datasets and rerun the recourse
commands to regenerate those records before validation. Public-facing documentation, citation
metadata, ignore rules and a small installation smoke check are added separately;
see [RELEASE_NOTES.md](RELEASE_NOTES.md) for the exact packaging changes.
No scientific experiment was rerun for this packaging task.

Recorded execution hashes still refer to the actual code used during computation;
they have not been rewritten to imply a fresh run. Personal source-path metadata
in JSON records was already made relative in the submitted code supplement.
`audit/submitted_package_manifest.json` preserves that archive's original file
manifest. `PACKAGE_MANIFEST.json` records this distribution's file sizes, SHA256
hashes and differences from the submitted archive. Current sources and scientific
execution snapshots can have different documentation while retaining the same
executable statements.

## Citation and licensing

Use the article DOI above when citing this work; `CITATION.cff` provides structured
citation metadata. The package omits the copyrighted manuscript, publisher proofs,
editorial correspondence and third-party raw datasets.

No software license is specified in this prepared package. The authors can add
their selected software license when publishing the repository. Public access to
source code by itself is not an additional license grant. Third-party dependencies
and datasets retain their respective licenses and terms.
