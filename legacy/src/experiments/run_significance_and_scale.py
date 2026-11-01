"""Round 1 Revision Experiments:
1. Statistical significance tests (P0-3) from existing 5-seed data
2. Scaled Criteo experiment at 200K (P0-4)
3. Knowledge component ablation across datasets (P1-1)

Outputs results to results/paper_tables/
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.datasets import make_semisynthetic_ihdp, load_hillstrom, load_criteo, CATEDataset
from src.models.cate_estimators import TLearner
from src.models.tesv import TESVExplainer
from src.models.baselines import (
    SHAPCATEBaseline, InterventionalSHAPBaseline, CausalSHAPBaseline,
    CausalTreeBaseline, CURLSBaseline,
)
from src.models.scrl import SCRL
from src.models.counterfactual import CausalCounterfactualExplainer
from src.metrics.fidelity import (
    top_k_effect_modifier_recall, causal_fidelity_score,
    attribution_stability, rule_coverage_fidelity,
)

OUT_DIR = "results/paper_tables"
N_SEEDS = 5
N_PERM = 20
N_BG = 50


def compute_significance_from_existing_data():
    """P0-3: Compute paired t-test and Wilcoxon p-values from existing 5-seed results."""
    print("\n" + "="*60)
    print("P0-3: STATISTICAL SIGNIFICANCE TESTS")
    print("="*60)
    
    # Load existing raw data
    with open(os.path.join(OUT_DIR, "tables_ihdp.json")) as f:
        ihdp_data = json.load(f)
    
    results = {}
    
    # Table 1: Attribution comparisons (IHDP)
    print("\n--- Table 1: Attribution (IHDP) ---")
    for scenario in ["A", "B"]:
        results[f"table1_{scenario}"] = {}
        # Compare TESV vs each baseline
        tesv_fid = ihdp_data["table1"][scenario]["tesv"]["fid"]
        for baseline in ["cond", "causal"]:
            bl_fid = ihdp_data["table1"][scenario][baseline]["fid"]
            if len(tesv_fid) >= 3 and len(bl_fid) >= 3:
                t_stat, t_pval = stats.ttest_rel(tesv_fid, bl_fid)
                w_stat, w_pval = stats.wilcoxon(
                    [a - b for a, b in zip(tesv_fid, bl_fid)],
                    alternative='two-sided'
                )
                results[f"table1_{scenario}"][f"tesv_vs_{baseline}_fidelity"] = {
                    "t_stat": float(t_stat), "t_pval": float(t_pval),
                    "w_stat": float(w_stat), "w_pval": float(w_pval),
                    "tesv_mean": float(np.mean(tesv_fid)),
                    "bl_mean": float(np.mean(bl_fid)),
                    "significant_005": t_pval < 0.05,
                    "significant_001": t_pval < 0.01,
                }
                sig = "**" if t_pval < 0.01 else ("*" if t_pval < 0.05 else "n.s.")
                print(f"  {scenario} TESV vs {baseline} (fid): t={t_stat:.3f}, p={t_pval:.4f} {sig}")
        
        # Spurious attribution (Scenario B only)
        if scenario == "B":
            tesv_spur = ihdp_data["table1"]["B"]["tesv"]["spur"]
            for baseline in ["cond", "causal"]:
                bl_spur = ihdp_data["table1"]["B"][baseline]["spur"]
                if len(tesv_spur) >= 3 and len(bl_spur) >= 3:
                    t_stat, t_pval = stats.ttest_rel(tesv_spur, bl_spur)
                    sig = "**" if t_pval < 0.01 else ("*" if t_pval < 0.05 else "n.s.")
                    results[f"table1_{scenario}"][f"tesv_vs_{baseline}_spurious"] = {
                        "t_stat": float(t_stat), "t_pval": float(t_pval),
                        "significant_005": t_pval < 0.05,
                    }
                    print(f"  B TESV vs {baseline} (spur): t={t_stat:.3f}, p={t_pval:.4f} {sig}")
    
    # Table 2: Rule discovery (IHDP)
    print("\n--- Table 2: Rule Discovery (IHDP) ---")
    scrl_fid = ihdp_data["table2"]["scrl"]["fid"]
    for baseline in ["ctree", "curls"]:
        bl_fid = ihdp_data["table2"][baseline]["fid"]
        if len(scrl_fid) >= 3 and len(bl_fid) >= 3:
            t_stat, t_pval = stats.ttest_rel(scrl_fid, bl_fid)
            try:
                w_stat, w_pval = stats.wilcoxon(
                    [a - b for a, b in zip(scrl_fid, bl_fid)],
                    alternative='two-sided'
                )
            except ValueError:
                w_stat, w_pval = 0, 1.0
            results[f"table2_scrl_vs_{baseline}"] = {
                "t_stat": float(t_stat), "t_pval": float(t_pval),
                "w_pval": float(w_pval),
                "scrl_mean": float(np.mean(scrl_fid)),
                "bl_mean": float(np.mean(bl_fid)),
                "significant_005": t_pval < 0.05,
            }
            sig = "**" if t_pval < 0.01 else ("*" if t_pval < 0.05 else "n.s.")
            print(f"  SCRL vs {baseline}: t={t_stat:.3f}, p={t_pval:.4f} {sig}")
            print(f"    SCRL mean={np.mean(scrl_fid):.3f}, {baseline} mean={np.mean(bl_fid):.3f}")
    
    return results


def run_scaled_criteo(sample_sizes=[50000, 100000, 200000], n_test=500):
    """P0-4: Run Criteo at larger scale."""
    print("\n" + "="*60)
    print("P0-4: SCALED CRITEO EXPERIMENTS")
    print("="*60)
    
    results = {}
    seeds = list(range(42, 42 + 3))  # 3 seeds for larger experiments (time constraint)
    
    for n_sample in sample_sizes:
        print(f"\n--- Criteo n={n_sample}, n_test={n_test} ---")
        results[str(n_sample)] = {
            "tesv_pfid": [], "tesv_stab": [],
            "scrl_fid": [], "fit_time": [], "tesv_time": [], "scrl_time": []
        }
        
        for seed in seeds:
            try:
                t0 = time.time()
                train, test = load_criteo(sample_n=n_sample, seed=seed)
                load_time = time.time() - t0
                print(f"  Seed {seed}: loaded in {load_time:.1f}s (train={train.n}, test={test.n})")
                
                # Fit CATE model
                t0 = time.time()
                model = TLearner()
                model.fit(train.X, train.T, train.Y)
                fit_time = time.time() - t0
                results[str(n_sample)]["fit_time"].append(fit_time)
                
                tau_test = model.predict_cate(test.X)
                n_exp = min(n_test, test.n)
                
                # TESV attribution
                from scipy.stats import spearmanr
                if hasattr(model, 'model_t'):
                    imp_t = model.model_t.feature_importances_
                    imp_c = model.model_c.feature_importances_
                    model_imp = imp_t + imp_c
                else:
                    model_imp = np.ones(train.d)
                
                tesv_exp = TESVExplainer(
                    cate_model=model, background_data=train.X,
                    n_background=N_BG, n_permutations=15, seed=seed
                )
                t0 = time.time()
                attr = tesv_exp.explain(test.X[:n_exp], show_progress=False)
                tesv_time = time.time() - t0
                results[str(n_sample)]["tesv_time"].append(tesv_time)
                
                glob_attr = np.abs(attr).mean(axis=0)
                rho, _ = spearmanr(glob_attr, model_imp)
                results[str(n_sample)]["tesv_pfid"].append(float(rho))
                
                # Stability (on subset for speed)
                stab = attribution_stability(
                    test.X[:min(20, n_exp)], tesv_exp,
                    noise_std=0.1, n_perturbations=3, seed=seed
                )
                results[str(n_sample)]["tesv_stab"].append(float(stab))
                
                # SCRL
                t0 = time.time()
                scrl = SCRL(
                    cate_model=model, n_rules=5, max_rule_length=2,
                    min_support=0.05, overlap_ratio=0.1,
                    n_candidates=10, n_bootstrap=5, seed=seed
                )
                scrl.fit(train.X, train.T, train.Y)
                scrl_time = time.time() - t0
                results[str(n_sample)]["scrl_time"].append(scrl_time)
                
                masks = [r.covers(test.X) for r in scrl.rules]
                fid = rule_coverage_fidelity(test.X, tau_test, masks) if masks else 0.0
                results[str(n_sample)]["scrl_fid"].append(float(fid))
                
                print(f"    TESV pfid={rho:.3f}, stab={stab:.3f}, "
                      f"SCRL fid={fid:.4f}, "
                      f"times: fit={fit_time:.1f}s tesv={tesv_time:.1f}s scrl={scrl_time:.1f}s")
                
            except Exception as e:
                print(f"    ERROR seed {seed}: {e}")
                continue
    
    # Summary
    print("\n--- Criteo Scale Summary ---")
    for n_str, data in results.items():
        if data["tesv_pfid"]:
            print(f"  n={n_str}: TESV pfid={np.mean(data['tesv_pfid']):.3f}±{np.std(data['tesv_pfid']):.3f}, "
                  f"stab={np.mean(data['tesv_stab']):.3f}, "
                  f"SCRL fid={np.mean(data['scrl_fid']):.4f}")
    
    return results


def run_knowledge_ablation_crossdataset():
    """P1-1: Knowledge component ablation on Hillstrom and Criteo."""
    print("\n" + "="*60)
    print("P1-1: KNOWLEDGE COMPONENT ABLATION (CROSS-DATASET)")
    print("="*60)
    
    results = {}
    seeds = list(range(42, 42 + N_SEEDS))
    
    for ds_name, loader_fn, loader_kw in [
        ("hillstrom", load_hillstrom, {}),
        ("criteo_50k", load_criteo, {"sample_n": 50000}),
    ]:
        print(f"\n--- {ds_name} ---")
        ds_results = {"full": {"targeting": [], "coverage": []},
                      "no_rules": {"targeting": [], "coverage": []},
                      "no_cf": {"targeting": [], "coverage": []},
                      "attr_only": {"targeting": [], "coverage": []}}
        
        for seed in seeds:
            try:
                train, test = loader_fn(seed=seed, **loader_kw)
                model = TLearner()
                model.fit(train.X, train.T, train.Y)
                tau_test = model.predict_cate(test.X)
                n_exp = min(50, test.n)
                
                # Full framework: SCRL targeting precision
                scrl = SCRL(cate_model=model, n_rules=5, max_rule_length=2,
                           min_support=0.05, overlap_ratio=0.1,
                           n_candidates=10, n_bootstrap=5, seed=seed)
                scrl.fit(train.X, train.T, train.Y)
                
                # Targeting: mean tau of rule-covered top-20%
                coverage_mask = np.zeros(test.n, dtype=bool)
                for r in scrl.rules:
                    coverage_mask |= r.covers(test.X)
                
                if coverage_mask.sum() > 0:
                    targeting_full = float(tau_test[coverage_mask].mean())
                else:
                    targeting_full = float(tau_test.mean())
                
                # TESV-based targeting (fallback for no-rules)
                tesv_exp = TESVExplainer(cate_model=model, background_data=train.X,
                                       n_background=N_BG, n_permutations=10, seed=seed)
                attr = tesv_exp.explain(test.X[:n_exp], show_progress=False)
                # Use sum of absolute attr as targeting score
                tesv_scores = np.abs(attr).sum(axis=1)
                top20_idx = np.argsort(-tesv_scores)[:max(1, int(0.2 * n_exp))]
                targeting_tesv = float(tau_test[:n_exp][top20_idx].mean())
                
                # CF actionable coverage
                cf_exp = CausalCounterfactualExplainer(
                    cate_model=model, actionable_features=list(range(train.d)),
                    max_changes=2, step_size=0.2, max_steps=10,
                    proximity_weight=0.05, seed=seed
                )
                cfs = cf_exp.explain(test.X[:n_exp])
                cf_coverage = float(np.mean([e.tau_gain > 0 for e in cfs]))
                
                # Rule actionable coverage
                rule_coverage = float(coverage_mask[:n_exp].mean()) if n_exp <= test.n else 0.0
                
                ds_results["full"]["targeting"].append(targeting_full)
                ds_results["full"]["coverage"].append(min(1.0, rule_coverage + cf_coverage))
                ds_results["no_rules"]["targeting"].append(targeting_tesv)
                ds_results["no_rules"]["coverage"].append(cf_coverage)
                ds_results["no_cf"]["targeting"].append(targeting_full)
                ds_results["no_cf"]["coverage"].append(rule_coverage)
                ds_results["attr_only"]["targeting"].append(targeting_tesv)
                ds_results["attr_only"]["coverage"].append(0.0)
                
                print(f"  Seed {seed}: full_tgt={targeting_full:.4f}, "
                      f"tesv_tgt={targeting_tesv:.4f}, cf_cov={cf_coverage:.2f}")
            except Exception as e:
                print(f"  ERROR seed {seed}: {e}")
                continue
        
        results[ds_name] = {}
        for variant, data in ds_results.items():
            if data["targeting"]:
                results[ds_name][variant] = {
                    "targeting_mean": float(np.mean(data["targeting"])),
                    "targeting_std": float(np.std(data["targeting"])),
                    "coverage_mean": float(np.mean(data["coverage"])),
                }
        
        # Print summary
        for variant in ["full", "no_rules", "no_cf", "attr_only"]:
            if variant in results[ds_name]:
                r = results[ds_name][variant]
                print(f"  {variant}: targeting={r['targeting_mean']:.4f}±{r['targeting_std']:.4f}, "
                      f"coverage={r['coverage_mean']:.2f}")
    
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}
    
    # P0-3: Statistical significance
    sig_results = compute_significance_from_existing_data()
    all_results["significance"] = sig_results
    
    # P0-4: Scaled Criteo (try 200K, fallback to smaller if memory issues)
    try:
        scale_results = run_scaled_criteo(sample_sizes=[50000, 200000], n_test=500)
        all_results["scaled_criteo"] = scale_results
    except Exception as e:
        print(f"\nWARNING: Scaled Criteo failed: {e}")
        print("Attempting smaller scale...")
        try:
            scale_results = run_scaled_criteo(sample_sizes=[50000, 100000], n_test=200)
            all_results["scaled_criteo"] = scale_results
        except Exception as e2:
            print(f"  Also failed: {e2}")
            all_results["scaled_criteo"] = {"error": str(e2)}
    
    # P1-1: Cross-dataset ablation
    try:
        ablation_results = run_knowledge_ablation_crossdataset()
        all_results["knowledge_ablation_crossdataset"] = ablation_results
    except Exception as e:
        print(f"\nWARNING: Cross-dataset ablation failed: {e}")
        all_results["knowledge_ablation_crossdataset"] = {"error": str(e)}
    
    # Save all results
    outpath = os.path.join(OUT_DIR, "round1_revision_results.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅ All Round 1 results saved to {outpath}")
    
    return all_results


if __name__ == "__main__":
    main()
