"""R1 rule evidence: paired version audit, normalized metrics, raw reusable outputs.

Run from the revision root. Each seed is checkpointed separately; existing completed
checkpoints are skipped unless --overwrite is explicit. No outcome-based selection.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import platform
import sys
import time
import os
from pathlib import Path
import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from src.data.datasets import load_hillstrom, load_megafon, make_semisynthetic_ihdp
from src.experiments.run_cross_skeleton import load_skeleton_covariates, make_semisynthetic_from_skeleton
from src.models.scrl import SCRL
from src.models.cate_estimators import TLearner
from src.models.baselines import CausalTreeBaseline, CURLSBaseline
from src.metrics.fidelity import coverage_fidelity_details, rule_coverage_fidelity

OUT = Path("results/rules_r1")
_SOURCE_PATHS = [Path(__file__), Path("src/models/scrl.py"), Path("src/data/datasets.py"),
                 Path("src/metrics/fidelity.py"), Path("src/experiments/run_r1_rule_extensions.py")]
_SOURCE_SHA256 = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in _SOURCE_PATHS}

def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value

def write_json(path, obj):
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False)+"\n")

def environment():
    return {"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
            "scipy":scipy.__version__,"sklearn":sklearn.__version__,
            "bootstrap_backend":os.environ.get("SCRL_BOOTSTRAP_BACKEND","numpy"),
            "source_hash_capture":"module_import", "source_sha256":_SOURCE_SHA256}

def rule_keys(learner):
    return [r._canonical_key for r in learner.rules]

def fit_pair(model, train, kwargs, force_hard=False):
    old = SCRL(cate_model=model,support_mode="legacy",**kwargs).fit(train.X,train.T,train.Y)
    # Exact path certificate: at EVERY historical argmax, the winner satisfies the
    # hard filter. Removing lower-scoring invalid candidates cannot change this argmax;
    # scoring consumes identical random draws before filtering in the revised version.
    if old.legacy_hard_equivalent_ and not force_hard:
        new = copy.deepcopy(old)
        new.support_mode = "hard"
        new.algorithm_version_ = "scrl-r1-hard-support-v1"
        certificate = "every_legacy_argmax_valid_identical_rng_path"
    else:
        new = SCRL(cate_model=model,support_mode="hard",**kwargs).fit(train.X,train.T,train.Y)
        certificate = "explicit_paired_fit"
    for a in new.support_audit_:
        assert all(a["selection"][k] for k in ("support_ok","overlap_ok","max_coverage_ok"))
    same = rule_keys(old)==rule_keys(new)
    if old.legacy_hard_equivalent_: assert same
    return old,new,{"same_conditions":same,"certificate":certificate,
                    "legacy_trace":old.selection_trace_,"legacy_trace_all_valid":old.legacy_hard_equivalent_}

def masks_of(learner, X):
    return np.asarray([r.covers(X) for r in learner.rules],dtype=bool).reshape(len(learner.rules),len(X))

def summarize_masks(pred, masks):
    masks = list(masks)
    out = coverage_fidelity_details(pred,masks)
    out["legacy_coverage_count_curve_mean"] = rule_coverage_fidelity(np.empty((len(pred),0)),pred,masks)
    return out

def outcome_difference(Y,T,mask):
    n1 = int(np.sum(mask & (T==1))); n0 = int(np.sum(mask & (T==0)))
    out = {"n":int(mask.sum()),"n_treated":n1,"n_control":n0,"coverage":float(mask.mean()),
           "difference_in_means":None,"conditional_se":None}
    if n1>=2 and n0>=2:
        a,b=Y[mask & (T==1)],Y[mask & (T==0)]
        out.update(difference_in_means=float(a.mean()-b.mean()),
                   conditional_se=float(np.sqrt(a.var(ddof=1)/n1+b.var(ddof=1)/n0)))
    return out

def describe_rules(learner,train,test,pred_tr,pred_te):
    records=[]
    for i,r in enumerate(learner.rules):
        records.append({"index":i,"conditions":r.conditions,"tau_mean_selection":r.tau_mean,
            "tau_std_selection":r.tau_std,"bootstrap_mean_std_selection":-r.stability_score,
            "discovery":learner.support_audit_[i]["discovery"],
            "selection":learner.support_audit_[i]["selection"],
            "test":learner.empirical_support(r,test.X,test.T)})
    mods=set(getattr(train,"effect_modifier_indices",None) or [])
    feats={int(c[0]) for r in learner.rules for c in r.conditions}
    return {"algorithm_version":learner.algorithm_version_,"n_rules":len(records),"rules":records,
        "modifier_hits":len(feats & mods) if mods else None,"n_distinct_features":len(feats),
        "train_fidelity":summarize_masks(pred_tr,masks_of(learner,train.X)),
        "test_fidelity":summarize_masks(pred_te,masks_of(learner,test.X))}

def add_rct_outcomes(report, arrays, learner, pred, test, prefix):
    masks={"population":np.ones(test.n,bool),"top20":pred>=np.percentile(pred,80)}
    rm=masks_of(learner,test.X)
    for i,m in enumerate(rm):masks[f"rule{i+1}"]=m
    masks["union"]=np.any(rm,axis=0) if len(rm) else np.zeros(test.n,bool)
    report["outcomes"]={k:outcome_difference(test.Y,test.T,m) for k,m in masks.items()}
    pop=report["outcomes"]["population"]["difference_in_means"]
    for k,v in report["outcomes"].items():
        v["ratio_to_population"]=v["difference_in_means"]/pop if pop and v["difference_in_means"] is not None else None
        arrays[f"{prefix}_{k}_mask"]=masks[k]

def run_case(experiment,dataset,seed,scope,force_hard=False,skeleton=None):
    start=time.time()
    if experiment=="cross":
        train,test=make_semisynthetic_from_skeleton(skeleton,seed,False)
        kw=dict(n_rules=5,max_rule_length=2,min_support=.1,overlap_ratio=.1,n_candidates=10,n_bootstrap=5,seed=seed)
        model=TLearner()
    elif experiment=="rct":
        loader=load_hillstrom if dataset=="hillstrom" else load_megafon
        train,test=loader(seed=seed,preprocess_scope=scope)
        kw=dict(n_rules=5,max_rule_length=2,n_candidates=20,n_bootstrap=10,max_coverage=.4,sample_split=True,seed=seed)
        model=TLearner(base_learner=GradientBoostingRegressor(n_estimators=100,max_depth=3,random_state=seed))
    else:
        train,test=make_semisynthetic_ihdp(n_modifiers=5,signal_strength=2.,noise_scale=.3,seed=seed)
        kw=dict(n_rules=5,n_candidates=20,n_bootstrap=10,seed=seed)
        model=TLearner()
    assert np.isfinite(train.X).all() and np.isfinite(train.Y).all()
    model.fit(train.X,train.T,train.Y)
    ptr,pte=model.predict_cate(train.X),model.predict_cate(test.X)
    assert np.isfinite(ptr).all() and np.isfinite(pte).all()
    print(f"start {experiment}/{dataset}/{scope}/s{seed} n={train.n}/{test.n}",flush=True)
    old,new,pair=fit_pair(model,train,kw,force_hard)
    print(f"rules ready s{seed}: equal={pair['same_conditions']} old/new={len(old.rules)}/{len(new.rules)} seconds={time.time()-start:.1f}",flush=True)
    arrays={"X_train":train.X,"T_train":train.T,"Y_train":train.Y,"X_test":test.X,"T_test":test.T,"Y_test":test.Y,
            "tau_pred_train":ptr,"tau_pred_test":pte,"discovery_indices":new.discovery_indices_,"selection_indices":new.selection_indices_}
    if train.tau is not None:arrays.update(true_tau_train=train.tau,true_tau_test=test.tau)
    report={"experiment":experiment,"dataset":dataset,"seed":seed,"preprocessing":scope,"config":kw,
            "pair":pair,"environment":environment(),"versions":{},
            "causal_interpretation":"RCT subgroup difference" if dataset=="hillstrom" and experiment=="rct" else
                                    "synthetic competition observed differences; randomization not verified" if dataset=="megafon" else "semi-synthetic benchmark"}
    for label,s in [("legacy",old),("hard",new)]:
        report["versions"][label]=describe_rules(s,train,test,ptr,pte)
        arrays[f"{label}_train_masks"]=masks_of(s,train.X)
        arrays[f"{label}_test_masks"]=masks_of(s,test.X)
        if experiment=="rct":add_rct_outcomes(report["versions"][label],arrays,s,pte,test,label)
    if experiment in {"cross","ihdp"}:
        ct=CausalTreeBaseline(max_depth=3,min_samples_leaf=20,seed=seed).fit(train.X,train.T,train.Y)
        cttr,ctte=ct.predict_cate(train.X),ct.predict_cate(test.X)
        values=sorted(np.unique(cttr),reverse=True)
        all_masks=np.asarray([ctte==v for v in values])
        top_masks=all_masks[:5]
        arrays.update(ct_train_prediction=cttr,ct_test_prediction=ctte,ct_top5_masks=top_masks,ct_all_masks=all_masks)
        mods=set(train.effect_modifier_indices or [])
        report["causal_tree"]={"top5":summarize_masks(pte,top_masks),"all_leaves":summarize_masks(pte,all_masks),
                              "modifier_hits":len({int(f) for f in ct.tree.tree_.feature if f>=0}&mods)}
        if experiment=="ihdp":
            cu=CURLSBaseline(cate_model=model,n_rules=5,max_depth=3,min_samples_leaf=30,n_bootstrap=50,seed=seed).fit(train.X,train.T,train.Y)
            cm=cu.get_rule_masks(test.X)
            arrays["curls_test_masks"]=np.asarray(cm,bool).reshape(len(cm),test.n)
            report["curls"]={"test_fidelity":summarize_masks(pte,cm)}
    for split,ds in [("train",train),("test",test)]:
        meta=ds.metadata or {}
        arrays[f"{split}_row_ids"]=np.asarray(meta.get("row_indices",np.arange(ds.n)))
        if meta:
            report[f"{split}_preprocessing"]={k:v for k,v in meta.items() if k not in {"row_indices","fit_row_indices"}}
            arrays[f"{split}_preprocess_fit_row_ids"]=np.asarray(meta["fit_row_indices"])
    if scope=="train_only" and experiment=="rct":
        assert not set(arrays["train_preprocess_fit_row_ids"])&set(arrays["test_row_ids"])
    report["elapsed_seconds"]=time.time()-start
    name=f"{experiment}_{dataset}_{scope}_seed{seed}"
    np.savez_compressed(OUT/f"{name}.npz",**arrays)
    joblib.dump(model,OUT/f"{name}_teacher.joblib")
    write_json(OUT/f"{name}.json",report)
    print(f"complete {name} seconds={report['elapsed_seconds']:.1f}",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--experiment",choices=["cross","rct","ihdp"],required=True)
    p.add_argument("--dataset",choices=["hillstrom","criteo","megafon","ihdp"],required=True)
    p.add_argument("--seeds",type=int,nargs="+",required=True)
    p.add_argument("--scopes",nargs="+",default=["legacy_full"])
    p.add_argument("--force-hard",action="store_true")
    p.add_argument("--overwrite",action="store_true")
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    skeleton=load_skeleton_covariates(args.dataset) if args.experiment=="cross" else None
    for seed in args.seeds:
        for scope in args.scopes:
            name=f"{args.experiment}_{args.dataset}_{scope}_seed{seed}"
            if (OUT/f"{name}.json").exists() and not args.overwrite:
                print("skip completed",name,flush=True);continue
            run_case(args.experiment,args.dataset,seed,scope,args.force_hard,skeleton)

if __name__=="__main__":main()
