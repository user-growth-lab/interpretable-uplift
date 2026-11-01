"""Necessary revised-SCRL ablation, non-additive and bootstrap-membership evidence.

Historical data constructions and seed schedules are retained to isolate rule filtering.
These are not newly claimed fully isolated real-data pipelines; E3 addresses that question.
"""
import argparse
import time
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from src.data.datasets import make_semisynthetic_ihdp,load_hillstrom,load_criteo
from src.models.cate_estimators import TLearner
from src.models.baselines import CausalTreeBaseline
from src.experiments.run_r1_rules import fit_pair,describe_rules,masks_of,write_json,environment
from src.experiments.run_nonlinear_modifiers import MODIFIERS,INTERACTIONS
from src.experiments.run_subgroup_stability import topk_mask,scrl_subgroup,mean_pairwise_jaccard

OUT=Path("results/rule_extensions_r1")

def single(family,label,seed,train,test,kwargs):
    name=f"{family}_{label}_seed{seed}"
    if (OUT/f"{name}.json").exists():return
    start=time.time();print("start",name,flush=True)
    model=TLearner().fit(train.X,train.T,train.Y)
    ptr,pte=model.predict_cate(train.X),model.predict_cate(test.X)
    old,new,pair=fit_pair(model,train,dict(kwargs,seed=seed))
    result={"family":family,"variant":label,"seed":seed,"config":kwargs,"pair":pair,"environment":environment(),
            "legacy":describe_rules(old,train,test,ptr,pte),"hard":describe_rules(new,train,test,ptr,pte),
            "elapsed_seconds":time.time()-start}
    arrays=dict(X_train=train.X,T_train=train.T,Y_train=train.Y,X_test=test.X,T_test=test.T,Y_test=test.Y,
                tau_pred_train=ptr,tau_pred_test=pte,true_tau_test=test.tau,
                discovery_indices=new.discovery_indices_,selection_indices=new.selection_indices_,
                legacy_train_masks=masks_of(old,train.X),legacy_test_masks=masks_of(old,test.X),
                hard_train_masks=masks_of(new,train.X),hard_test_masks=masks_of(new,test.X))
    np.savez_compressed(OUT/f"{name}.npz",**arrays);write_json(OUT/f"{name}.json",result)
    print("complete",name,"same",pair["same_conditions"],"seconds",result["elapsed_seconds"],flush=True)

def ablation(seeds):
    for seed in seeds:
        tr,te=make_semisynthetic_ihdp(n_modifiers=5,signal_strength=2.,noise_scale=.3,seed=seed)
        for name,kw in [("no_div_dedup",dict(use_diversity=False,use_dedup=False)),("variance_based",dict(variance_scoring=True))]:
            single("ablation",name,seed,tr,te,dict(n_rules=5,n_candidates=20,n_bootstrap=10,**kw))

def nonlinear(seeds):
    for seed in seeds:
        for name,nonlin,inter in [("linear",False,None),("nonlinear",True,None),("interaction",False,INTERACTIONS),("nonlinear_interaction",True,INTERACTIONS)]:
            tr,te=make_semisynthetic_ihdp(replication=seed-42,effect_modifier_indices=MODIFIERS,nonlinear=nonlin,
                    interaction_terms=inter,signal_strength=2.,noise_scale=.3,seed=seed)
            single("nonlinear",name,seed,tr,te,dict(n_rules=5,max_rule_length=2,min_support=.05,max_coverage=.4,
                    overlap_ratio=.1,n_candidates=6,n_bootstrap=3,sample_split=True))

def stability(dataset):
    name=f"stability_{dataset}"
    if (OUT/f"{name}.json").exists():return
    if dataset=="ihdp":tr,_=make_semisynthetic_ihdp(replication=0,n_modifiers=5,seed=42)
    elif dataset=="hillstrom":tr,_=load_hillstrom(seed=42)
    else:tr,_=load_criteo(sample_n=6000,seed=42)
    if dataset!="ihdp" and tr.n>2500:
        idx=np.random.default_rng(42).choice(tr.n,2500,replace=False)
        tr.X,tr.T,tr.Y=tr.X[idx],tr.T[idx],tr.Y[idx]
    X,T,Y=tr.X,tr.T,tr.Y;n=len(X)
    teacher_ref=TLearner().fit(X,T,Y);tau_ref=teacher_ref.predict_cate(X)
    true_top=topk_mask(tr.tau,int(round(.3*n))) if tr.tau is not None else None
    rng=np.random.default_rng(42);arrays={"X_reference":X,"T_reference":T,"Y_reference":Y,"tau_reference":tau_ref}
    if tr.tau is not None:arrays["true_tau_reference"]=tr.tau
    rows=[];covers={v:{m:[] for m in ["SCRL","CDT","CausalTree"]} for v in ["legacy","hard"]}
    for b in range(6):
        start=time.time();print("start",name,b,flush=True)
        idx=rng.choice(n,n,replace=True)
        from src.data.datasets import CATEDataset
        boot=CATEDataset(X=X[idx],T=T[idx],Y=Y[idx],effect_modifier_indices=tr.effect_modifier_indices)
        teacher=TLearner().fit(boot.X,boot.T,boot.Y);tau=teacher.predict_cate(X)
        seed=int(rng.integers(1e9))
        old,new,pair=fit_pair(teacher,boot,dict(n_rules=5,max_rule_length=2,min_support=.05,max_coverage=.4,
                   overlap_ratio=.1,n_candidates=6,n_bootstrap=3,sample_split=True,seed=seed))
        cdt=DecisionTreeRegressor(random_state=int(rng.integers(1e9)),max_depth=3,min_samples_leaf=30)
        cdt.fit(boot.X,teacher.predict_cate(boot.X))
        ct=CausalTreeBaseline(seed=int(rng.integers(1e9)),max_depth=3,min_samples_leaf=30).fit(boot.X,boot.T,boot.Y)
        row={"bootstrap":b,"scrl_seed":seed,"pair":pair,"versions":{}}
        arrays[f"b{b}_bootstrap_indices"]=idx;arrays[f"b{b}_teacher_predictions"]=tau
        for version,s in [("legacy",old),("hard",new)]:
            m=scrl_subgroup(s,tau,X);k=int(round(m.mean()*n))
            method_masks={"SCRL":m,"CDT":topk_mask(cdt.predict(X),k),"CausalTree":topk_mask(ct.tree.predict(X),k)}
            if k==0:method_masks={m:np.zeros(n,bool) for m in method_masks}
            metrics={}
            for method,mask in method_masks.items():
                covers[version][method].append(mask);arrays[f"b{b}_{version}_{method}_mask"]=mask
                metrics[method]={"coverage":float(mask.mean()),"targeting_precision":float(true_top[mask].mean()) if mask.any() and true_top is not None else None,
                   "subgroup_uplift":float((tr.tau if tr.tau is not None else tau_ref)[mask].mean()) if mask.any() else None}
            row["versions"][version]={"metrics":metrics,"rules":[{"conditions":r.conditions,"support":a} for r,a in zip(s.rules,s.support_audit_)]}
        rows.append(row);print("complete",name,b,"seconds",time.time()-start,flush=True)
    summary={v:{m:{"jaccard_mean":mean_pairwise_jaccard(ms)[0],"jaccard_pair_std":mean_pairwise_jaccard(ms)[1]} for m,ms in methods.items()} for v,methods in covers.items()}
    np.savez_compressed(OUT/f"{name}.npz",**arrays)
    write_json(OUT/f"{name}.json",dict(family="stability",dataset=dataset,protocol="6 teacher-refit bootstraps, original training-reference population; legacy-full preprocessing",rows=rows,summary=summary,environment=environment()))

def main():
    p=argparse.ArgumentParser();p.add_argument("--family",choices=["ablation","nonlinear","stability"],required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[42,43,44,45,46]);p.add_argument("--dataset",choices=["ihdp","hillstrom","criteo"],default="ihdp")
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.family=="ablation":ablation(a.seeds)
    elif a.family=="nonlinear":nonlinear(a.seeds)
    else:stability(a.dataset)

if __name__=="__main__":main()
