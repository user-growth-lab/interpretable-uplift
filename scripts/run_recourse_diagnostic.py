"""E2 raw-pair audit; see audit/protocol_recourse.md."""
from __future__ import annotations
import argparse, hashlib, json, pickle, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.datasets import make_semisynthetic_ihdp, load_hillstrom
from src.models.cate_estimators import TLearner
from src.models.counterfactual import CausalCounterfactualExplainer
from src.models.baselines import GradientBasedUpliftCF, RandomSearchUplift
TOL=1e-7; GAIN_TOL=1e-9

def save(path,obj):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda v: v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
    temporary.replace(path)

class Domain:
    def __init__(self,label,X,names,scaler=None,categories=None):
        self.label=label;self.names=names;self.scaler=scaler;self.categories=categories
        self.bounds=np.stack([np.percentile(X,1,axis=0),np.percentile(X,99,axis=0)],axis=1)
        self.groups=[]
        if label=='IHDP':
            for j,name in enumerate(names):self.groups.append(dict(name=name,columns=[j],kind='continuous' if j<6 else 'categorical'))
        else:
            for name in ['recency','history','mens','womens','newbie']:
                self.groups.append(dict(name=name,columns=[names.index(name)],kind='continuous' if name=='history' else 'categorical'))
            for name in ['history_segment','zip_code','channel']:
                self.groups.append(dict(name=name,columns=[j for j,v in enumerate(names) if v.startswith(name+'_')],kind='categorical'))
        self.frozen=[j for g in self.groups if g['kind']=='categorical' for j in g['columns']]
        self.allowed=[j for j in range(X.shape[1]) if j not in self.frozen]
        self.levels={};self.scales={};self.fallbacks={}
        for gi,g in enumerate(self.groups):
            cols=g['columns']
            if g['kind']=='categorical':self.levels[gi]=np.unique(X[:,cols],axis=0)
            else:
                iqr=np.subtract(*np.percentile(X[:,cols],[75,25],axis=0));ranges=np.ptp(X[:,cols],axis=0)
                scale=np.where(iqr<=1e-12,np.where(ranges>1e-12,ranges,1),iqr);self.scales[gi]=scale
                if np.any(iqr<=1e-12):self.fallbacks[g['name']]=dict(iqr=iqr.tolist(),scale=scale.tolist())
    def checks(self,x,z):
        box=bool(np.all(z>=self.bounds[:,0]-TOL) and np.all(z<=self.bounds[:,1]+TOL))
        types=all(np.any(np.all(np.abs(levels-z[self.groups[gi]['columns']])<=TOL,axis=1)) for gi,levels in self.levels.items())
        frozen=bool(np.max(np.abs(z[self.frozen]-x[self.frozen]),initial=0)<=TOL)
        derived=True
        if self.label=='Hillstrom':
            raw=self.scaler.inverse_transform(z.reshape(1,-1))[0]
            g=next(g for g in self.groups if g['name']=='history_segment');vals=raw[g['columns']]
            if np.all(np.abs(vals)<=TOL):cat=0
            elif np.sum(np.abs(vals-1)<=TOL)==1 and np.all((np.abs(vals)<=TOL)|(np.abs(vals-1)<=TOL)):cat=int(np.argmax(vals))+1
            else:cat=-1
            amount=raw[self.names.index('history')];limits=[0.,100.,200.,350.,500.,750.,1000.,float('inf')]
            derived=cat>=0 and amount>=-TOL and amount<=limits[cat+1]+TOL and (cat==0 or amount>limits[cat]-TOL)
        return dict(box_valid=box,type_valid=bool(types),frozen_valid=frozen,derived_valid=bool(derived),legal=bool(box and types and frozen and derived))
    def validator(self,x,z):return self.checks(x,z)['legal']
    def changed(self,x,z):return [g['name'] for g in self.groups if np.any(np.abs(x[g['columns']]-z[g['columns']])>TOL)]
    def distances(self,query,reference,k=5):
        result=[]
        for start in range(0,len(query),20):
            q=query[start:start+20];dist=np.zeros((len(q),len(reference)))
            for gi,g in enumerate(self.groups):
                diff=np.abs(q[:,None,g['columns']]-reference[None,:,g['columns']])
                dist+=np.mean(diff/self.scales[gi],axis=2) if g['kind']=='continuous' else np.any(diff>TOL,axis=2)
            dist/=len(self.groups);result.extend(np.partition(dist,k-1,axis=1)[:,k-1].tolist())
        return np.array(result)
    def serializable(self):
        return dict(groups=self.groups,frozen_indices=self.frozen,allowed_target_indices=self.allowed,bounds=self.bounds.tolist(),
                    iqr_scales={str(k):v.tolist() for k,v in self.scales.items()},iqr_fallbacks=self.fallbacks,
                    categorical_levels={str(k):v.tolist() for k,v in self.levels.items()},
                    scaler_mean=None if self.scaler is None else self.scaler.mean_.tolist(),
                    scaler_scale=None if self.scaler is None else self.scaler.scale_.tolist(),categories=self.categories)

def data_domain(label,seed):
    if label=='IHDP':
        tr,te=make_semisynthetic_ihdp(replication=0,n_modifiers=5,seed=seed)
        return tr,te,np.arange(tr.n),np.arange(te.n),Domain(label,tr.X,tr.feature_names)
    tr,te=load_hillstrom(seed=seed)
    raw=pd.read_csv(ROOT/'data/hillstrom/hillstrom.csv');cov=raw.drop(columns=['segment','visit','conversion','spend'])
    cats=cov.select_dtypes(include=['object','category']).columns.tolist();nums=[c for c in cov.columns if c not in cats]
    enc=OneHotEncoder(sparse_output=False,drop='first');whole=np.hstack([cov[nums].values.astype(float),enc.fit_transform(cov[cats])])
    scaler=StandardScaler().fit(whole);T=(raw['segment']!='No E-Mail').values.astype(int)
    train_ids,test_ids=train_test_split(np.arange(len(raw)),test_size=1-.8,random_state=seed,stratify=T)
    np.testing.assert_allclose(tr.X,scaler.transform(whole)[train_ids],atol=1e-12,rtol=0)
    idx=np.random.default_rng(seed).choice(tr.n,8000,replace=False);tr.X,tr.T,tr.Y=tr.X[idx],tr.T[idx],tr.Y[idx]
    domain=Domain(label,tr.X,tr.feature_names,scaler,{c:[str(v) for v in a] for c,a in zip(cats,enc.categories_)})
    assert all(domain.checks(x,x)['type_valid'] and domain.checks(x,x)['derived_valid'] for x in tr.X)
    return tr,te,train_ids[idx],test_ids,domain

def summarize(rows):
    keys=['box_valid','type_valid','frozen_valid','derived_valid','legal','initial_box_valid','no_op','strict_positive','nonnegative','diagnostic_success','orig_exceeds','cf_exceeds','gain','changed_groups','l2','orig_fifth_nn','cf_fifth_nn','nn_change']
    out={'n':len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in keys}}
    valid=[r for r in rows if r['legal'] and not r['no_op']];out['changed_legal_n']=len(valid)
    out['support_given_changed_legal']=float(np.mean([not r['cf_exceeds'] for r in valid])) if valid else None
    return out

def run_one(label,seed,n_test=50):
    directory=ROOT/'results'/'recourse'/f'{label.lower()}_seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists():print('complete',directory,flush=True);return
    start=time.time();tr,te,tr_ids,te_ids,domain=data_domain(label,seed);teacher=TLearner().fit(tr.X,tr.T,tr.Y)
    selection=np.random.default_rng(seed).choice(te.n,min(n_test,te.n),replace=False);X=te.X[selection]
    partition=np.random.default_rng(seed+10000).permutation(tr.n);nref=int(.8*tr.n);refidx=partition[:nref];calidx=partition[nref:];reference=tr.X[refidx]
    caldist=domain.distances(tr.X[calidx],reference);threshold=float(np.quantile(caldist,.95));orig_dist=domain.distances(X,reference);preds=teacher.predict_cate(X)
    np.savez_compressed(directory/'data.npz',train_X=tr.X,train_T=tr.T,train_Y=tr.Y,train_row_ids=tr_ids,test_X=X,test_row_ids=te_ids[selection],test_indices=selection,reference_indices=refidx,calibration_indices=calidx,calibration_distances=caldist,orig_distances=orig_dist)
    with (directory/'teacher.pkl').open('wb') as f:pickle.dump(teacher,f)
    save(directory/'schema.json',domain.serializable())
    configs=dict(dataset=label,seed=seed,n_test=len(X),n_train=tr.n,k=5,calibration_quantile=.95,threshold=threshold,feature_names=tr.feature_names,modifier_indices=tr.effect_modifier_indices,
                 preprocessing='legacy_full covariates Hillstrom; training-only IHDP',protocol_sha256=hashlib.sha256((ROOT/'audit/protocol_recourse.md').read_bytes()).hexdigest(),tolerance=TOL,gain_tolerance=GAIN_TOL,python=platform.python_version())
    save(directory/'config.json',configs);summaries={}
    for version in ['historical','filtered']:
        extra={} if version=='historical' else dict(actionable_features=domain.allowed,candidate_validator=domain.validator)
        methods={'Greedy':CausalCounterfactualExplainer(cate_model=teacher,feature_bounds=domain.bounds,max_changes=3,step_size=.1,max_steps=20,seed=seed,**extra),
                 'Gradient':GradientBasedUpliftCF(cate_model=teacher,feature_bounds=domain.bounds,step_size=.1,max_steps=50,seed=seed,**extra),
                 'RandomSearch':RandomSearchUplift(cate_model=teacher,feature_bounds=domain.bounds,n_candidates=2000,max_distance=3.,seed=seed,**extra)}
        for name,method in methods.items():
            path=directory/f'{version}_{name}.json'
            if path.exists():summaries[f'{version}/{name}']=json.loads(path.read_text())['summary'];continue
            outputs=[]
            for x in X:
                result=method.explain_instance(x);outputs.append(result.x_cf if hasattr(result,'x_cf') else result['x_cf'])
            cf=np.array(outputs);cfpred=teacher.predict_cate(cf);dist=domain.distances(cf,reference);rows=[]
            for i,(x,z) in enumerate(zip(X,cf)):
                valid=domain.checks(x,z);gain=float(cfpred[i]-preds[i]);no_op=bool(np.max(np.abs(z-x))<=TOL);changed=domain.changed(x,z)
                row=dict(row_id=int(te_ids[selection[i]]),x=x.tolist(),x_cf=z.tolist(),tau_orig=float(preds[i]),tau_cf=float(cfpred[i]),gain=gain,**valid,
                         initial_box_valid=domain.checks(x,x)['box_valid'],no_op=no_op,strict_positive=gain>GAIN_TOL,nonnegative=gain>=-GAIN_TOL,
                         changed_groups=len(changed),changed_fields=changed,l2=float(np.linalg.norm(z-x)),orig_fifth_nn=float(orig_dist[i]),cf_fifth_nn=float(dist[i]),
                         nn_change=float(dist[i]-orig_dist[i]),orig_exceeds=bool(orig_dist[i]>threshold),cf_exceeds=bool(dist[i]>threshold),diagnostic_success=bool(not no_op and valid['legal'] and gain>GAIN_TOL))
                row['failure_reasons']=[k for k in ['box_valid','type_valid','frozen_valid','derived_valid'] if not row[k]]
                if no_op:row['failure_reasons'].append('no_change')
                if gain<=GAIN_TOL:row['failure_reasons'].append('no_strict_positive_gain')
                rows.append(row)
            summary=summarize(rows);summaries[f'{version}/{name}']=summary
            save(path,dict(dataset=label,seed=seed,version=version,method=name,threshold=threshold,summary=summary,rows=rows))
            print(label,seed,version,name,'success',summary['diagnostic_success'],'noop',summary['no_op'],'types',summary['type_valid'],'support',summary['support_given_changed_legal'],flush=True)
    save(directory/'result.json',dict(config=configs,summary=summaries,elapsed_sec=time.time()-start))

def aggregate():
    output={}
    for label in ['IHDP','Hillstrom']:
        records=[json.loads(p.read_text()) for p in sorted((ROOT/'results/recourse').glob(f'{label.lower()}_seed*/result.json'))]
        if not records:continue
        output[label]={'seeds':[r['config']['seed'] for r in records],'methods':{}}
        for key in records[0]['summary']:
            vals={}
            for metric in records[0]['summary'][key]:
                xs=[r['summary'][key][metric] for r in records if r['summary'][key][metric] is not None]
                vals[metric]=dict(mean=float(np.mean(xs)) if xs else None,sd=float(np.std(xs,ddof=1)) if len(xs)>1 else None,defined_seeds=len(xs))
            output[label]['methods'][key]=vals
    save(ROOT/'results/recourse_summary.json',output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['IHDP','Hillstrom','all'],default='all');p.add_argument('--seeds',type=int,nargs='+',default=list(range(42,47)));p.add_argument('--n-test',type=int,default=50);args=p.parse_args()
    for label in (['IHDP','Hillstrom'] if args.dataset=='all' else [args.dataset]):
        for seed in args.seeds:run_one(label,seed,args.n_test)
    aggregate()
