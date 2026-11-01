"""Recompute saved R1 scientific quantities independently, without refitting models."""
import json
import argparse
import joblib
from pathlib import Path
import numpy as np
from src.experiments.run_r1_rules import write_json


def union_fidelity(pred, masks):
    union=np.any(masks,axis=0) if len(masks) else np.zeros(len(pred),bool)
    k=int(union.sum());n=len(pred)
    if k==0:return None,"empty_coverage"
    if k==n:return None,"full_coverage"
    den=float(np.sort(pred)[-k:].mean()-pred.mean())
    if abs(den)<=1e-12:return None,"near_zero_denominator"
    return float((pred[union].mean()-pred.mean())/den),"ok"


def validate(path):
    d=json.loads(path.read_text());a=np.load(path.with_suffix('.npz'))
    versions=d.get('versions') or {v:d[v] for v in ('legacy','hard')}
    p=a['tau_pred_test'];nt=len(a['X_train']);checks=0
    disc=a['discovery_indices'];sel=a['selection_indices']
    assert not set(disc)&set(sel) and sorted(np.r_[disc,sel])==list(range(nt))
    checks+=1
    if d['pair']['same_conditions']:
        assert [r['conditions'] for r in versions['legacy']['rules']]==[r['conditions'] for r in versions['hard']['rules']]
        for split in ['train','test']:
            np.testing.assert_array_equal(a[f'legacy_{split}_masks'],a[f'hard_{split}_masks'])
        checks+=1
    model_path=path.with_name(path.stem+'_teacher.joblib')
    if model_path.exists():
        model=joblib.load(model_path)
        for split in ['train','test']:
            np.testing.assert_allclose(model.predict_cate(a[f'X_{split}']),a[f'tau_pred_{split}'],rtol=1e-12,atol=1e-12)
            checks+=1
    for version,report in versions.items():
        conditions=report['rules']
        for split in ('train','test'):
            X=a[f'X_{split}'];masks=a[f'{version}_{split}_masks']
            assert masks.shape==(len(conditions),len(X))
            for i,r in enumerate(conditions):
                mask=np.ones(len(X),bool)
                for j,op,thr in r['conditions']:
                    assert op in ('>=','<=')
                    mask &= X[:,j]>=thr if op=='>=' else X[:,j]<=thr
                np.testing.assert_array_equal(mask,masks[i]);checks+=1
            pred=a[f'tau_pred_{split}'];value,status=union_fidelity(pred,masks)
            recorded=report[f'{split}_fidelity']
            assert recorded['status']==status,(path,version,split,status,recorded)
            if value is None:assert recorded['value'] is None
            else:np.testing.assert_allclose(recorded['value'],value,rtol=1e-12,atol=1e-12)
            checks+=1
        for i,r in enumerate(conditions):
            for split,idx in (('discovery',disc),('selection',sel),('test',np.arange(len(p)))):
                masks=a[f'{version}_test_masks'] if split=='test' else a[f'{version}_train_masks'][:,idx]
                T=a['T_test'] if split=='test' else a['T_train'][idx]
                diag=r[split];mask=masks[i]
                n=int(mask.sum());n1=int(T[mask].sum());n0=n-n1
                assert (diag['n'],diag['n_treated'],diag['n_control'])==(n,n1,n0)
                if split=='selection' and version=='hard':
                    kw=d['config'];assert n>=kw.get('min_support',.05)*len(T) and n>=2
                    assert min(n1,n0)>=kw.get('overlap_ratio',.1)*n
                    assert n/len(T)<=kw.get('max_coverage',.4)
                checks+=1
        if 'outcomes' in report:
            for name,entry in report['outcomes'].items():
                m=a[f'{version}_{name}_mask'];T=a['T_test'];Y=a['Y_test']
                n1=int(np.sum(m&(T==1)));n0=int(np.sum(m&(T==0)))
                assert (entry['n'],entry['n_treated'],entry['n_control'])==(int(m.sum()),n1,n0)
                if n1>=2 and n0>=2:
                    y1,y0=Y[m&(T==1)],Y[m&(T==0)]
                    np.testing.assert_allclose(entry['difference_in_means'],y1.mean()-y0.mean(),rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(entry['conditional_se'],np.sqrt(y1.var(ddof=1)/n1+y0.var(ddof=1)/n0),rtol=1e-12,atol=1e-12)
                else:assert entry['difference_in_means'] is None
                checks+=1
    for name in ('top5','all_leaves'):
        if 'causal_tree' in d:
            masks=a['ct_top5_masks' if name=='top5' else 'ct_all_masks']
            value,status=union_fidelity(p,masks);r=d['causal_tree'][name]
            assert r['status']==status
            if value is None:assert r['value'] is None
            else:np.testing.assert_allclose(value,r['value'],rtol=1e-12,atol=1e-12)
            checks+=1
    if d.get('preprocessing')=='train_only' and d.get('experiment')=='rct':
        assert not set(a['train_preprocess_fit_row_ids'])&set(a['test_row_ids'])
        assert set(a['train_preprocess_fit_row_ids'])==set(a['train_row_ids']);checks+=1
    return {'file':str(path),'checks':checks,'n_train':nt,'n_test':len(p),'selection_hard_violations':0}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    paths=sorted(Path('results/rules_r1').glob('*_seed*.json'))+sorted(Path('results/rule_extensions_r1').glob('*_seed*.json'))
    expected={f'cross_{dataset}_legacy_full_seed{seed}.json' for dataset in ['hillstrom','criteo'] for seed in range(5)}
    expected|={f'ihdp_ihdp_legacy_full_seed{seed}.json' for seed in range(42,52)}
    expected|={f'rct_hillstrom_{scope}_seed{seed}.json' for scope in ['legacy_full','train_only'] for seed in range(5)}
    expected|={f'rct_megafon_train_only_seed{seed}.json' for seed in range(10)}
    expected|={f'nonlinear_{variant}_seed{seed}.json' for variant in ['linear','nonlinear','interaction','nonlinear_interaction'] for seed in range(42,45)}
    expected|={f'ablation_{variant}_seed{seed}.json' for variant in ['no_div_dedup','variance_based'] for seed in range(42,47)}
    missing=sorted(expected-{p.name for p in paths})
    if args.require_complete:assert not missing, f'Incomplete required evidence: {missing}'
    rows=[validate(p) for p in paths]
    out={'status':'passed','complete':not missing,'expected_checkpoint_count':len(expected),'missing_checkpoints':missing,
         'checkpoint_count':len(rows),'checks':sum(r['checks'] for r in rows),'checkpoints':rows,
         'scope':'All completed checkpoints at validation time; independent saved-array recomputation, no model refit. Does not validate unfinished experiments.'}
    write_json(Path('audit/rule_validation.json'),out)
    print(json.dumps({k:v for k,v in out.items() if k!='checkpoints'},indent=2))

if __name__=='__main__':main()
