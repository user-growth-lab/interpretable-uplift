"""Rebuild nonlinearity, component-ablation and stability summaries from checkpoints."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from src.experiments.run_r1_rules import write_json
from src.experiments.summarize_r1_rules import stats

OUT=Path('results/rule_extensions_r1')


def nonlinear():
    groups=defaultdict(list)
    for p in OUT.glob('nonlinear*_seed*.json'):
        d=json.loads(p.read_text());groups[d['variant']].append(d)
    summary={}
    for variant,rows in groups.items():
        f=stats([r['hard']['test_fidelity']['value'] for r in rows])
        h=stats([r['hard']['modifier_hits'] for r in rows])
        summary[variant]={'n':len(rows),'seeds':sorted(r['seed'] for r in rows),
            'changed':sum(not r['pair']['same_conditions'] for r in rows),
            'fidelity_mean':f['mean'],'fidelity_std':f['std'],'fidelity_valid':f['valid'],
            'hits_mean':h['mean'],'hits_std':h['std'],'n_rules':[r['hard']['n_rules'] for r in rows]}
    write_json(OUT/'nonlinear_summary.json',summary)


def ablation():
    groups=defaultdict(list)
    for p in OUT.glob('ablation*_seed*.json'):
        d=json.loads(p.read_text());groups[d['variant']].append(d)
    for seed in range(42,47):
        p=Path(f'results/rules_r1/ihdp_ihdp_legacy_full_seed{seed}.json')
        if p.exists():
            d=json.loads(p.read_text());groups['full'].append(dict(seed=seed,pair=d['pair'],**d['versions']))
    summary={}
    for variant,rows in groups.items():
        rec={'seeds':sorted(r['seed'] for r in rows),'complete':len(rows)==5,
             'changed':sum(not r['pair']['same_conditions'] for r in rows),'versions':{}}
        for version in ['legacy','hard']:
            vr=[r[version] for r in rows]
            rec['versions'][version]={'train_fidelity':stats([r['train_fidelity']['value'] for r in vr]),
                 'test_fidelity':stats([r['test_fidelity']['value'] for r in vr]),
                 'modifier_hits':stats([r['modifier_hits'] for r in vr]),
                 'distinct_features':stats([r['n_distinct_features'] for r in vr]),
                 'n_rules':stats([r['n_rules'] for r in vr]),
                 'selection_violations':sum(not all(q['selection'][k] for k in ['support_ok','overlap_ok','max_coverage_ok']) for r in vr for q in r['rules']),
                 'degeneracy':[{'seed':r['seed'],'status':r[version]['test_fidelity']['status']} for r in rows if r[version]['test_fidelity']['status']!='ok']}
        summary[variant]=rec
    write_json(OUT/'ablation_summary.json',summary)


def stability():
    summary={}
    for p in OUT.glob('stability_*.json'):
        if p.name=='stability_summary.json':continue
        d=json.loads(p.read_text());a=np.load(p.with_suffix('.npz'));n=len(a['X_reference']);checks=0
        result={'changed_bootstraps':sum(not r['pair']['same_conditions'] for r in d['rows']),'methods':{}}
        for version in ['legacy','hard']:
            result['methods'][version]={}
            for method in ['SCRL','CDT','CausalTree']:
                masks=[a[f'b{b}_{version}_{method}_mask'] for b in range(6)]
                sims=[]
                for i in range(6):
                    for j in range(i+1,6):
                        u=np.sum(masks[i]|masks[j]);sims.append(np.sum(masks[i]&masks[j])/u if u else 1.)
                assert np.isclose(np.mean(sims),d['summary'][version][method]['jaccard_mean'])
                metrics=[row['versions'][version]['metrics'][method] for row in d['rows']]
                rec={'jaccard':float(np.mean(sims)),'jaccard_pair_sd':float(np.std(sims))}
                for metric in ['coverage','targeting_precision','subgroup_uplift']:
                    vals=[m[metric] for m in metrics if m[metric] is not None]
                    rec[metric]={'mean':float(np.mean(vals)) if vals else None,'std':float(np.std(vals)) if vals else None,'valid':len(vals)}
                for b,mask in enumerate(masks):
                    assert int(mask.sum())==int(a[f'b{b}_{version}_SCRL_mask'].sum())
                    assert np.isclose(metrics[b]['coverage'],mask.mean())
                    if 'true_tau_reference' in a and mask.any():
                        true=a['true_tau_reference'];top=np.argsort(-true)[:int(round(.3*n))]
                        target=np.zeros(n,bool);target[top]=True
                        assert np.isclose(metrics[b]['targeting_precision'],target[mask].mean())
                    assert np.isfinite(a[f'b{b}_teacher_predictions']).all();checks+=1
                result['methods'][version][method]=rec
            # Independently reconstruct selection means and the positive-rule union.
            for b,row in enumerate(d['rows']):
                tau=a[f'b{b}_teacher_predictions'];idx=a[f'b{b}_bootstrap_indices']
                shuffled=np.arange(n);np.random.default_rng(row['scrl_seed']).shuffle(shuffled)
                val=shuffled[n//2:];masks=[];means=[]
                for rule in row['versions'][version]['rules']:
                    mask=np.ones(n,bool)
                    for j,op,thr in rule['conditions']:
                        mask &= a['X_reference'][:,j]>=thr if op=='>=' else a['X_reference'][:,j]<=thr
                    mval=mask[idx][val];means.append(tau[idx][val][mval].mean());masks.append(mask)
                keep=[mask for mask,mean in zip(masks,means) if mean>=tau.mean()] or masks
                union=np.any(keep,axis=0) if keep else np.zeros(n,bool)
                np.testing.assert_array_equal(union,a[f'b{b}_{version}_SCRL_mask']);checks+=1
        result['raw_checks']=checks;summary[d['dataset']]=result
    write_json(OUT/'stability_summary.json',summary)


def main():
    nonlinear();ablation();stability()
    print('Saved nonlinear_summary.json, ablation_summary.json, stability_summary.json from completed checkpoints.')

if __name__=='__main__':main()
