"""E3 outcome and preprocessing evidence, separating split SD from conditional SE."""
from pathlib import Path
import json
import numpy as np
from src.experiments.run_r1_rules import write_json
from src.experiments.summarize_r1_rules import stats

OUT=Path('results/rules_r1')


def scope_summary(rows):
    out={'seeds':[r['seed'] for r in rows],'legacy_hard_changed':sum(not r['pair']['same_conditions'] for r in rows)}
    for version in ['legacy','hard']:
        vr=[r['versions'][version] for r in rows]
        names=sorted(set().union(*(v['outcomes'] for v in vr)))
        outcomes={}
        for name in names:
            entries=[v['outcomes'].get(name,{}) for v in vr]
            outcomes[name]={key:stats([e.get(key) for e in entries]) for key in
               ['difference_in_means','conditional_se','ratio_to_population','coverage','n_treated','n_control']}
            outcomes[name]['conditional_se_by_seed']=[{'seed':r['seed'],'conditional_se':e.get('conditional_se')} for r,e in zip(rows,entries)]
            for key in ['conditional_se','n_treated','n_control']:
                values=[e[key] for e in entries if e.get(key) is not None]
                outcomes[name][key]['range']=[min(values),max(values)] if values else None
        rules=[q for r in vr for q in r['rules']]
        out[version]={'outcomes':outcomes,'normalized_fidelity':stats([v['test_fidelity']['value'] for v in vr]),
             'fidelity_degeneracy':[{'seed':r['seed'],'status':v['test_fidelity']['status']} for r,v in zip(rows,vr) if v['test_fidelity']['status']!='ok'],
             'rule_count':len(rules),'constraints':{split:{key:sum(not q[split][key] for q in rules) for key in
                ['support_ok','overlap_ok','max_coverage_ok']} for split in ['discovery','selection','test']}}
    return out


def main():
    hill={scope:[] for scope in ['legacy_full','train_only']};mega=[]
    for scope in hill:
        for seed in range(5):
            p=OUT/f'rct_hillstrom_{scope}_seed{seed}.json'
            if p.exists():hill[scope].append(json.loads(p.read_text()))
    for seed in range(10):
        p=OUT/f'rct_megafon_train_only_seed{seed}.json'
        if p.exists():mega.append(json.loads(p.read_text()))
    report={'uncertainty':'Outcome-contrast std is descriptive variation across repeated splits; conditional_se is within a fixed fitted subgroup and does not account for fitting/selection. Reused records do not create independent trials.',
            'constraint_scope':'Minimum support and treatment fraction apply to discovery and selection; the maximum-coverage cap is enforced only on selection. Test values are diagnostics, not filters.',
            'hillstrom':{'complete':all(len(x)==5 for x in hill.values()),'counts':{k:len(v) for k,v in hill.items()}},
            'megafon':{'complete':len(mega)==10,'seeds':[r['seed'] for r in mega],'interpretation':'Provider-generated synthetic competition data; contrasts are descriptive; treatment randomization not independently verified.'}}
    if report['hillstrom']['complete']:
        report['hillstrom']['scopes']={k:scope_summary(v) for k,v in hill.items()}
        pairs=[]
        for old,new in zip(hill['legacy_full'],hill['train_only']):
            seed=old['seed'];assert seed==new['seed']
            a=np.load(OUT/f'rct_hillstrom_legacy_full_seed{seed}.npz')
            b=np.load(OUT/f'rct_hillstrom_train_only_seed{seed}.npz')
            for key in ['train_row_ids','test_row_ids','T_train','T_test','Y_train','Y_test']:
                np.testing.assert_array_equal(a[key],b[key])
            assert not set(b['train_preprocess_fit_row_ids'])&set(b['test_row_ids'])
            assert set(b['train_preprocess_fit_row_ids'])==set(b['train_row_ids'])
            assert set(a['train_preprocess_fit_row_ids'])==set(a['train_row_ids'])|set(a['test_row_ids'])
            pair={'seed':seed,'raw_rows_T_Y_identical':True,'train_only_fit_excludes_test':True,
                 'max_abs_teacher_prediction_change':float(np.max(np.abs(a['tau_pred_test']-b['tau_pred_test']))),
                 'hard_test_masks_identical':bool(np.array_equal(a['hard_test_masks'],b['hard_test_masks'])),
                 'outcome_changes':{},'pipeline_outcome_changes':{}}
            common=set(old['versions']['hard']['outcomes'])&set(new['versions']['hard']['outcomes'])
            for name in sorted(common):
                x=old['versions']['hard']['outcomes'][name];y=new['versions']['hard']['outcomes'][name]
                pair['outcome_changes'][name]={key:(y[key]-x[key] if x[key] is not None and y[key] is not None else None) for key in
                    ['difference_in_means','conditional_se','coverage','ratio_to_population']}
                historical=old['versions']['legacy']['outcomes'].get(name,{})
                pair['pipeline_outcome_changes'][name]={key:(y[key]-historical[key] if historical.get(key) is not None and y[key] is not None else None) for key in
                    ['difference_in_means','conditional_se','coverage','ratio_to_population']}
            pairs.append(pair)
        names=sorted(set().union(*(p['outcome_changes'] for p in pairs)))
        changes={name:{key:stats([p['outcome_changes'].get(name,{}).get(key) for p in pairs]) for key in
                    ['difference_in_means','conditional_se','coverage','ratio_to_population']} for name in names}
        report['hillstrom']['paired']={'rows':pairs,'train_only_minus_legacy_full':changes,
              'contrast_definition':'train_only_minus_legacy_full holds hard filtering fixed; revised_pipeline_minus_historical also changes legacy selection to hard selection.',
              'revised_pipeline_minus_historical':{name:{key:stats([p['pipeline_outcome_changes'].get(name,{}).get(key) for p in pairs]) for key in
                    ['difference_in_means','conditional_se','coverage','ratio_to_population']} for name in names},
              'all_raw_splits_identical':True,'all_train_only_fit_excludes_test':True,
              'identical_hard_mask_seeds':sum(p['hard_test_masks_identical'] for p in pairs)}
    if report['megafon']['complete']:report['megafon']['train_only']=scope_summary(mega)
    write_json(OUT/'e3_comparison_summary.json',report)
    print(json.dumps({k:{f:v[f] for f in v if f in ['complete','counts','seeds']} for k,v in report.items() if isinstance(v,dict)},indent=2))

if __name__=='__main__':main()
