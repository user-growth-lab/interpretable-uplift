"""Complete table columns from saved teachers/rules; no new SCRL or teacher fit."""
from pathlib import Path
import json
import joblib
import numpy as np
from src.models.baselines import CURLSBaseline
from src.metrics.fidelity import coverage_fidelity_details
from src.experiments.run_r1_rules import write_json
from src.experiments.summarize_r1_rules import stats


def main():
    rows=[];arrays={}
    for seed in range(42,52):
        p=Path(f'results/rules_r1/ihdp_ihdp_legacy_full_seed{seed}.json')
        if not p.exists():continue
        d=json.loads(p.read_text());a=np.load(p.with_suffix('.npz'))
        row={'seed':seed,'methods':{}}
        for label,version in [('SCRL_legacy','legacy'),('SCRL_hard','hard')]:
            v=d['versions'][version]
            row['methods'][label]={'train':v['train_fidelity'],'test':v['test_fidelity'],'modifier_hits':v['modifier_hits']}
        top=sorted(np.unique(a['ct_train_prediction']),reverse=True)[:5]
        ct_tr=np.array([a['ct_train_prediction']==v for v in top])
        row['methods']['RP_tree_top5']={'train':coverage_fidelity_details(a['tau_pred_train'],ct_tr),
            'test':d['causal_tree']['top5'],'modifier_hits':d['causal_tree']['modifier_hits']}
        model=joblib.load(p.with_name(p.stem+'_teacher.joblib'))
        bp=CURLSBaseline(cate_model=model,n_rules=5,max_depth=3,min_samples_leaf=30,n_bootstrap=50,seed=seed)
        bp.fit(a['X_train'],a['T_train'],a['Y_train'])
        bp_tr=np.asarray(bp.get_rule_masks(a['X_train']),bool).reshape(-1,len(a['X_train']))
        bp_te=np.asarray(bp.get_rule_masks(a['X_test']),bool).reshape(-1,len(a['X_test']))
        np.testing.assert_array_equal(bp_te,a['curls_test_masks'])
        row['methods']['BP_tree']={'train':coverage_fidelity_details(a['tau_pred_train'],bp_tr),
            'test':d['curls']['test_fidelity'],'modifier_hits':None}
        arrays[f's{seed}_RP_train_masks']=ct_tr;arrays[f's{seed}_BP_train_masks']=bp_tr
        arrays[f's{seed}_BP_test_masks']=bp_te;rows.append(row)
    summary={}
    for method in ['SCRL_legacy','SCRL_hard','RP_tree_top5','BP_tree']:
        report={split:stats([r['methods'][method][split]['value'] for r in rows]) for split in ['train','test']}
        report['modifier_hits']=stats([r['methods'][method]['modifier_hits'] for r in rows])
        report['degeneracy']=[{'seed':r['seed'],'split':split,'status':r['methods'][method][split]['status']}
               for r in rows for split in ['train','test'] if r['methods'][method][split]['status']!='ok']
        summary[method]=report
    out=Path('results/rule_extensions_r1');out.mkdir(parents=True,exist_ok=True)
    write_json(out/'ihdp_table_summary.json',{'complete':len(rows)==10,'seeds':[r['seed'] for r in rows],
        'protocol':'Saved teacher and SCRL, exact-size normalized fidelity. BP comparator refitted only to restore omitted training masks; saved test masks checked exactly equal. BP modifier-hit unavailable rather than historical hardcoded zero.',
        'rows':rows,'summary':summary})
    np.savez_compressed(out/'ihdp_baseline_masks.npz',**arrays)
    print(json.dumps({'complete':len(rows)==10,'summary':summary},indent=2))

if __name__=='__main__':main()
