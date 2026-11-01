"""Rule-assisted versus direct teacher targeting from saved IHDP predictions.

The stored tau is the simulated individual effect used to generate outcomes,
including its individual noise term, rather than the noise-free conditional CATE.
No teacher, attribution, rule or recourse model is refitted here.
"""
from pathlib import Path
import json
import numpy as np
from src.experiments.run_r1_rules import write_json
from src.experiments.summarize_r1_rules import stats


def main():
    rows=[];raw={}
    for seed in range(42,47):
        p=Path(f'results/rules_r1/ihdp_ihdp_legacy_full_seed{seed}.json')
        if not p.exists():continue
        d=json.loads(p.read_text());a=np.load(p.with_suffix('.npz'))
        theta=a['true_tau_test'];pred=a['tau_pred_test'];n=len(pred);budget=max(1,int(.2*n))
        teacher_top=np.argsort(-pred)[:budget]
        row={'seed':seed,'test_n':n,'budget':budget,'population_simulated_effect':float(theta.mean()),
             'teacher_top20_effect':float(theta[teacher_top].mean()),'versions':{}}
        raw[f's{seed}_teacher_top20_indices']=teacher_top
        for version in ['legacy','hard']:
            rules=d['versions'][version]['rules'];masks=a[f'{version}_test_masks']
            order=sorted(range(len(rules)),key=lambda i:rules[i]['tau_mean_selection'],reverse=True)
            chosen=np.zeros(n,bool)
            for i in order:
                if chosen.sum()>=budget:break
                chosen|=masks[i]
            selected=np.where(chosen)[0]
            if len(selected)>budget:selected=selected[np.argsort(-pred[selected])[:budget]]
            k=len(selected);matched=np.argsort(-pred)[:k]
            effect=float(theta[selected].mean()) if k else None
            matched_effect=float(theta[matched].mean()) if k else None
            row['versions'][version]={'selected_n':k,'coverage':k/n,'budget_filled':k==budget,
                 'rule_targeting_effect':effect,'teacher_matched_coverage_effect':matched_effect,
                 'paired_difference_rule_minus_teacher':effect-matched_effect if k else None,
                 'status':'ok' if k else 'no_selected_rows'}
            raw[f's{seed}_{version}_selected_indices']=selected
            raw[f's{seed}_{version}_teacher_matched_indices']=matched
        rows.append(row)
    summary={v:{metric:stats([r['versions'][v][metric] for r in rows]) for metric in
               ['selected_n','rule_targeting_effect','teacher_matched_coverage_effect','paired_difference_rule_minus_teacher']}
             for v in ['legacy','hard']}
    report={'protocol':'Fixed seeds 42..46; rank rules by selection mean, accumulate to 20% budget, truncate within rule-selected candidates by teacher prediction; compare teacher top-k at exactly the same selected count. Never fill insufficient rule coverage from test outcomes.',
            'target':'simulated individual effect theta, with individual effect noise; not noise-free conditional CATE',
            'expected_seeds':list(range(42,47)),'completed_seeds':[r['seed'] for r in rows],
            'complete':len(rows)==5,'rows':rows,'summary':summary}
    out=Path('results/rule_extensions_r1');out.mkdir(parents=True,exist_ok=True)
    write_json(out/'targeting_summary.json',report);np.savez_compressed(out/'targeting_selected_indices.npz',**raw)
    print(json.dumps({'complete':report['complete'],'seeds':report['completed_seeds'],'summary':summary},indent=2))

if __name__=='__main__':main()
