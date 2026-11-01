from pathlib import Path
import os,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'logs/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=ROOT/'results/rules_r1/ihdp_ihdp_legacy_full_seed42';a=np.load(str(p)+'.npz');d=json.loads(p.with_suffix('.json').read_text());tau=a['tau_pred_test'];masks=a['hard_test_masks'];union=np.zeros(len(tau),dtype=bool);coverage=[];means=[]
for mask in masks:
 union|=mask;coverage.append(union.mean());means.append(tau[union].mean())
order=np.sort(tau)[::-1];frac=np.arange(1,len(tau)+1)/len(tau)
fig,ax=plt.subplots(figsize=(6.8,4.3));ax.plot(frac,np.cumsum(order)/np.arange(1,len(tau)+1),color='#637D98',label='Highest model predictions at each size')
ax.plot(coverage,means,'o-',color='#B55B3D',label='Union of the first k selected rules')
for i,(x,y) in enumerate(zip(coverage,means)):ax.annotate(str(i+1),(x,y),xytext=(4,7 if i%2==0 else -13),textcoords='offset points',fontsize=9)
ax.axhline(tau.mean(),color='#666666',ls='--',lw=1,label='All test rows')
ax.set(xlabel='Fraction of test rows covered',ylabel='Mean fitted CATE prediction',xlim=(0,1));ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(ROOT/'paper/figures/rule_lift_curve.pdf',bbox_inches='tight');fig.savefig(ROOT/'paper/figures/rule_lift_curve.png',dpi=170,bbox_inches='tight')
(ROOT/'results/figure3_config.json').write_text(json.dumps(dict(source=str(p.relative_to(ROOT)),algorithm='hard',seed=42,n_test=len(tau),rule_order='training selection order, unchanged on test',coverage=coverage,mean_model_prediction=means,population_prediction=float(tau.mean())),indent=2)+'\n')
rows=[];mods={1,10,15,17,24}
for r in d['versions']['hard']['rules']:
 terms=[]
 for j,op,v in r['conditions']:
  name=r'\mathbf{x}_{'+str(j)+'}' if j in mods else 'x_{'+str(j)+'}'
  operator=r'\geq' if op=='>=' else r'\leq'
  terms.append(f'{name}{operator}{v:.2f}')
 sel=r['selection'];test=r['test']
 rows.append(f"{r['index']+1} & $"+r'\wedge '.join(terms)+f"$ & {r['tau_mean_selection']:.2f} & {100*sel['coverage']:.1f}\\% & {sel['n_treated']}/{sel['n_control']} & {test['n_treated']}/{test['n_control']} \\\\")
(ROOT/'results/rule_example_rows.tex').write_text('\n'.join(rows)+'\n')
