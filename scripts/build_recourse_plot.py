from pathlib import Path
import os,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'logs/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(9,3.7))
methods=['Greedy','Gradient','RandomSearch'];colors=['#B55B3D','#637D98','#558369']
for ax,label in zip(axes,['ihdp','hillstrom']):
 for j,(method,color) in enumerate(zip(methods,colors)):
  vals=[json.loads((ROOT/f'results/recourse/{label}_seed{s}/filtered_{method}.json').read_text())['summary']['gain'] for s in range(42,47)]
  ax.scatter(j+np.linspace(-.10,.10,5),vals,color=color,zorder=3,s=32)
  ax.plot([j-.18,j+.18],[np.mean(vals)]*2,color='black',lw=1.5)
 ax.set_xticks(range(3),['Greedy','Gradient','Random search']);ax.set_title('IHDP (semi-synthetic)' if label=='ihdp' else 'Hillstrom (visit outcome)')
 ax.set_ylabel('Mean predicted CATE gain');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(ROOT/'paper/figures/counterfactual_gains.pdf',bbox_inches='tight');fig.savefig(ROOT/'paper/figures/counterfactual_gains.png',dpi=170,bbox_inches='tight')
