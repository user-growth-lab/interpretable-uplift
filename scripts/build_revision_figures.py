from __future__ import annotations
import os
from pathlib import Path
import sys,json,pickle
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'logs/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Patch
from src.models.tesv import TESVExplainer

class IgnoreAppended:
    def __init__(self,model):self.model=model
    def predict_cate(self,X):return self.model.predict_cate(X[:,:25])

def main():
    out=ROOT/'paper/figures';out.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(10.8,6.1));ax.set(xlim=(0,12),ylim=(0,7));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03,rounding_size=.12',facecolor=color,edgecolor='#435469',linewidth=1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#435469',lw=1.2))
    box(.15,5.3,3.3,1.15,'Outer training rows\nCovariates, treatment, outcome','#E8EEF5')
    box(4.35,5.3,3.3,1.15,'One fitted CATE function\nTrained on all outer training rows','#E4EFF4')
    box(8.55,5.3,3.3,1.15,'Outer test rows\nReserved for final evaluation','#F4EFE5')
    arrow((3.48,5.88),(4.32,5.88))
    for center in [1.8,6,10.2]:arrow((6,5.26),(center,4.35))
    box(.15,2.62,3.3,1.7,'Attribution\nMarginal replacement of covariates\nTraining background samples\n\nOutput: contribution to fitted CATE','#E8EEF5')
    box(4.35,2.62,3.3,1.7,'Subgroup rules\nTraining rows split for\ndiscovery / selection\n\nOutput: conditions and support','#E6F0E8')
    box(8.55,2.62,3.3,1.7,'Feature-space recourse\nTraining bounds and declared\ncoordinate restrictions\n\nOutput: a candidate target state','#F4E9E6')
    ax.text(6,2.19,'The fitted CATE model uses both rule subsets; no cross-fitting.',ha='center',fontsize=10,color='#534B45')
    for center in [1.8,10.2]:arrow((center,2.55),(center,1.73))
    arrow((6,2.55),(6,2.39));arrow((6,1.99),(6,1.73))
    ax.plot([11.85,11.97,11.97],[5.88,5.88,1.1],color='#435469',lw=1,ls='--')
    arrow((11.97,1.1),(11.84,1.1))
    box(.15,.58,11.7,1.1,'Final evaluation uses a stated reference:\nknown semi-synthetic modifiers / fitted-model agreement / observed outcomes\nPreprocessing exceptions and diagnostic splits are specified per experiment.','#F4F5F7')
    fig.tight_layout();fig.savefig(out/'framework.pdf',bbox_inches='tight');fig.savefig(out/'framework.png',dpi=170,bbox_inches='tight');plt.close(fig)
    if '--framework-only' in sys.argv:return
    source=ROOT/'results/recourse/ihdp_seed42';d=np.load(source/'data.npz');config=json.loads((source/'config.json').read_text())
    with (source/'teacher.pkl').open('rb') as f:model=pickle.load(f)
    modifiers=config['modifier_indices'];rng=np.random.default_rng(20260928)
    train=d['train_X'];test=d['test_X'];rho=.9
    trproxy=rho*train[:,modifiers]+np.sqrt(1-rho*rho)*rng.normal(size=(len(train),5))
    teproxy=rho*test[:,modifiers]+np.sqrt(1-rho*rho)*rng.normal(size=(len(test),5))
    train_aug=np.c_[train,trproxy];test_aug=np.c_[test,teproxy]
    teacher=IgnoreAppended(model);explainer=TESVExplainer(teacher,train_aug,n_background=50,n_permutations=20,seed=42)
    values=explainer.explain(test_aug,show_progress=False);importance=np.abs(values).mean(axis=0)
    assert np.max(np.abs(values[:,25:]))<1e-10
    fig,ax=plt.subplots(figsize=(8.6,7.4));order=np.r_[np.argsort(-importance[:25]),np.arange(25,30)]
    colors=['#B55B3D' if j in modifiers else '#637D98' if j<25 else '#A3A8AD' for j in order]
    labels=[f'x{j} *' if j in modifiers else f'x{j}' if j<25 else f'proxy {j-24} (ignored)' for j in order]
    bars=ax.barh(np.arange(30),importance[order],color=colors)
    ax.set_yticks(np.arange(30),labels,fontsize=9);ax.invert_yaxis();ax.set_xlabel('Mean absolute attribution to the fitted CATE function')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color='#B55B3D',label='* Known effect modifier'),Patch(color='#637D98',label='Other original covariate'),Patch(color='#A3A8AD',label='Appended, model-ignored proxy')],loc='lower right',fontsize=9,frameon=False)
    fig.tight_layout();fig.savefig(out/'tesv_attributions.pdf',bbox_inches='tight');fig.savefig(out/'tesv_attributions.png',dpi=170,bbox_inches='tight');plt.close(fig)
    np.savez_compressed(ROOT/'results/figure2_attributions.npz',attributions=values,train_X=train_aug,test_X=test_aug,background=explainer.background_data,modifiers=np.array(modifiers))
    (ROOT/'results/figure2_config.json').write_text(json.dumps(dict(dataset='IHDP semi-synthetic',seed=42,modifiers=modifiers,n_test=len(test),B=50,M=20,proxy_rho=rho,proxy_seed=20260928,teacher_source=str(source.relative_to(ROOT)),note='Illustrative fixed model from E2; not pooled Table attribution recall.'),indent=2)+'\n')
    print('saved',out,flush=True)
if __name__=='__main__':main()
