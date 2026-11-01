from __future__ import annotations
from pathlib import Path
import sys,itertools,json,pickle,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.tesv import TESVExplainer
ATOL=1e-10;RTOL=1e-8

def direct_path(predict,x,background,permutations):
    d=len(x);hybrids=[]
    for order in permutations:
        z=background.copy();hybrids.append(z.copy())
        for j in order:
            z[:,j]=x[j];hybrids.append(z.copy())
    values=predict(np.concatenate(hybrids)).reshape(len(permutations),d+1,len(background)).mean(axis=2)
    steps=np.diff(values,axis=1);phis=np.zeros((len(permutations),d))
    for k,order in enumerate(permutations):phis[k,order]=steps[k]
    return phis.mean(axis=0)

def exact(predict,x,background):
    d=len(x);values={}
    for mask in range(2**d):
        idx=[j for j in range(d) if mask&(1<<j)];z=background.copy();z[:,idx]=x[idx];values[mask]=predict(z).mean()
    phi=np.zeros(d)
    for j in range(d):
        for mask in range(2**d):
            if mask&(1<<j):continue
            s=bin(mask).count('1');weight=math.factorial(s)*math.factorial(d-s-1)/math.factorial(d)
            phi[j]+=weight*(values[mask|(1<<j)]-values[mask])
    return phi

class Function:
    def predict_cate(self,X):return 2*X[:,0]+X[:,0]*X[:,1]

def error(a,b):
    return dict(max_absolute=float(np.max(np.abs(a-b))),max_scaled_relative=float(np.max(np.abs(a-b)/np.maximum(1,np.abs(b)))),passed=bool(np.allclose(a,b,atol=ATOL,rtol=RTOL)))

def main():
    out=ROOT/'results/shap_equivalence';out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20260928);bg=rng.normal(size=(7,3));x=np.array([.7,-.3,8.]);perms=np.array(list(itertools.permutations(range(3))))
    fun=Function();phi_exact=exact(fun.predict_cate,x,bg);phi_tesv=TESVExplainer(fun,bg,n_background=7).explain_instance(x,permutations=perms)
    lowdim=dict(exact=phi_exact.tolist(),tesv=phi_tesv.tolist(),comparison=error(phi_tesv,phi_exact),dummy_absolute=float(abs(phi_tesv[2])),efficiency_residual=float(abs(phi_tesv.sum()-(fun.predict_cate(x[None])[0]-fun.predict_cate(bg).mean()))))
    np.savez_compressed(out/'exact_inputs.npz',background=bg,x=x,permutations=perms)
    assert lowdim['comparison']['passed'] and lowdim['dummy_absolute']<ATOL
    results={'tolerances':dict(absolute=ATOL,relative=RTOL),'exact_lowdim':lowdim,'datasets':{}}
    for label in ['ihdp','hillstrom']:
        source=ROOT/'results/recourse'/f'{label}_seed42';data=np.load(source/'data.npz')
        with (source/'teacher.pkl').open('rb') as f:model=pickle.load(f)
        X=data['test_X'][:10];indices=rng.choice(len(data['train_X']),50,replace=False);background=data['train_X'][indices]
        orders=np.array([[rng.permutation(X.shape[1]) for _ in range(20)] for _ in range(10)])
        explainer=TESVExplainer(model,background,n_background=50,n_permutations=20)
        ours=np.array([explainer.explain_instance(row,permutations=order) for row,order in zip(X,orders)])
        direct=np.array([direct_path(model.predict_cate,row,background,order) for row,order in zip(X,orders)])
        paired=np.array([direct_path(model.model_t.predict,row,background,order)-direct_path(model.model_c.predict,row,background,order) for row,order in zip(X,orders)])
        residual=float(np.max(np.abs(ours.sum(axis=1)-(model.predict_cate(X)-model.predict_cate(background).mean()))))
        record=dict(n_instances=10,background_size=50,permutations=20,tesv_vs_direct=error(ours,direct),tesv_vs_paired=error(ours,paired),efficiency_residual=residual)
        results['datasets'][label]=record
        np.savez_compressed(out/f'{label}_inputs_outputs.npz',background_indices=indices,background=background,x=X,permutations=orders,tesv=ours,direct=direct,paired=paired)
        print(label,record,flush=True)
        assert record['tesv_vs_direct']['passed'] and record['tesv_vs_paired']['passed'] and residual<ATOL
    (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
