"""Exact-value audit of optional acceleration; run with SCRL_BOOTSTRAP_BACKEND=numba."""
import importlib.util
import sys
import hashlib
from pathlib import Path
import numpy as np
from sklearn.utils import resample
from src.models.scrl import SCRL,CausalRule,_BOOTSTRAP_BACKEND,_bootstrap_index_matrix
from src.experiments.run_r1_rules import write_json


class Teacher:
    def predict_cate(self,X):return X[:,0]**2+X[:,1]


def main():
    assert _BOOTSTRAP_BACKEND=='numba'
    paths=['audit/scrl_before_exact_bootstrap_optimization.py','audit/scrl_before_numba_optimization.py']
    references=[]
    for i,path in enumerate(paths):
        name=f'scrl_reference_{i}'
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        references.append(module)
    seeds=np.array([0,1,42,982734,999999999,2147483647,4294967295],np.int64)
    index_checks=0;score_checks=0;fit_checks=0
    globalstate=np.random.get_state()
    for n in [2,3,10,67,336,1000,10000,40000]:
        actual=_bootstrap_index_matrix(seeds,n)
        for b,seed in enumerate(seeds):
            expected=resample(np.arange(n),random_state=int(seed))
            np.testing.assert_array_equal(actual[b],expected);index_checks+=1
        X=np.random.default_rng(n).normal(size=(n,3));T=np.arange(n)%2;tau=Teacher().predict_cate(X)
        for seed in [0,1,42,982734,999999999]:
            new=SCRL(Teacher(),seed=seed,n_bootstrap=10);new._global_tau_mean=float(tau.mean())
            r1=CausalRule([],0,0,0,0,0,0);score=new._score_rule(r1,X,tau,T)
            for ref in references:
                old=ref.SCRL(Teacher(),seed=seed,n_bootstrap=10);old._global_tau_mean=float(tau.mean())
                r0=ref.CausalRule([],0,0,0,0,0,0)
                assert old._score_rule(r0,X,tau,T)==score
                assert r0.stability_score==r1.stability_score
                assert old.rng.bit_generator.state==new.rng.bit_generator.state;score_checks+=1
    assert np.array_equal(np.random.get_state()[1],globalstate[1])
    for seed in [0,1,42]:
        X=np.random.default_rng(seed).normal(size=(220,3));T=np.arange(220)%2;Y=np.zeros(220)
        kw=dict(n_rules=3,max_rule_length=2,n_candidates=3,n_bootstrap=4,seed=seed)
        new=SCRL(Teacher(),**kw).fit(X,T,Y)
        for ref in references:
            old=ref.SCRL(Teacher(),**kw).fit(X,T,Y)
            assert [vars(r) for r in old.rules]==[vars(r) for r in new.rules]
            assert old.rng.bit_generator.state==new.rng.bit_generator.state
            assert old.selection_trace_==new.selection_trace_;fit_checks+=1
    out={'status':'passed','backend':'optional Numba indices only at n_R<=10000; NumPy means/std unchanged',
         'sklearn_resample_index_checks':index_checks,'score_and_rng_exact_checks':score_checks,
         'full_fit_exact_checks':fit_checks,'global_numpy_rng_unchanged':True,
         'numba_version':__import__('numba').__version__,'numpy_version':np.__version__,
         'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths+['src/models/scrl.py']}}
    write_json(Path('audit/bootstrap_acceleration_validation.json'),out);print(out)

if __name__=='__main__':main()
