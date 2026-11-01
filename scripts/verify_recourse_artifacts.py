"""Check persisted predictions, failure denominators and mixed-distance definition."""
from pathlib import Path
import sys,json,pickle
import numpy as np
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_recourse_diagnostic import Domain

def independent_distances(query,reference,schema):
    matrix=np.zeros((len(query),len(reference)))
    for gi,group in enumerate(schema['groups']):
        cols=group['columns']
        if group['kind']=='continuous':
            scale=np.array(schema['iqr_scales'][str(gi)])
            matrix+=cdist(query[:,cols]/scale,reference[:,cols]/scale,metric='cityblock')/len(cols)
        else:
            matrix+=(cdist(query[:,cols],reference[:,cols],metric='chebyshev')>1e-7)
    return np.sort(matrix/len(schema['groups']),axis=1)[:,4]

def main():
    checked=0;max_pred_error=0;max_distance_error=0;counts={}
    for directory in sorted((ROOT/'results/recourse').glob('*_seed*')):
        data=np.load(directory/'data.npz');schema=json.loads((directory/'schema.json').read_text());config=json.loads((directory/'config.json').read_text())
        with (directory/'teacher.pkl').open('rb') as f:model=pickle.load(f)
        reference=data['train_X'][data['reference_indices']]
        # Independently recompute every calibration distance, not just the reported quantile.
        distance=independent_distances(data['train_X'][data['calibration_indices']],reference,schema)
        np.testing.assert_allclose(distance,data['calibration_distances'],atol=1e-12,rtol=1e-12)
        assert abs(float(np.quantile(distance,.95))-config['threshold'])<1e-12
        for version in ['historical','filtered']:
            for method in ['Greedy','Gradient','RandomSearch']:
                result=json.loads((directory/f'{version}_{method}.json').read_text());rows=result['rows'];n=len(rows);assert n==50
                X=np.array([r['x'] for r in rows]);Z=np.array([r['x_cf'] for r in rows]);gains=model.predict_cate(Z)-model.predict_cate(X)
                error=np.max(np.abs(gains-[r['gain'] for r in rows]));max_pred_error=max(max_pred_error,float(error));assert error<1e-12
                dist=independent_distances(Z,reference,schema);error=np.max(np.abs(dist-[r['cf_fifth_nn'] for r in rows]));max_distance_error=max(max_distance_error,float(error));assert error<1e-12
                for i,r in enumerate(rows):
                    assert r['no_op']==bool(np.max(np.abs(Z[i]-X[i]))<=1e-7)
                    assert r['diagnostic_success']==bool(not r['no_op'] and r['legal'] and gains[i]>1e-9)
                    assert r['cf_exceeds']==bool(dist[i]>config['threshold'])
                    # Every changed filtered output must respect frozen fields, types and history relation.
                    if version=='filtered' and not r['no_op']:
                        assert r['legal'] and np.all(np.abs(Z[i,schema['frozen_indices']]-X[i,schema['frozen_indices']])<=1e-7)
                checked+=n
                counts[f'{directory.name}/{version}/{method}']=n
    report=dict(checked_point_pairs=checked,expected_point_pairs=3000,max_prediction_discrepancy=max_pred_error,max_fifth_neighbor_discrepancy=max_distance_error,
                independent_implementation='SciPy cityblock/Chebyshev group distances with complete sorting; fifth order statistic',
                validation='Every calibration distance, saved gain, test distance, no-op and success denominator checked; all changed filtered outputs legal.',counts=counts)
    assert checked==3000
    (ROOT/'audit/recourse_verification.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='counts'},flush=True)
if __name__=='__main__':main()
