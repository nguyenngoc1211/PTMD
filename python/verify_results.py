"""Read-only integration checks; write a compact verification receipt."""
import hashlib
import json
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from common import CONFIG, ROOT, setup, inference, adjust_bh
from compute_cfi import aligned_values, evaluate


def main():
    setup()
    checks=[]
    def check(condition,message):
        if not condition: raise AssertionError(message)
        checks.append(message)
    split=joblib.load('results/intermediate/train_test_split.joblib')
    model=joblib.load('results/models/xgboost_model.joblib')
    info=json.loads(Path('results/intermediate/split_r.json').read_text())
    features=split['feature_names']
    check(not set(split['train_indices']) & set(split['test_indices']),'Train/test indices disjoint')
    hashes=lambda X:pd.util.hash_pandas_object(pd.DataFrame(X),index=False)
    check(not hashes(split['X_test']).isin(set(hashes(split['X_train']))).any(),'No train/test feature duplicates')
    for part in ['X_train','X_test','y_train','y_test']:
        check(np.array_equal(split[part],model[part]),f'Model {part} agrees with saved split')
    check(info['train_indices']==split['train_indices'].tolist(),'R train indices agree')
    check(info['test_indices']==split['test_indices'].tolist(),'R test indices agree')
    full=pd.read_csv('results/intermediate/full_data.csv')
    np.testing.assert_allclose(full.iloc[info['train_indices']][features],split['X_train'])
    np.testing.assert_allclose(full.iloc[info['test_indices']][features],split['X_test'])
    checks.append('R exported values agree with Python split')
    indices=info['importance_test_positions']; repeats=CONFIG['conditional_repeats']
    draws=pd.read_csv('results/intermediate/X_cond_resampled.csv')
    cases=pd.read_csv('results/intermediate/X_case_studies.csv')
    cfi=pd.read_csv('results/tables/conditional_feature_importance.csv').set_index('feature').loc[features]
    dif=np.load('results/intermediate/cfi_differences.npz')['differences']
    check(dif.shape==(len(indices),len(features)),'CFI loss matrix dimensions')
    for j,feature in enumerate(features):
        aligned_values(draws,feature,indices,repeats)
        stats=inference(dif[:,j],CONFIG['alpha'])
        np.testing.assert_allclose(stats['importance'],cfi.loc[feature,'conditional_importance'],atol=1e-12,equal_nan=True)
    checks.append('All feature samples aligned and CFI means agree')
    np.testing.assert_allclose(adjust_bh(cfi.p_value),cfi.q_value,atol=1e-12,equal_nan=True)
    checks.append('BH adjusted p-values agree')
    for col in cases:
        if col.endswith('|cond=all'):
            feature=col.split('|')[0]
            np.testing.assert_allclose(cases[col],draws[feature],equal_nan=True)
    checks.append('Case-study all-features draws identical to main CFI')
    X,y=model['X_test'][indices],model['y_test'][indices]
    for feature in cfi.sort_values('conditional_importance',ascending=False).head(3).index:
        j=features.index(feature)
        recalculated=evaluate(model['model'],X,y,j,aligned_values(draws,feature,indices,repeats))
        np.testing.assert_allclose(recalculated,dif[:,j],atol=1e-12,equal_nan=True)
    checks.append('Top-3 CFI independently recomputed from model and resamples')
    metrics=pd.read_csv('results/tables/model_metrics.csv').iloc[0]
    prediction=model['model'].predict(model['X_test'])
    probability=model['model'].predict_proba(model['X_test'])[:,1]
    for key,value in dict(accuracy=accuracy_score(model['y_test'],prediction),
                          f1=f1_score(model['y_test'],prediction),roc_auc=roc_auc_score(model['y_test'],probability)).items():
        check(np.isclose(metrics[key],value,rtol=0,atol=1e-12),f'{key} recomputed from saved model')
    report=Path('report/research_results.md').read_text()
    links=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report)
    check(all((ROOT/'report'/link).is_file() for link in links),'All report figures exist')
    manifest=json.loads(Path('results/intermediate/run_manifest.json').read_text())
    check(manifest['config']==CONFIG,'Report manifest uses current configuration')
    dataset_digest=hashlib.sha256((ROOT/CONFIG['data']).read_bytes()).hexdigest()
    check(manifest['dataset_sha256']==dataset_digest,'Dataset fingerprint agrees with run manifest')
    check(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in manifest['source_sha256'].items()),'Source fingerprints agree with run manifest')
    receipt=dict(status='passed',checks=checks,n_checks=len(checks),
                 coverage_min=float(cfi.coverage.min()),coverage_max=float(cfi.coverage.max()),
                 limitations='Single-seed benchmark; conditional sampler quality requires further validation.')
    Path('results/intermediate/verification.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__': main()
