"""CPI inference from repeated conditional draws, with explicit row alignment."""
import json
import joblib
import numpy as np
import pandas as pd
from common import CONFIG, setup, logloss_rows, inference, adjust_bh


def aligned_values(frame, column, indices, repeats):
    expected = pd.MultiIndex.from_product([indices, range(1,repeats+1)],names=['test_position','replicate'])
    actual = pd.MultiIndex.from_frame(frame[['test_position','replicate']])
    if actual.has_duplicates or len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError('Conditional samples do not match test positions and replicate IDs')
    return frame.set_index(['test_position','replicate'])[column].reindex(expected).to_numpy().reshape(len(indices),repeats)


def evaluate(model, X, y, j, draws):
    baseline = logloss_rows(y,model.predict_proba(X)[:,1])
    valid = np.isfinite(draws).all(axis=1)
    differences = np.full(len(y),np.nan)
    if valid.any():
        repeated = np.repeat(X[valid], draws.shape[1], axis=0)
        repeated[:,j] = draws[valid].ravel()
        loss = logloss_rows(np.repeat(y[valid],draws.shape[1]),model.predict_proba(repeated)[:,1])
        differences[valid] = loss.reshape(valid.sum(),draws.shape[1]).mean(axis=1)-baseline[valid]
    return differences


def main():
    setup()
    saved = joblib.load('results/models/xgboost_model.joblib')
    model, features = saved['model'], saved['features']
    indices = json.load(open('results/intermediate/split_r.json'))['importance_test_positions']
    X,y = saved['X_test'][indices],saved['y_test'][indices]
    conditional = pd.read_csv('results/intermediate/X_cond_resampled.csv')
    repeats = CONFIG['conditional_repeats']
    rows, differences = [], []
    for j, feature in enumerate(features):
        draws = aligned_values(conditional,feature,indices,repeats)
        dif = evaluate(model,X,y,j,draws)
        differences.append(dif)
        stat = inference(dif,CONFIG['alpha'])
        stat['conditional_importance'] = stat.pop('importance')
        stat['conditional_importance_se'] = stat.pop('se')
        rows.append(dict(feature=feature,coverage=np.isfinite(dif).mean(),**stat))
    out = pd.DataFrame(rows)
    out['q_value'] = adjust_bh(out.p_value)
    out['conditional_rank'] = out.conditional_importance.rank(ascending=False,method='min').astype('Int64')
    out.sort_values('conditional_rank').to_csv('results/tables/conditional_feature_importance.csv',index=False)
    np.savez_compressed('results/intermediate/cfi_differences.npz',differences=np.array(differences).T,features=features)
    cases = pd.read_csv('results/intermediate/X_case_studies.csv')
    labels = {'cond=all':'All other features','cond=same_group':'Same forensic group','cond=other_groups':'Other forensic groups'}
    case_rows=[]
    for feature in dict.fromkeys(c.split('|')[0] for c in cases if '|' in c):
        cols = [c for c in cases if c.startswith(feature+'|')]
        all_dif = [evaluate(model,X,y,features.index(feature),aligned_values(cases,col,indices,repeats)) for col in cols]
        # Every conditioning set for a feature uses the exact same valid observations.
        common = np.isfinite(np.array(all_dif)).all(axis=0)
        for col,dif in zip(cols,all_dif):
            stat = inference(dif[common],CONFIG['alpha'])
            case_rows.append(dict(feature=feature,conditioning_set=labels[col.split('|')[1]],
                                  coverage=common.mean(),**stat))
    case_out=pd.DataFrame(case_rows); case_out['q_value']=adjust_bh(case_out.p_value)
    case_out.to_csv('results/tables/conditioning_case_studies.csv',index=False)
    print(out.sort_values('conditional_rank').head(10).to_string(index=False))


if __name__ == '__main__': main()
