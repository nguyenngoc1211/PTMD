"""Full-test F1 MFI plus matched-subset log-loss MFI for comparison with CFI."""
from common import CONFIG, setup, logloss_rows, inference, adjust_bh
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def main():
    setup()
    saved = joblib.load('results/models/xgboost_model.joblib')
    model, features = saved['model'], saved['features']
    X, y = saved['X_test'], saved['y_test']
    result = permutation_importance(model,X,y,scoring='f1',n_repeats=CONFIG['permutation_repeats'],
                                    random_state=CONFIG['seed'],n_jobs=1)
    out = pd.DataFrame(dict(feature=features,marginal_importance_mean=result.importances_mean,
                            marginal_importance_std=result.importances_std))
    out['marginal_rank'] = out.marginal_importance_mean.rank(ascending=False,method='min').astype(int)
    out = out.sort_values('marginal_rank')
    out.to_csv('results/tables/marginal_feature_importance.csv',index=False)
    indices = json.load(open('results/intermediate/split_r.json'))['importance_test_positions']
    X, y = X[indices], y[indices]
    baseline = logloss_rows(y,model.predict_proba(X)[:,1])
    rng = np.random.default_rng(CONFIG['seed'])
    rows, loss_differences = [], []
    for j, feature in enumerate(features):
        differences = []
        for _ in range(CONFIG['permutation_repeats']):
            perturbed = X.copy(); perturbed[:,j] = rng.permutation(X[:,j])
            differences.append(logloss_rows(y,model.predict_proba(perturbed)[:,1])-baseline)
        mean_differences = np.mean(differences,axis=0)
        loss_differences.append(mean_differences)
        rows.append(dict(feature=feature, **inference(mean_differences,CONFIG['alpha'])))
    matched = pd.DataFrame(rows)
    matched['q_value'] = adjust_bh(matched.p_value)
    matched['rank'] = matched.importance.rank(ascending=False,method='min').astype(int)
    matched.sort_values('rank').to_csv('results/tables/marginal_logloss_importance.csv',index=False)
    np.savez_compressed('results/intermediate/mfi_logloss_differences.npz',
                        differences=np.array(loss_differences).T,features=features)
    top = out.head(10)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.barh(top.feature,top.marginal_importance_mean,xerr=top.marginal_importance_std)
    ax.invert_yaxis(); ax.set(xlabel='F1 decrease (error bars: SD over permutations)',title='Top-10 full-test marginal importance')
    fig.tight_layout(); fig.savefig('results/figures/03_marginal_importance_top10.png',dpi=160); plt.close(fig)
    print(matched.sort_values('rank').head(10).to_string(index=False))


if __name__ == '__main__': main()
