"""Validate data, remove exact duplicate feature vectors, split, fit imputation."""
from common import CONFIG, ROOT, setup
import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def prepare(df, seed=42, test_size=.2):
    labels = df['Class'].str.strip().str.lower()
    if not labels.isin(['benign', 'malware']).all():
        raise ValueError('Class must contain only Benign and Malware')
    features = df.drop(columns=['Class', 'Category']).select_dtypes(include='number').columns.tolist()
    X = df[features].replace([np.inf, -np.inf], np.nan)
    y = labels.eq('malware').astype(int)
    # Reject ambiguous duplicate vectors instead of arbitrarily selecting a label.
    if len(X.drop_duplicates()) != len(X.assign(__label=y).drop_duplicates()):
        raise ValueError('Identical feature vectors have conflicting labels')
    keep = ~X.duplicated()
    source_rows = np.flatnonzero(keep)
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].to_numpy()
    train, test = train_test_split(np.arange(len(X)), stratify=y,
                                   test_size=test_size, random_state=seed)
    medians = X.iloc[train].median()
    if medians.isna().any():
        raise ValueError('A feature is entirely missing in training data')
    missing = int(X.isna().sum().sum())
    X = X.fillna(medians)
    groups = {}
    for name in features:
        groups.setdefault(name.split('.')[0], []).append(name)
    hashes = pd.util.hash_pandas_object(X, index=False)
    overlap = int(hashes.iloc[test].isin(set(hashes.iloc[train])).sum())
    if overlap:
        raise ValueError('Imputation produced feature duplicates across train/test')
    split = dict(X_train=X.iloc[train].to_numpy(), X_test=X.iloc[test].to_numpy(),
                 y_train=y[train], y_test=y[test], feature_names=features,
                 feature_groups=groups, train_indices=train, test_indices=test,
                 source_rows=source_rows, imputation_medians=medians.to_dict(),
                 constant_features=X.iloc[train].columns[X.iloc[train].nunique() == 1].tolist())
    profile = dict(samples_raw=len(df), samples_total=len(X), samples_train=len(train),
                   samples_test=len(test), duplicates_removed=int((~keep).sum()),
                   train_test_duplicate_overlap=overlap, missing_values_before_imputation=missing,
                   features_numeric=len(features), feature_groups={k: len(v) for k,v in groups.items()},
                   constant_features=split['constant_features'],
                   class_counts={str(i): int((y == i).sum()) for i in (0,1)},
                   seed=seed, split_strategy='deduplicate feature vectors, then stratified holdout')
    return split, profile, X.assign(label=y)


def main():
    setup()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=CONFIG['data'])
    args = parser.parse_args()
    path = ROOT / args.data
    split, profile, full = prepare(pd.read_csv(path), CONFIG['seed'], CONFIG['test_size'])
    profile.update(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    joblib.dump(split, 'results/intermediate/train_test_split.joblib')
    full.to_csv('results/intermediate/full_data.csv', index=False)
    full.drop(columns='label').describe().T.to_csv('results/tables/descriptive_stats.csv')
    with open('results/intermediate/data_profile.json','w') as f:
        json.dump(profile, f, indent=2)
    fig, ax = plt.subplots(figsize=(6,4))
    counts = [profile['class_counts'][str(i)] for i in (0,1)]
    ax.bar(['Benign','Malware'], counts, color=['#279b69','#df6255'])
    for i, n in enumerate(counts): ax.text(i,n,str(n),ha='center',va='bottom')
    ax.set(ylabel='Samples',title='Class distribution after deduplication')
    fig.tight_layout(); fig.savefig('results/figures/01_class_distribution.png',dpi=160); plt.close(fig)
    print(json.dumps(profile, indent=2))


if __name__ == '__main__':
    main()
