"""Export the actual saved split and deterministic subsets; never reconstruct a split."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from common import CONFIG, setup


def subset(y, size):
    indices = np.arange(len(y))
    if size >= len(y): return indices
    selected, _ = train_test_split(indices, train_size=size, stratify=y, random_state=CONFIG['seed'])
    return np.sort(selected)


def main():
    setup()
    split = joblib.load('results/intermediate/train_test_split.joblib')
    train_sub = subset(split['y_train'], CONFIG['arf_train_samples'])
    test_sub = subset(split['y_test'], CONFIG['importance_test_samples'])
    features = split['feature_names']
    info = dict(train_indices=split['train_indices'].tolist(), test_indices=split['test_indices'].tolist(),
                arf_train_positions=train_sub.tolist(), importance_test_positions=test_sub.tolist(),
                feature_names=features, random_state=CONFIG['seed'], indexing='zero-based',
                config=CONFIG)
    with open('results/intermediate/split_r.json','w') as f: json.dump(info,f,indent=2)
    pd.DataFrame({'sub_idx': test_sub + 1}).to_csv('results/intermediate/test_sub_indices.csv',index=False)
    pd.DataFrame(split['X_test'],columns=features).to_csv('results/intermediate/X_test_original.csv',index=False)
    pd.DataFrame({'y_test':split['y_test']}).to_csv('results/intermediate/y_test.csv',index=False)
    print(f'Exported {len(train_sub)} ARF train rows, {len(test_sub)} importance test rows')


if __name__ == '__main__': main()
