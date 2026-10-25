"""Leave-one-family-out generalization study with engineered features.

Scientific question
-------------------
Do (a) derived features and (b) a stricter evaluation protocol change how well
a model detects malware families that never appear during training?

Protocol
--------
For every malware family F (15 families across Ransomware / Spyware / Trojan):
    test  = all deduplicated malware rows of F  +  the benign test split
    train = malware of the other 14 families    +  the benign train split
The benign split and the model seed are repeated over ``--seeds``. F never enters
fitting, imputation or early stopping. A random stratified holdout with the same
feature set is reported as an optimistic control.

Two feature sets are compared on identical folds:
    baseline    = the 55 raw forensic features
    engineered  = baseline + per-process ratios + forensic-group aggregates

Outputs (new, isolated from the verified main pipeline):
    results/generalization/lofo_runs.csv
    results/generalization/lofo_by_family.csv
    results/generalization/summary.csv
    results/generalization/control_random_split.csv
    results/generalization/cv_reliability.csv        (when --cv-folds > 0)
    results/generalization/lofo_f1_by_family.png
    results/generalization/lofo_summary.png
    report/generalization_results.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

from common import CONFIG, ROOT, setup
from generate_report import table

# Raw counts worth normalising by the process count. Aggregated means such as
# handles.avg_handles_per_proc already exist and are kept in the baseline set.
RATIO_BASES = [
    'dlllist.ndlls',
    'handles.nhandles', 'handles.nfile', 'handles.nevent', 'handles.nkey',
    'handles.nthread', 'handles.ndirectory', 'handles.nsemaphore',
    'handles.ntimer', 'handles.nsection', 'handles.nmutant', 'handles.nport',
    'handles.ndesktop',
    'ldrmodules.not_in_load', 'ldrmodules.not_in_init', 'ldrmodules.not_in_mem',
    'malfind.ninjections', 'malfind.commitCharge', 'malfind.uniqueInjections',
    'malfind.protection',
    'psxview.not_in_pslist', 'psxview.not_in_eprocess_pool',
    'psxview.not_in_ethread_pool', 'psxview.not_in_pspcid_list',
    'psxview.not_in_csrss_handles', 'psxview.not_in_session',
    'psxview.not_in_deskthrd',
    'modules.nmodules',
    'svcscan.nservices', 'svcscan.kernel_drivers', 'svcscan.fs_drivers',
    'svcscan.process_services', 'svcscan.shared_process_services',
    'svcscan.interactive_process_services', 'svcscan.nactive',
    'callbacks.ncallbacks', 'callbacks.nanonymous', 'callbacks.ngeneric',
]


def load_frame():
    """Load the raw CSV, parse Type/Family, deduplicate exact feature vectors."""
    df = pd.read_csv(ROOT / CONFIG['data'])
    labels = df['Class'].str.strip().str.lower()
    if not labels.isin(['benign', 'malware']).all():
        raise ValueError('Class must contain only Benign and Malware')
    parts = df['Category'].fillna('Benign').str.split('-')
    types = parts.str[0]
    families = parts.str[1].fillna('Benign')
    features = (df.drop(columns=['Class', 'Category'])
                  .select_dtypes(include='number').columns.tolist())
    X = df[features].replace([np.inf, -np.inf], np.nan)
    y = labels.eq('malware').astype(int)
    # Reject ambiguous duplicate vectors instead of arbitrarily selecting a label.
    if len(X.drop_duplicates()) != len(X.assign(__label=y).drop_duplicates()):
        raise ValueError('Identical feature vectors have conflicting labels')
    keep = ~X.duplicated()
    frame = X.loc[keep].reset_index(drop=True)
    frame['__label'] = y.loc[keep].to_numpy()
    frame['__type'] = types.loc[keep].to_numpy()
    frame['__family'] = families.loc[keep].to_numpy()
    missing = int(frame[features].isna().sum().sum())
    if missing:
        raise ValueError(f'{missing} missing values after infinity replacement')
    return frame, features


def engineer(frame, features):
    """Add per-process ratios and forensic-group aggregates to the raw features."""
    X = frame[features]
    out = X.copy()
    denom = X['pslist.nproc'].clip(lower=0) + 1.0
    for name in RATIO_BASES:
        if name in X.columns:
            out[f'ratio.{name}'] = X[name] / denom
    groups = {}
    for name in features:
        groups.setdefault(name.split('.')[0], []).append(name)
    for group, columns in groups.items():
        if len(columns) < 2:  # a single-column group would duplicate itself
            continue
        out[f'group.{group}.sum'] = X[columns].sum(axis=1)
        out[f'group.{group}.mean'] = X[columns].mean(axis=1)
    psxview = [c for c in features if c.startswith('psxview.not_in_')
               and not c.endswith('_false_avg')]
    ldrmodules = [c for c in features if c.startswith('ldrmodules.not_in_')
                  and not c.endswith('_avg')]
    malfind = [c for c in features if c.startswith('malfind.')]
    out['psxview.discrepancy_sum'] = X[psxview].sum(axis=1)
    out['ldrmodules.not_in_sum'] = X[ldrmodules].sum(axis=1)
    out['malfind.sum'] = X[malfind].sum(axis=1)
    return out


def lofo_folds(frame, seed, test_size):
    """Yield (family, train_positions, test_positions) with the family held out."""
    label = frame['__label'].to_numpy()
    type_ = frame['__type'].to_numpy()
    family = frame['__family'].to_numpy()
    benign = np.flatnonzero(type_ == 'Benign')
    malware = label == 1
    benign_train, benign_test = train_test_split(benign, test_size=test_size,
                                                 random_state=seed)
    for name in sorted(set(family[malware])):
        held = np.flatnonzero(malware & (family == name))
        rest = np.flatnonzero(malware & (family != name))
        train = np.sort(np.concatenate([benign_train, rest]))
        test = np.sort(np.concatenate([benign_test, held]))
        if np.intersect1d(train, test).size:
            raise AssertionError('LOFO train/test overlap')
        yield name, train, test


def control_split(frame, seed, test_size):
    """Random stratified holdout, the optimistic control protocol."""
    index = np.arange(len(frame))
    train, test = train_test_split(index, test_size=test_size,
                                   stratify=frame['__label'].to_numpy(),
                                   random_state=seed)
    return train, test


def fit_model(X, y, seed):
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, eval_metric='logloss',
        random_state=seed, n_jobs=CONFIG['threads'],
    )
    model.fit(X, y)
    return model


def score(model, X, y):
    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return dict(
        accuracy=accuracy_score(y, prediction),
        precision=precision_score(y, prediction, zero_division=0),
        recall=recall_score(y, prediction, zero_division=0),
        f1=f1_score(y, prediction, zero_division=0),
        roc_auc=roc_auc_score(y, probability),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        n_test=int(len(y)),
    )


def load_artifacts(out):
    """Reload saved tables so the report can be rebuilt without retraining."""
    runs = pd.read_csv(out / 'lofo_runs.csv')
    lofo = runs[runs.family != '__random_control__']
    by_family = pd.read_csv(out / 'lofo_by_family.csv', header=[0, 1], index_col=[0, 1])
    summary = pd.read_csv(out / 'summary.csv', header=[0, 1], index_col=0)
    control = pd.read_csv(out / 'control_random_split.csv', header=[0, 1], index_col=0)
    cv_path = out / 'cv_reliability.csv'
    cv = pd.read_csv(cv_path) if cv_path.exists() else None
    return runs, lofo, by_family, summary, control, cv


def run(args):
    setup()
    out = ROOT / 'results' / 'generalization'
    out.mkdir(parents=True, exist_ok=True)
    frame, features = load_frame()
    feature_sets = {
        'baseline': frame[features],
        'engineered': engineer(frame, features),
    }
    if args.report_only:
        runs, lofo, by_family, summary, control, cv = load_artifacts(out)
        figures(out, lofo, summary)
        write_report(out, frame, features, feature_sets, runs, lofo, by_family,
                     summary, control, cv, args)
        print(f'Report rebuilt from {out.relative_to(ROOT)}', flush=True)
        return
    y = frame['__label'].to_numpy()
    test_size = CONFIG['test_size']

    runs = []
    for seed in args.seeds:
        for name, X in feature_sets.items():
            for family, train, test in lofo_folds(frame, seed, test_size):
                model = fit_model(X.iloc[train], y[train], seed)
                runs.append(dict(featureset=name, family=family, seed=seed,
                                 n_features=X.shape[1],
                                 **score(model, X.iloc[test], y[test])))
            train, test = control_split(frame, seed, test_size)
            model = fit_model(X.iloc[train], y[train], seed)
            runs.append(dict(featureset=name, family='__random_control__', seed=seed,
                             n_features=X.shape[1],
                             **score(model, X.iloc[test], y[test])))
        print(f'[seed {seed}] done', flush=True)

    runs = pd.DataFrame(runs)
    runs.to_csv(out / 'lofo_runs.csv', index=False)

    metric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc',
                   'fp', 'fn', 'tn', 'tp', 'n_test']
    lofo = runs[runs.family != '__random_control__']
    by_family = (lofo.groupby(['featureset', 'family'])[metric_cols]
                     .agg(['mean', 'std']).round(6))
    by_family.to_csv(out / 'lofo_by_family.csv')

    summary = (lofo.groupby('featureset')[metric_cols]
                   .agg(['mean', 'std']).round(6))
    summary.to_csv(out / 'summary.csv')

    control = (runs[runs.family == '__random_control__']
               .groupby('featureset')[metric_cols].agg(['mean', 'std']).round(6))
    control.to_csv(out / 'control_random_split.csv')

    cv = None
    if args.cv_folds > 0:
        cv = reliability(frame, feature_sets, y, test_size, args)
        cv.to_csv(out / 'cv_reliability.csv', index=False)

    figures(out, lofo, summary)
    write_report(out, frame, features, feature_sets, runs, lofo, by_family,
                 summary, control, cv, args)
    print(summary.to_string(), flush=True)
    print(f'Generalization artifacts written to {out.relative_to(ROOT)}', flush=True)


def reliability(frame, feature_sets, y, test_size, args):
    """Within-training stratified k-fold F1 to separate model from held-out family."""
    rows = []
    seed = args.seeds[0]
    for family, train, _ in lofo_folds(frame, seed, test_size):
        for name, X in feature_sets.items():
            Xtr, ytr = X.iloc[train], y[train]
            folds = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                                    random_state=seed)
            scores = []
            for tr, va in folds.split(Xtr, ytr):
                model = fit_model(Xtr.iloc[tr], ytr[tr], seed)
                scores.append(score(model, Xtr.iloc[va], ytr[va])['f1'])
            rows.append(dict(featureset=name, family=family, seed=seed,
                             cv_f1_mean=float(np.mean(scores)),
                             cv_f1_std=float(np.std(scores, ddof=1))))
        print(f'[cv] {family} done', flush=True)
    return pd.DataFrame(rows)


def figures(out, lofo, summary):
    families = sorted(lofo.family.unique())
    pivot = (lofo.groupby(['family', 'featureset']).f1.mean().unstack())
    pivot = pivot.loc[families]
    error = (lofo.groupby(['family', 'featureset']).f1.std().unstack()).loc[families]
    fig, ax = plt.subplots(figsize=(12, 6))
    position = np.arange(len(families))
    width = 0.38
    ax.bar(position - width / 2, pivot['baseline'], width, yerr=error['baseline'],
           capsize=3, label='baseline (55)')
    ax.bar(position + width / 2, pivot['engineered'], width, yerr=error['engineered'],
           capsize=3, label='engineered')
    ax.set_xticks(position, families, rotation=45, ha='right')
    ax.set(ylabel='F1 on held-out family (mean ± SD over seeds)',
           title='Leave-one-family-out F1')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'lofo_f1_by_family.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in ('baseline', 'engineered'):
        ax.hist(lofo[lofo.featureset == name].f1, bins=12, alpha=0.6, label=name)
    ax.set(xlabel='Per-fold F1', ylabel='Fold count',
           title='Distribution of LOFO F1 across families and seeds')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'lofo_summary.png', dpi=160)
    plt.close(fig)


def write_report(out, frame, features, feature_sets, runs, lofo, by_family,
                 summary, control, cv, args):
    benign = int(frame['__type'].eq('Benign').sum())
    malware = int(frame['__label'].eq(1).sum())
    families = sorted(lofo.family.unique())
    engineered_cols = [c for c in feature_sets['engineered'].columns
                       if c not in features]
    metric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'fp', 'fn']
    summary_flat = summary[metric_cols].copy()
    summary_flat.columns = ['_'.join(c).strip('_') for c in summary_flat.columns]
    summary_flat = summary_flat.reset_index()

    family_mean = (lofo.groupby(['featureset', 'family'])[['f1', 'fn', 'fp']]
                       .mean().reset_index())
    comparison = (family_mean.pivot(index='family', columns='featureset',
                                    values='f1').loc[families].round(6))
    comparison['delta'] = (comparison['engineered'] - comparison['baseline']).round(6)
    comparison = comparison.reset_index()

    delta_overall = (summary.loc['engineered', ('f1', 'mean')]
                     - summary.loc['baseline', ('f1', 'mean')])

    lines = []
    lines.append('# Kiểm tra khả năng tổng quát hóa theo family chưa thấy\n')
    lines.append('Báo cáo được sinh bởi `python/run_generalization.py`. Đây là thí '
                 'nghiệm độc lập với pipeline chính, dùng cùng dataset và cùng thiết '
                 'lập XGBoost, nhưng đổi cách chia dữ liệu sang leave-one-family-out.\n')
    lines.append('## 1. Câu hỏi và thiết kế\n')
    lines.append(f'- Dataset sau loại trùng vector: {benign:,} benign, {malware:,} malware, '
                 f'{len(features)} feature gốc.')
    lines.append(f'- Có {len(families)} family malware: ' + ', '.join(f'`{f}`' for f in families) + '.')
    lines.append(f'- Số seed lặp: {len(args.seeds)} ({", ".join(map(str, args.seeds))}); '
                 f'k-fold CV trong tập train: {args.cv_folds if args.cv_folds else "không chạy"}.')
    lines.append(f'- Bộ feature engineered thêm {len(engineered_cols)} cột: tỉ lệ theo số tiến '
                 'trình (`ratio.*`) và tổng/trung bình theo nhóm forensic (`group.*`).\n')
    lines.append('Mỗi fold giữ trọn một family làm test, cộng phần benign test; train bằng 14 '
                 'family còn lại cộng phần benign train. Family giữ lại không tham gia huấn '
                 'luyện, không dùng để chọn feature hay ngưỡng. Đối chứng là chia stratified '
                 'ngẫu nhiên (giao thức của pipeline chính) trên cùng bộ feature.\n')
    lines.append('## 2. Tổng hợp theo bộ feature\n')
    lines.append(table(summary_flat) + '\n')
    lines.append(f'Chênh lệch F1 trung bình (engineered − baseline): **{delta_overall:+.4f}**.\n')
    lines.append('## 3. Đối chứng chia ngẫu nhiên\n')
    control_flat = control.copy()
    control_flat.columns = ['_'.join(c).strip('_') for c in control_flat.columns]
    lines.append(table(control_flat.reset_index()) + '\n')
    lines.append('Khoảng cách giữa F1 đối chứng và F1 leave-one-family-out là thước đo mức độ '
                 'lạc quan của cách chia ngẫu nhiên đối với họ chưa thấy.\n')
    lines.append('## 4. F1 theo từng family giữ lại\n')
    lines.append(table(comparison) + '\n')
    lines.append('![LOFO F1 theo family](../results/generalization/lofo_f1_by_family.png)\n')
    lines.append('![Phân bố LOFO F1](../results/generalization/lofo_summary.png)\n')
    lines.append('## 5. Chi tiết theo family và bộ feature\n')
    by_family_flat = by_family.copy()
    by_family_flat.columns = ['_'.join(c).strip('_') for c in by_family_flat.columns]
    by_family_flat = by_family_flat.reset_index()
    lines.append(table(by_family_flat) + '\n')
    if cv is not None:
        cv_summary = (cv.groupby('featureset')[['cv_f1_mean', 'cv_f1_std']]
                        .mean().round(6).reset_index())
        lines.append('## 6. Độ ổn định trong tập train (k-fold CV)\n')
        lines.append('CV chạy bên trong tập train của mỗi fold, không dùng family giữ lại. '
                     'F1 CV cao nhưng F1 LOFO thấp cho thấy vấn đề nằm ở khả năng tổng quát '
                     'hóa sang họ chưa thấy, không phải ở độ ổn định của mô hình.\n')
        lines.append(table(cv_summary) + '\n')
    worst = comparison.sort_values('baseline').head(2)
    worst_text = ', '.join(f'`{row.family}` (F1 {row.baseline:.4f})'
                           for row in worst.itertuples())
    control_f1 = float(control.loc['baseline', ('f1', 'mean')])
    lofo_f1 = float(summary.loc['baseline', ('f1', 'mean')])
    n_better = int((comparison['delta'] > 1e-6).sum())
    n_worse = int((comparison['delta'] < -1e-6).sum())
    lines.append('## 7. Kết luận và giới hạn\n')
    lines.append(f'- Đối chứng chia ngẫu nhiên đạt F1 trung bình {control_f1:.4f}, còn '
                 f'leave-one-family-out chỉ {lofo_f1:.4f}: chia ngẫu nhiên lạc quan hơn '
                 f'khoảng {control_f1 - lofo_f1:.4f} khi gặp họ chưa thấy.')
    lines.append(f'- Sai số LOFO tập trung ở {worst_text}; các family còn lại gần như hoàn '
                 'hảo. Đây là những họ khó nhận diện khi vắng mặt trong train.')
    if cv is not None:
        cv_train = float(cv[cv.featureset == 'baseline'].cv_f1_mean.mean())
        lines.append(f'- F1 k-fold bên trong tập train ({cv_train:.4f}) cao hơn F1 LOFO '
                     f'({lofo_f1:.4f}), nên khoảng cách đến từ việc tổng quát hóa sang họ '
                     'mới chứ không phải mô hình thiếu ổn định.')
    lines.append(f'- Bộ engineered {"cải thiện" if delta_overall > 0 else "không cải thiện"} '
                 f'F1 trung bình {delta_overall:+.5f} (thắng {n_better}, thua {n_worse} '
                 f'trên {len(comparison)} family), tức '
                 f'{"có" if abs(delta_overall) > 1e-3 else "không có"} lợi ích thực chất.')
    lines.append('- Exact deduplication không loại được các dump gần giống hoặc cùng nguồn; '
                 'một phần khoảng cách có thể đến từ đặc điểm nguồn thu thập.')
    lines.append('- Số seed và k-fold hữu hạn; chưa đo bất định do huấn luyện lại XGBoost '
                 'với siêu tham số khác hoặc dùng model khác.\n')
    lines.append('## 8. Sản phẩm\n')
    for name in ('lofo_runs.csv', 'lofo_by_family.csv', 'summary.csv',
                 'control_random_split.csv', 'cv_reliability.csv'):
        if (out / name).exists():
            lines.append(f'- `results/generalization/{name}`')
    (ROOT / 'report').mkdir(exist_ok=True)
    (ROOT / 'report' / 'generalization_results.md').write_text('\n'.join(lines))
    print('Generated report/generalization_results.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='42,43,44',
                        help='Comma-separated model/benign-split seeds')
    parser.add_argument('--cv-folds', type=int, default=0,
                        help='Stratified folds inside each training set (0 = skip)')
    parser.add_argument('--report-only', action='store_true',
                        help='Rebuild the report and figures from saved tables')
    args = parser.parse_args()
    args.seeds = [int(s) for s in str(args.seeds).split(',') if s.strip()]
    run(args)


if __name__ == '__main__':
    main()
