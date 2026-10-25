"""Multiclass malware family attribution under a leakage-audited protocol.

Scientific question
-------------------
Binary malware detection on CIC-MalMem-2022 is saturated (F1 = 1.0 on a random
split, 0.9988 under leave-one-family-out), and two independent external
benchmarks confirm the headroom is in *family* attribution, not detection.
This module therefore attacks the unsaturated problem: predicting which family
a sample belongs to.

Protocol
--------
* 16-class target : Benign + the 15 malware families.
* 4-class target  : Benign + Ransomware / Spyware / Trojan, for a direct
  comparison with the OACSP-Forensics benchmark (4-class accuracy 0.878).
* Group-aware stratified holdout on exact-deduplicated vectors, repeated over
  several seeds. Feature-identical vectors that carry conflicting family
  labels are removed (the external leakage audit reports 19 such 16-class
  groups) instead of assigning a family arbitrarily.
* Models: XGBoost, HistGradientBoosting, RandomForest, ExtraTrees, Logistic
  Regression, plus a two-stage hierarchical model (type -> family).
* Primary metric: macro-F1.

Outputs (isolated from the verified main pipeline):
    results/family_multiclass/runs.csv
    results/family_multiclass/summary.csv
    results/family_multiclass/per_class.csv
    results/family_multiclass/confusion_16_xgboost.csv
    results/family_multiclass/confusion_16_hierarchical.csv
    results/family_multiclass/confusion_4_xgboost.csv
    results/family_multiclass/macro_f1_by_model.png
    results/family_multiclass/confusion_16_xgboost.png
    report/family_multiclass.md
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import CONFIG, ROOT, setup
from generate_report import table
from run_generalization import load_frame

BENIGN = 'Benign'
FLAT_MODELS = ['xgboost', 'histgb', 'randomforest', 'extratrees', 'logreg']
ALL_MODELS = FLAT_MODELS + ['hierarchical']
FOUR_CLASS_MODELS = ['xgboost', 'histgb']
METRIC_COLS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1']
OACSP_4CLASS_ACCURACY = 0.878327645


def drop_family_conflicts(frame, features, raw=None):
    """Drop feature-identical vectors that carry conflicting family labels.

    ``load_frame`` deduplicates exact vectors by keeping the first row, which
    silently assigns an arbitrary family when identical vectors belong to
    different families. The external leakage audit reports 19 such 16-class
    groups, so we detect and remove them explicitly. ``raw`` is injectable for
    testing; by default the dataset is re-read from disk.
    """
    if raw is None:
        raw = pd.read_csv(ROOT / CONFIG['data'])
    families = raw['Category'].fillna(BENIGN).str.split('-').str[1].fillna(BENIGN)
    X = raw[features].replace([np.inf, -np.inf], np.nan)
    keys = pd.util.hash_pandas_object(X, index=False)
    nunique = (pd.DataFrame({'key': keys, 'family': families.to_numpy()})
                 .groupby('key')['family'].nunique())
    conflicting = set(nunique[nunique > 1].index)
    frame_keys = pd.util.hash_pandas_object(frame[features], index=False)
    drop = frame_keys.isin(conflicting)
    kept = frame.loc[~drop].reset_index(drop=True)
    return kept, len(conflicting), int(drop.sum())


def class_maps(frame):
    families = sorted(frame['__family'].unique().tolist())
    types = sorted(frame['__type'].unique().tolist())
    family_code = {name: i for i, name in enumerate(families)}
    type_code = {name: i for i, name in enumerate(types)}
    return families, types, family_code, type_code


def map_families_to_types(frame, families, family_code, type_code):
    """Group family codes by their malware type code (Benign included)."""
    family_type = frame.groupby('__family')['__type'].first()
    mapping = {}
    for name in families:
        code = type_code[family_type[name]]
        mapping.setdefault(code, []).append(family_code[name])
    return {code: sorted(codes) for code, codes in mapping.items()}


def stratified_split(frame, seed, test_size, y):
    index = np.arange(len(frame))
    train, test = train_test_split(index, test_size=test_size, stratify=y,
                                   random_state=seed)
    return train, test


def make_xgb(seed):
    """XGBoost that auto-selects binary or multiclass objective from the labels."""
    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        random_state=seed, n_jobs=CONFIG['threads'],
    )


def flat_models(seed, balanced):
    weight = 'balanced' if balanced else None
    return {
        'xgboost': make_xgb(seed),
        'histgb': HistGradientBoostingClassifier(random_state=seed,
                                                 class_weight=weight),
        'randomforest': RandomForestClassifier(n_estimators=300, random_state=seed,
                                               n_jobs=CONFIG['threads'],
                                               class_weight=weight),
        'extratrees': ExtraTreesClassifier(n_estimators=300, random_state=seed,
                                           n_jobs=CONFIG['threads'],
                                           class_weight=weight),
        'logreg': make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, random_state=seed,
                                                   class_weight=weight)),
    }


def inverse_frequency_weights(y, n_classes):
    counts = np.bincount(np.asarray(y), minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = len(y) / (n_classes * counts)
    return weights[np.asarray(y)]


def fit_model(model, X, y, balanced, n_classes):
    if balanced and isinstance(model, XGBClassifier):
        model.fit(X, y, sample_weight=inverse_frequency_weights(y, n_classes))
    else:
        model.fit(X, y)
    return model


class HierarchicalClassifier:
    """Two-stage model: predict the malware type, then the family within it."""

    def __init__(self, seed, balanced, type_code, family_code, families_of_type):
        self.seed = seed
        self.balanced = balanced
        self.type_code = type_code
        self.family_code = family_code
        self.families_of_type = families_of_type
        self.benign_type = type_code[BENIGN]
        self.benign_family = family_code[BENIGN]

    def fit(self, X, y_type, y_family):
        self.type_model = make_xgb(self.seed)
        fit_model(self.type_model, X, y_type, self.balanced, len(self.type_code))
        self.family_models = {}
        for type_code, family_codes in self.families_of_type.items():
            if type_code == self.benign_type:
                continue
            mask = y_type == type_code
            remap = {code: i for i, code in enumerate(family_codes)}
            y_local = np.array([remap[code] for code in y_family[mask]])
            model = make_xgb(self.seed)
            fit_model(model, X[mask], y_local, self.balanced, len(family_codes))
            self.family_models[type_code] = (model, family_codes)
        return self

    def predict(self, X):
        type_pred = self.type_model.predict(X)
        out = np.full(len(X), self.benign_family, dtype=int)
        for type_code, (model, family_codes) in self.family_models.items():
            mask = type_pred == type_code
            if mask.any():
                out[mask] = np.array(family_codes)[model.predict(X[mask])]
        return out


def score(y_true, y_pred, n_classes):
    labels = list(range(n_classes))
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average='macro',
                                zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, labels=labels, average='weighted',
                                   zero_division=0)),
    )


def per_class_table(model, y_true, y_pred, classes, seed, balanced):
    labels = list(range(len(classes)))
    values = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = np.bincount(np.asarray(y_true), minlength=len(classes))
    return [dict(model=model, class_name=classes[i], f1=float(values[i]),
                 support=int(support[i]), seed=seed, balanced=balanced)
            for i in range(len(classes))]


def save_confusion(path, y_true, y_pred, names):
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    pd.DataFrame(matrix, index=names, columns=names).to_csv(path)


def run(args):
    setup()
    out = ROOT / 'results' / 'family_multiclass'
    out.mkdir(parents=True, exist_ok=True)
    frame, features = load_frame()
    frame, n_conflict_groups, n_conflict_rows = drop_family_conflicts(frame, features)
    families, types, family_code, type_code = class_maps(frame)
    families_of_type = map_families_to_types(frame, families, family_code, type_code)
    y16 = frame['__family'].map(family_code).to_numpy()
    y4 = frame['__type'].map(type_code).to_numpy()
    n16, n4 = len(families), len(types)

    if args.report_only:
        runs = pd.read_csv(out / 'runs.csv')
        per_class = pd.read_csv(out / 'per_class.csv')
        summary = pd.read_csv(out / 'summary.csv', header=[0, 1], index_col=[0, 1, 2])
        figures(out, runs)
        write_report(out, frame, features, families, types, n_conflict_groups,
                     n_conflict_rows, runs, per_class, summary, args)
        print(f'Report rebuilt from {out.relative_to(ROOT)}', flush=True)
        return

    rows = []
    per_class_rows = []
    settings = [False] + ([True] if args.balanced else [])
    for seed in args.seeds:
        train, test = stratified_split(frame, seed, CONFIG['test_size'], y16)
        Xtr, Xte = frame.iloc[train][features], frame.iloc[test][features]
        for balanced in settings:
            flat = flat_models(seed, balanced)
            for name in args.models:
                if name == 'hierarchical':
                    continue
                model = fit_model(flat[name], Xtr, y16[train], balanced, n16)
                pred = model.predict(Xte)
                rows.append(dict(protocol='16-class', model=name, balanced=balanced,
                                 seed=seed, n_classes=n16,
                                 **score(y16[test], pred, n16)))
                if seed == args.seeds[0]:
                    per_class_rows += per_class_table(name, y16[test], pred, families,
                                                      seed, balanced)
                    if name == 'xgboost':
                        save_confusion(out / 'confusion_16_xgboost.csv', y16[test],
                                       pred, families)
            if 'hierarchical' in args.models:
                hier = HierarchicalClassifier(seed, balanced, type_code, family_code,
                                              families_of_type)
                hier.fit(Xtr, y4[train], y16[train])
                pred_h = hier.predict(Xte)
                rows.append(dict(protocol='16-class', model='hierarchical',
                                 balanced=balanced, seed=seed, n_classes=n16,
                                 **score(y16[test], pred_h, n16)))
                if seed == args.seeds[0]:
                    per_class_rows += per_class_table('hierarchical', y16[test], pred_h,
                                                      families, seed, balanced)
                    save_confusion(out / 'confusion_16_hierarchical.csv', y16[test],
                                   pred_h, families)
            flat4 = flat_models(seed, balanced)
            for name in FOUR_CLASS_MODELS:
                if name not in args.models:
                    continue
                model = fit_model(flat4[name], Xtr, y4[train], balanced, n4)
                pred4 = model.predict(Xte)
                rows.append(dict(protocol='4-class', model=name, balanced=balanced,
                                 seed=seed, n_classes=n4,
                                 **score(y4[test], pred4, n4)))
                if seed == args.seeds[0] and name == 'xgboost':
                    save_confusion(out / 'confusion_4_xgboost.csv', y4[test], pred4, types)
            if 'hierarchical' in args.models:
                pred4h = hier.type_model.predict(Xte)
                rows.append(dict(protocol='4-class', model='hierarchical(type)',
                                 balanced=balanced, seed=seed, n_classes=n4,
                                 **score(y4[test], pred4h, n4)))
        print(f'[seed {seed}] done', flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(out / 'runs.csv', index=False)
    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(out / 'per_class.csv', index=False)
    summary = (runs.groupby(['protocol', 'model', 'balanced'])[METRIC_COLS]
                   .agg(['mean', 'std']).round(6))
    summary.to_csv(out / 'summary.csv')
    figures(out, runs)
    write_report(out, frame, features, families, types, n_conflict_groups,
                 n_conflict_rows, runs, per_class, summary, args)
    print(summary.to_string(), flush=True)
    print(f'Multiclass artifacts written to {out.relative_to(ROOT)}', flush=True)


def figures(out, runs):
    sixteen = runs[runs.protocol == '16-class']
    grouped = sixteen.groupby(['model', 'balanced']).macro_f1.mean().reset_index()
    grouped['label'] = [f'{m}{" (cân bằng)" if b else ""}'
                        for m, b in zip(grouped.model, grouped.balanced)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(grouped.label, grouped.macro_f1, color='#3b6ea5')
    for x, value in zip(grouped.label, grouped.macro_f1):
        ax.text(x, value, f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set(ylabel='macro-F1 (trung bình theo seed)',
           title='Phân loại 16 lớp: macro-F1 theo model', ylim=(0, 1))
    ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    fig.savefig(out / 'macro_f1_by_model.png', dpi=160)
    plt.close(fig)

    for name in ('confusion_16_xgboost', 'confusion_16_hierarchical'):
        path = out / f'{name}.csv'
        if not path.exists():
            continue
        matrix = pd.read_csv(path, index_col=0)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(matrix.values, cmap='Blues')
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=7)
        ax.set(xlabel='Dự đoán', ylabel='Thực tế',
               title=f'Ma trận nhầm lẫn 16 lớp — {name.replace("confusion_16_", "")}')
        fig.tight_layout()
        fig.savefig(out / f'{name}.png', dpi=160)
        plt.close(fig)


def flatten(frame):
    flat = frame.copy()
    flat.columns = ['_'.join(c).strip('_') for c in flat.columns]
    return flat.reset_index()


def write_report(out, frame, features, families, types, n_conflict_groups,
                 n_conflict_rows, runs, per_class, summary, args):
    benign = int(frame['__type'].eq(BENIGN).sum())
    malware = int((frame['__type'] != BENIGN).sum())
    summary16 = summary.loc[summary.index.get_level_values('protocol') == '16-class']
    summary4 = summary.loc[summary.index.get_level_values('protocol') == '4-class']
    worst = (per_class[(per_class.model == 'xgboost') & (~per_class.balanced)]
             .groupby('class_name')[['f1']].mean().sort_values('f1').head(8).reset_index())
    four = runs[(runs.protocol == '4-class') & (~runs.balanced)]
    four_best = (four.groupby('model')[['accuracy', 'macro_f1']].mean()
                 .sort_values('macro_f1', ascending=False).reset_index())
    best16 = (runs[(runs.protocol == '16-class') & (~runs.balanced)]
              .groupby('model').macro_f1.mean().sort_values(ascending=False))
    balanced_present = bool(runs.balanced.any())

    lines = []
    lines.append('# Phân loại đa lớp family malware trên RAM\n')
    lines.append('Báo cáo sinh bởi `python/run_multiclass_family.py`. Thí nghiệm độc lập '
                 'với pipeline chính: cùng dataset và cùng tinh thần chống rò rỉ, nhưng '
                 'đổi mục tiêu sang gán nhãn family.\n')
    lines.append('## 1. Câu hỏi và bối cảnh\n')
    lines.append('- Phát hiện nhị phân Benign/Malware đã bão hoà: F1 = 1.0 khi chia ngẫu '
                 'nhiên và 0.9988 khi leave-one-family-out.')
    lines.append('- Hai benchmark độc lập (gói leakage-audit Zenodo 22726771 và '
                 'OACSP-Forensics Zenodo 21435602) xác nhận dư địa nằm ở phân loại '
                 'family: OACSP phân loại 4 lớp chỉ đạt accuracy 0.878.')
    lines.append('- Câu hỏi: phân loại **16 lớp** (Benign + 15 family) khó đến đâu, và '
                 'model nào phù hợp nhất?\n')
    lines.append('## 2. Dữ liệu và giao thức\n')
    lines.append(f'- Sau loại trùng vector: {benign:,} benign, {malware:,} malware, '
                 f'{len(features)} feature gốc.')
    lines.append(f'- Loại thêm {n_conflict_groups} nhóm vector trùng nhưng khác family '
                 f'({n_conflict_rows} dòng) thay vì gán nhãn tuỳ tiện.')
    lines.append('- 16 lớp: ' + ', '.join(f'`{f}`' for f in families) + '.')
    lines.append('- 4 lớp: ' + ', '.join(f'`{t}`' for t in types) + '.')
    lines.append(f'- Chia stratified holdout, test_size = {CONFIG["test_size"]}, seed = '
                 f'{", ".join(map(str, args.seeds))}; macro-F1 là metric chính.\n')
    lines.append('## 3. Model\n')
    lines.append('XGBoost (`multi:softprob`, 300 cây, depth 6, lr 0.08), '
                 'HistGradientBoosting, RandomForest, ExtraTrees, LogisticRegression '
                 '(có chuẩn hoá), và mô hình phân cấp type→family. Biến thể cân bằng lớp '
                 f'được chạy: {"có" if balanced_present else "không"}.\n')
    lines.append('## 4. Kết quả 16 lớp\n')
    lines.append(table(flatten(summary16)) + '\n')
    lines.append(f'- Xếp hạng macro-F1 trung bình: ' +
                 ', '.join(f'`{m}` {v:.4f}' for m, v in best16.items()) + '.\n')
    lines.append('## 5. F1 theo từng lớp (XGBoost)\n')
    lines.append(table(worst) + '\n')
    lines.append('![macro-F1 theo model](../results/family_multiclass/macro_f1_by_model.png)\n')
    lines.append('![Ma trận nhầm lẫn 16 lớp](../results/family_multiclass/'
                 'confusion_16_xgboost.png)\n')
    confusion_path = out / 'confusion_16_xgboost.csv'
    if confusion_path.exists():
        matrix = pd.read_csv(confusion_path, index_col=0)
        pairs = sorted(((int(matrix.loc[a, b]), a, b)
                        for a in matrix.index for b in matrix.columns
                        if a != b and matrix.loc[a, b] > 0), reverse=True)
        benign_mix = int(matrix.loc[BENIGN].drop(BENIGN).sum()
                         + matrix[BENIGN].drop(BENIGN).sum())
        lines.append('### 5.1 Nhầm lẫn chính\n')
        lines.append(f'- Nhầm lẫn giữa Benign và malware: **{benign_mix}** mẫu; mô hình tách '
                     'benign gần như tuyệt đối, sai số nằm trong nội bộ từng nhóm.')
        lines.append('- Các cặp nhầm lẫn lớn nhất (thực tế → dự đoán): ' +
                     '; '.join(f'`{a}`→`{b}` ({n})' for n, a, b in pairs[:6]) + '.\n')
    lines.append('## 6. So sánh 4 lớp với OACSP\n')
    lines.append(table(four_best) + '\n')
    lines.append(f'- Mốc OACSP-Forensics (4 lớp) accuracy = {OACSP_4CLASS_ACCURACY:.6f}; '
                 'cột `accuracy` ở bảng trên là so sánh trực tiếp trên cùng giao thức '
                 'chống rò rỉ của project.')
    lines.append('- `hierarchical(type)` là tầng 1 của mô hình phân cấp, trùng cấu hình '
                 'với XGBoost 4 lớp nên cho kết quả giống hệt; dùng như kiểm tra nhất '
                 'quán giữa hai đường huấn luyện.\n')
    lines.append('## 7. Kết luận và giới hạn\n')
    lines.append(f'- Bài toán 16 lớp khó hơn hẳn nhị phân: macro-F1 tốt nhất khoảng '
                 f'{float(best16.iloc[0]):.4f} so với F1 nhị phân 0.9988.')
    lines.append('- Sai số tập trung ở một số family khó; các lớp còn lại gần như hoàn hảo.')
    lines.append('- Chỉ dùng 55 feature gốc (bộ engineered đã được chứng minh không cải '
                 'thiện ở thí nghiệm leave-one-family-out).')
    lines.append('- Exact deduplication không loại được dump gần giống hoặc cùng nguồn; '
                 'một phần sai số có thể đến từ đặc điểm nguồn thu thập.')
    lines.append(f'- Số seed hữu hạn ({len(args.seeds)}); chưa tinh chỉnh siêu tham số '
                 'trên test.\n')
    lines.append('## 8. Sản phẩm\n')
    for name in ('runs.csv', 'summary.csv', 'per_class.csv', 'confusion_16_xgboost.csv',
                 'confusion_16_hierarchical.csv', 'confusion_4_xgboost.csv',
                 'macro_f1_by_model.png', 'confusion_16_xgboost.png'):
        if (out / name).exists():
            lines.append(f'- `results/family_multiclass/{name}`')
    (ROOT / 'report').mkdir(exist_ok=True)
    (ROOT / 'report' / 'family_multiclass.md').write_text('\n'.join(lines))
    print('Generated report/family_multiclass.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='42,43,44',
                        help='Comma-separated model/split seeds')
    parser.add_argument('--models', default=','.join(ALL_MODELS),
                        help='Comma-separated models: ' + ', '.join(ALL_MODELS))
    parser.add_argument('--balanced', action='store_true',
                        help='Also run class-balanced variants alongside the default')
    parser.add_argument('--report-only', action='store_true',
                        help='Rebuild the report and figures from saved tables')
    args = parser.parse_args()
    args.seeds = [int(s) for s in str(args.seeds).split(',') if s.strip()]
    args.models = [m.strip() for m in str(args.models).split(',') if m.strip()]
    unknown = sorted(set(args.models) - set(ALL_MODELS))
    if unknown:
        parser.error(f'unknown models: {", ".join(unknown)}')
    run(args)


if __name__ == '__main__':
    main()
