"""Shared protocol for forensic-group experiments.

Why this module exists
----------------------
The verified pipeline answers questions at the *feature* level. The research
design (``research_report/05_experimental_design.md``, experiments E2-E6) needs
questions asked at the level of the nine Volatility plugin groups. Every one of
those experiments must use the same data frame, the same deduplication rule,
the same split rule and the same model configuration, otherwise the group
comparisons are confounded.

This module is the single source of truth for:

* the nine forensic groups (``pslist`` ... ``callbacks``);
* three prediction targets (binary detection, 4-class type, 16-class family);
* deduplication and the stratified holdout, per target and seed;
* the specimen identifier parsed from ``Category`` (used by the source-aware
  sensitivity split, never as a model input);
* model factories with identical hyper-parameters for every condition;
* metric definitions and rank-agreement statistics.

Nothing here writes results. The experiment scripts own their outputs.

Protocol notes
--------------
* Hyper-parameters are identical for every ablation condition. The tuning
  budget is therefore structurally zero and equal for all conditions; no
  condition can benefit from extra model selection. This mirrors the verified
  pipeline, which never selects hyper-parameters on test.
* ``binary`` uses every deduplicated row and stratifies on the binary label,
  which reproduces the verified split bit-for-bit at seed 42.
* ``type4`` and ``family16`` drop feature-identical vectors whose family label
  conflicts, following ``run_multiclass_family.drop_family_conflicts``.
* Seeds are split seeds *and* model seeds. Runs sharing a split are not
  independent samples and must not be treated as such downstream.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, log_loss,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import CONFIG, ROOT

BENIGN = 'Benign'
GROUPS = ['pslist', 'dlllist', 'handles', 'ldrmodules', 'malfind',
          'psxview', 'modules', 'svcscan', 'callbacks']
TASKS = ['binary', 'type4', 'family16']
MODELS = ['xgboost', 'randomforest', 'histgb', 'logreg']

# ``Category`` is ``Type-Family-<file hash>-<run>.raw``. Most hashes are
# SHA-256 (64 hex chars), some are MD5 (32), and the Reconyc entries use a
# truncated 21-character prefix. We accept 20-64 hex chars and keep the raw
# string as the specimen key. This is an identifier, never a model input.
SPECIMEN_PATTERN = re.compile(
    r'^(?P<type>[^-]+)-(?P<family>[^-]+)-(?P<specimen>[0-9a-f]{20,64})-(?P<run>\d+)\.raw$')


@dataclass
class TaskData:
    """Everything an experiment needs for one prediction target."""

    name: str
    frame: pd.DataFrame
    features: list
    y: np.ndarray
    classes: list
    label_column: str
    conflicts_dropped: int = 0
    conflict_groups: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def split(self, seed: int, test_size: float | None = None, strategy: str = 'random'):
        """Stratified holdout on this task's own target.

        ``random`` is the verified pipeline's rule (stratified on rows) and stays
        the default so nothing downstream changes silently. ``source_aware``
        keeps every specimen on one side of the boundary; see ``split_manager``.
        """
        if test_size is None:
            test_size = CONFIG['test_size']
        if strategy == 'source_aware':
            from split_manager import source_aware_indices, units_for
            train, test = source_aware_indices(units_for(self), self.y,
                                               test_size, seed)
        elif strategy == 'random':
            index = np.arange(len(self.frame))
            train, test = train_test_split(index, test_size=test_size, stratify=self.y,
                                           random_state=seed)
            train, test = np.sort(train), np.sort(test)
        else:
            raise ValueError(f'unknown split strategy {strategy!r}')
        if np.intersect1d(train, test).size:
            raise AssertionError('train/test overlap')
        return train, test

    def features_of(self, group: str) -> list:
        return [f for f in self.features if f.split('.')[0] == group]

    def complement_of(self, group: str) -> list:
        return [f for f in self.features if f.split('.')[0] != group]


def group_map(features) -> dict:
    """Map group name -> ordered feature list, preserving CSV column order."""
    mapping = {}
    for name in features:
        mapping.setdefault(name.split('.')[0], []).append(name)
    unknown = sorted(set(mapping) - set(GROUPS))
    if unknown:
        raise ValueError(f'unexpected forensic groups: {unknown}')
    return {group: mapping.get(group, []) for group in GROUPS}


def parse_category(category: str):
    """Return (type, family, specimen, run) or Nones for Benign/unparsable."""
    if category == BENIGN:
        return BENIGN, BENIGN, None, None
    match = SPECIMEN_PATTERN.match(category)
    if match is None:
        return None, None, None, None
    return (match.group('type'), match.group('family'),
            match.group('specimen'), int(match.group('run')))


def load_base_frame(path=None):
    """Deduplicate exact feature vectors and attach non-predictive metadata.

    Mirrors ``preprocess.prepare`` and ``run_generalization.load_frame``:
    identical feature vectors with conflicting binary labels abort the run
    instead of being relabelled arbitrarily.
    """
    frame = pd.read_csv(path or (ROOT / CONFIG['data']))
    labels = frame['Class'].str.strip().str.lower()
    if not labels.isin(['benign', 'malware']).all():
        raise ValueError('Class must contain only Benign and Malware')
    features = (frame.drop(columns=['Class', 'Category'])
                     .select_dtypes(include='number').columns.tolist())
    X = frame[features].replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        raise ValueError('missing values after infinity replacement')
    y = labels.eq('malware').astype(int)
    if len(X.drop_duplicates()) != len(X.assign(__label=y).drop_duplicates()):
        raise ValueError('Identical feature vectors have conflicting labels')
    keep = ~X.duplicated()
    out = X.loc[keep].reset_index(drop=True)
    parsed = [parse_category(c) for c in frame.loc[keep, 'Category']]
    out['__label'] = y.loc[keep].to_numpy()
    out['__type'] = [p[0] for p in parsed]
    out['__family'] = [p[1] for p in parsed]
    out['__specimen'] = [p[2] for p in parsed]
    out['__run'] = [p[3] for p in parsed]
    out['__category'] = frame.loc[keep, 'Category'].to_numpy()
    # Feature-vector conflicts must be detected on the *raw* rows: after
    # deduplication every vector appears once, so the conflict is invisible.
    raw_families = pd.Series([parse_category(c)[1] for c in frame['Category']])
    raw_keys = pd.util.hash_pandas_object(X, index=False)
    nunique = (pd.DataFrame({'key': raw_keys, 'family': raw_families})
                 .groupby('key')['family'].nunique())
    conflicting = set(nunique[nunique > 1].index)
    out['__conflict'] = pd.util.hash_pandas_object(out[features], index=False).isin(conflicting)
    # Benign rows and 430 truncated-id rows cannot be grouped; each becomes its
    # own singleton unit so a grouped split still covers every row.
    singleton = [f'__row_{i}' for i in range(len(out))]
    out['__unit'] = [s if s is None else s for s in out['__specimen']]
    out['__unit'] = out['__unit'].where(out['__unit'].notna(),
                                        pd.Series(singleton, index=out.index))
    out.attrs['conflicting_vectors'] = len(conflicting)
    return out, features


def conflicting_vector_rows(frame, features) -> np.ndarray:
    """Boolean mask of rows whose exact feature vector spans several families.

    Computed in ``load_base_frame`` from the raw rows and stored as
    ``__conflict``; this reproduces ``run_multiclass_family.drop_family_conflicts``
    so the 16-class row count stays comparable with the saved artifacts.
    """
    if '__conflict' not in frame.columns:
        raise ValueError('frame is missing the __conflict column')
    return frame['__conflict'].to_numpy(), int(frame.attrs.get('conflicting_vectors', 0))


def conflicting_specimens(frame) -> set:
    """Specimen hashes whose rows carry more than one family label.

    Reported by the provenance audit only; it is a broader notion than the
    feature-vector conflict above because one specimen contributes many runs.
    """
    malware = frame[frame['__label'] == 1]
    known = malware[malware['__specimen'].notna()]
    nunique = known.groupby('__specimen')['__family'].nunique()
    return set(nunique[nunique > 1].index)


def build_task(frame, features, task: str) -> TaskData:
    """Slice the base frame into one prediction target."""
    if task == 'binary':
        data = frame.copy()
        classes = [BENIGN, 'Malware']
        column = '__label'
        conflicts = 0
        groups = 0
    elif task in ('type4', 'family16'):
        drop, n_conflict_groups = conflicting_vector_rows(frame, features)
        data = frame.loc[~drop].reset_index(drop=True)
        conflicts = int(drop.sum())
        groups = n_conflict_groups
        column = '__type' if task == 'type4' else '__family'
        classes = sorted(data[column].unique().tolist())
        expected = 4 if task == 'type4' else 16
        if len(classes) != expected:
            raise ValueError(f'{task} expected {expected} classes, got {len(classes)}')
    else:
        raise ValueError(f'unknown task {task!r}')
    if task == 'binary':
        y = data['__label'].to_numpy().astype(int)
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError('binary: labels outside {0, 1}')
    else:
        codes = {name: i for i, name in enumerate(classes)}
        mapped = data[column].map(codes)
        if mapped.isna().any():
            raise ValueError(f'{task}: unmapped label')
        y = mapped.to_numpy().astype(int)
    return TaskData(name=task, frame=data, features=list(features), y=y,
                    classes=classes, label_column=column,
                    conflicts_dropped=conflicts, conflict_groups=groups)


def load_tasks(tasks=None) -> dict:
    frame, features = load_base_frame()
    return {task: build_task(frame, features, task) for task in (tasks or TASKS)}


def make_model(name: str, task: TaskData, seed: int):
    """Identical hyper-parameters for every ablation condition."""
    threads = CONFIG['threads']
    if name == 'xgboost':
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85, eval_metric='logloss',
            tree_method='hist', random_state=seed, n_jobs=threads,
        )
    if name == 'randomforest':
        return RandomForestClassifier(n_estimators=300, random_state=seed,
                                      n_jobs=threads)
    if name == 'histgb':
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=6,
            random_state=seed, early_stopping=False,
        )
    if name == 'logreg':
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed),
        )
    raise ValueError(f'unknown model {name!r}')


def score(model, X, y, task: TaskData) -> dict:
    """Metrics with a sensitive primary: log-loss for all three targets.

    Binary F1 is saturated at 1.0 on this benchmark, so it cannot rank
    ablation conditions; log-loss and PR-AUC remain informative.
    """
    labels = list(range(task.n_classes))
    probability = model.predict_proba(X)
    if task.n_classes == 2:
        probability = probability[:, 1]
        prediction = (probability >= 0.5).astype(int)
        recall = recall_score(y, prediction, labels=[0, 1], average=None,
                              zero_division=0)
        return dict(
            log_loss=float(log_loss(y, probability, labels=[0, 1])),
            accuracy=float(accuracy_score(y, prediction)),
            f1=float(f1_score(y, prediction, zero_division=0)),
            roc_auc=float(roc_auc_score(y, probability)),
            pr_auc=float(average_precision_score(y, probability)),
            per_class_recall=json.dumps({name: round(float(value), 4)
                                         for name, value in zip(task.classes, recall)}),
        )
    prediction = probability.argmax(axis=1)
    recall = recall_score(y, prediction, labels=labels, average=None, zero_division=0)
    # Macro one-vs-rest AUCs, computed only over classes present in y so a
    # small held-out class cannot turn the whole metric into an exception.
    present = [k for k in labels if np.any(y == k)]
    roc_auc = float(np.mean([roc_auc_score((y == k).astype(int), probability[:, k])
                             for k in present]))
    pr_auc = float(np.mean([average_precision_score((y == k).astype(int),
                                                    probability[:, k])
                            for k in present]))
    return dict(
        log_loss=float(log_loss(y, probability, labels=labels)),
        accuracy=float(accuracy_score(y, prediction)),
        macro_f1=float(f1_score(y, prediction, labels=labels, average='macro',
                                zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
        per_class_recall=json.dumps({name: round(float(value), 4)
                                     for name, value in zip(task.classes, recall)}),
    )


# Direction of improvement for each metric; used so that "importance" is always
# "how much worse the model gets when the group is removed".
METRIC_DIRECTION = {'log_loss': -1, 'accuracy': +1, 'f1': +1, 'macro_f1': +1,
                    'roc_auc': +1, 'pr_auc': +1, 'balanced_accuracy': +1}


def degradation(full: dict, ablated: dict, metric: str = 'log_loss') -> float:
    """Positive value means the ablation made the model worse."""
    direction = METRIC_DIRECTION[metric]
    return direction * (full[metric] - ablated[metric])


def rank_values(values, features_or_groups) -> pd.Series:
    """Rank descending (1 = most important)."""
    return (pd.Series(np.asarray(values, dtype=float), index=list(features_or_groups))
              .rank(ascending=False, method='min'))


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rho on the common index, NaN if fewer than three items."""
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return float('nan')
    return float(a.loc[common].corr(b.loc[common], method='spearman'))


def jaccard_topk(a: pd.Series, b: pd.Series, k: int) -> float:
    """Jaccard overlap of the top-k sets, ties broken by index order."""
    top_a = set(a.sort_values(ascending=False, kind='mergesort').head(k).index)
    top_b = set(b.sort_values(ascending=False, kind='mergesort').head(k).index)
    union = top_a | top_b
    return float(len(top_a & top_b) / len(union)) if union else float('nan')


def random_feature_sets(features, size: int, n_draws: int, seed: int,
                        exclude=()) -> list:
    """Matched-size random feature subsets drawn outside ``exclude``.

    The null control for "drop these k features" is "drop k *other* features",
    so the control never partially ablates the group under test.
    """
    pool = [f for f in features if f not in set(exclude)]
    if size > len(pool):
        raise ValueError(f'cannot draw {size} features from {len(pool)}')
    rng = np.random.default_rng(seed)
    return [sorted(rng.choice(pool, size=size, replace=False).tolist())
            for _ in range(n_draws)]
