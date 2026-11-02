"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

def make_folds(train_features, train_targets, n_splits=5, shuffle=True, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return [(tr, va) for tr, va in kf.split(train_features, train_targets)]

def build_meta_features(train_features, train_targets, folds, random_seed=42):
    n_samples = train_features.shape[0]
    oof = np.zeros((n_samples, 3))
    for idx_train, idx_val in folds:
        X_tr, y_tr = train_features[idx_train], train_targets[idx_train]
        X_va = train_features[idx_val]
        for m_idx, model in enumerate(make_base_models(random_seed)):
            model.fit(X_tr, y_tr)
            oof[idx_val, m_idx] = model.predict_proba(X_va)[:, 1]
    return oof

def train_base_models(train_features, train_targets, random_seed=42):
    models = make_base_models(random_seed)
    for model in models:
        model.fit(train_features, train_targets)
    return models

def train_meta_model(oof, train_targets, random_seed=42):
    meta_model = LogisticRegression(max_iter=1000, random_state=random_seed)
    meta_model.fit(oof, train_targets)
    return meta_model


def make_base_models(random_seed=42):
    return [
        GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=random_seed,
        ),
        RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=random_seed,
        ),
        LogisticRegression(
            max_iter=1000,
            random_state=random_seed,
        ),
    ]

def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    train_features = np.asarray(data.train_features)
    train_target = np.asarray(data.train_target)
    test_features = np.asarray(data.test_features)
    test_target = np.asarray(data.test_target)
    folds = make_folds(
        train_features, train_target, n_splits=5, random_state=data.random_seed
    )
    oof = build_meta_features(
        train_features, train_target, folds, random_seed=data.random_seed
    )
    meta_model = train_meta_model(oof, train_target, random_seed=data.random_seed)
    base_models = train_base_models(
        train_features, train_target, random_seed=data.random_seed
    )
    test_meta = np.column_stack(
        [model.predict_proba(test_features)[:, 1] for model in base_models]
    )
    probabilities = meta_model.predict_proba(test_meta)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return EnsembleResult(
        model_name="Stacking GB+RF+LR -> LogReg)",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(test_target, probabilities)),
        accuracy=float(accuracy_score(test_target, predictions)),
        precision=float(precision_score(test_target, predictions)),
        recall=float(recall_score(test_target, predictions)),
        f1=float(f1_score(test_target, predictions)),
    )

def print_ensemble_result(result: EnsembleResult):
    print(f"Model: {result.model_name}")
    print(f"ROC AUC : {result.roc_auc:.4f}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall  : {result.recall:.4f}")
    print(f"F1      : {result.f1:.4f}")

