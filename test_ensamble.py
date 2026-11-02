import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ensemble_models import print_ensemble_result, run_ensemble_experiment

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from grader_contracts.ensembles import EnsembleInput


def standardize_data(df, targets_column):
    scaler = StandardScaler()
    features = scaler.fit_transform(df.loc[:, df.columns != targets_column].to_numpy(dtype=np.float64))
    targets = df[targets_column].to_numpy(dtype=np.float64)
    return features, targets


def get_classifier_metrics(preds, probs, targets):
    metrics = {'accuracy': accuracy_score(targets, preds), 'precision': precision_score(targets, preds),
               'recall': recall_score(targets, preds), 'f1': f1_score(targets, preds),
               'roc_auc': roc_auc_score(targets, probs)}
    return metrics


def prepare_dataset(source_df, quantile_proc, train_set_proc, random_seed=42):
    clean_df = source_df.dropna().copy()

    low_boundary = clean_df.quantile(quantile_proc)
    high_boundary = clean_df.quantile(1 - quantile_proc)
    clean_df = clean_df[((clean_df >= low_boundary) & (clean_df <= high_boundary)).all(axis=1)]

    random_df = clean_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    train_count = int(random_df.shape[0] * train_set_proc)
    train_df = random_df[:train_count]
    test_df = random_df[train_count:random_df.shape[0]]

    return train_df, test_df

def test_ensemble():
    original_df = pd.read_csv('german.csv', sep=';')
    train_df, test_df = prepare_dataset(original_df, 0.02, 0.8)
    train_features, train_targets = standardize_data(train_df, 'Creditability')
    test_features, test_targets = standardize_data(test_df, 'Creditability')

    result = run_ensemble_experiment(EnsembleInput(
        train_features = train_features,
        train_target = train_targets,
        test_features = test_features,
        test_target = test_targets,
        random_seed = 42
    ))

    print_ensemble_result(result)

test_ensemble()

# Результат
# Model: Stacking GB+RF+LR -> LR
# ROC AUC : 0.8329
# Accuracy: 0.7889
# Precision: 0.7959
# Recall  : 0.9360
# F1      : 0.8603
