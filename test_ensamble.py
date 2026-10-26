import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

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


