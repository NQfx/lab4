import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)

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

def get_models():
    return [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("KNN", KNeighborsClassifier()),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
        ("Random Forest", RandomForestClassifier(random_state=42)),
        ("Extra Trees", ExtraTreesClassifier(random_state=42)),
        ("Hist Gradient Boosting", HistGradientBoostingClassifier(random_state=42)),
    ]


def evaluate_models(models, train_features, train_targets, test_features, test_targets):
    results = []
    for name, model in models:
        model.fit(train_features, train_targets)
        preds = model.predict(test_features)
        proba = model.predict_proba(test_features)[:, 1]
        results.append({
            "model": name,
            "ROC-AUC": roc_auc_score(test_targets, proba),
            "Accuracy": accuracy_score(test_targets, preds),
            "Precision": precision_score(test_targets, preds),
            "Recall": recall_score(test_targets, preds),
            "F1": f1_score(test_targets, preds),
        })

    return results


def print_stat(stat):
    df = pd.DataFrame(stat).set_index("model")
    print(df.round(4).to_string())


def test_models():
    original_df = pd.read_csv('german.csv', sep=';')
    train_df, test_df = prepare_dataset(original_df, 0.02, 0.8)
    train_features, train_targets = standardize_data(train_df, 'Creditability')
    test_features, test_targets = standardize_data(test_df, 'Creditability')

    stat = evaluate_models(
        get_models(),
        train_features, train_targets,
        test_features, test_targets,
    )

    print_stat(stat)

test_models()

# Результат
#                         ROC-AUC  Accuracy  Precision  Recall      F1
# model
# Logistic Regression      0.8215    0.7944     0.8099   0.920  0.8614
# KNN                      0.7943    0.7944     0.7857   0.968  0.8674
# Decision Tree            0.5971    0.6667     0.7519   0.776  0.7638
# Gradient Boosting        0.8198    0.7778     0.7931   0.920  0.8519
# Random Forest            0.8233    0.7889     0.7959   0.936  0.8603
# Extra Trees              0.8089    0.7611     0.7628   0.952  0.8470
# Hist Gradient Boosting   0.8122    0.7611     0.7929   0.888  0.8377