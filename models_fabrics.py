from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

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
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_seed,
        ),
        ExtraTreesClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_seed,
        ),
    ]
