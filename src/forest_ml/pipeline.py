from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def make_model(scaler, log_max_depth, log_n_estimators, random_state=42):


    steps = []
    if scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(
            ("RandomForest", RandomForestClassifier(
                max_depth=log_max_depth, n_estimators=log_n_estimators,
                random_state=random_state)
            )
        )
    return Pipeline(steps)
