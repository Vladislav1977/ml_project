from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(scaler=True, log_penalty="l1", log_max_iter=1000, log_C=1.0, random_state=42):

    steps = []
    if scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(
        ("regresion", LogisticRegression(
            penalty=log_penalty, C=log_C, max_iter=log_max_iter, random_state=random_state
        )
         )
    )
    return Pipeline(steps)
