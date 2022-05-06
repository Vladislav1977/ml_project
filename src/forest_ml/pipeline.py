from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(scaler=True, log_penalty="l1", log_max_iter=1000, log_C=1.0, random_state=42):

    """Logistic Regression classifier.

            Args:
                scaler: bool, default=True
                    Data standartization.
                log_penalty: string, default="L1"
                    Type of regularization
                log_max_iter: int, default=1000
                    Maximum number of iterations taken for the solvers to converge.
                log_C: float, default=1.0
                    Inverse of regularization strength; must be a positive float.
                    Smaller values specify stronger regularization.
                random_state: int, default=42
                    Random state.
            """

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
