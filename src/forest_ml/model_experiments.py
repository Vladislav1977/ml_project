import mlflow
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import click
from .data import data_process
from pathlib import Path


with mlflow.start_run():
    for c in [0.1, 0.5, 0.9]:
        for selection in [-1, 0, 1]:
            main_model = LogisticRegression(C=c, max_iter=2000)
            if selection == -1:
                model = make_pipeline(StandardScaler(), main_model)
            elif selection == 0:
                model = make_pipeline(StandardScaler(),
                                      VarianceThreshold(0.2),
                                      main_model)
            else:
                selection_model = RandomForestClassifier(random_state=42)
                model = make_pipeline(StandardScaler(),
                                      SelectFromModel(selection_model),
                                      main_model)
            accuracy = cross_val_score(model, feature, target, scoring='accuracy', cv=5).mean()
            model_accuracy[model] = accuracy
            mlflow.log_param('C', c)
            mlflow.log_param('accuracy', accuracy)
            if selection != -1:
                mlflow.log_param('selection', model[1])
            mlflow.log_param('model_type', 'LogisticRegression')
            mlflow.end_run()