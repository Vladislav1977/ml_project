from pathlib import Path
from joblib import dump
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV, cross_val_score
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import click
from sklearn.metrics import accuracy_score, precision_score, f1_score

from .data import data_process
from .pipeline import make_model

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path)
)

@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)

@click.option(
    "--val-size",
    default=0.3,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)

@click.option(
    "--scaler",
    default=True,
    type=bool,
    show_default=True,
)

@click.option(
    "--log-max-depth",
    default=100,
    type=int,
    show_default=True,
)

@click.option(
    "--log-n-estimators",
    default=300,
    type=int,
    show_default=True,
)

@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)

def train(
        dataset_path,
        scaler,
        random_state,
        val_size,
        log_max_depth,
        log_n_estimators,
        save_model_path
):

    x_train, x_test, y_train, y_test = data_process(
        dataset_path).split(random_state, val_size)
    pipeline = make_model(log_max_depth, log_n_estimators, random_state)
    pipeline.fit(x_train, y_train)
    cv = cross_validate(pipeline,
                        x_train, y_train, cv=5,
                        return_estimator=True,
                        scoring=['accuracy', "f1_weighted"])
    accuracy = cv["test_accuracy"].mean()
    f1_weighted = cv["test_f1_weighted"].mean()
    click.echo('Accuracy %.2f%% (average over CV test folds)' %
               (100 * accuracy))
    click.echo('f1 %.2f%% (average over CV test folds)' %
               (100 * f1_weighted))
    test_accuracy = accuracy_score(y_test, pipeline.predict(x_test))
    click.echo('Test Accuracy: %.2f%%' % (100 * test_accuracy))
    dump(pipeline, save_model_path)



@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path))

@click.option(
    '--model-type',
    type=click.Choice(['LogisticRegression', 'RandomForest'],
    case_sensitive=False),
    default="RandomForest")

def model_tuning(model_type, dataset_path):
    x_train, x_test, y_train, y_test = data_process(
        dataset_path).split()

    if model_type == "LogisticRegression":
        clf = LogisticRegression(max_iter=2000)
        pipe = Pipeline([('std', StandardScaler()),
                          ('clf', clf)])
        param_grid = [{'clf__C': np.power(10., np.arange(-4, 4))}]
    elif model_type == "RandomForest":
        clf = RandomForestClassifier(random_state=1)
        pipe = Pipeline([('std', StandardScaler()),
                         ('clf', clf)])
        param_grid = [{'clf__max_depth': list(range(100, 200, 10)) + [None],
                        'clf__n_estimators': [50, 100, 200, 300]}]

    gridcvs = {}
    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)

    pgrid, est, name = (param_grid, pipe, model_type)
    gcv = GridSearchCV(estimator=est,
                        param_grid=pgrid,
                        scoring='accuracy',
                        n_jobs=-1,
                        cv=inner_cv,
                        verbose=0,
                        refit=True)
    gridcvs[name] = gcv

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    nested_score_dict = {}

    for name, gs_est in gridcvs.items():
        nested_score = cross_val_score(gs_est,
                                   X=x_train,
                                   y=y_train,
                                   cv=outer_cv,
                                   n_jobs=-1)
        nested_score_dict[name] = nested_score.mean()

    best_algo = gridcvs[model_type]
    best_algo.fit(x_train, y_train)
    best_params = best_algo.best_params_
    test_acc = accuracy_score(y_true=y_test, y_pred=best_algo.predict(x_test))
    test_f1 = f1_score(y_test, best_algo.predict(x_test), average='weighted')
    test_precision = precision_score(y_test, best_algo.predict(x_test), average='weighted')
    click.echo('Accuracy %.2f%% (average over CV test folds)' %
          (100 * best_algo.best_score_))
    click.echo('Best Parameters: %s' % best_algo.best_params_)
    click.echo('Test Accuracy: %.2f%%' % (100 * test_acc))
    click.echo('Test f1: %.2f%%' % (100 * test_f1))
    click.echo('Test precision: %.2f%%' % (100 * test_precision))
