from pathlib import Path
from joblib import dump
from sklearn.model_selection import cross_validate
import sys

import click
import pandas as pd
from sklearn.metrics import accuracy_score

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
    "--log-penalty",
    default="l2",
    type=str,
    show_default=True,
)

@click.option(
    "--log-max_iter",
    default=1000,
    type=int,
    show_default=True,
)

@click.option(
    "--log-c",
    default=1.0,
    type=float,
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
        random_state,
        val_size,
        scaler,
        log_penalty,
        log_max_iter,
        log_c,
        save_model_path
):

    x_train, x_val, y_train, y_val = data_process(
        dataset_path).split(random_state, val_size)
    pipeline = make_model(scaler,   log_penalty, log_max_iter, log_c, random_state)
    pipeline.fit(x_train, y_train)
    feature, target = data_process(dataset_path).extract()
    cv = cross_validate(pipeline,
                        feature, target, cv=5,
                        return_estimator=True,
                        scoring=['accuracy', "f1_weighted", "neg_log_loss"])
    accuracy = cv["test_accuracy"].mean()
    f1_weighted = cv["test_f1_weighted"].mean()
    neg_log_loss = cv["test_neg_log_loss"].mean()
    click.echo(f"accuracy score  {accuracy}.")
    click.echo(f"f1_weighted score  {f1_weighted}.")
    click.echo(f"neg_log_loss score  {neg_log_loss}.")
    dump(pipeline, save_model_path)
