from pathlib import Path
from joblib import dump

import click
import pandas as pd
from sklearn.metrics import accuracy_score

from data import data_process
from pipeline import make_model

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


def train(
        dataset_path,
        random_state,
        val_size,
        scaler,
        log_penalty,
        log_max_iter,
        log_C
):

    x_train, x_val, y_train, y_val = data_process(
        dataset_path, random_state, val_size
    )
    pipeline = make_model(scaler, log_penalty, log_max_iter, log_C, random_state)
    pipeline.fit(x_train, y_train)
    accuracy = accuracy_score(y_val, pipeline.predict(x_val))
    dump(pipeline, save_model_path)