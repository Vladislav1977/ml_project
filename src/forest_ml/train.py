from pathlib import Path

import click
import pandas as pd

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path))

def train(dataset_path):
    df = pd.read_csv(dataset_path)
