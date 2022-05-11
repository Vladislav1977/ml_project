Final Machine Learning project for RS School Machine Learning course.

This project uses [Forest Train](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset.

## Usage
This package allows you to predict forest cover type from strictly cartographic variables.
1. Clone this repository to your machine.
2. Download [Forest Train](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (Poetry version 1.1.13).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
There are 3 python scripts in that rpoject: ***train***, ***model_tuning***, ***model_selection***.
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
This script fits the model and return validation metrics.
6. Run model_tuning with the following command:
```sh
poetry run model_tuning -d <path to csv with data> --model-type <type of the model to validate>
```
7. Run model_selection with the following command:
```sh
poetry run model_selection -d <path to csv with data> 
```
It tunes hyperparameters manualy and tracks each experiment into MLFlow.
8. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
The result should be similar to:

![MLFlow experiments example][https://github.com/Vladislav1977/git_2/blob/main/ml.PNG?raw=true]
