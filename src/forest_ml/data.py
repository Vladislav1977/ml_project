from sklearn.model_selection import train_test_split
import click
import pandas as pd


class data_process:

    def __init__(self, dataset_path):
        self.path = dataset_path

    def extract(self):
        data = pd.read_csv(self.path)
        feature = data.iloc[:, 1:-1]
        target = data.iloc[:, -1]
        return feature, target

    def split(self, random_state, val_size):
        feature, target = self.extract()
        x_train, x_val, y_train, y_val = train_test_split(
        feature, target, test_size=val_size, random_state=random_state
    )
        return (x_train, x_val, y_train, y_val)
