from sklearn.model_selection import train_test_split
import click
import pandas as pd




def data_process(dataset_path, random_state, val_size):
    data = pd.read_csv(dataset_path)
    feature = data.iloc[:, :-1]
    target = data.iloc[:, 1]
    x_train, x_val, y_train, y_val = train_test_split(
        feature, target, test_size=val_size, random_state=random_state
    )
    return (x_train, x_val, y_train, y_val)
