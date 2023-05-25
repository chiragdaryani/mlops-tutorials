import argparse
import os
import pickle
import click


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



import mlflow

EXPERIMENT_NAME = "random-forest-experiment"
mlflow.set_experiment(EXPERIMENT_NAME) #create experiment
mlflow.sklearn.autolog() #enable autolog







def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)







@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)

def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run(): # required to track each run of the experiment


        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        #mlflow autolog will capture info about training set auomatically. but we have to manually track it for validation and test sets:

        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse) #track validation rmse



if __name__ == '__main__':
    run_train()















