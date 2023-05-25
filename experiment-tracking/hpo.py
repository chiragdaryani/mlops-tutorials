import argparse
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


import click
import optuna
from optuna.samplers import TPESampler


import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000") #running mlflow server
mlflow.set_experiment("random-forest-hyperopt") #create new experiment




def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)




@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=10,
    help="The number of parameter evaluations for the optimizer to explore"
)




def run_optimization(data_path: str, num_trials: int):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(trial):
        
        with mlflow.start_run(): # required to track each run of the experiment

            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 50, 1),
                'max_depth': trial.suggest_int('max_depth', 1, 20, 1),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, 1),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, 1),
                'random_state': 42,
                'n_jobs': -1
            }

            mlflow.log_params(params) #log hyper-parameters

        
        

        
            

            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)

            
            
            mlflow.log_metric("rmse", rmse) #log validation rmse



        return rmse

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


if __name__ == '__main__':
    run_optimization()



 







