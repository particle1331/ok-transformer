from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.metrics import mean_squared_error
from functools import partial

import mlflow
import xgboost as xgb
import time

from utils import (
    preprocess_datasets, 
    plot_duration_distribution, 
    artifacts, 
    data_path
)


def setup():

    global xgb_train, y_train, xgb_valid, y_valid
    global train_data_path, valid_data_path

    # Preprocessing
    train_data_path = data_path / 'green_tripdata_2021-01.parquet'
    valid_data_path = data_path / 'green_tripdata_2021-02.parquet'
    X_train, y_train, X_valid, y_valid = preprocess_datasets(train_data_path, valid_data_path)
    
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_valid = xgb.DMatrix(X_valid, label=y_valid)

    # Set experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")
    mlflow.xgboost.autolog()


def objective(params):
    """Compute validation RMSE (one trial = one run)."""

    with mlflow.start_run():
        
        model = xgb.train(
            params=params,
            dtrain=xgb_train,
            num_boost_round=1000,
            evals=[(xgb_valid, 'validation')],
            early_stopping_rounds=50
        )

        # MLflow logging
        start_time = time.time()
        y_pred_train = model.predict(xgb_train)
        y_pred_valid = model.predict(xgb_valid)
        inference_time = time.time() - start_time

        rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
        rmse_valid = mean_squared_error(y_valid, y_pred_valid, squared=False)

        fig = plot_duration_distribution(model, xgb_train, y_train, xgb_valid, y_valid)
        fig.savefig(artifacts / 'plot.svg')

        mlflow.set_tag('author', 'particle')
        mlflow.set_tag('model', 'baseline')
        
        mlflow.log_param('train_data_path', train_data_path)
        mlflow.log_param('valid_data_path', valid_data_path)
        
        mlflow.log_metric('rmse_train', rmse_train)
        mlflow.log_metric('rmse_valid', rmse_valid)
        mlflow.log_metric(
            'inference_time', 
            inference_time / (len(y_pred_train) + len(y_pred_valid))
        )
        
        mlflow.log_artifact(artifacts / 'plot.svg')
        mlflow.log_artifact(artifacts / 'preprocessor.pkl', artifact_path='preprocessing')
    
    return {'loss': rmse_valid, 'status': STATUS_OK}


search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:squarederror',
    'seed': 42
}


def main(num_runs):
    best_result = fmin(
        fn=partial(objective),
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_runs,
        trials=Trials()
    )


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", default=1, type=int)
    args = parser.parse_args()
    
    # Experiment runs
    setup()
    main(num_runs=args.num_runs)
