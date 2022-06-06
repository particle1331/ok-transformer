from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.metrics import mean_squared_error
from functools import partial

import mlflow
import xgboost as xgb
import time

from utils import (
    create_features, 
    read_training_dataframes, 
    artifacts, 
    data_path
)

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner


def objective(params, xgb_train, y_train, xgb_valid, y_valid):
    """Compute validation RMSE (one trial = one run)."""

    with mlflow.start_run():
        
        model = xgb.train(
            params=params,
            dtrain=xgb_train,
            num_boost_round=100,
            evals=[(xgb_valid, 'validation')],
            early_stopping_rounds=5
        )

        # MLflow logging
        start_time = time.time()
        y_pred_train = model.predict(xgb_train)
        y_pred_valid = model.predict(xgb_valid)
        inference_time = time.time() - start_time

        rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
        rmse_valid = mean_squared_error(y_valid, y_pred_valid, squared=False)

        mlflow.set_tag('author', 'particle')
        mlflow.set_tag('model', 'baseline')
                
        mlflow.log_metric('rmse_train', rmse_train)
        mlflow.log_metric('rmse_valid', rmse_valid)
        mlflow.log_metric(
            'inference_time', 
            inference_time / (len(y_pred_train) + len(y_pred_valid))
        )
        
        mlflow.log_artifact(artifacts / 'preprocessor.pkl', artifact_path='preprocessing')
        mlflow.xgboost.log_model(model, artifact_path="models")

    return {'loss': rmse_valid, 'status': STATUS_OK}


@task
def search_xgboost_params(num_runs, xgb_train, y_train, xgb_valid, y_valid):
    """Run TPE algorithm on search space to minimize objective."""

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:squarederror',
        'seed': 42
    }

    best_result = fmin(
        fn=partial(
            objective, 
            xgb_train=xgb_train, y_train=y_train, 
            xgb_valid=xgb_valid, y_valid=y_valid,
        ),
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_runs,
        trials=Trials()
    )


@flow(task_runner=SequentialTaskRunner())
def main(num_runs):

    # Preprocessing
    train_data_path = data_path / 'green_tripdata_2021-01.parquet'
    valid_data_path = data_path / 'green_tripdata_2021-02.parquet'
    
    X_train, y_train, X_valid, y_valid = create_features(
        *read_training_dataframes(
            train_data_path=train_data_path, 
            valid_data_path=valid_data_path
        ).result()
    ).result()
    
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_valid = xgb.DMatrix(X_valid, label=y_valid)

    # Set and run experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")
    search_xgboost_params(num_runs, xgb_train, y_train, xgb_valid, y_valid)


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", default=1, type=int)
    args = parser.parse_args()
    
    # Experiment runs
    main(num_runs=args.num_runs)
