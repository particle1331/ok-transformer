import time
from functools import partial
from pathlib import Path

import mlflow
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from utils import artifacts, data_path, preprocess_data


def objective(params, xgb_train, y_train, xgb_valid, y_valid):
    """Compute validation RMSE (one trial = one run)."""

    with mlflow.start_run():
        
        model = xgb.train(
            params=params,
            dtrain=xgb_train,
            num_boost_round=100,
            evals=[(xgb_valid, 'validation')],
            early_stopping_rounds=5,
            verbose_eval=False,
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
def xgboost_runs(num_runs, training_packet):
    """Run TPE algorithm on search space to minimize objective."""

    X_train, y_train, X_valid, y_valid = training_packet
    Xgb_train = xgb.DMatrix(X_train, label=y_train)
    Xgb_valid = xgb.DMatrix(X_valid, label=y_valid)


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
            xgb_train=Xgb_train, y_train=y_train, 
            xgb_valid=Xgb_valid, y_valid=y_valid,
        ),
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_runs,
        trials=Trials()
    )


@task
def linreg_runs(training_packet):
    """Run linear regression training."""

    X_train, y_train, X_valid, y_valid = training_packet
    
    with mlflow.start_run():

        model = LinearRegression()
        model.fit(X_train, y_train)

        # MLflow logging
        start_time = time.time()
        y_pred_train = model.predict(X_train)
        y_pred_valid = model.predict(X_valid)
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
        mlflow.sklearn.log_model(model, artifact_path="models")

@task
def stage_model(tracking_uri, experiment_name):
    """Register and stage best model."""

    # Get best model from current experiment
    client = MlflowClient(tracking_uri=tracking_uri)
    candidates = client.search_runs(
        experiment_ids=client.get_experiment_by_name(experiment_name).experiment_id,
        filter_string='metrics.rmse_valid < 6.5 and metrics.inference_time < 20e-6',
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.rmse_valid ASC"]
    )

    # Register and stage best model
    best_model = candidates[0]
    registered_model = mlflow.register_model(
        model_uri=f"runs:/{best_model.info.run_id}/model", 
        name='NYCRideDurationModel'
    )

    client.transition_model_version_stage(
        name='NYCRideDurationModel',
        version=registered_model.version, 
        stage='Staging',
    )

    # Update description of staged model
    client.update_model_version(
        name=model_name,
        version=model_version,
        description=f"[{datetime.datetime.now()}] The model version {model_version} from experiment '{experiment_name}' was transitioned to {new_stage}.\n{old_description}"
    )


@flow(task_runner=SequentialTaskRunner())
def main(
    train_data_path, 
    valid_data_path, 
    num_xgb_runs=1, 
    experiment_name="nyc-taxi-experiment",
    tracking_uri="sqlite:///mlflow.db",
):

    # Set and run experiment
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    future = preprocess_data(train_data_path, valid_data_path)
    linreg_runs(future)
    xgboost_runs(num_xgb_runs, future)


@flow(name='mlflow-staging', task_runner=SequentialTaskRunner())
def mlflow_staging(train_data_path, valid_data_path, datetime, num_xgb_runs=1):

    # Setup experiment
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT_NAME = f"nyc-taxi-experiment-{datetime}"

    # Make experiment runs
    main(
        train_data_path=train_data_path, 
        valid_data_path=valid_data_path, 
        num_xgb_runs=num_xgb_runs, 
        experiment_name=EXPERIMENT_NAME,
        tracking_uri=MLFLOW_TRACKING_URI,
    )
    
    # Stage best model
    stage_model(
        tracking_uri=MLFLOW_TRACKING_URI, 
        experiment_name=EXPERIMENT_NAME
    )


if __name__ == "__main__":

    from datetime import datetime

    parameters={
        "train_data_path": './data/green_tripdata_2021-01.parquet',
        "valid_data_path": './data/green_tripdata_2021-02.parquet',
        "num_xgb_runs": 3,
        "datetime": str(datetime.now())
    }

    mlflow_staging(**parameters)
