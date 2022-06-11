from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from prefect.task_runners import SequentialTaskRunner
from prefect import flow, task

from datetime import timedelta, datetime
from functools import partial
from pathlib import Path

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import mlflow
import xgboost as xgb
import time


# Config variables
root = Path(__file__).parent.resolve()
artifacts = root / 'artifacts'
artifacts.mkdir(exist_ok=True)
runs = root / 'mlruns'
data_path = root / 'data'


class ConvertToString(BaseEstimator, TransformerMixin):
    """Convert columns of DataFrame to type string."""

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.features] = X[self.features].astype(str)
        return X


class AddPickupDropoffPair(BaseEstimator, TransformerMixin):
    """Add product of pickup and dropoff locations."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['PU_DO'] = X['PULocationID'].astype(str) + '_' + X['DOLocationID'].astype(str)
        return X


class ConvertToDict(BaseEstimator, TransformerMixin):
    """Convert tabular data to feature dictionaries."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.to_dict(orient='records')


class SelectFeatures(BaseEstimator, TransformerMixin):
    """Convert tabular data to feature dictionaries."""

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features]


@task
def load_training_dataframe(file_path, y_min=1, y_max=60):
    """Load data from disk and preprocess for training."""
    
    # Load data from disk
    data = pd.read_parquet(file_path)

    # Create target column and filter outliers
    data['duration'] = data.lpep_dropoff_datetime - data.lpep_pickup_datetime
    data['duration'] = data.duration.dt.total_seconds() / 60
    data = data[(data.duration >= y_min) & (data.duration <= y_max)]

    return data


@task
def fit_preprocessor(train_data):
    """Fit and save preprocessing pipeline."""

    # Unpack passed data
    y_train = train_data.duration.values
    X_train = train_data.drop('duration', axis=1)    

    # Initialize pipeline
    cat_features = ['PU_DO']
    num_features = ['trip_distance']

    preprocessor = make_pipeline(
        AddPickupDropoffPair(),
        SelectFeatures(cat_features + num_features),
        ConvertToString(cat_features),
        ConvertToDict(),
        DictVectorizer(),
    )

    # Fit only on train set
    preprocessor.fit(X_train, y_train)
    joblib.dump(preprocessor, artifacts / 'preprocessor.pkl')
    
    return preprocessor


@task
def create_model_features(preprocessor, train_data, valid_data):
    """Fit feature engineering pipeline. Transform training dataframes."""

    # Unpack passed data
    y_train = train_data.duration.values
    y_valid = valid_data.duration.values
    X_train = train_data.drop('duration', axis=1)
    X_valid = valid_data.drop('duration', axis=1)
    
    # Feature engineering
    X_train = preprocessor.transform(X_train)
    X_valid = preprocessor.transform(X_valid)

    return X_train, y_train, X_valid, y_valid


@flow
def preprocess_data(train_data_path, valid_data_path):
    """Preprocess data for model training."""

    train_data = load_training_dataframe(train_data_path)
    valid_data = load_training_dataframe(valid_data_path)
    
    preprocessor = fit_preprocessor(train_data)
    
    return create_model_features(preprocessor, train_data, valid_data).result()


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


@flow(task_runner=SequentialTaskRunner())
def deploy_main(train_data_path, valid_data_path, num_xgb_runs=1):

    # Set and run experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(f"nyc-taxi-experiment-{str(datetime.datetime.now())}")

    future = preprocess_data(train_data_path, valid_data_path)
    linreg_runs(future)
    xgboost_runs(num_xgb_runs, future)


DeploymentSpec(
    flow=deploy_main,
    name="mlflow_staging",
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    flow_runner=SubprocessFlowRunner(),
    parameters={
        "train_data_path": data_path / 'green_tripdata_2021-01.parquet',
        "valid_data_path": data_path / 'green_tripdata_2021-02.parquet',
        "num_xgb_runs": 10
    },
    tags=["ml"]
)
