import xgboost as xgb
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from utils import set_datasets, plot_duration_distribution
from functools import partial

import mlflow


# Set datasets
train_data_path = '../data/green_tripdata_2021-01.parquet'
valid_data_path = '../data/green_tripdata_2021-02.parquet'
X_train, y_train, X_valid, y_valid = set_datasets(train_data_path, valid_data_path)

xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_valid = xgb.DMatrix(X_valid, label=y_valid)

# Set experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")


def objective(params, autolog):
    """Compute validation RMSE (one trial = one run)."""
    
    mlflow.xgboost.autolog(disable=not(autolog))

    with mlflow.start_run():
        
        # Train model
        booster = xgb.train(
            params=params,
            dtrain=xgb_train,
            num_boost_round=1000,
            evals=[(xgb_valid, 'validation')],
            early_stopping_rounds=50
        )

        # Plot predictions vs ground truth
        fig = plot_duration_distribution(booster, xgb_train, y_train, xgb_valid, y_valid)
        fig.savefig('plot.svg')

        # Compute metrics
        rmse_valid = mean_squared_error(y_valid, booster.predict(xgb_valid), squared=False)
        rmse_train = mean_squared_error(y_train, booster.predict(xgb_train), squared=False)
        
        # Logging
        mlflow.set_tag('author', 'particle')
        mlflow.set_tag('model', 'xgboost')
        
        mlflow.log_param('train_data_path', train_data_path)
        mlflow.log_param('valid_data_path', valid_data_path)
        
        mlflow.log_metric('rmse_train', rmse_train)
        mlflow.log_metric('rmse_valid', rmse_valid)
        
        mlflow.log_artifact('preprocessor.b', artifact_path='preprocessor')
        mlflow.log_artifact('plot.svg')

        if not autolog:
            mlflow.xgboost.log_model(booster, 'model')
            mlflow.log_params(params)
    
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


def main(autolog, num_runs):
    best_result = fmin(
        fn=partial(objective, autolog=autolog),
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_runs,
        trials=Trials()
    )


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--autolog", choices=["False", "True"])
    parser.add_argument("--num_runs", default=1, type=int)
    
    args = parser.parse_args()
    
    main(autolog=(args.autolog == 'True'), num_runs=args.num_runs)
