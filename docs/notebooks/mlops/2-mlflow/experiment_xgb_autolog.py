import xgboost as xgb
import mlflow
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from utils import set_datasets, plot_duration_distribution
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error


# Set datasets
train_data_path = '../data/green_tripdata_2021-01.parquet'
valid_data_path = '../data/green_tripdata_2021-02.parquet'
X_train, y_train, X_valid, y_valid = set_datasets(train_data_path, valid_data_path)
xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_valid = xgb.DMatrix(X_valid, label=y_valid)

# Set experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")
mlflow.xgboost.autolog()


def objective(params):
    """Compute validation RMSE (one trial = one run)."""
    
    with mlflow.start_run():
        
        # Train model
        booster = xgb.train(
            params=params,
            dtrain=xgb_train,
            num_boost_round=1000,
            evals=[(xgb_valid, 'validation')],
            early_stopping_rounds=50
        )

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

def main():
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials()
    )

if __name__ == "__main__":
    main()
