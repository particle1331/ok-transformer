from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import xgboost as xgb

import mlflow

from utils import *
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error


train_data_path = '../data/green_tripdata_2021-01.parquet'
valid_data_path = '../data/green_tripdata_2021-02.parquet'

# In-between transformations
transforms = [add_pickup_dropoff_pair]
categorical = ['PU_DO']
numerical = ['trip_distance']

train_dicts, y_train = preprocess_dataset(train_data_path, transforms, categorical, numerical)
valid_dicts, y_valid = preprocess_dataset(valid_data_path, transforms, categorical, numerical)

# Fit all possible categories
dv = DictVectorizer()
dv.fit(train_dicts + valid_dicts)

X_train = dv.transform(train_dicts)
X_valid = dv.transform(valid_dicts)

xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_valid = xgb.DMatrix(X_valid, label=y_valid)


def objective(params):
    
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        
        booster = xgb.train(
            params=params,
            dtrain=xgb_train,
            num_boost_round=1000,
            evals=[(xgb_valid, 'validation')],
            early_stopping_rounds=50
        )
        
        y_pred = booster.predict(xgb_valid)
        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    return {'loss': rmse, 'status': STATUS_OK}


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

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
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials()
)
