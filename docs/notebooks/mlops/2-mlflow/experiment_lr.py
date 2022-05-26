from utils import *

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

import mlflow


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

with mlflow.start_run(run_name='lr'):

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

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train);

    # Plot predictions vs ground truth
    fig = plot_duration_distribution(model, X_train, y_train, X_valid, y_valid)
    fig.savefig('plot.svg')

    # Print metric
    rmse_train = mean_squared_error(y_train, model.predict(X_train), squared=False)
    rmse_valid = mean_squared_error(y_valid, model.predict(X_valid), squared=False)

    # MLFlow logging
    mlflow.set_tag("author", "particle")
    mlflow.log_param('train_data_path', train_data_path)
    mlflow.log_param('valid_data_path', valid_data_path)
    mlflow.log_metric('rmse_train', rmse_train)
    mlflow.log_metric('rmse_valid', rmse_valid)
    mlflow.log_artifact('plot.svg')
