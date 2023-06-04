import time

import mlflow
from utils import fixtures, create_pipeline, setup_experiment
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from ride_duration.utils import plot_duration_histograms
from ride_duration.processing import prepare_features

setup_experiment()
data = fixtures()

with mlflow.start_run():
    train_data_path = data["train_data_path"]
    valid_data_path = data["valid_data_path"]
    train_data = data["train_data"]
    valid_data = data["valid_data"]

    X_train, y_train = prepare_features(train_data, train=True)
    X_valid, y_valid = prepare_features(valid_data, train=True)

    # Train model
    model = LinearRegression()
    pipe = create_pipeline(model)
    pipe.fit(X_train, y_train)

    # Computing metrics
    start_time = time.time()
    yp_train = pipe.predict(X_train)
    yp_valid = pipe.predict(X_valid)
    inference_time = time.time() - start_time

    rmse_train = mean_squared_error(y_train, yp_train, squared=False)
    rmse_valid = mean_squared_error(y_valid, yp_valid, squared=False)

    fig = plot_duration_histograms(y_train, yp_train, y_valid, yp_valid)

    # Tags
    mlflow.set_tag("author", "particle")
    mlflow.set_tag("model", "baseline")

    # MLflow logging
    mlflow.log_param("train_data_path", train_data_path)
    mlflow.log_param("valid_data_path", valid_data_path)

    mlflow.log_metric("rmse_train", rmse_train)
    mlflow.log_metric("rmse_valid", rmse_valid)
    mlflow.log_metric(
        "inference_time",
        inference_time / (len(yp_train) + len(yp_valid)),
    )

    mlflow.log_figure(fig, "plot.svg")
