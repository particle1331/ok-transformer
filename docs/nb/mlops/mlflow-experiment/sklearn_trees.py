import time

import mlflow
from utils import fixtures, create_pipeline, setup_experiment
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from ride_duration.utils import plot_duration_histograms
from ride_duration.config import config
from ride_duration.processing import prepare_features


def add_endpoints_interaction(df):
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    return df


setup_experiment()
data = fixtures()
mlflow.sklearn.autolog()  # (!)


def run(model_class):
    with mlflow.start_run():
        train_data_path = data["train_data_path"]
        valid_data_path = data["valid_data_path"]
        train_data = data["train_data"]
        valid_data = data["valid_data"]

        # Feature engg + selection
        transforms = [
            add_endpoints_interaction,
            lambda df: df[["PU_DO"] + config.NUM_FEATURES],
        ]

        X_train, y_train = prepare_features(train_data, transforms, train=True)
        X_valid, y_valid = prepare_features(valid_data, transforms, train=True)

        # Train model
        model = model_class()
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
        mlflow.set_tag("model", model_class.__name__)

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


if __name__ == "__main__":
    for model_class in [
        RandomForestRegressor,
        GradientBoostingRegressor,
        ExtraTreesRegressor,
    ]:
        run(model_class)
