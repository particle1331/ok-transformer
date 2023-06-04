import time

import mlflow
import optuna
from utils import fixtures, setup_experiment, create_feature_pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from ride_duration.utils import plot_duration_histograms
from ride_duration.config import config
from ride_duration.processing import prepare_features


def add_endpoints_interaction(df):
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    return df


# setup_experiment()
mlflow.set_tracking_uri("http://0.0.0.0:5001")
mlflow.set_experiment("test")
data = fixtures()
mlflow.xgboost.autolog()


def run(params, pudo: bool):
    with mlflow.start_run():
        train_data_path = data["train_data_path"]
        valid_data_path = data["valid_data_path"]
        train_data = data["train_data"]
        valid_data = data["valid_data"]

        # Feature engg + selection
        transforms = [
            add_endpoints_interaction,
            lambda df: df[["PU_DO"] + config.NUM_FEATURES],
        ] if pudo else []  # fmt: skip

        X_train, y_train = prepare_features(train_data, transforms, train=True)
        X_valid, y_valid = prepare_features(valid_data, transforms, train=True)

        # XGBoost specific code
        feature_pipe = create_feature_pipeline()

        X_train = feature_pipe.fit_transform(X_train)
        X_valid = feature_pipe.transform(X_valid)

        # Train model
        model = XGBRegressor(early_stopping_rounds=50, **params)
        model.fit(
            X_train, y_train,   # fmt: skip
            eval_set=[(X_valid, y_valid)],
        )

        # Computing metrics
        start_time = time.time()
        yp_train = model.predict(X_train)
        yp_valid = model.predict(X_valid)
        predict_time = time.time() - start_time

        rmse_train = mean_squared_error(y_train, yp_train, squared=False)
        rmse_valid = mean_squared_error(y_valid, yp_valid, squared=False)

        fig = plot_duration_histograms(y_train, yp_train, y_valid, yp_valid)

        # MLflow logging
        mlflow.set_tag("author", "particle")
        mlflow.set_tag("model", "xgboost")

        mlflow.log_param("interaction", pudo)
        mlflow.log_param("train_data_path", train_data_path)
        mlflow.log_param("valid_data_path", valid_data_path)

        mlflow.log_metric("rmse_train", rmse_train)
        mlflow.log_metric("rmse_valid", rmse_valid)

        mlflow.log_metric(
            "inference_time", predict_time / (len(yp_train) + len(yp_valid))
        )

        mlflow.log_figure(fig, "plot.svg")

        # Log feature pipeline as artifact
        mlflow.sklearn.log_model(feature_pipe, "preprocessor")

    return rmse_valid


def objective(trial, pudo: bool):
    # fmt: off
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 1, 10, step=1) * 100,
        "max_depth":        trial.suggest_int("max_depth", 4, 100),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-5, 0.1, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-6, 0.1, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 1000, log=True),
        "objective": "reg:squarederror",
        "seed": 42,
    }

    return run(params, pudo)


if __name__ == "__main__":
    import sys
    from functools import partial

    N_TRIALS = int(sys.argv[1])
    USE_PUDO = int(sys.argv[2])

    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, pudo=USE_PUDO), n_trials=N_TRIALS)
