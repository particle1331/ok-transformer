import mlflow 
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

from ride_duration.utils import load_training_dataframe, prepare_features


def setup(tracking_server_host):
    TRACKING_URI = f"http://{tracking_server_host}:5000"
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("nyc-taxi-experiment")


def run_training(X_train, y_train, X_valid, y_valid):
    with mlflow.start_run():
        params = {
            'n_estimators': 100,
            'max_depth': 20
        }
        
        pipeline = make_pipeline(
            DictVectorizer(), 
            RandomForestRegressor(**params, n_jobs=-1)
        )
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_valid)
        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        
        mlflow.log_params(params)
        mlflow.log_metric("rmse_valid", rmse)
        mlflow.sklearn.log_model(pipeline, artifact_path='model')


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-server-host", type=str)
    args = parser.parse_args()

    # Getting data from disk
    train_path = './data/green_tripdata_2021-01.parquet'
    valid_path = './data/green_tripdata_2021-02.parquet'
    train_data = load_training_dataframe(train_path)
    valid_data = load_training_dataframe(valid_path)

    # Preprocessing dataset
    X_train = prepare_features(train_data.drop(['duration'], axis=1))
    X_valid = prepare_features(valid_data.drop(['duration'], axis=1))
    y_train = train_data.duration.values
    y_valid = valid_data.duration.values

    # Push training to server
    setup(args.tracking_server_host)
    run_training(X_train, y_train, X_valid, y_valid)
