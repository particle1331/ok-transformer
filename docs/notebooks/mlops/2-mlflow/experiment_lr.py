from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow

from utils import (
    preprocess_datasets, 
    plot_duration_distribution, 
    artifacts, 
    data_path
)


def setup():

    global X_train, y_train, X_valid, y_valid
    global train_data_path, valid_data_path

    # Set datasets
    train_data_path = data_path / 'green_tripdata_2021-01.parquet'
    valid_data_path = data_path / 'green_tripdata_2021-02.parquet'
    X_train, y_train, X_valid, y_valid = preprocess_datasets(train_data_path, valid_data_path)

    # Set experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")


def run():
    with mlflow.start_run():

        # Training run
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Logging
        rmse_train = mean_squared_error(y_train, model.predict(X_train), squared=False)
        rmse_valid = mean_squared_error(y_valid, model.predict(X_valid), squared=False)

        fig = plot_duration_distribution(model, X_train, y_train, X_valid, y_valid)
        fig.savefig('plot.svg')

        mlflow.set_tag('author', 'particle')
        mlflow.set_tag('model', 'baseline')
        
        mlflow.log_param('train_data_path', train_data_path)
        mlflow.log_param('valid_data_path', valid_data_path)
        
        mlflow.log_metric('rmse_train', rmse_train)
        mlflow.log_metric('rmse_valid', rmse_valid)
        
        mlflow.log_artifact(artifacts / 'plot.svg')
        mlflow.log_artifact(artifacts / 'preprocessor.pkl', artifact_path='preprocessing')


def main():
    setup()
    run()


if __name__ == "__main__":
    main()
