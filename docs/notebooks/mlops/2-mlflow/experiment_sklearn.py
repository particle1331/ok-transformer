from utils import set_datasets, plot_duration_distribution

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error

import mlflow


# Set datasets
train_data_path = '../data/green_tripdata_2021-01.parquet'
valid_data_path = '../data/green_tripdata_2021-02.parquet'
X_train, y_train, X_valid, y_valid = set_datasets(train_data_path, valid_data_path)

# Set experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")
mlflow.sklearn.autolog()


def run(model_class):
    with mlflow.start_run():

        # Train model
        model = model_class()
        model.fit(X_train, y_train)

        # Plot predictions vs ground truth
        fig = plot_duration_distribution(model, X_train, y_train, X_valid, y_valid)
        fig.savefig('plot.svg')

        # Print metric
        rmse_train = mean_squared_error(y_train, model.predict(X_train), squared=False)
        rmse_valid = mean_squared_error(y_valid, model.predict(X_valid), squared=False)

        # Logging
        mlflow.set_tag('author', 'particle')
        mlflow.set_tag('model', 'sklearn')
        
        mlflow.log_param('train_data_path', train_data_path)
        mlflow.log_param('valid_data_path', valid_data_path)
        
        mlflow.log_metric('rmse_train', rmse_train)
        mlflow.log_metric('rmse_valid', rmse_valid)
        
        mlflow.log_artifact('preprocessor.b', artifact_path='preprocessor')
        mlflow.log_artifact('plot.svg')


def main():
    for model_class in [
        Lasso,
        Ridge,
        ExtraTreesRegressor,
        RandomForestRegressor, 
        GradientBoostingRegressor,
    ]:
        run(model_class)


if __name__ == "__main__":
    main()
