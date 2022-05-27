import mlflow

from utils import set_datasets, plot_duration_distribution

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR


# Set datasets
train_data_path = '../data/green_tripdata_2021-01.parquet'
valid_data_path = '../data/green_tripdata_2021-02.parquet'
X_train, y_train, X_valid, y_valid = set_datasets(train_data_path, valid_data_path)

# Set experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")
mlflow.sklearn.autolog()

for model_class in (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, LinearSVR):

    with mlflow.start_run(run_name='sklearn'):

        # Train model
        model = model_class()
        model.fit(X_train, y_train)

        # Plot predictions vs ground truth
        fig = plot_duration_distribution(model, X_train, y_train, X_valid, y_valid)
        fig.savefig('plot.svg')

        # Compute metrics
        rmse_valid = mean_squared_error(y_valid, booster.predict(xgb_valid), squared=False)
        rmse_train = mean_squared_error(y_train, booster.predict(xgb_train), squared=False)

        # MLFlow logging
        mlflow.set_tag('author', 'particle')
        mlflow.set_tag('model', 'sklearn')
        
        mlflow.log_param('train-data-path', train_data_path)
        mlflow.log_param('valid-data-path', valid_data_path)
        
        mlflow.log_metric('rmse_valid', rmse_valid)
        mlflow.log_metric('rmse_train', rmse_train)
        
        mlflow.log_artifact('preprocessor.b', artifact_path='preprocessor')
        mlflow.log_artifact('plot.svg')
