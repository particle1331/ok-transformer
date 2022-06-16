from ride_duration.utils import load_training_dataframe, prepare_features, package_dir

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

import joblib


def train_pipeline(X_train, y_train):
    """Fit and save preprocessing pipeline."""

    pipe = make_pipeline(
        DictVectorizer(),
        LinearRegression()
    )
    
    pipe.fit(X_train, y_train)
    return pipe


def run_training(train_path, valid_path):
    """Train model and pickle model file."""

    train_data = load_training_dataframe(train_path)
    valid_data = load_training_dataframe(valid_path)

    X_train = train_data.drop(['duration'], axis=1)
    y_train = train_data.duration.values

    # Persist trained pipeline
    pipeline = train_pipeline(prepare_features(X_train), y_train)
    joblib.dump(pipeline, package_dir / 'pipeline.pkl')

    # Evaluation
    X_valid = valid_data.drop(['duration'], axis=1)
    y_valid = valid_data.duration.values

    print("RMSE (train):", mean_squared_error(y_train, pipeline.predict(prepare_features(X_train)), squared=False))
    print("RMSE (valid):", mean_squared_error(y_valid, pipeline.predict(prepare_features(X_valid)), squared=False))
    

if __name__ == "__main__":
    
    pipeline = run_training(
        train_path='./data/green_tripdata_2021-01.parquet', 
        valid_path='./data/green_tripdata_2021-02.parquet'
    )
