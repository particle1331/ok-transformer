from ride_model.utils import load_training_dataframe, prepare_features

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

import joblib


def train_lr_pipeline(X_train, y_train):
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
    pipeline = train_lr_pipeline(prepare_features(X_train), y_train)
    joblib.dump(pipeline, package_dir / 'pipeline.pkl')
    
    # Evaluation
    X_valid = valid_data.drop(['duration'], axis=1)
    y_valid = valid_data.duration.values

    print("RMSE (train):", mean_squared_error(y_train, pipeline.predict(prepare_features(X_train)), squared=False))
    print("RMSE (valid):", mean_squared_error(y_valid, pipeline.predict(prepare_features(X_valid)), squared=False))


if __name__ == "__main__":
    
    from utils import package_dir
    import argparse
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    parser = argparse.ArgumentParser()
    parser.add_argument("--taxi_type", default='green', type=str)
    parser.add_argument("--year", default=2021, type=int)
    parser.add_argument("--train_month", default=1, type=int)
    parser.add_argument("--valid_month", default=2, type=int)
    args = parser.parse_args()

    source_url = 'https://s3.amazonaws.com/nyc-tlc/trip+data'
    train_path = f'{source_url}/{args.taxi_type}_tripdata_{args.year:04d}-{args.train_month:02d}.parquet'
    valid_path = f'{source_url}/{args.taxi_type}_tripdata_{args.year:04d}-{args.valid_month:02d}.parquet'

    run_training(train_path, valid_path)
