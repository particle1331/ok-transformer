from ride_model.utils import load_training_dataframe, data_path
from ride_model.preprocessing import prepare_features

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


if __name__ == "__main__":
    train_path = data_path / 'green_tripdata_2021-01.parquet'
    valid_path = data_path / 'green_tripdata_2021-02.parquet'

    train_data = load_training_dataframe(train_path)
    valid_data = load_training_dataframe(valid_path)

    X_train = train_data.drop(['duration'], axis=1)
    y_train = train_data.duration.values

    # Persist trained pipeline
    pipeline = train_lr_pipeline(prepare_features(X_train), y_train)
    joblib.dump(pipeline, 'pipeline.pkl')
    
    # Evaluation
    X_valid = valid_data.drop(['duration'], axis=1)
    y_valid = valid_data.duration.values

    print("RMSE (train):", mean_squared_error(y_train, pipeline.predict(prepare_features(X_train)), squared=False))
    print("RMSE (valid):", mean_squared_error(y_valid, pipeline.predict(prepare_features(X_valid)), squared=False))
