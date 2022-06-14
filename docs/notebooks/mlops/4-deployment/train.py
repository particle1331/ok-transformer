import pandas as pd

import joblib
import warnings
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# Config variables
root = Path(__file__).parent.resolve()
data_path = root / 'data'
models_path = root / 'models'


class ConvertToString(BaseEstimator, TransformerMixin):
    """Convert columns of DataFrame to type string."""

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.features] = X[self.features].astype(str)
        return X


class AddPickupDropoffPair(BaseEstimator, TransformerMixin):
    """Add product of pickup and dropoff locations."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['PU_DO'] = X['PULocationID'].astype(str) + '_' + X['DOLocationID'].astype(str)
        return X


class ConvertToDict(BaseEstimator, TransformerMixin):
    """Convert tabular data to feature dictionaries."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.to_dict(orient='records')


class SelectFeatures(BaseEstimator, TransformerMixin):
    """Convert tabular data to feature dictionaries."""

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features]


def load_training_dataframe(file_path, y_min=1, y_max=60):
    """Load data from disk and preprocess for training."""
    
    # Load data from disk
    data = pd.read_parquet(file_path)

    # Create target column and filter outliers
    data['duration'] = data.lpep_dropoff_datetime - data.lpep_pickup_datetime
    data['duration'] = data.duration.dt.total_seconds() / 60
    data = data[(data.duration >= y_min) & (data.duration <= y_max)]

    return data


def fit_preprocessor(train_data):
    """Fit and save preprocessing pipeline."""

    # Unpack passed data
    y_train = train_data.duration.values
    X_train = train_data.drop('duration', axis=1)    

    # Initialize pipeline
    cat_features = ['PU_DO']
    num_features = ['trip_distance']

    preprocessor = make_pipeline(
        AddPickupDropoffPair(),
        SelectFeatures(cat_features + num_features),
        ConvertToString(cat_features),
        ConvertToDict(),
        DictVectorizer(),
    )

    # Fit only on train set
    preprocessor.fit(X_train, y_train)
    
    return preprocessor


def pipeline(preprocessor, X_train, y_train):
    """Return trained linear regression model pipeline."""

    pipeline = make_pipeline(
        preprocessor, 
        LinearRegression()
    )

    pipeline.fit(X_train, y_train)
    return pipeline


if __name__ == "__main__":

    # Training linear regression model
    train_data_path = data_path / 'green_tripdata_2021-01.parquet'
    train_data = load_training_dataframe(train_data_path)
    
    lr_pipe = pipeline(
        preprocessor=fit_preprocessor(train_data),
        X_train=train_data.drop(['duration'], axis=1), 
        y_train=train_data.duration.values
    )
 
    joblib.dump(lr_pipe, models_path / 'lin_reg.bin')


    # Evaluation
    valid_data_path = data_path / 'green_tripdata_2021-02.parquet'
    valid_data = load_training_dataframe(valid_data_path)
    
    print("RMSE (train):", mean_squared_error(train_data.duration.values, lr_pipe.predict(train_data), squared=False))
    print("RMSE (valid):", mean_squared_error(valid_data.duration.values, lr_pipe.predict(valid_data), squared=False))
