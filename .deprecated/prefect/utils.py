import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings

from pandas.core.common import SettingWithCopyWarning
from matplotlib_inline import backend_inline

from toolz import compose
from pathlib import Path
from prefect import task, flow

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
backend_inline.set_matplotlib_formats('svg')


# Config variables
root = Path(__file__).parent.resolve()
artifacts = root / 'artifacts'
artifacts.mkdir(exist_ok=True)
runs = root / 'mlruns'
data_path = root / 'data'


class PrepareFeatures(BaseEstimator, TransformerMixin):
    """Prepare features for dict vectorizer."""

    def __init__(self, categorical, numerical):
        self.categorical = categorical
        self.numerical = numerical
        self.features = categorical + numerical

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['PU_DO'] = X['PULocationID'].astype(str) + '_' + X['DOLocationID'].astype(str)
        X[self.categorical] = X[self.categorical].astype(str)
        X = X[self.features].to_dict(orient='records')
        
        return X


@task
def load_training_dataframe(file_path, y_min=1, y_max=60):
    """Load data from disk and preprocess for training."""
    
    # Load data from disk
    data = pd.read_parquet(file_path)

    # Create target column and filter outliers
    data['duration'] = data.lpep_dropoff_datetime - data.lpep_pickup_datetime
    data['duration'] = data.duration.dt.total_seconds() / 60
    data = data[(data.duration >= y_min) & (data.duration <= y_max)]

    return data


@task
def fit_preprocessor(train_data):
    """Fit and save preprocessing pipeline."""

    # Unpack passed data
    y_train = train_data.duration.values
    X_train = train_data.drop('duration', axis=1)    

    # Initialize pipeline
    categorical = ['PU_DO']
    numerical = ['trip_distance']

    preprocessor = make_pipeline(
        PrepareFeatures(categorical, numerical),
        DictVectorizer(),
    )

    # Fit only on train set
    preprocessor.fit(X_train, y_train)
    joblib.dump(preprocessor, artifacts / 'preprocessor.pkl')
    
    return preprocessor


@task
def create_model_features(preprocessor, train_data, valid_data):
    """Fit feature engineering pipeline. Transform training dataframes."""

    # Unpack passed data
    y_train = train_data.duration.values
    y_valid = valid_data.duration.values
    X_train = train_data.drop('duration', axis=1)
    X_valid = valid_data.drop('duration', axis=1)
    
    # Feature engineering
    X_train = preprocessor.transform(X_train)
    X_valid = preprocessor.transform(X_valid)

    return X_train, y_train, X_valid, y_valid


@flow
def preprocess_data(train_data_path, valid_data_path):
    """Return feature and target arrays from paths. 
    Note: This just combines all the functions above in a single step."""

    train_data = load_training_dataframe(train_data_path)
    valid_data = load_training_dataframe(valid_data_path)
    preprocessor = fit_preprocessor(train_data)

    # X_train, y_train, X_valid, y_valid
    return create_model_features(preprocessor, train_data, valid_data).result()
