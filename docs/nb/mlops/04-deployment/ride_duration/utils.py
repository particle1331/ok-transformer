import pandas as pd
import warnings
import uuid

from pathlib import Path
from typing import Union

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def load_training_dataframe(file_path, y_min=1, y_max=60):
    """Load data from disk and preprocess for training."""
    
    # Load data from disk
    data = pd.read_parquet(file_path)

    # Create target column and filter outliers
    data['duration'] = data.lpep_dropoff_datetime - data.lpep_pickup_datetime
    data['duration'] = data.duration.dt.total_seconds() / 60
    data = data[(data.duration >= y_min) & (data.duration <= y_max)]

    return data


def prepare_features(input_data: Union[list[dict], pd.DataFrame]):
    """Prepare features for dict vectorizer."""

    X = pd.DataFrame(input_data)
    X['PU_DO'] = X['PULocationID'].astype(str) + '_' + X['DOLocationID'].astype(str)
    X = X[['PU_DO', 'trip_distance']].to_dict(orient='records')
    
    return X
