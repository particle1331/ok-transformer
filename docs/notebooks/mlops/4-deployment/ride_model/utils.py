import pandas as pd
import warnings

from pathlib import Path
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin


# Config variables
package_dir = Path(__file__).parent.resolve()
runs = package_dir / 'mlruns'
data_path = package_dir / 'data'


def load_training_dataframe(file_path, y_min=1, y_max=60):
    """Load data from disk and preprocess for training."""
    
    # Load data from disk
    data = pd.read_parquet(file_path)

    # Create target column and filter outliers
    data['duration'] = data.lpep_dropoff_datetime - data.lpep_pickup_datetime
    data['duration'] = data.duration.dt.total_seconds() / 60
    data = data[(data.duration >= y_min) & (data.duration <= y_max)]

    return data
