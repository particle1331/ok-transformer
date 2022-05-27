import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')

from pandas.core.common import SettingWithCopyWarning
import warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from toolz import compose
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer


# Config variables
ROOT_DIR = Path(__file__).parent.resolve()
ARTIFACTS_DIR = ROOT_DIR / 'artifacts'
RUNS_DIR = ROOT_DIR / 'mlruns'
DATA_DIR = Path(__file__).parents[1].resolve() / 'data'


def add_pickup_dropoff_pair(df):
    """Add product of pickup and dropoff locations."""
    
    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    return df


def preprocess_test(df, dict_vectorizer, transforms, categorical, numerical):
    """Preprocess raw data in dataframe for inference."""
    
    # Apply in-between transformations
    df = compose(*transforms[::-1])(df)

    # For dict vectorizer: int = ignored, str = one-hot
    df[categorical] = df[categorical].astype(str)

    # Convert dataframe to feature dictionaries and transform
    feature_dicts = df[categorical + numerical].to_dict(orient='records')
    X = dict_vectorizer.transform(feature_dicts)

    return X


def preprocess_train(df, transforms, categorical, numerical):
    """Return processed features dict and target."""

    # Add target column; filter outliers
    # New data has no access to these datetime columns
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    # Apply in-between transformations
    df = compose(*transforms[::-1])(df)

    # For dict vectorizer: int = ignored, str = one-hot
    df[categorical] = df[categorical].astype(str)

    # Convert dataframe to feature dictionaries
    feature_dicts = df[categorical + numerical].to_dict(orient='records')
    target = df.duration.values

    return feature_dicts, target


def plot_duration_distribution(model, X_train, y_train, X_valid, y_valid):
    """Plot true and prediction distribution."""
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    sns.histplot(model.predict(X_train), ax=ax[0], label='pred', color='C0', stat='density', kde=True)
    sns.histplot(y_train, ax=ax[0], label='true', color='C1', stat='density', kde=True)
    ax[0].set_title("Train")
    ax[0].legend()

    sns.histplot(model.predict(X_valid), ax=ax[1], label='pred', color='C0', stat='density', kde=True)
    sns.histplot(y_valid, ax=ax[1], label='true', color='C1', stat='density', kde=True)
    ax[1].set_title("Valid")
    ax[1].legend()

    fig.tight_layout()
    return fig


def set_datasets(train_data_path, valid_data_path):
    """Processes datasets for model training and saves artifacts."""

    # In-between transformations
    transforms = [add_pickup_dropoff_pair]
    categorical = ['PU_DO']
    numerical = ['trip_distance']

    train_dicts, y_train = preprocess_train(pd.read_parquet(train_data_path), transforms, categorical, numerical)
    valid_dicts, y_valid = preprocess_train(pd.read_parquet(valid_data_path), transforms, categorical, numerical)

    # Fit all possible categories
    dv = DictVectorizer()
    dv.fit(train_dicts + valid_dicts)

    X_train = dv.transform(train_dicts)
    X_valid = dv.transform(valid_dicts)

    # Save artifacts
    joblib.dump(dv, ARTIFACTS_DIR / 'dict_vectorizer.pkl')
    joblib.dump(transforms, ARTIFACTS_DIR / 'transforms.pkl')
    joblib.dump(categorical, ARTIFACTS_DIR / 'categorical.pkl')
    joblib.dump(numerical, ARTIFACTS_DIR / 'numerical.pkl')

    return X_train, y_train, X_valid, y_valid
