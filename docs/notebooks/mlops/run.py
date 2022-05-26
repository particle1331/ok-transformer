import pandas as pd
import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt

from toolz import compose
from functools import partial

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

import warnings
from pandas.core.common import SettingWithCopyWarning
from matplotlib_inline import backend_inline
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
backend_inline.set_matplotlib_formats('svg')


def add_pickup_dropoff_pair(df):
    """Add product of pickup and dropoff locations."""
    
    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    return df


def preprocess_dataset(filename, transforms, categorical, numerical):
    """Return processed features dict and target."""
    
    # Load dataset
    df = pd.read_parquet(filename)

    # Add target column; filter outliers
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
    ax[0].legend();

    sns.histplot(model.predict(X_valid), ax=ax[1], label='pred', color='C0', stat='density', kde=True)
    sns.histplot(y_valid, ax=ax[1], label='true', color='C1', stat='density', kde=True)
    ax[1].set_title("Valid")
    ax[1].legend();

    fig.tight_layout()
    return fig


import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")


with mlflow.start_run(run_name='demo'):

    train_data_path = './data/green_tripdata_2021-01.parquet'
    valid_data_path = './data/green_tripdata_2021-02.parquet'
    alpha = 0.01

    # In-between transformations
    transforms = [add_pickup_dropoff_pair]
    categorical = ['PU_DO']
    numerical = ['trip_distance']

    train_dicts, y_train = preprocess_dataset(train_data_path, transforms, categorical, numerical)
    valid_dicts, y_valid = preprocess_dataset(valid_data_path, transforms, categorical, numerical)

    # Fit all possible categories
    dv = DictVectorizer()
    dv.fit(train_dicts + valid_dicts)

    X_train = dv.transform(train_dicts)
    X_valid = dv.transform(valid_dicts)

    # Train model
    model = Lasso(alpha)
    model.fit(X_train, y_train);

    # Plot predictions vs ground truth
    fig = plot_duration_distribution(model, X_train, y_train, X_valid, y_valid)
    fig.savefig('models/plot.svg');

    # Print metric
    rmse_train = mean_squared_error(y_train, model.predict(X_train), squared=False)
    rmse_valid = mean_squared_error(y_valid, model.predict(X_valid), squared=False)

    # MLFlow logging
    mlflow.set_tag("author", "particle")
    mlflow.log_param('train_data_path', train_data_path)
    mlflow.log_param('valid_data_path', valid_data_path)
    mlflow.log_param('alpha', alpha)
    mlflow.log_metric('rmse_train', rmse_train)
    mlflow.log_metric('rmse_valid', rmse_valid)
    mlflow.log_artifact('models/plot.svg')
