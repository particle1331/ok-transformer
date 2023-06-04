from pathlib import Path

import mlflow
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer

from ride_duration.utils import convert_to_dict

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"
EXPERIMENT_NAME = "nyc-green-taxi"
TRACKING_URI = f"http://127.0.0.1:5001"


def setup_experiment():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def fixtures():
    train_data_path = DATA_DIR / "green_tripdata_2021-01.parquet"
    valid_data_path = DATA_DIR / "green_tripdata_2021-02.parquet"
    train_data = pd.read_parquet(train_data_path)
    valid_data = pd.read_parquet(valid_data_path)

    return {
        "train_data_path": train_data_path,
        "valid_data_path": valid_data_path,
        "train_data": train_data,
        "valid_data": valid_data,
    }


def create_pipeline(model):
    return make_pipeline(
        FunctionTransformer(convert_to_dict),
        DictVectorizer(),
        model,
    )


def create_feature_pipeline():
    return make_pipeline(
        FunctionTransformer(convert_to_dict),
        DictVectorizer(),
    )
