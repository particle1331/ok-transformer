import joblib
import mlflow
import pandas as pd
import requests

from ride_duration.utils import package_dir, prepare_features
from typing import Union
from mlflow.tracking import MlflowClient


def load_model(experiment_id, run_id):
    """Get model from our S3 artifacts store."""

    source = f"s3://mlflow-models-ron/{experiment_id}/{run_id}/artifacts/model"
    model = mlflow.pyfunc.load_model(source)
    
    return model


def make_prediction(model, input_data: Union[list[dict], pd.DataFrame]):
    """Make prediction from features dict or DataFrame."""
    
    X = prepare_features(input_data)
    preds = model.predict(X)

    return preds
