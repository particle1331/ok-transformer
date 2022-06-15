import joblib
import mlflow
import pandas as pd
from ride_model.utils import package_dir, prepare_features
from typing import Union


def load_model():
    return joblib.load(package_dir / 'pipeline.pkl')


def load_mlflow_model(run_id):
    logged_model = f's3://mlflow-models-alexey/1/{run_id}/artifacts/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def make_prediction(model, input: Union[list[dict], pd.DataFrame]):
    X = prepare_features(input)
    preds = model.predict(X)
    return preds
