import joblib
import mlflow
import pandas as pd
import os 
import requests

from ride_duration.utils import package_dir, prepare_features
from typing import Union
from mlflow.tracking import MlflowClient


def load_model():
    """Get latest production model from tracking server 
    or specific model from S3 bucket if server is down."""

    try:
        TRACKING_SERVER_HOST = os.getenv('TRACKING_SERVER_HOST')
        TRACKING_URI = f"http://{TRACKING_SERVER_HOST}:5000"

        # Check availability of API
        response = requests.head(TRACKING_URI)
        if response.status_code != 200:
            raise Exception(f"Tracking server unavailable: HTTP response code {response.status_code}")
            
    except:
        EXPERIMENT_ID = os.getenv('EXPERIMENT_ID')
        RUN_ID = os.getenv('MODEL_RUN_ID')
        
        print(f"Downloading model {RUN_ID}...")
        source = f"s3://mlflow-models-ron/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model"

    else:
        mlflow.set_tracking_uri(TRACKING_URI)
        client = MlflowClient(tracking_uri=TRACKING_URI)
        logged_model = client.get_latest_versions(name='NYCRideDurationModel', stages=['Production'])[0]
        
        print("Downloading model {RUN_ID} (latest, production)...")
        RUN_ID = logged_model.run_id
        source = logged_model.source
    
    model = mlflow.pyfunc.load_model(source)
    return model, RUN_ID


def make_prediction(model, input_data: Union[list[dict], pd.DataFrame]):
    """Make prediction from features dict or DataFrame."""
    
    X = prepare_features(input_data)
    preds = model.predict(X)

    return preds
