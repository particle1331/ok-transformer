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
            
        # Fetch production model from client
        mlflow.set_tracking_uri(TRACKING_URI)
        client = MlflowClient(tracking_uri=TRACKING_URI)
        prod_model = client.get_latest_versions(name='NYCRideDurationModel', stages=['Production'])[0]

    except:
        EXPERIMENT_ID = os.getenv('EXPERIMENT_ID')
        RUN_ID = os.getenv('MODEL_RUN_ID')
        source = f"s3://mlflow-models-ron/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model"
        print(f"Downloading model {RUN_ID} from S3...")
        
    else:
        RUN_ID = prod_model.run_id
        source = prod_model.source
        print(f"Downloading model {RUN_ID} (latest, production)...")
    
    model = mlflow.pyfunc.load_model(source)
    return model, RUN_ID


def make_prediction(model, input_data: Union[list[dict], pd.DataFrame]):
    """Make prediction from features dict or DataFrame."""
    
    X = prepare_features(input_data)
    preds = model.predict(X)

    return preds
