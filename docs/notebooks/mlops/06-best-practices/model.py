import json
import mlflow
import base64


def load_model(run_id):
    model_path = f's3://mlflow-models-ron/1/{run_id}/artifacts/model'
    model = mlflow.pyfunc.load_model(model_path)
    return model


def base64_decode(encoded_data):
    decoded_data = base64.b64decode(encoded_data).decode('utf-8')
    ride_event = json.loads(decoded_data)
    return ride_event


class ModelService:

    def __init__(self, model, model_version=None):
        self.model = model
        self.model_version = model_version

    def prepare_features(self, ride):
        features = {}
        features['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
        features['trip_distance'] = ride['trip_distance']
        return features

    def predict(self, features):
        pred = self.model.predict(features)
        return float(pred[0])

    def lambda_handler(self, event):
        """Predict on batch of input events."""

        predictions_events = []

        for record in event['Records']:

            # Decode data from input kinesis stream
            encoded_data = record['kinesis']['data']
            ride_event = base64_decode(encoded_data)

            # Pickout id to match input to output
            ride = ride_event['ride']
            ride_id = ride_event['ride_id']

            # Make predictions using model
            features = self.prepare_features(ride)
            prediction = self.predict(features)

            # Package prediction event for output stream
            prediction_event = {
                'model': 'ride_duration_prediction_model',
                'version': self.model_version,
                'prediction': {'ride_duration': prediction, 'ride_id': ride_id},
            }

            predictions_events.append(prediction_event)

        return {'predictions': predictions_events}


def init(run_id: str):
    """Initialize model service."""

    model = load_model(run_id)
    model_service = ModelService(model=model, model_version=run_id)

    return model_service
