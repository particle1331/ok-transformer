import base64
import json

import boto3
import mlflow


def load_model(model_location):
    """Load MLflow model from path."""
    model = mlflow.pyfunc.load_model(model_location)
    return model


def base64_decode(encoded_data):
    decoded_data = base64.b64decode(encoded_data).decode("utf-8")
    ride_event = json.loads(decoded_data)
    return ride_event


class ModelService:
    def __init__(self, model, model_version, callbacks=None):
        self.model = model
        self.model_version = model_version
        self.callbacks = callbacks or []

    def prepare_features(self, ride):
        features = {}
        features["PU_DO"] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
        features["trip_distance"] = ride["trip_distance"]
        return features

    def predict(self, features):
        pred = self.model.predict(features)
        return float(pred[0])

    def lambda_handler(self, event):

        predictions_events = []

        for record in event["Records"]:
            encoded_data = record["kinesis"]["data"]
            ride_event = base64_decode(encoded_data)

            ride = ride_event["ride"]
            ride_id = ride_event["ride_id"]

            features = self.prepare_features(ride)
            prediction = self.predict(features)

            prediction_event = {
                "model": "ride_duration_prediction_model",
                "version": self.model_version,
                "prediction": {
                    "ride_duration": prediction,
                    "ride_id": ride_id,
                },
            }

            for callback in self.callbacks:
                callback(prediction_event)

            predictions_events.append(prediction_event)

        return {"predictions": predictions_events}


class KinesisCallback:
    # pylint: disable=too-few-public-methods

    def __init__(self, kinesis_client, predictions_stream_name):
        self.kinesis_client = kinesis_client
        self.predictions_stream_name = predictions_stream_name

    def put_record(self, prediction_event):
        ride_id = prediction_event["prediction"]["ride_id"]
        self.kinesis_client.put_record(
            StreamName=self.predictions_stream_name,
            Data=json.dumps(prediction_event),
            PartitionKey=str(ride_id),
        )


def init(
    predictions_stream_name: str,
    model_location: str,
    model_version: str,
    test_run: bool,
):
    """Initialize model_service for lambda_function module."""

    model = load_model(model_location)

    callbacks = []
    if not test_run:
        kinesis_client = boto3.client("kinesis")
        kinesis_callback = KinesisCallback(kinesis_client, predictions_stream_name)
        callbacks.append(kinesis_callback.put_record)

    model_service = ModelService(
        model=model,
        model_version=model_version,
        callbacks=callbacks,
    )

    return model_service
