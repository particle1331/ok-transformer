import model
import os


PREDICTIONS_STREAM_NAME = os.getenv("PREDICTIONS_STREAM_NAME", "ride_predictions")
MODEL_LOCATION = os.getenv("MODEL_LOCATION")
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "mlflow-models-ron")
EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID", "1")
RUN_ID = os.getenv("RUN_ID")
TEST_RUN = os.getenv("TEST_RUN", "False") == "True"


if MODEL_LOCATION is None:
    logged_model = f"s3://{MODEL_BUCKET}/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model"
    model_version = RUN_ID
else:
    model_location = MODEL_LOCATION
    model_version = "localtest"


model_service = model.init(
    predictions_stream_name=PREDICTIONS_STREAM_NAME,
    model_location=model_location,
    model_version=model_version,
    test_run=TEST_RUN,
)


def lambda_handler(event, context):
    return model_service.lambda_handler(event)
