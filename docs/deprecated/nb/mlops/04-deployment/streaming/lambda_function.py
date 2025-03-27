# lambda_function.py
import json
import base64
import boto3
import os
import mlflow

from ride_duration.predict import load_model
from ride_duration.utils import prepare_features


# Load environmental variables
PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'ride_predictions')
TEST_RUN = os.getenv('TEST_RUN', 'False')
RUN_ID = os.getenv('RUN_ID')
EXPERIMENT_ID = os.getenv('EXPERIMENT_ID')

# Load the model from S3
model = load_model(experiment_id=EXPERIMENT_ID, run_id=RUN_ID)


def predict(features):
    prediction = model.predict(features)
    return float(prediction[0])


def lambda_handler(event, context):
    
    prediction_events = []

    for record in event['Records']:
        encoded_data = record['kinesis']['data']
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        
        ride_event = json.loads(decoded_data)
        ride_data = ride_event['ride']
        ride_id = ride_event['ride_id']
    
        features = prepare_features([ride_data])
        prediction = predict(features)
        
        prediction_event = {
            'model': 'ride_duration_prediction_model',
            'version': RUN_ID,
            'prediction': {
                'ride_duration': prediction,
                'ride_id': ride_id
            }
        }

        if TEST_RUN == 'False':
            kinesis_client = boto3.client('kinesis')

            # This is just the Python client version of the `aws kinesis put-record` CLI command.
            # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/kinesis.html#Kinesis.Client.put_record
            kinesis_client.put_record(
                StreamName=PREDICTIONS_STREAM_NAME,
                Data=json.dumps(prediction_event),
                PartitionKey=str(ride_id)
            )
        
        prediction_events.append(prediction_event)

    return {
        'predictions': prediction_events
    }
