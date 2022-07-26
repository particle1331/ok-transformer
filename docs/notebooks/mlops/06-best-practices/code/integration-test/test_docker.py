# test_docker.py
import json
import requests

from deepdiff import DeepDiff

event = {
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49630706038424016596026506533782471779140474214180454402",
                "data": "eyAgICAgICAgICAicmlkZSI6IHsgICAgICAgICAgICAgICJQVUxvY2F0aW9uSUQiOiAxMzAsICAgICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LCAgICAgICAgICAgICAgInRyaXBfZGlzdGFuY2UiOiAzLjY2ICAgICAgICAgIH0sICAgICAgICAgICJyaWRlX2lkIjogMTIzICAgICAgfQ==",
                "approximateArrivalTimestamp": 1655944485.718
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49630706038424016596026506533782471779140474214180454402",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::241297376613:role/lambda-kinesis-role",
            "awsRegion": "us-east-1",
            "eventSourceARN": "arn:aws:kinesis:us-east-1:241297376613:stream/ride_events"
        }
    ]
}


url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
actual_response = requests.post(url, json=event).json()

print('actual response:')
print(json.dumps(actual_response, indent=4))

expected_response = {
    'predictions': [
        {
            'model': 'ride_duration_prediction_model', 
            'version': 'localtest',
            'prediction': {
                'ride_duration': 18.2536313889,
                'ride_id': 123
            }
        }
    ]
}

diff = DeepDiff(expected_response, actual_response, math_epsilon=1e-7)
print('\ndiff:')
print(json.dumps(diff, indent=4))

assert len(diff) == 0
