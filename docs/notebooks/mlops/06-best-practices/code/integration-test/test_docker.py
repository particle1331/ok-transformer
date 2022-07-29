import json
import requests

from deepdiff import DeepDiff


with open("event.json", "rt", encoding="utf-8") as f_in:
    event = json.load(f_in)


url = "http://localhost:8080/2015-03-31/functions/function/invocations"
actual_response = requests.post(url, json=event).json()

print("actual response:")
print(json.dumps(actual_response, indent=4))

expected_response = {
    "predictions": [
        {
            "model": "ride_duration_prediction_model",
            "version": "localtest",
            "prediction": {
                "ride_duration": 18.2536313889,
                "ride_id": 123,
            },
        }
    ]
}

diff = DeepDiff(expected_response, actual_response, math_epsilon=1e-7)
print("\ndiff:")
print(json.dumps(diff, indent=4))

assert len(diff) == 0
