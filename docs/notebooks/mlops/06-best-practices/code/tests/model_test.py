import model
import pathlib


def read_text(file):
    test_directory = pathlib.Path(__file__).parent
    with open(test_directory / file, 'rt', encoding='utf-8') as f_in:
        return f_in.read().strip()


class ModelMock:
    # pylint: disable=too-few-public-methods
    
    def __init__(self, value):
        self.value = value 

    def predict(self, X):
        n = len(X)
        return [self.value] * n


def test_prepare_features():
    """Test preprocessing."""

    ride = {
        "PULocationID": 130,
        "DOLocationID": 205,
        "trip_distance": 3.66
    }

    model_service = model.ModelService(model=None, model_version=None)
    actual_features = model_service.prepare_features(ride)
    
    expected_features = {
        'PU_DO': '130_205',
        'trip_distance': 3.66,
    }

    assert actual_features == expected_features


def test_base64_decode():

    base64_input = read_text('data.b64')
    
    actual_result = model.base64_decode(base64_input)
    
    expected_result = {
        "ride": {
            "PULocationID": 130,
            "DOLocationID": 205,
            "trip_distance": 3.66
        }, 
        "ride_id": 123
    }

    assert actual_result == expected_result


def test_predict():
    
    features = {
        'PU_DO': '130_205',
        'trip_distance': 3.66,
    }

    model_mock = ModelMock(value=10.0)
    model_service = model.ModelService(model=model_mock, model_version=None)

    actual_result = model_service.predict(features)
    expected_result = 10.0

    assert actual_result == expected_result


def test_lambda_handler():
    
    event = {
        "Records": [
            {
                "kinesis": {
                    "data": read_text("data.b64"),
                },
            }
        ]
    }

    model_mock = ModelMock(value=10.0)
    model_service = model.ModelService(model=model_mock, model_version="model-mock")

    actual_result = model_service.lambda_handler(event)
    expected_result = {
        'predictions': [
            {
                'model': 'ride_duration_prediction_model', 
                'version': 'model-mock', 
                'prediction': {
                    'ride_duration': 10.0,
                    'ride_id': 123
                }
            }
        ]
    }

    assert actual_result == expected_result
