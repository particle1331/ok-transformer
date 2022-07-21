from model import ModelService
from model import base64_decode


def test_base64_decode():
    base64_input = "eyAgICAgICAgICAicmlkZSI6IHsgICAgICAgICAgICAgICJQVUxvY2F0aW9uSUQiOiAxMzAsICAgICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LCAgICAgICAgICAgICAgInRyaXBfZGlzdGFuY2UiOiAzLjY2ICAgICAgICAgIH0sICAgICAgICAgICJyaWRlX2lkIjogMTIzICAgICAgfQ=="

    actual_result = base64_decode(base64_input)
    expected_result = {
        "ride": {
            "PULocationID": 130,
            "DOLocationID": 205,
            "trip_distance": 3.66,
        },
        "ride_id": 123,
    }

    assert actual_result == expected_result


def test_prepare_features():
    """Test preprocessing."""

    ride = {
        "PULocationID": 140,
        "DOLocationID": 205,
        "trip_distance": 2.05
    }

    model_service = ModelService(model=None)

    actual_features = model_service.prepare_features(ride)
    
    expected_features = {
        'PU_DO': '140_205',
        'trip_distance': 2.05,
    }

    assert actual_features == expected_features
