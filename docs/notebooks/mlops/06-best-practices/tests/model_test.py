from model import ModelService



def test_prepare_features():
    """Test preprocessing."""

    ride = {
        "PULocationID": 140,
        "DOLocationID": 205,
        "trip_distance": 2.05
    }

    model = ModelService(model=None)

    actual_features = model.prepare_features(ride)
    
    expected_features = {
        'PU_DO': '140_205',
        'trip_distance': 2.05,
    }

    assert actual_features == expected_features
